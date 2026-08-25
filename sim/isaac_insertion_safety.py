"""Live no-actuation safety evaluation for one direct insertion proposal."""

from __future__ import annotations

from time import time
from typing import Any

import numpy as np

from jepa_wm.control_safety import (
    INSERTION_TARGET_PROGRESS,
    SimulatorSafetyLimits,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.direct_safety import (
    DirectInsertionSafetyEvidence,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.training_artifact import ArtifactIdentity
from sim.control_context import recording_task
from sim.control_session import CONTROL_ROOT, RECORDING_ROOT, ControlSession
from sim.isaac_control_execution import ExecutionSafetyContext, select_safe_projection
from sim.isaac_control_runtime import (
    live_runtime_for,
    synchronized_insertion_safety_snapshot,
)
from sim.isaac_demo_runtime import JointCommand


async def evaluate_direct_insertion_candidate(session_id: str) -> dict[str, Any]:
    """Evaluate a fresh direct proposal against live insertion state without motion."""

    import omni.kit.app
    import omni.timeline
    import omni.usd

    session = ControlSession.at(CONTROL_ROOT, session_id)
    observation, persisted_state = session.load_capture()
    proposal = session.load_response()
    if recording_task(RECORDING_ROOT / persisted_state.reference_recording) != INSERTION_TASK_ID:
        raise ValueError("direct insertion safety requires an insertion reference")
    if proposal.proposal_fingerprint is None:
        raise ValueError("direct insertion safety requires an exact proposal identity")
    captured_state = persisted_state.require_safety_snapshot()
    if (
        persisted_state.execution_policy
        is not ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION
    ):
        raise ValueError("direct insertion safety requires its no-actuation policy")
    if not captured_state.plug_attached:
        raise ValueError("direct insertion safety requires an attached plug")

    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(session_id, stage)
    if runtime is None:
        raise RuntimeError("live insertion runtime was lost before safety evaluation")

    limits = SimulatorSafetyLimits()
    live_state = await synchronized_insertion_safety_snapshot(
        runtime,
        omni.timeline.get_timeline_interface(),
        omni.kit.app.get_app().next_update_async,
        captured_state,
        limits,
        operation="insertion safety synchronization",
    )

    evaluated_at = time()
    safety = ExecutionSafetyContext(
        observation=observation,
        current=JointCommand(
            np.asarray(live_state.joint_positions),
            live_state.gripper_width_m,
        ),
        observed_joint_positions=captured_state.joint_positions,
        contact_force_newtons=live_state.contact_force_newtons,
        collision_detected=live_state.collision_detected,
        limits=limits,
    )
    attempts, selected = select_safe_projection(
        safety,
        proposal,
        now_unix_seconds=evaluated_at,
        target_progress=INSERTION_TARGET_PROGRESS,
    )
    evidence = DirectInsertionSafetyEvidence(
        observation_id=observation.observation_id,
        evaluated_at_unix_seconds=evaluated_at,
        proposed_actions=proposal.actions,
        proposal=ArtifactIdentity(proposal.proposal, proposal.proposal_fingerprint),
        attempts=attempts,
        selected_action_scale=(attempts[-1].scale if selected is not None else None),
        live_state=live_state,
    )
    session.write_direct_safety(evidence)
    return evidence.to_dict()
