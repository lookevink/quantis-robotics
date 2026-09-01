"""Counterfactual Isaac safety evaluation for a shadow-only JEPA-WM candidate."""

from __future__ import annotations

from time import time
from typing import Any

import numpy as np

from jepa_wm.action import MAX_GRIPPER_WIDTH_M
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from sim.control_session import CONTROL_ROOT, ControlSession
from sim.isaac_control_execution import ExecutionSafetyContext, select_safe_projection
from sim.isaac_demo_runtime import JointCommand


async def evaluate_shadow_candidate(session_id: str) -> dict[str, Any]:
    """Evaluate the planned candidate against the captured state without actuation."""

    import omni.usd
    from isaacsim.core.simulation_manager import SimulationManager

    from sim.isaac_unknown_start_shadow import (
        reauthenticate_unknown_start_shadow_session,
    )

    reauthenticate_unknown_start_shadow_session(session_id)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    observation, persisted_state = session.load_capture()
    direct = session.load_response()
    shadow = session.load_shadow()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    omni.usd.get_context().get_stage()

    current = JointCommand(
        np.asarray(persisted_state.current_joint_positions, dtype=np.float64),
        (1.0 - observation.pose.values[6]) * MAX_GRIPPER_WIDTH_M,
    )
    safety = ExecutionSafetyContext(
        observation=observation,
        current=current,
        observed_joint_positions=persisted_state.current_joint_positions,
        contact_force_newtons=persisted_state.contact_force_newtons,
        collision_detected=persisted_state.collision_detected,
        limits=SimulatorSafetyLimits(),
    )
    planned = direct.with_actions(shadow.planned.actions)
    attempts, selected = select_safe_projection(
        safety,
        planned,
        now_unix_seconds=direct.created_at_unix_seconds,
    )
    evidence = ShadowSafetyEvidence(
        observation_id=observation.observation_id,
        evaluated_at_unix_seconds=time(),
        counterfactual_as_of_unix_seconds=direct.created_at_unix_seconds,
        planned_actions=shadow.planned.actions,
        attempts=attempts,
        selected_action_scale=(attempts[-1].scale if selected is not None else None),
    )
    session.write_shadow_safety(evidence)
    return evidence.to_dict()
