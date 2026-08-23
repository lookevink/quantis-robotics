"""Capture a fresh follow-up observation without resetting the live stage."""

from __future__ import annotations

from time import time
from typing import Any

from jepa.contract import ObservationStage
from jepa_wm.action import action_between
from jepa_wm.control_protocol import ControlObservation
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    ControlCaptureResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
)
from sim.demo_sequence import Phase
from sim.control_identity import control_proposal_path, observation_id_for_session
from sim.isaac_control_runtime import contact_sensor, read_contact
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, capture_camera_frame
from sim.isaac_demo_runtime import create_actuators, prepare_plug, recording_snapshot
from sim.isaac_demo_scene import ROBOT_PATH
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id


async def capture_followup_observation(
    session_id: str,
    previous_session_id: str,
    proposal_name: str,
) -> dict[str, Any]:
    """Persist the current live pose/frame as the next single-use request."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager

    validate_recording_id(proposal_name)
    previous = ControlSession.at(CONTROL_ROOT, previous_session_id)
    if previous.result_status() != ControlResultStatus.APPLIED:
        raise ValueError("follow-up control requires an applied previous step")
    previous_observation, previous_state = previous.load_capture()
    expected_proposal = control_proposal_path(proposal_name)
    if previous_observation.expected_proposal != expected_proposal:
        raise ValueError("follow-up proposal differs from the rollout checkpoint")

    session = ControlSession.at(CONTROL_ROOT, session_id)
    session.create()
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    try:
        stage = omni.usd.get_context().get_stage()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        attachment = prepare_plug(stage)
        sensor = contact_sensor(stage, create=False)
        await omni.kit.app.get_app().next_update_async()
        current = actuators.actual_command()
        collision_detected, contact_force = read_contact(sensor)
        snapshot = recording_snapshot(
            RecordingLabel(RecordingMoment.MOTION, Phase.READY),
            ObservationStage.APPROACHING_CABLE,
            current,
            attachment,
        )
        context_path = session.path / "context.png"
        await capture_camera_frame(JEPA_WM_CAMERA_SPECS[0], context_path)
    finally:
        timeline.pause()

    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=time(),
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target_frame=previous_observation.target_frame,
        expected_proposal=expected_proposal,
        pose=snapshot.end_effector_pose,
        previous_action=action_between(
            previous_observation.pose, snapshot.end_effector_pose
        ),
        warmup_frames=previous_observation.warmup_frames + 1,
    )
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=previous_state.reference_recording,
        seed=previous_state.seed,
        recording=previous_state.recording,
        current_joint_positions=tuple(current.arm_positions),
        collision_detected=collision_detected,
        contact_force_newtons=contact_force,
        previous_session_id=previous_session_id,
    )
    session.write_capture(observation, state)
    return ControlCaptureResult(
        session_id,
        observation,
        session.request_path,
        contact_force,
        collision_detected,
    ).to_dict()
