"""Capture a fresh follow-up observation without resetting the live stage."""

from __future__ import annotations

from time import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from jepa.contract import ObservationStage
from jepa_wm.action import DroidPose, action_between
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.control_tracking import ActionTrackingLimits
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    ControlCaptureResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
    PostActionEvidence,
)
from sim.demo_sequence import Phase
from sim.control_identity import control_proposal_path, observation_id_for_session
from sim.isaac_control_runtime import (
    bind_live_runtime,
    contact_sensor,
    live_runtime_for,
    read_contact,
)
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, capture_camera_frame
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    prepare_plug,
    recording_snapshot,
)
from sim.isaac_demo_scene import ROBOT_PATH
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id


def validate_followup_continuity(
    previous: PostActionEvidence,
    current: JointCommand,
    current_pose: DroidPose,
    *,
    safety_limits: SimulatorSafetyLimits = SimulatorSafetyLimits(),
    tracking_limits: ActionTrackingLimits = ActionTrackingLimits(),
) -> None:
    """Fail if the live articulation no longer matches the last applied result."""

    joint_drift = float(
        np.max(
            np.abs(
                np.asarray(current.arm_positions)
                - np.asarray(previous.joint_positions)
            )
        )
    )
    pose_drift = action_between(previous.pose, current_pose)
    translation_drift = float(np.linalg.norm(pose_drift.values[:3]))
    rotation_drift = float(
        Rotation.from_euler("xyz", pose_drift.values[3:6]).magnitude()
    )
    if (
        joint_drift > safety_limits.maximum_observation_joint_drift_radians
        or translation_drift > tracking_limits.maximum_translation_error_meters
        or rotation_drift > tracking_limits.maximum_rotation_error_radians
        or abs(pose_drift.values[6]) > tracking_limits.maximum_gripper_error
    ):
        raise ValueError("live stage no longer matches the previous applied result")


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
    previous_result = previous.load_result()
    if previous_result.status != ControlResultStatus.APPLIED:
        raise ValueError("follow-up control requires an applied previous step")
    if previous_result.post_action is None:
        raise ValueError("applied previous step has no post-action evidence")
    previous_observation, previous_state = previous.load_capture()
    expected_proposal = control_proposal_path(proposal_name)
    if previous_observation.expected_proposal != expected_proposal:
        raise ValueError("follow-up proposal differs from the rollout checkpoint")

    session = ControlSession.at(CONTROL_ROOT, session_id)
    session.create()
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(previous_session_id, stage)
    timeline.play()
    try:
        if runtime is None:
            if SimulationManager.get_physics_sim_view() is None:
                SimulationManager.initialize_physics()
            actuators = create_actuators(stage, Articulation(ROBOT_PATH))
            attachment = prepare_plug(stage)
            sensor = contact_sensor(stage, create=False)
        else:
            actuators = runtime.actuators
            attachment = runtime.attachment
            sensor = runtime.sensor
        context_path = session.path / "context.png"
        await omni.kit.app.get_app().next_update_async()
        if not actuators.articulation.is_physics_tensor_entity_valid():
            actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        await capture_camera_frame(JEPA_WM_CAMERA_SPECS[0], context_path)
        current = actuators.actual_command()
        collision_detected, contact_force = read_contact(sensor)
        snapshot = recording_snapshot(
            RecordingLabel(RecordingMoment.MOTION, Phase.READY),
            ObservationStage.APPROACHING_CABLE,
            current,
            attachment,
        )
        validate_followup_continuity(
            previous_result.post_action,
            current,
            snapshot.end_effector_pose,
        )
        captured_at = time()
    finally:
        timeline.pause()

    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=captured_at,
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=previous_observation.target,
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
        execution_policy=previous_state.execution_policy,
    )
    session.write_capture(observation, state)
    bind_live_runtime(session_id, stage, actuators, attachment, sensor)
    return ControlCaptureResult(
        session_id,
        observation,
        session.request_path,
        contact_force,
        collision_detected,
    ).to_dict()
