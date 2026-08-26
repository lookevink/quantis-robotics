"""Capture a fresh follow-up observation without resetting the live stage."""

from __future__ import annotations

from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from jepa.contract import ObservationStage
from jepa_wm.action import ActionSelectionBounds, DroidPose, action_between
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.control_policy import (
    ControlExecutionPolicy,
    is_insertion_trial_execution_policy,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.trajectory import load_rollout_at
from sim.control_context import recording_task
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.direct_safety import ControlSafetySnapshot
from jepa_wm.control_tracking import ActionTrackingLimits
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    ControlCaptureResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
    InsertionFollowupLineage,
    PostActionEvidence,
)
from sim.control_identity import control_proposal_path, observation_id_for_session
from sim.demo_sequence import Phase
from sim.isaac_control_runtime import (
    bind_live_runtime,
    contact_sensor,
    live_runtime_for,
    read_control_contact,
    synchronized_insertion_safety_snapshot,
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

if TYPE_CHECKING:
    from jepa_wm.control_rollout import ControlStepSummary


def build_insertion_followup_capture(
    session_id: str,
    lineage: InsertionFollowupLineage,
    *,
    captured_at_unix_seconds: float,
    context_frame: Path,
    target: ControlTarget,
    current: ControlSafetySnapshot,
    current_pose: DroidPose,
    active_drive_target: JointDriveTarget,
) -> tuple[ControlObservation, ControlSessionState]:
    """Bind one no-actuation observation to an exact applied insertion result."""

    validate_recording_id(session_id)
    if (
        current.collision_detected
        or current.contact_force_newtons
        > SimulatorSafetyLimits().maximum_contact_force_newtons
        or not current.plug_attached
    ):
        raise ValueError("follow-up capture requires one safe applied insertion result")
    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=captured_at_unix_seconds,
        context_frame=context_frame,
        target=target,
        expected_proposal=lineage.observation.expected_proposal,
        pose=current_pose,
        previous_action=action_between(lineage.observation.pose, current_pose),
        warmup_frames=lineage.observation.warmup_frames + 1,
    )
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=lineage.state.reference_recording,
        seed=lineage.state.seed,
        recording=lineage.state.recording,
        current_joint_positions=current.joint_positions,
        collision_detected=current.collision_detected,
        contact_force_newtons=current.contact_force_newtons,
        previous_session_id=lineage.result.session_id,
        execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
        plug_position=current.plug_position,
        plug_attached=current.plug_attached,
        current_gripper_width_m=current.gripper_width_m,
        insertion_target_policy=lineage.state.insertion_target_policy,
        active_drive_target=active_drive_target,
    )
    lineage.validate_source(observation, state)
    return observation, state


def validate_followup_continuity(
    previous: PostActionEvidence,
    current: JointCommand,
    current_pose: DroidPose,
    *,
    current_plug_position: tuple[float, ...] | None = None,
    current_plug_attached: bool = False,
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
    plug_drift = (
        float(
            np.linalg.norm(
                np.asarray(current_plug_position)
                - np.asarray(previous.plug_position)
            )
        )
        if previous.plug_position is not None and current_plug_position is not None
        else 0.0
    )
    if (
        joint_drift > safety_limits.maximum_observation_joint_drift_radians
        or translation_drift > tracking_limits.maximum_translation_error_meters
        or rotation_drift > tracking_limits.maximum_rotation_error_radians
        or abs(pose_drift.values[6]) > tracking_limits.maximum_gripper_error
        or previous.plug_attached != current_plug_attached
        or (previous.plug_position is None) != (current_plug_position is None)
        or plug_drift > safety_limits.maximum_observation_plug_drift_meters
    ):
        raise ValueError("live stage no longer matches the previous applied result")


async def _capture_generic_followup_observation(
    session: ControlSession,
    previous_step: ControlStepSummary,
) -> dict[str, Any]:
    """Preserve the established non-insertion receding-horizon workflow."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager

    previous_observation = previous_step.observation
    previous_state = previous_step.state
    previous_result = previous_step.result
    if previous_result.post_action is None:
        raise ValueError("applied previous step has no post-action evidence")
    previous_session_id = previous_state.session_id
    reference_path = (
        QUANTIS_DATA_ROOT / "recordings" / previous_state.reference_recording
    )
    next_context_index = previous_observation.warmup_frames + 1
    target = previous_observation.target
    if recording_task(reference_path) == GRASP_TASK_ID:
        reference_rollout = load_rollout_at(
            reference_path,
            camera="wrist",
            context_index=next_context_index,
            bounds=ActionSelectionBounds(minimum_action_norm=0.0),
        )
        target = ControlTarget(
            reference_rollout.target.path.relative_to(QUANTIS_DATA_ROOT),
            reference_rollout.target_pose,
        )

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
        collision_detected, contact_force = read_control_contact(sensor)
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
            current_plug_position=tuple(float(value) for value in snapshot.plug_position),
            current_plug_attached=snapshot.plug_attached,
        )
        captured_at = time()
    finally:
        timeline.pause()

    observation = ControlObservation(
        observation_id=observation_id_for_session(session.session_id),
        captured_at_unix_seconds=captured_at,
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=target,
        expected_proposal=previous_observation.expected_proposal,
        pose=snapshot.end_effector_pose,
        previous_action=action_between(
            previous_observation.pose, snapshot.end_effector_pose
        ),
        warmup_frames=next_context_index,
    )
    state = ControlSessionState(
        session_id=session.session_id,
        reference_recording=previous_state.reference_recording,
        seed=previous_state.seed,
        recording=previous_state.recording,
        current_joint_positions=tuple(current.arm_positions),
        collision_detected=collision_detected,
        contact_force_newtons=contact_force,
        previous_session_id=previous_session_id,
        execution_policy=previous_state.execution_policy,
        plug_position=tuple(float(value) for value in snapshot.plug_position),
        plug_attached=snapshot.plug_attached,
        current_gripper_width_m=current.gripper_width_m,
    )
    session.write_capture(observation, state)
    bind_live_runtime(session.session_id, stage, actuators, attachment, sensor)
    return ControlCaptureResult(
        session.session_id,
        observation,
        session.request_path,
        contact_force,
        collision_detected,
    ).to_dict()


async def capture_followup_observation(
    session_id: str,
    previous_session_id: str,
    proposal_name: str,
) -> dict[str, Any]:
    """Persist the current live pose/frame as the next single-use request."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    validate_recording_id(proposal_name)
    previous = ControlSession.at(CONTROL_ROOT, previous_session_id)
    from jepa_wm.control_rollout import ControlStepSummary

    previous_step = ControlStepSummary.from_session(previous)
    previous_result = previous_step.result
    previous_observation = previous_step.observation
    previous_state = previous_step.state
    if previous_result.status is not ControlResultStatus.APPLIED:
        raise ValueError("follow-up control requires an applied previous step")
    expected_proposal = control_proposal_path(proposal_name)
    if previous_observation.expected_proposal != expected_proposal:
        raise ValueError("follow-up proposal differs from the rollout checkpoint")
    next_context_index = previous_observation.warmup_frames + 1
    reference_path = QUANTIS_DATA_ROOT / "recordings" / previous_state.reference_recording
    reference_task = recording_task(reference_path)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    if reference_task != INSERTION_TASK_ID:
        return await _capture_generic_followup_observation(
            session,
            previous_step,
        )
    lineage = InsertionFollowupLineage(
        previous_observation,
        previous_state,
        previous_result,
    )
    target_policy = previous_state.insertion_target_policy
    if target_policy is None:
        raise ValueError("insertion follow-up requires its persisted target policy")
    reference_rollout = target_policy.select(
        reference_path,
        context_index=next_context_index,
    )
    target = ControlTarget(
        reference_rollout.target.path.relative_to(QUANTIS_DATA_ROOT),
        reference_rollout.target_pose,
    )

    if session.path.exists():
        raise ValueError(f"control session already exists: {session_id}")
    timeline = omni.timeline.get_timeline_interface()
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(previous_session_id, stage)
    if runtime is None:
        raise RuntimeError("live insertion runtime was lost before follow-up capture")
    if previous_result.post_action is None:
        raise ValueError("applied previous step has no post-action evidence")
    captured_state = previous_result.post_action.require_safety_snapshot()
    synchronized = await synchronized_insertion_safety_snapshot(
        runtime,
        timeline,
        omni.kit.app.get_app().next_update_async,
        captured_state,
        SimulatorSafetyLimits(),
        operation="insertion follow-up capture synchronization",
    )
    if synchronized.pose is None:
        raise RuntimeError("live insertion pose was not refreshed")
    if synchronized.active_drive_target is None:
        raise RuntimeError("live insertion drive target was not refreshed")
    context_path = session.path / "context.png"
    await capture_camera_frame(JEPA_WM_CAMERA_SPECS[0], context_path)
    observation, state = build_insertion_followup_capture(
        session_id,
        lineage,
        captured_at_unix_seconds=time(),
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=target,
        current=synchronized.safety,
        current_pose=synchronized.pose,
        active_drive_target=synchronized.active_drive_target,
    )
    session.write_capture(observation, state)
    bind_live_runtime(
        session_id,
        stage,
        synchronized.runtime.actuators,
        synchronized.runtime.attachment,
        synchronized.runtime.sensor,
    )
    return ControlCaptureResult(
        session_id,
        observation,
        session.request_path,
        synchronized.safety.contact_force_newtons,
        synchronized.safety.collision_detected,
    ).to_dict()
