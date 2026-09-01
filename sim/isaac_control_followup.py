"""Capture a fresh follow-up observation without resetting the live stage."""

from __future__ import annotations

from base64 import b64decode
import json
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial.transform import Rotation

from jepa.contract import ObservationStage
from jepa_wm.action import (
    ActionSelectionBounds,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.contact_grasp_acquisition_handoff import (
    PROPOSAL_NAME as ACQUISITION_PROPOSAL_NAME,
    SOURCE_FINGERPRINTS as ACQUISITION_SOURCE_FINGERPRINTS,
    SOURCE_SESSION_ID as ACQUISITION_SOURCE_SESSION_ID,
    ContactGraspAcquisitionHandoff,
    runtime_fingerprint as acquisition_runtime_fingerprint,
)
from jepa_wm.contact_grasp_acquisition_continuation import (
    HANDOFF_SCHEMA as ACQUISITION_CONTINUATION_SCHEMA,
    PROPOSAL_NAME as ACQUISITION_CONTINUATION_PROPOSAL_NAME,
    SOURCE_FINGERPRINTS as ACQUISITION_CONTINUATION_SOURCE_FINGERPRINTS,
    SOURCE_SESSION_ID as ACQUISITION_CONTINUATION_SOURCE_SESSION_ID,
    ContactGraspAcquisitionContinuation,
    runtime_fingerprint as acquisition_continuation_runtime_fingerprint,
)
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.control_policy import (
    ControlExecutionPolicy,
    is_insertion_trial_execution_policy,
)
from jepa_wm.contact_grasp_target import ContactGraspTargetPolicy
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    INSERTION_TASK_ID,
    INSERTION_CONTROL_TARGET_POLICY,
    ContactInsertionSegment,
    InsertionControlTargetPolicy,
)
from jepa_wm.insertion_rollout import (
    DEMO_INSERTION_ROLLOUT,
    TWO_STEP_INSERTION_ROLLOUT,
    InsertionRolloutPosition,
    InsertionRolloutRoster,
)
from jepa_wm.insertion_trial import InsertionTrialRollbackEvidence
from jepa_wm.insertion_transition import (
    insertion_proposal_continuation_from_dict,
    resolve_insertion_followup_proposal,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.trajectory import load_rollout_at
from jepa_wm.training_artifact import artifact_fingerprint
from sim.control_context import recording_task
from jepa_wm.control_safety import ControlGateReason, SimulatorSafetyLimits
from jepa_wm.insertion_refresh import (
    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
    ControlSafetySnapshot,
)
from jepa_wm.control_tracking import ActionTrackingLimits
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    RECORDING_ROOT,
    ControlCaptureResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
    GraspToInsertionLineage,
    InsertionFollowupLineage,
    PostActionEvidence,
)
from sim.control_identity import control_proposal_path, observation_id_for_session
from sim.demo_sequence import Phase
from sim.isaac_control_runtime import (
    bind_live_runtime,
    contact_sensor,
    live_runtime_for,
    pause_control_timeline,
    read_control_contact,
    synchronized_insertion_frame_capture,
)
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, capture_camera_frame
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    prepare_plug,
    recording_snapshot,
    resume_live_simulation,
)
from sim.isaac_demo_scene import ROBOT_PATH
from sim.isaac_demo_kinematics import solve_droid_pose
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id

if TYPE_CHECKING:
    from jepa_wm.control_rollout import ControlRolloutReport, ControlStepSummary


def persist_insertion_proposal_handoff(
    previous_session_id: str,
    followup_session_id: str,
    encoded_evidence: str,
) -> dict[str, Any]:
    """Persist host-authenticated proposal provenance as the Isaac owner."""

    validate_recording_id(previous_session_id)
    validate_recording_id(followup_session_id)
    try:
        payload = json.loads(b64decode(encoded_evidence, validate=True).decode())
        handoff = insertion_proposal_continuation_from_dict(payload)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("insertion proposal handoff payload is invalid") from error
    previous = ControlSession.at(CONTROL_ROOT, previous_session_id)
    observation, _ = previous.load_capture()
    response = previous.load_response()
    handoff.resolve(
        observation.expected_proposal,
        response.proposal_fingerprint,
        handoff.requested.path,
    )
    output = previous.insertion_proposal_handoff_path(followup_session_id)
    if output.exists():
        raise ValueError("insertion proposal handoff already exists")
    write_json_atomic(output, handoff.to_dict())
    return {
        "status": "insertion_proposal_handoff_ready",
        "previous_session_id": previous_session_id,
        "followup_session_id": followup_session_id,
        "previous": handoff.previous.to_dict(),
        "requested": handoff.requested.to_dict(),
    }


def diagnose_control_ik_scales(session_id: str) -> dict[str, Any]:
    """Re-solve one blocked command at smaller scales without moving physics."""

    validate_recording_id(session_id)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    observation, state = session.load_capture()
    response = session.load_response()
    result = session.load_result()
    if (
        result.status is not ControlResultStatus.BLOCKED
        or result.gate.reasons
        != (ControlGateReason.JOINT_VELOCITY_VIOLATION,)
        or result.selected_action_scale is not None
        or result.post_action is not None
    ):
        raise ValueError("IK scale diagnosis requires one no-motion velocity block")
    scales = (
        DroidActionScale(0.03125, 0.125, 0.125),
        DroidActionScale(0.03125, 0.0625, 0.125),
        DroidActionScale(0.03125, 0.03125, 0.125),
        DroidActionScale(0.03125, 0.0, 0.125),
    )
    attempts = []
    current = np.asarray(state.current_joint_positions, dtype=np.float64)
    for scale in scales:
        candidate = observation.pose.applied(scale.apply(response.first_action))
        try:
            solved = solve_droid_pose(candidate, current)
            attempts.append(
                {
                    "scale": scale.to_dict(),
                    "solved": True,
                    "maximum_joint_delta_rad": float(
                        np.max(np.abs(solved.arm_positions - current))
                    ),
                    "position_error_m": solved.position_error_m,
                    "orientation_error_rad": solved.orientation_error_rad,
                    "joint_positions": solved.arm_positions.tolist(),
                }
            )
        except (RuntimeError, ValueError) as error:
            attempts.append(
                {
                    "scale": scale.to_dict(),
                    "solved": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return {
        "status": "diagnosed_no_actuation",
        "session": session_id,
        "source_reasons": [reason.value for reason in result.gate.reasons],
        "attempts": attempts,
        "simulator_action_authorized": False,
    }


async def capture_contact_grasp_acquisition_handoff(
    session_id: str,
    source_session_id: str,
    proposal_name: str,
    encoded_evidence: str,
) -> dict[str, Any]:
    """Start one frozen acquisition chain from an authenticated no-motion block."""

    import omni.kit.app
    import omni.timeline
    import omni.usd

    validate_recording_id(session_id)
    validate_recording_id(source_session_id)
    validate_recording_id(proposal_name)
    try:
        payload = json.loads(b64decode(encoded_evidence, validate=True).decode())
        continuation = payload.get("schema") == ACQUISITION_CONTINUATION_SCHEMA
        if continuation:
            handoff = ContactGraspAcquisitionContinuation.from_dict(payload)
            expected_source_session = ACQUISITION_CONTINUATION_SOURCE_SESSION_ID
            expected_proposal = ACQUISITION_CONTINUATION_PROPOSAL_NAME
            expected_fingerprints = ACQUISITION_CONTINUATION_SOURCE_FINGERPRINTS
            expected_runtime_fingerprint = (
                acquisition_continuation_runtime_fingerprint()
            )
            expected_gate_reason = ControlGateReason.JOINT_VELOCITY_VIOLATION
        else:
            handoff = ContactGraspAcquisitionHandoff.from_dict(payload)
            expected_source_session = ACQUISITION_SOURCE_SESSION_ID
            expected_proposal = ACQUISITION_PROPOSAL_NAME
            expected_fingerprints = ACQUISITION_SOURCE_FINGERPRINTS
            expected_runtime_fingerprint = acquisition_runtime_fingerprint()
            expected_gate_reason = ControlGateReason.GRIPPER_VIOLATION
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("contact-grasp acquisition handoff is invalid") from error
    if (
        source_session_id != expected_source_session
        or proposal_name != expected_proposal
        or handoff.followup_session_id != session_id
        or handoff.runtime_fingerprint != expected_runtime_fingerprint
    ):
        raise ValueError("contact-grasp acquisition handoff authority changed")

    source = ControlSession.at(CONTROL_ROOT, source_session_id)
    if any(
        artifact_fingerprint(source.path / name) != expected
        for name, expected in expected_fingerprints.items()
    ):
        raise ValueError("contact-grasp acquisition source changed")
    source_observation, source_state = source.load_capture()
    source_result = source.load_result()
    refresh = source_result.insertion_trial_refresh
    if (
        source_result.status is not ControlResultStatus.BLOCKED
        or source_result.gate.reasons != (expected_gate_reason,)
        or source_result.selected_action_scale is not None
        or source_result.post_action is not None
        or refresh is None
        or source_state.plug_attached
        or source_state.collision_detected
        or source_state.contact_force_newtons
        > SimulatorSafetyLimits().maximum_contact_force_newtons
        or source_state.active_drive_target is None
    ):
        raise ValueError("contact-grasp acquisition source is not a no-motion block")
    if continuation:
        expected_source_pose = refresh.live_pose
        expected_source_safety = refresh.live_state
        if (
            expected_source_safety.plug_attached
            or expected_source_safety.collision_detected
            or expected_source_safety.contact_force_newtons
            > SimulatorSafetyLimits().maximum_contact_force_newtons
        ):
            raise ValueError("contact-grasp continuation refresh is unsafe")
    else:
        if (
            refresh.live_pose != source_observation.pose
            or refresh.live_state != source_state.require_safety_snapshot()
        ):
            raise ValueError("contact-grasp acquisition source refresh changed")
        expected_source_pose = source_observation.pose
        expected_source_safety = source_state.require_safety_snapshot()

    session = ControlSession.at(CONTROL_ROOT, session_id)
    if session.path.exists():
        raise ValueError(f"control session already exists: {session_id}")
    handoff_path = source.contact_grasp_acquisition_handoff_path(session_id)
    if handoff_path.exists():
        raise ValueError("contact-grasp acquisition handoff already exists")
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(source_session_id, stage)
    if runtime is None:
        raise RuntimeError("live contact-grasp acquisition runtime was lost")
    context_path = session.path / "context.png"

    async def capture_frame(observe_safety) -> None:
        await capture_camera_frame(
            JEPA_WM_CAMERA_SPECS[0],
            context_path,
            observe_safety=observe_safety,
        )

    synchronized = await synchronized_insertion_frame_capture(
        runtime,
        omni.timeline.get_timeline_interface(),
        omni.kit.app.get_app().next_update_async,
        expected_source_safety,
        SimulatorSafetyLimits(),
        capture_frame,
        expected_active_drive_target=source_state.active_drive_target,
        operation="contact-grasp acquisition handoff capture",
        maximum_gripper_error_meters=(
            MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
        ),
    )
    if (
        synchronized.pose is None
        or synchronized.active_drive_target != source_state.active_drive_target
        or synchronized.safety.plug_attached
    ):
        raise RuntimeError("contact-grasp acquisition handoff state changed")
    source_policy = source_state.require_current_contact_grasp_policy()
    target_policy = ContactGraspTargetPolicy.for_scene_translation(
        source_policy.scene_translation_m
    )
    reference_path = QUANTIS_DATA_ROOT / "recordings" / source_state.reference_recording
    target = target_policy.initial_target(
        reference_path,
        frame_root=QUANTIS_DATA_ROOT,
        live_pose=synchronized.pose,
    )
    context_index = target_policy.context_index_for_target(target.frame)
    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=time(),
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=target,
        expected_proposal=control_proposal_path(proposal_name),
        pose=synchronized.pose,
        previous_action=action_between(expected_source_pose, synchronized.pose),
        warmup_frames=context_index,
    )
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=source_state.reference_recording,
        seed=source_state.seed,
        recording=source_state.recording,
        current_joint_positions=synchronized.safety.joint_positions,
        collision_detected=synchronized.safety.collision_detected,
        contact_force_newtons=synchronized.safety.contact_force_newtons,
        previous_session_id=source_session_id,
        execution_policy=ControlExecutionPolicy.DIRECT,
        plug_position=synchronized.safety.plug_position,
        plug_attached=synchronized.safety.plug_attached,
        current_gripper_width_m=synchronized.safety.gripper_width_m,
        active_drive_target=synchronized.active_drive_target,
        contact_grasp_target_policy=target_policy,
    )
    session.write_capture(observation, state)
    write_json_atomic(handoff_path, handoff.to_dict())
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


def restore_insertion_no_actuation_retry(
    previous_session_id: str,
    failed_safety_session_id: str,
    next_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Rebind a target-progress-only no-actuation capture to its predecessor."""

    import omni.usd

    previous = ControlSession.at(CONTROL_ROOT, previous_session_id)
    failed = ControlSession.at(CONTROL_ROOT, failed_safety_session_id)
    from jepa_wm.control_rollout import ControlStepSummary

    previous_step = ControlStepSummary.from_session(previous)
    lineage = InsertionFollowupLineage(
        previous_step.observation,
        previous_step.state,
        previous_step.result,
        next_maximum_steps,
    )
    observation, state = failed.load_capture()
    response = failed.load_response()
    safety = failed.load_direct_safety()
    proposal_changed = (
        previous_step.observation.expected_proposal
        != observation.expected_proposal
    )
    handoff = (
        previous.load_insertion_proposal_handoff(failed.session_id)
        if proposal_changed
        else None
    )
    expected_proposal = resolve_insertion_followup_proposal(
        previous_step.observation.expected_proposal,
        observation.expected_proposal,
        previous_proposal_fingerprint=previous_step.response.proposal_fingerprint,
        handoff=handoff,
    )
    lineage.validate_source(
        observation,
        state,
        expected_proposal=expected_proposal,
    )
    if (
        response.proposal != safety.proposal.path
        or response.proposal_fingerprint != safety.proposal.fingerprint
        or safety.proposal.path != expected_proposal
        or safety.passed
        or safety.selected_action_scale is not None
        or any(
            attempt.gate.reasons
            != (ControlGateReason.TARGET_PROGRESS_INSUFFICIENT,)
            for attempt in safety.attempts
        )
        or failed.execution_path.exists()
        or failed.result_path.exists()
    ):
        raise ValueError("insertion no-actuation retry source is invalid")
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(failed.session_id, stage)
    runtime_owner_session_id = failed.session_id
    if runtime is None:
        runtime = live_runtime_for(previous.session_id, stage)
        runtime_owner_session_id = previous.session_id
    if runtime is None:
        raise RuntimeError("insertion no-actuation retry runtime was lost")
    bind_live_runtime(
        previous.session_id,
        stage,
        runtime.actuators,
        runtime.attachment,
        runtime.sensor,
    )
    return {
        "status": "insertion_no_actuation_retry_ready",
        "previous_session_id": previous.session_id,
        "failed_safety_session_id": failed.session_id,
        "runtime_owner_session_id": runtime_owner_session_id,
        "requested": safety.proposal.to_dict(),
    }


def restore_insertion_rollback_retry(
    previous_session_id: str,
    rolled_back_session_id: str,
    next_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Return runtime ownership after an exact safe insertion rollback."""

    import omni.usd
    from jepa_wm.control_rollout import ControlStepSummary

    previous = ControlStepSummary.from_session(
        ControlSession.at(CONTROL_ROOT, previous_session_id)
    )
    lineage = InsertionFollowupLineage(
        previous.observation,
        previous.state,
        previous.result,
        next_maximum_steps,
    )
    rolled_back = ControlStepSummary.from_session(
        ControlSession.at(CONTROL_ROOT, rolled_back_session_id)
    )
    rollback = rolled_back.result.insertion_trial_rollback
    if (
        rolled_back.result.status
        not in (
            ControlResultStatus.ROLLED_BACK_TRACKING,
            ControlResultStatus.ROLLED_BACK_PROGRESS,
        )
        or rolled_back.state.previous_session_id != previous_session_id
        or not isinstance(rollback, InsertionTrialRollbackEvidence)
        or rollback.drive_target != lineage.active_drive_target
        or not rollback.plug_attached
    ):
        raise ValueError("insertion rollback retry source is invalid")
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(rolled_back_session_id, stage)
    if runtime is None:
        raise RuntimeError("live rolled-back insertion runtime was lost")
    bind_live_runtime(
        previous_session_id,
        stage,
        runtime.actuators,
        runtime.attachment,
        runtime.sensor,
    )
    return {
        "status": "insertion_rollback_retry_ready",
        "previous_session_id": previous_session_id,
        "rolled_back_session_id": rolled_back_session_id,
        "active_drive_target": lineage.active_drive_target.to_dict(),
    }


def restore_insertion_retry(
    previous_session_id: str,
    failed_session_id: str,
    next_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Restore the exact runtime for either supported fail-closed retry."""

    failed = ControlSession.at(CONTROL_ROOT, failed_session_id)
    if failed.direct_safety_path.is_file():
        return restore_insertion_no_actuation_retry(
            previous_session_id,
            failed_session_id,
            next_maximum_steps,
        )
    return restore_insertion_rollback_retry(
        previous_session_id,
        failed_session_id,
        next_maximum_steps,
    )


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
    target_policy: InsertionControlTargetPolicy,
    expected_proposal: Path,
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
        expected_proposal=expected_proposal,
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
        insertion_target_policy=target_policy,
        active_drive_target=active_drive_target,
        insertion_rollout_position=lineage.followup_position,
    )
    lineage.validate_source(
        observation,
        state,
        expected_proposal=expected_proposal,
    )
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


def verify_insertion_followup_source(session_id: str) -> dict[str, Any]:
    """Require one reconstructed applied action with passing realized progress."""

    from jepa_wm.control_rollout import ControlStepSummary

    previous = ControlSession.at(CONTROL_ROOT, session_id)
    step = ControlStepSummary.from_session(previous)
    lineage = InsertionFollowupLineage(
        step.observation,
        step.state,
        step.result,
    )
    return {
        "status": "followup_ready",
        "session_id": session_id,
        "realized_target_progress": lineage.realized_target_progress.to_dict(),
    }


def verify_grasp_to_insertion_source(session_id: str) -> dict[str, Any]:
    """Require one applied attached contact-aware grasp endpoint."""

    from jepa_wm.control_rollout import ControlStepSummary

    session = ControlSession.at(CONTROL_ROOT, session_id)
    step = ControlStepSummary.from_session(session)
    lineage = GraspToInsertionLineage(
        step.observation,
        step.state,
        step.result,
    )
    if recording_task(RECORDING_ROOT / step.state.reference_recording) != INSERTION_TASK_ID:
        raise ValueError("grasp-to-insertion source is not contact-aware")
    return {
        "status": "grasp_to_insertion_ready",
        "session_id": session_id,
        "active_drive_target": lineage.active_drive_target.to_dict(),
    }


def restore_grasp_transition_retry(
    grasp_session_id: str,
    rolled_back_session_id: str,
) -> dict[str, Any]:
    """Return runtime ownership after one authenticated, settled tracking rollback."""

    import omni.usd
    from jepa_wm.control_rollout import ControlStepSummary

    grasp = ControlStepSummary.from_session(
        ControlSession.at(CONTROL_ROOT, grasp_session_id)
    )
    lineage = GraspToInsertionLineage(
        grasp.observation,
        grasp.state,
        grasp.result,
    )
    rolled_back = ControlStepSummary.from_session(
        ControlSession.at(CONTROL_ROOT, rolled_back_session_id)
    )
    rollback = rolled_back.result.insertion_trial_rollback
    if (
        rolled_back.result.status is not ControlResultStatus.ROLLED_BACK_TRACKING
        or rolled_back.state.previous_session_id != grasp_session_id
        or not isinstance(rollback, InsertionTrialRollbackEvidence)
        or rollback.drive_target != lineage.active_drive_target
        or not rollback.plug_attached
    ):
        raise ValueError("grasp transition retry source is invalid")
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(rolled_back_session_id, stage)
    if runtime is None:
        raise RuntimeError("live rolled-back transition runtime was lost")
    bind_live_runtime(
        grasp_session_id,
        stage,
        runtime.actuators,
        runtime.attachment,
        runtime.sensor,
    )
    return {
        "status": "grasp_transition_retry_ready",
        "grasp_session_id": grasp_session_id,
        "rolled_back_session_id": rolled_back_session_id,
        "active_drive_target": lineage.active_drive_target.to_dict(),
    }


def _verify_insertion_rollout_result(
    roster: InsertionRolloutRoster,
    reference_name: str,
    exploration_seed: int,
    *,
    predecessor_session_id: str | None = None,
) -> ControlRolloutReport:
    """Reconstruct one exact bounded rollout and require every action applied."""

    from jepa_wm.control_rollout import ControlRolloutReport, ControlStepSummary

    steps = tuple(
        ControlStepSummary.from_session(ControlSession.at(CONTROL_ROOT, session_id))
        for session_id in roster.session_ids
    )
    if tuple(
        step.state.resolved_insertion_rollout_position() for step in steps
    ) != roster.positions:
        raise ValueError("insertion rollout positions are invalid")
    report = ControlRolloutReport.from_sessions(
        QUANTIS_DATA_ROOT,
        roster.session_ids[-1],
        roster.session_ids,
        reference_recording=reference_name,
        seed=exploration_seed,
        proposal=steps[0].observation.expected_proposal,
        requested_steps=roster.maximum_steps,
        predecessor_session_id=predecessor_session_id,
    )
    report.require_all_steps_applied()
    return report


def verify_grasp_to_insertion_result(
    run_id: str,
    grasp_rollout_id: str,
    insertion_session_roster: str,
    reference_name: str,
    exploration_seed: int,
) -> dict[str, Any]:
    """Reconstruct and persist one bounded task-terminal grasp plus four actions."""

    from jepa_wm.control_rollout import ControlRolloutReport, ControlStepSummary
    from jepa_wm.grasp_to_insertion import (
        GRASP_ACTIONS,
        GraspToInsertionReport,
    )

    validate_recording_id(run_id)
    validate_recording_id(grasp_rollout_id)
    prior_report_path = (
        QUANTIS_DATA_ROOT / "control_rollouts" / grasp_rollout_id / "report.json"
    )
    prior_report = json.loads(prior_report_path.read_text())
    reported_steps = prior_report.get("steps") if isinstance(prior_report, dict) else None
    if (
        not isinstance(reported_steps, list)
        or not reported_steps
        or not all(isinstance(step, dict) for step in reported_steps)
    ):
        raise ValueError("grasp rollout report has no session roster")
    grasp_sessions = tuple(step.get("session") for step in reported_steps)
    if (
        not all(isinstance(session, str) for session in grasp_sessions)
        or len(grasp_sessions) > GRASP_ACTIONS
        or grasp_sessions
        != tuple(
            f"{grasp_rollout_id}-{index:02d}"
            for index in range(len(grasp_sessions))
        )
    ):
        raise ValueError("grasp rollout session roster is invalid")
    first_grasp = ControlStepSummary.from_session(
        ControlSession.at(CONTROL_ROOT, grasp_sessions[0])
    )
    grasp = ControlRolloutReport.from_sessions(
        QUANTIS_DATA_ROOT,
        grasp_rollout_id,
        grasp_sessions,
        reference_recording=reference_name,
        seed=exploration_seed,
        proposal=first_grasp.observation.expected_proposal,
        requested_steps=GRASP_ACTIONS,
    )
    insertion_roster = InsertionRolloutRoster.from_csv(
        insertion_session_roster,
        DEMO_INSERTION_ROLLOUT.maximum_steps,
    )
    insertion = _verify_insertion_rollout_result(
        insertion_roster,
        reference_name,
        exploration_seed,
        predecessor_session_id=grasp_sessions[-1],
    )
    report = GraspToInsertionReport(run_id, grasp, insertion)
    output = QUANTIS_DATA_ROOT / "control_rollouts" / run_id / "report.json"
    output.parent.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output, report.to_dict())
    return report.to_dict()


def verify_insertion_two_step_result(
    first_session_id: str,
    second_session_id: str,
    reference_name: str,
    exploration_seed: int,
) -> dict[str, Any]:
    """Reconstruct and require exactly two applied insertion actions."""

    roster = InsertionRolloutRoster(
        (first_session_id, second_session_id),
        TWO_STEP_INSERTION_ROLLOUT.maximum_steps,
    )
    report = _verify_insertion_rollout_result(
        roster,
        reference_name,
        exploration_seed,
    )
    return {
        "status": "two_step_applied",
        "first_session_id": first_session_id,
        "second_session_id": second_session_id,
        "report": report.to_dict(),
    }


def verify_insertion_demo_rollout_result(
    session_roster: str,
    reference_name: str,
    exploration_seed: int,
) -> dict[str, Any]:
    """Reconstruct and require the complete hard-capped demo rollout."""

    roster = InsertionRolloutRoster.from_csv(
        session_roster,
        DEMO_INSERTION_ROLLOUT.maximum_steps,
    )
    report = _verify_insertion_rollout_result(
        roster,
        reference_name,
        exploration_seed,
    )
    return {
        "status": "demo_rollout_applied",
        "sessions": list(roster.session_ids),
        "report": report.to_dict(),
    }


async def _capture_contact_grasp_followup(
    session: ControlSession,
    previous_step: ControlStepSummary,
) -> dict[str, Any]:
    """Capture one interlocked direct grasp continuation on insertion geometry."""

    import omni.kit.app
    import omni.timeline
    import omni.usd

    previous = previous_step.result.post_action
    if previous is None:
        raise ValueError("contact grasp follow-up has no post-action evidence")
    previous_state = previous_step.state
    previous_observation = previous_step.observation
    reference_path = (
        QUANTIS_DATA_ROOT / "recordings" / previous_state.reference_recording
    )
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(previous_step.result.session_id, stage)
    if runtime is None:
        raise RuntimeError("live contact grasp runtime was lost before follow-up")
    context_path = session.path / "context.png"

    async def capture_frame(observe_safety) -> None:
        await capture_camera_frame(
            JEPA_WM_CAMERA_SPECS[0],
            context_path,
            observe_safety=observe_safety,
        )

    active_drive_target = previous_step.contact_grasp_drive_target()
    synchronized = await synchronized_insertion_frame_capture(
        runtime,
        omni.timeline.get_timeline_interface(),
        omni.kit.app.get_app().next_update_async,
        previous.require_safety_snapshot(),
        SimulatorSafetyLimits(),
        capture_frame,
        expected_active_drive_target=active_drive_target,
        operation="contact grasp follow-up capture",
        maximum_gripper_error_meters=(
            MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
        ),
    )
    if synchronized.pose is None:
        raise RuntimeError("contact grasp follow-up pose was not refreshed")
    if synchronized.active_drive_target != active_drive_target:
        raise RuntimeError("contact grasp follow-up drive target changed")
    target_policy = _contact_grasp_followup_policy(previous_step)
    target = target_policy.select(
        reference_path,
        frame_root=QUANTIS_DATA_ROOT,
        live_pose=synchronized.pose,
        plug_attached=synchronized.safety.plug_attached,
        previous_target=previous_observation.target_frame,
    )
    next_context_index = target_policy.context_index_for_target(target.frame)
    observation = ControlObservation(
        observation_id=observation_id_for_session(session.session_id),
        captured_at_unix_seconds=time(),
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=target,
        expected_proposal=previous_observation.expected_proposal,
        pose=synchronized.pose,
        previous_action=action_between(previous_observation.pose, synchronized.pose),
        warmup_frames=next_context_index,
    )
    state = ControlSessionState(
        session_id=session.session_id,
        reference_recording=previous_state.reference_recording,
        seed=previous_state.seed,
        recording=previous_state.recording,
        current_joint_positions=synchronized.safety.joint_positions,
        collision_detected=synchronized.safety.collision_detected,
        contact_force_newtons=synchronized.safety.contact_force_newtons,
        previous_session_id=previous_step.result.session_id,
        execution_policy=ControlExecutionPolicy.DIRECT,
        plug_position=synchronized.safety.plug_position,
        plug_attached=synchronized.safety.plug_attached,
        current_gripper_width_m=synchronized.safety.gripper_width_m,
        active_drive_target=active_drive_target,
        contact_grasp_target_policy=target_policy,
    )
    previous.validate_followup_capture(
        observation,
        state,
        maximum_gripper_error_meters=(
            MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
        ),
    )
    session.write_capture(observation, state)
    bind_live_runtime(
        session.session_id,
        stage,
        synchronized.runtime.actuators,
        synchronized.runtime.attachment,
        synchronized.runtime.sensor,
    )
    return ControlCaptureResult(
        session.session_id,
        observation,
        session.request_path,
        synchronized.safety.contact_force_newtons,
        synchronized.safety.collision_detected,
    ).to_dict()


def _contact_grasp_followup_policy(
    previous_step: ControlStepSummary,
) -> ContactGraspTargetPolicy:
    """Recover current grasp authority, including one consumed reset candidate."""

    previous_state = previous_step.state
    if previous_state.execution_policy is ControlExecutionPolicy.DIRECT:
        return previous_state.require_current_contact_grasp_policy()
    if (
        previous_state.execution_policy
        is not ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
    ):
        raise ValueError("contact grasp follow-up source policy is invalid")

    previous_session = ControlSession.at(
        CONTROL_ROOT,
        previous_step.result.session_id,
    )
    binding = previous_session.load_candidate_binding(previous_step.response)
    source_session = ControlSession.at(CONTROL_ROOT, binding.source_session_id)
    source_observation, source_state = source_session.load_capture()
    policy = source_state.require_current_contact_grasp_policy()
    if (
        source_state.reference_recording != previous_state.reference_recording
        or source_state.seed != previous_state.seed
        or source_state.recording != previous_state.recording
        or source_observation.target != previous_step.observation.target
        or source_observation.warmup_frames
        != previous_step.observation.warmup_frames
        or source_observation.expected_proposal
        != previous_step.observation.expected_proposal
        or previous_step.result.status is not ControlResultStatus.APPLIED
        or previous_step.result.post_action is None
        or not previous_step.result.gate.passed
        or not previous_step.result.post_action.tracking.passed
        or previous_step.result.post_action.collision_detected
        or previous_step.result.post_action.contact_force_newtons
        > SimulatorSafetyLimits().maximum_contact_force_newtons
    ):
        raise ValueError(
            "unknown-start contact grasp handoff is not bound to its source"
        )
    return policy


async def verify_unknown_start_grasp_continuation_source(
    session_id: str,
) -> dict[str, Any]:
    """Prove the resident generation can continue one safe reset candidate."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from jepa_wm.control_rollout import ControlStepSummary

    session = ControlSession.at(CONTROL_ROOT, session_id)
    step = ControlStepSummary.from_session(session)
    _contact_grasp_followup_policy(step)
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(session_id, stage)
    timeline = omni.timeline.get_timeline_interface()
    if runtime is None:
        raise RuntimeError("unknown-start grasp continuation runtime was lost")
    await pause_control_timeline(
        timeline,
        omni.kit.app.get_app().next_update_async,
    )
    current = runtime.actuators.actual_command()
    collision, force = read_control_contact(runtime.sensor)
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.MOTION, Phase.READY),
        ObservationStage.APPROACHING_CABLE,
        current,
        runtime.attachment,
    )
    live = ControlSafetySnapshot(
        tuple(float(value) for value in current.arm_positions),
        current.gripper_width_m,
        tuple(float(value) for value in snapshot.plug_position),
        force,
        collision,
        snapshot.plug_attached,
    )
    live.validate_followup_continuity(
        step.result.post_action.require_safety_snapshot(),
        step.contact_grasp_drive_target(),
        maximum_gripper_error_meters=(
            MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
        ),
    )
    return {
        "status": "unknown_start_grasp_continuation_ready",
        "session_id": session_id,
        "result_status": step.result.status.value,
        "plug_attached": live.plug_attached,
        "contact_force_newtons": live.contact_force_newtons,
        "collision_detected": live.collision_detected,
        "timeline_playing": False,
    }


async def capture_insertion_transition_observation(
    session_id: str,
    previous_session_id: str,
    proposal_name: str,
    *,
    maximum_steps: int = DEMO_INSERTION_ROLLOUT.maximum_steps,
) -> dict[str, Any]:
    """Capture the attached live grasp endpoint as insertion action one."""

    import omni.kit.app
    import omni.timeline
    import omni.usd

    validate_recording_id(proposal_name)
    previous_session = ControlSession.at(CONTROL_ROOT, previous_session_id)
    from jepa_wm.control_rollout import ControlStepSummary

    previous_step = ControlStepSummary.from_session(previous_session)
    lineage = GraspToInsertionLineage(
        previous_step.observation,
        previous_step.state,
        previous_step.result,
    )
    reference_path = (
        QUANTIS_DATA_ROOT
        / "recordings"
        / previous_step.state.reference_recording
    )
    if recording_task(reference_path) != INSERTION_TASK_ID:
        raise ValueError("grasp-to-insertion transition requires insertion geometry")
    session = ControlSession.at(CONTROL_ROOT, session_id)
    if session.path.exists():
        raise ValueError(f"control session already exists: {session_id}")
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(previous_session_id, stage)
    if runtime is None:
        raise RuntimeError("live grasp runtime was lost before insertion transition")
    context_path = session.path / "context.png"

    async def capture_frame(observe_safety) -> None:
        await capture_camera_frame(
            JEPA_WM_CAMERA_SPECS[0],
            context_path,
            observe_safety=observe_safety,
        )

    synchronized = await synchronized_insertion_frame_capture(
        runtime,
        omni.timeline.get_timeline_interface(),
        omni.kit.app.get_app().next_update_async,
        lineage.post_action.require_safety_snapshot(),
        SimulatorSafetyLimits(),
        capture_frame,
        expected_active_drive_target=lineage.active_drive_target,
        operation="grasp-to-insertion transition capture",
    )
    if synchronized.pose is None:
        raise RuntimeError("grasp-to-insertion transition pose was not refreshed")
    if synchronized.active_drive_target != lineage.active_drive_target:
        raise RuntimeError("grasp-to-insertion transition drive target changed")
    context_index = CONTACT_INSERTION_RECORDING.start_index(
        ContactInsertionSegment.GRASP_ATTACH
    )
    target_policy = INSERTION_CONTROL_TARGET_POLICY.for_followup()
    selected = target_policy.select(
        reference_path,
        context_index=context_index,
        current_pose=synchronized.pose,
    )
    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=time(),
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=ControlTarget(
            selected.target.path.relative_to(QUANTIS_DATA_ROOT),
            selected.target_pose,
        ),
        expected_proposal=control_proposal_path(proposal_name),
        pose=synchronized.pose,
        previous_action=action_between(
            previous_step.observation.pose,
            synchronized.pose,
        ),
        warmup_frames=context_index,
    )
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=previous_step.state.reference_recording,
        seed=previous_step.state.seed,
        recording=previous_step.state.recording,
        current_joint_positions=synchronized.safety.joint_positions,
        collision_detected=synchronized.safety.collision_detected,
        contact_force_newtons=synchronized.safety.contact_force_newtons,
        previous_session_id=previous_session_id,
        execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
        plug_position=synchronized.safety.plug_position,
        plug_attached=synchronized.safety.plug_attached,
        current_gripper_width_m=synchronized.safety.gripper_width_m,
        insertion_target_policy=target_policy,
        active_drive_target=lineage.active_drive_target,
        insertion_rollout_position=InsertionRolloutPosition.initial(maximum_steps),
    )
    lineage.validate_source(observation, state)
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
    if recording_task(reference_path) in (GRASP_TASK_ID, INSERTION_TASK_ID):
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
    resume_live_simulation(timeline)
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
    next_maximum_steps: int | None = None,
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
    proposal_changed = previous_observation.expected_proposal != expected_proposal
    next_context_index = previous_observation.warmup_frames + 1
    reference_path = QUANTIS_DATA_ROOT / "recordings" / previous_state.reference_recording
    reference_task = recording_task(reference_path)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    contact_grasp_source = (
        reference_task == INSERTION_TASK_ID
        and previous_state.execution_policy
        in (
            ControlExecutionPolicy.DIRECT,
            ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
        )
        and previous_state.insertion_target_policy is None
    )
    if contact_grasp_source:
        if proposal_changed:
            if (
                previous_state.execution_policy
                is ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
            ):
                raise ValueError(
                    "unknown-start contact grasp cannot change proposal at handoff"
                )
            return await capture_insertion_transition_observation(
                session_id,
                previous_session_id,
                proposal_name,
            )
        return await _capture_contact_grasp_followup(session, previous_step)
    if reference_task != INSERTION_TASK_ID:
        if proposal_changed:
            raise ValueError("follow-up proposal differs from the rollout checkpoint")
        return await _capture_generic_followup_observation(
            session,
            previous_step,
        )
    expected_proposal = resolve_insertion_followup_proposal(
        previous_observation.expected_proposal,
        expected_proposal,
        previous_proposal_fingerprint=previous_step.response.proposal_fingerprint,
        handoff=(
            previous.load_insertion_proposal_handoff(session_id)
            if proposal_changed
            else None
        ),
    )
    lineage = InsertionFollowupLineage(
        previous_observation,
        previous_state,
        previous_result,
        next_maximum_steps,
    )
    target_policy = previous_state.insertion_target_policy
    if target_policy is None:
        raise ValueError("insertion follow-up requires its persisted target policy")
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
    context_path = session.path / "context.png"

    async def capture_frame(observe_safety) -> None:
        await capture_camera_frame(
            JEPA_WM_CAMERA_SPECS[0],
            context_path,
            observe_safety=observe_safety,
        )

    synchronized = await synchronized_insertion_frame_capture(
        runtime,
        timeline,
        omni.kit.app.get_app().next_update_async,
        captured_state,
        SimulatorSafetyLimits(),
        capture_frame,
        expected_active_drive_target=lineage.active_drive_target,
        operation="insertion follow-up capture synchronization",
    )
    if synchronized.pose is None:
        raise RuntimeError("live insertion pose was not refreshed")
    if synchronized.active_drive_target is None:
        raise RuntimeError("live insertion drive target was not refreshed")
    followup_target_policy = target_policy.for_adaptive_followup()
    reference_rollout = followup_target_policy.select(
        reference_path,
        context_index=next_context_index,
        current_pose=synchronized.pose,
    )
    target = ControlTarget(
        reference_rollout.target.path.relative_to(QUANTIS_DATA_ROOT),
        reference_rollout.target_pose,
    )
    observation, state = build_insertion_followup_capture(
        session_id,
        lineage,
        captured_at_unix_seconds=time(),
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=target,
        current=synchronized.safety,
        current_pose=synchronized.pose,
        active_drive_target=synchronized.active_drive_target,
        target_policy=followup_target_policy,
        expected_proposal=expected_proposal,
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
