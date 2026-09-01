"""Safety-gated reset recovery after a terminal unknown-start candidate."""

from __future__ import annotations

import json
from typing import Any

from jepa_wm.persistence import write_json_atomic
from jepa_wm.training_artifact import artifact_fingerprint


def _host_array(values: Any) -> Any:
    """Convert one Isaac tensor-like value to a NumPy-compatible host value."""

    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        values = values.numpy()
    return values


async def diagnose_unknown_start_candidate_rollback(
    session_id: str,
) -> dict[str, Any]:
    """Read exact paused recovery drift without mutating simulator state."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    import numpy as np
    from isaacsim.core.experimental.prims import Articulation

    from jepa.contract import ObservationStage
    from sim.control_session import CONTROL_ROOT, ControlSession
    from sim.isaac_control_runtime import (
        live_runtime_for,
        pause_control_timeline,
        read_control_contact,
    )
    from sim.isaac_demo_kinematics import _solver_for_stage
    from sim.isaac_demo_runtime import create_actuators, recording_snapshot
    from sim.isaac_demo_scene import ROBOT_PATH
    from sim.isaac_unknown_start_shadow import _load_authenticated_reset
    from sim.recording import RecordingLabel, RecordingMoment
    from sim.unknown_start_shadow import UnknownStartControlHandoff

    session = ControlSession.at(CONTROL_ROOT, session_id)
    _, state = session.load_capture()
    handoff = UnknownStartControlHandoff.from_dict(
        json.loads((session.path / "unknown_start_handoff.json").read_text())
    )
    if (
        handoff.session_id != session_id
        or artifact_fingerprint(session.request_path) != handoff.request_fingerprint
        or artifact_fingerprint(session.state_path) != handoff.state_fingerprint
    ):
        raise ValueError("unknown-start recovery diagnostic source is invalid")
    _, _, evidence, _ = _load_authenticated_reset(
        handoff.reset_recording_id,
        handoff.reset_result_fingerprint,
    )
    timeline = omni.timeline.get_timeline_interface()
    await pause_control_timeline(timeline, omni.kit.app.get_app().next_update_async)
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(session_id, stage)
    if runtime is None:
        raise RuntimeError("unknown-start recovery diagnostic runtime was lost")
    actual = runtime.actuators.actual_command()
    authored = runtime.actuators.current_command()
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.INITIAL),
        ObservationStage.APPROACHING_CABLE,
        actual,
        runtime.attachment,
    )
    velocities = np.asarray(
        _host_array(runtime.actuators.articulation.get_dof_velocities()),
        dtype=np.float64,
    )
    if velocities.ndim == 2 and velocities.shape[0] == 1:
        velocities = velocities[0]
    solver, _, _ = _solver_for_stage(stage)
    actual_fk = solver.compute_forward_kinematics(
        "right_gripper", actual.arm_positions
    )[0]
    rebound_actual = create_actuators(
        stage,
        Articulation(ROBOT_PATH),
    ).actual_command()
    rebound_fk = solver.compute_forward_kinematics(
        "right_gripper", rebound_actual.arm_positions
    )[0]
    expected_joints = np.asarray(
        evidence.observed_arm_positions_radians,
        dtype=np.float64,
    )
    expected_fk = solver.compute_forward_kinematics(
        "right_gripper", expected_joints
    )[0]
    expected_gripper_frame = np.asarray(
        evidence.workspace.gripper_control_frame_position_m,
        dtype=np.float64,
    )
    collision, force = read_control_contact(runtime.sensor)
    return {
        "schema": "quantis.unknown_start_rollback_diagnostic.v1",
        "session_id": session_id,
        "timeline_playing": timeline.is_playing(),
        "actual_joint_positions": actual.arm_positions.tolist(),
        "expected_joint_positions": expected_joints.tolist(),
        "maximum_joint_error_radians": float(
            np.max(np.abs(actual.arm_positions - expected_joints))
        ),
        "authored_joint_target": authored.arm_positions.tolist(),
        "captured_drive_target": (
            list(state.active_drive_target.joint_positions)
            if state.active_drive_target is not None
            else None
        ),
        "dof_velocities": velocities.tolist(),
        "maximum_dof_velocity": float(np.max(np.abs(velocities))),
        "usd_gripper_frame_position": (
            np.asarray(snapshot.gripper_frame_world_position).tolist()
        ),
        "actual_joint_fk_gripper_position": np.asarray(actual_fk).tolist(),
        "rebound_joint_positions": rebound_actual.arm_positions.tolist(),
        "rebound_joint_fk_gripper_position": np.asarray(rebound_fk).tolist(),
        "expected_joint_fk_gripper_position": np.asarray(expected_fk).tolist(),
        "expected_gripper_frame_position": expected_gripper_frame.tolist(),
        "usd_gripper_frame_error_meters": float(
            np.max(
                np.abs(
                    np.asarray(snapshot.gripper_frame_world_position)
                    - expected_gripper_frame
                )
            )
        ),
        "actual_joint_fk_error_meters": float(
            np.max(np.abs(np.asarray(actual_fk) - expected_gripper_frame))
        ),
        "rebound_joint_fk_error_meters": float(
            np.max(np.abs(np.asarray(rebound_fk) - expected_gripper_frame))
        ),
        "expected_joint_fk_error_meters": float(
            np.max(np.abs(np.asarray(expected_fk) - expected_gripper_frame))
        ),
        "collision_detected": collision,
        "contact_force_newtons": force,
        "plug_attached": runtime.attachment.attached,
    }


async def recover_unknown_start_candidate_rollback(
    session_id: str,
) -> dict[str, Any]:
    """Finish one failed drive rollback, then initialize the exact reset."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    import numpy as np

    from jepa_wm.control_policy import ControlExecutionPolicy
    from jepa_wm.joint_drive import JointDriveTarget
    from sim.control_session import CONTROL_ROOT, ControlResultStatus, ControlSession
    from sim.isaac_control_execution import (
        UNKNOWN_START_ROLLBACK_SETTLEMENT,
        rollback_control_command,
    )
    from sim.isaac_control_runtime import (
        live_runtime_for,
        pause_control_timeline,
        read_control_contact,
        refresh_paused_live_control_articulation,
    )
    from sim.isaac_demo_runtime import (
        ContactReading,
        JointCommand,
        resume_live_simulation,
    )
    from sim.isaac_unknown_start_shadow import (
        _load_authenticated_reset,
        reauthenticate_unknown_start_shadow_session,
    )
    from sim.unknown_start_shadow import UnknownStartControlHandoff

    session = ControlSession.at(CONTROL_ROOT, session_id)
    recovery_path = session.path / "rollback_recovery.json"
    _, state = session.load_capture()
    result = session.load_result()
    handoff_path = session.path / "unknown_start_handoff.json"
    handoff = UnknownStartControlHandoff.from_dict(json.loads(handoff_path.read_text()))
    refresh = result.insertion_trial_refresh
    interlock = result.execution_interlock
    if (
        state.execution_policy is not ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
        or result.status is not ControlResultStatus.ROLLBACK_FAILED
        or result.post_action is None
        or result.post_action.collision_detected
        or result.post_action.contact_force_newtons > 2.0
        or result.post_action.plug_attached
        or state.current_gripper_width_m is None
        or refresh is None
        or refresh.live_state.plug_attached
        or interlock is None
        or state.active_drive_target is None
        or interlock.collision_detected
        or interlock.maximum_contact_force_newtons > 2.0
        or handoff.session_id != session_id
        or artifact_fingerprint(session.path / "context.png")
        != handoff.context_fingerprint
        or artifact_fingerprint(session.request_path) != handoff.request_fingerprint
        or artifact_fingerprint(session.state_path) != handoff.state_fingerprint
    ):
        raise ValueError("unknown-start rollback recovery source is invalid")
    _, _, reset_evidence, _ = _load_authenticated_reset(
        handoff.reset_recording_id,
        handoff.reset_result_fingerprint,
    )
    target = JointCommand(
        np.asarray(
            reset_evidence.observed_arm_positions_radians,
            dtype=np.float64,
        ),
        reset_evidence.observed_gripper_width_m,
    )
    reset_drive_target = JointCommand(
        np.asarray(state.active_drive_target.joint_positions, dtype=np.float64),
        state.active_drive_target.gripper_width_m,
    )
    if recovery_path.exists():
        existing = json.loads(recovery_path.read_text())
        existing_joints = np.asarray(
            existing.get("joint_positions", ()),
            dtype=np.float64,
        )
        existing_gripper = existing.get("gripper_width_m")
        existing_is_authentic = (
            existing.get("schema")
            == "quantis.unknown_start_rollback_recovery.v1"
            and existing.get("session_id") == session_id
            and existing.get("source_result_fingerprint")
            == artifact_fingerprint(session.result_path)
            and existing.get("recovered") is True
            and existing.get("applied_model_actions") == 0
            and existing.get("collision_detected") is False
            and existing.get("plug_attached") is False
            and existing.get("timeline_playing") is False
            and existing_joints.shape == (7,)
            and np.all(np.isfinite(existing_joints))
            and isinstance(existing_gripper, (int, float))
            and np.isfinite(existing_gripper)
        )
        if not existing_is_authentic:
            raise ValueError("existing unknown-start rollback recovery is invalid")
        existing_is_exact = (
            float(np.max(np.abs(existing_joints - target.arm_positions))) <= 1e-7
            and abs(float(existing_gripper) - target.gripper_width_m)
            <= 1e-7
        )
        if existing_is_exact:
            return existing
        superseded_path = recovery_path.with_name(
            "rollback_recovery.superseded-"
            f"{artifact_fingerprint(recovery_path)}.json"
        )
        if superseded_path.exists():
            raise ValueError("unknown-start superseded recovery already exists")
        recovery_path.replace(superseded_path)
    timeline = omni.timeline.get_timeline_interface()
    await pause_control_timeline(timeline, omni.kit.app.get_app().next_update_async)
    if timeline.is_playing():
        raise RuntimeError("unknown-start rollback recovery could not pause timeline")
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(session_id, stage)
    if runtime is None:
        raise RuntimeError("unknown-start rollback recovery runtime was lost")

    def observe_safety() -> ContactReading:
        collision, force = read_control_contact(runtime.sensor)
        if collision or force > 2.0 or runtime.attachment.attached:
            raise RuntimeError(
                "unknown-start rollback recovery exceeded its live safety boundary"
            )
        return ContactReading(collision, force)

    runtime = await refresh_paused_live_control_articulation(
        runtime,
        timeline,
        omni.kit.app.get_app().next_update_async,
        observe_safety,
    )
    active = runtime.actuators.current_command()
    active_positions = tuple(float(value) for value in active.arm_positions)
    allowed_active_targets = (
        JointDriveTarget.for_command(
            refresh.live_state.joint_positions,
            refresh.live_state.gripper_width_m,
        ),
        JointDriveTarget.for_command(
            state.current_joint_positions,
            state.current_gripper_width_m,
        ),
        state.active_drive_target,
    )
    for expected_active in allowed_active_targets:
        try:
            expected_active.validate_active(
                active_positions,
                active.gripper_width_m,
            )
            break
        except ValueError:
            continue
    else:
        raise ValueError("unknown-start recovery active drive target changed")
    resume_live_simulation(timeline)
    try:
        await rollback_control_command(
            runtime.actuators,
            reset_drive_target,
            runtime.attachment,
            omni.kit.app.get_app().next_update_async,
            expected_attachment=False,
            observe_safety=observe_safety,
            settlement=UNKNOWN_START_ROLLBACK_SETTLEMENT,
        )
    finally:
        await pause_control_timeline(
            timeline,
            omni.kit.app.get_app().next_update_async,
        )
    drive_recovered = runtime.actuators.actual_command()
    drive_arm_error = float(
        np.max(np.abs(drive_recovered.arm_positions - target.arm_positions))
    )
    drive_gripper_error = abs(drive_recovered.gripper_width_m - target.gripper_width_m)
    observe_safety()
    if (
        drive_arm_error > UNKNOWN_START_ROLLBACK_SETTLEMENT.maximum_arm_error_radians
        or drive_gripper_error
        > UNKNOWN_START_ROLLBACK_SETTLEMENT.maximum_gripper_error_meters
        or timeline.is_playing()
    ):
        raise RuntimeError("unknown-start drive recovery did not reach reset floor")
    # The authenticated artifact records positions but not the transient DOF
    # velocities that produced them. Finish recovery at the explicit paused
    # initialization boundary and authenticate its deterministic joint FK
    # without an app update or physics tick.
    runtime.actuators.set_reset_state(
        target,
        drive_target=reset_drive_target,
    )
    observe_safety()
    if timeline.is_playing():
        raise RuntimeError("unknown-start reset initialization resumed timeline")
    reauthenticate_unknown_start_shadow_session(session_id)
    actual = runtime.actuators.actual_command()
    collision, force = read_control_contact(runtime.sensor)
    payload = {
        "schema": "quantis.unknown_start_rollback_recovery.v1",
        "session_id": session_id,
        "source_result_fingerprint": artifact_fingerprint(session.result_path),
        "recovered": True,
        "recovery_mode": "drive_then_paused_reset_initialization",
        "drive_arm_error_radians": drive_arm_error,
        "drive_gripper_error_meters": drive_gripper_error,
        "applied_model_actions": 0,
        "joint_positions": [float(value) for value in actual.arm_positions],
        "gripper_width_m": actual.gripper_width_m,
        "contact_force_newtons": force,
        "collision_detected": collision,
        "plug_attached": runtime.attachment.attached,
        "timeline_playing": False,
    }
    write_json_atomic(recovery_path, payload)
    return payload
