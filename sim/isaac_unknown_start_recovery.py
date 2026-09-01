"""Safety-gated reset recovery after a terminal unknown-start candidate."""

from __future__ import annotations

import json
from typing import Any

from jepa_wm.persistence import write_json_atomic
from jepa_wm.training_artifact import artifact_fingerprint


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
    )
    from sim.isaac_demo_runtime import (
        ContactReading,
        JointCommand,
        advance_physics_updates,
        resume_live_simulation,
    )
    from sim.isaac_unknown_start_shadow import (
        _load_authenticated_reset,
        reauthenticate_unknown_start_shadow_session,
    )
    from sim.unknown_start_shadow import UnknownStartControlHandoff

    session = ControlSession.at(CONTROL_ROOT, session_id)
    recovery_path = session.path / "rollback_recovery.json"
    if recovery_path.exists():
        raise ValueError("unknown-start rollback recovery already exists")
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
    _load_authenticated_reset(
        handoff.reset_recording_id,
        handoff.reset_result_fingerprint,
    )
    timeline = omni.timeline.get_timeline_interface()
    await pause_control_timeline(timeline, omni.kit.app.get_app().next_update_async)
    if timeline.is_playing():
        raise RuntimeError("unknown-start rollback recovery could not pause timeline")
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(session_id, stage)
    if runtime is None:
        raise RuntimeError("unknown-start rollback recovery runtime was lost")
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
    target = JointCommand(
        np.asarray(state.current_joint_positions, dtype=np.float64),
        state.current_gripper_width_m,
    )
    reset_drive_target = JointCommand(
        np.asarray(state.active_drive_target.joint_positions, dtype=np.float64),
        state.active_drive_target.gripper_width_m,
    )

    def observe_safety() -> ContactReading:
        collision, force = read_control_contact(runtime.sensor)
        if collision or force > 2.0 or runtime.attachment.attached:
            raise RuntimeError(
                "unknown-start rollback recovery exceeded its live safety boundary"
            )
        return ContactReading(collision, force)

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
    # One physics update refreshes articulation-linked world transforms.  The
    # reset initializer clears DOF velocities first so this update cannot carry
    # rollback momentum into the exact authenticated state.
    resume_live_simulation(timeline)
    try:
        runtime.actuators.set_reset_state(
            target,
            drive_target=reset_drive_target,
        )
        await advance_physics_updates(1, observe_safety)
    finally:
        await pause_control_timeline(
            timeline,
            omni.kit.app.get_app().next_update_async,
        )
    if timeline.is_playing():
        raise RuntimeError("unknown-start reset initialization could not pause timeline")
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
