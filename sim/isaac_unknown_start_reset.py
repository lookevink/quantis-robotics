"""One zero-actuation realization of the frozen milestone-20 reset contract."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DROID_FPS
from jepa_wm.persistence import write_json_atomic
from jepa_wm.insertion_task import InsertionTaskLimits
from jepa_wm.unknown_start_reset_runtime import authenticate_runtime_source
from sim.isaac_control_runtime import (
    connector_contact_sensor,
    contact_sensor,
    read_contact,
)
from sim.isaac_demo_camera import (
    JEPA_WM_CAMERA_SPECS,
    WRIST_CAMERA_TRANSLATION_METERS,
    DemoRecorder,
)
from sim.isaac_demo_kinematics import solve_waypoints
from sim.isaac_demo_runtime import (
    ContactReading,
    JointCommand,
    advance_physics_updates,
    create_actuators,
    prepare_fixed_joint_plug,
    recording_safety_telemetry,
    recording_snapshot,
)
from sim.isaac_demo_scene import (
    PLUG_PATH,
    ROBOT_PATH,
    SOCKET_PATH,
    WRIST_CAMERA_PATH,
    world_pose,
)
from sim.isaac_exploration import (
    ExplorationRecordingMode,
    ExplorationRecordingProfile,
    apply_variant,
    prepare_recording_stage,
)


def _apply_variant_with_readback(stage: Any, plan: Any) -> dict[str, Any]:
    """Apply the shared variant, then authenticate its authored USD state."""

    from pxr import UsdGeom

    exposure_baseline: dict[str, float] = {}
    for prim in stage.Traverse():
        exposure = prim.GetAttribute("inputs:exposure")
        current = exposure.Get() if exposure.IsValid() else None
        if isinstance(current, (int, float)):
            exposure_baseline[str(prim.GetPath())] = float(current)
    apply_variant(stage, plan)
    camera_translation = stage.GetPrimAtPath(WRIST_CAMERA_PATH).GetAttribute(
        "xformOp:translate"
    )
    if not camera_translation.IsValid():
        raise RuntimeError("unknown-start wrist camera translation is missing")
    realized_camera_offset = np.asarray(
        camera_translation.Get(),
        dtype=np.float64,
    ) - np.asarray(WRIST_CAMERA_TRANSLATION_METERS, dtype=np.float64)
    socket = UsdGeom.Xformable(stage.GetPrimAtPath(SOCKET_PATH))
    scale_operations = [
        operation
        for operation in socket.GetOrderedXformOps()
        if operation.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if len(scale_operations) != 1:
        raise RuntimeError("unknown-start socket scale realization is ambiguous")
    realized_scale = np.asarray(scale_operations[0].Get(), dtype=np.float64)
    realized_exposure_deltas = []
    for prim_path, baseline in exposure_baseline.items():
        exposure = stage.GetPrimAtPath(prim_path).GetAttribute("inputs:exposure")
        realized_exposure_deltas.append(float(exposure.Get()) - baseline)
    if (
        realized_camera_offset.shape != (3,)
        or not np.all(np.isfinite(realized_camera_offset))
        or realized_scale.shape != (3,)
        or not np.all(np.isfinite(realized_scale))
        or not np.allclose(realized_scale, realized_scale[0], atol=1e-12, rtol=0.0)
        or not realized_exposure_deltas
        or any(
            abs(delta - plan.light_exposure_delta) > 1e-9
            for delta in realized_exposure_deltas
        )
    ):
        raise RuntimeError("unknown-start variant did not realize its plan")
    return {
        "camera_offset_m": realized_camera_offset.tolist(),
        "light_exposure_delta": realized_exposure_deltas[0],
        "socket_scale": float(realized_scale[0]),
    }
from sim.exploration import build_exploration_plan
from sim.recording import RecordingLabel, RecordingMoment
from sim.unknown_start_reset import (
    UNKNOWN_START_RESET_CONTRACT,
    UnknownStartResetEvidence,
    UnknownStartResetPhase,
    UnknownStartSampleRealization,
    UnknownStartWorkspaceState,
)


async def authenticate_unknown_start_reset(
    recording_id: str,
    seed: int,
    source_revision: str,
    runtime_source_fingerprint: str,
) -> dict[str, Any]:
    """Capture and authenticate one reset without running inference or motion."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("unknown-start reset source revision is invalid")
    authenticate_runtime_source(runtime_source_fingerprint)
    sample = UNKNOWN_START_RESET_CONTRACT.draw(seed, forbidden_seeds=set())
    profile = ExplorationRecordingProfile.for_mode(
        ExplorationRecordingMode.CONTACT_INSERTION
    )
    plan = profile.apply_to_plan(
        build_exploration_plan(seed, sample.split)
    )
    original_rendering_dt = await prepare_recording_stage(plan.sample_period_seconds)
    recorder = None
    timeline = None
    completed = False
    try:
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        variant = _apply_variant_with_readback(stage, plan)
        preparation = prepare_fixed_joint_plug(stage)
        recorder = DemoRecorder(
            recording_id,
            fps=DROID_FPS,
            minimum_stage_frames=0,
            camera_specs=JEPA_WM_CAMERA_SPECS,
            metadata={
                "task": "unknown_start_reset_authentication",
                "contract_fingerprint": UNKNOWN_START_RESET_CONTRACT.fingerprint,
                "sample": sample.to_dict(),
                "sample_fingerprint": sample.fingerprint,
                "source_revision": source_revision,
                "runtime_source_fingerprint": runtime_source_fingerprint,
                "applied_actions": 0,
                "prefix_replay_frames": 0,
            },
        )
        timeline = omni.timeline.get_timeline_interface()
        await recorder.initialize()
        hand_sensor = contact_sensor(stage, create=True)
        plug_sensor = connector_contact_sensor(stage, create=True)

        def observe_safety() -> ContactReading:
            hand_collision, hand_force = read_contact(hand_sensor)
            plug_collision, plug_force = read_contact(plug_sensor)
            reading = ContactReading(
                hand_collision or plug_collision,
                max(hand_force, plug_force),
            )
            if (
                reading.collision_detected
                or reading.force_newtons
                > InsertionTaskLimits().maximum_contact_force_newtons
            ):
                raise RuntimeError("unknown-start reset exceeded its safety gate")
            return reading

        await omni.kit.app.get_app().next_update_async()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        attachment = preparation.bind_physics(RigidPrim(PLUG_PATH))
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        ready = solve_waypoints()[0]
        command = JointCommand(
            ready.arm_positions + np.asarray(sample.initial_arm_offset_radians),
            ready.waypoint.gripper_width_m,
        )
        timeline.play()
        actuators.set_reset_state(command)
        safety = await advance_physics_updates(16, observe_safety)
        actual = actuators.actual_command()
        snapshot = recording_snapshot(
            RecordingLabel(RecordingMoment.INITIAL),
            ObservationStage.APPROACHING_CABLE,
            actual,
            attachment,
            safety=recording_safety_telemetry(command, actual, safety),
        )
        await recorder.capture_current(snapshot)
        connector_position, _ = attachment.world_pose()
        socket_position, _ = world_pose(stage.GetPrimAtPath(SOCKET_PATH))
        evidence = UnknownStartResetEvidence(
            sample=sample,
            workspace=UnknownStartWorkspaceState(
                connector_position_m=tuple(float(value) for value in connector_position),
                socket_position_m=tuple(float(value) for value in socket_position),
                end_effector_position_m=tuple(
                    float(value) for value in snapshot.end_effector_world_position
                ),
                socket_scale=float(variant["socket_scale"]),
            ),
            realization=UnknownStartSampleRealization(
                initial_arm_offset_radians=tuple(
                    float(value)
                    for value in actual.arm_positions - ready.arm_positions
                ),
                camera_offset_m=tuple(
                    float(value) for value in variant["camera_offset_m"]
                ),
                light_exposure_delta=float(variant["light_exposure_delta"]),
            ),
            observed_arm_positions_radians=tuple(
                float(value) for value in actual.arm_positions
            ),
            observed_gripper_width_m=float(actual.gripper_width_m),
            realized_sample_fingerprint=sample.fingerprint,
            plug_attached=attachment.attached,
            collision_detected=safety.collision_detected,
            contact_force_newtons=safety.force_newtons,
            direct_state_setting_count=1,
            prefix_replay_frames=0,
            applied_actions=0,
            phase=UnknownStartResetPhase.RESET_AUTHENTICATION,
        )
        evidence.validate(UNKNOWN_START_RESET_CONTRACT)
        completed = True
    except Exception:
        if recorder is not None:
            recorder.abort()
        raise
    finally:
        RenderingManager.set_dt(original_rendering_dt)
        if timeline is not None:
            timeline.stop()

    if recorder is None or not completed:
        raise RuntimeError("unknown-start reset did not complete")
    output = recorder.finish()
    evidence_path = output / "unknown_start_reset_evidence.json"
    write_json_atomic(evidence_path, evidence.to_dict())
    result = {
        "schema": "quantis.unknown_start_reset_capture.v1",
        "status": "captured",
        "recording_id": recording_id,
        "source_revision": source_revision,
        "runtime_source_fingerprint": runtime_source_fingerprint,
        "contract_fingerprint": UNKNOWN_START_RESET_CONTRACT.fingerprint,
        "sample_fingerprint": sample.fingerprint,
        "evidence": str(evidence_path),
        "evidence_fingerprint": sha256(evidence_path.read_bytes()).hexdigest(),
        "applied_actions": 0,
    }
    write_json_atomic(output / "CAPTURE.json", result)
    return result
