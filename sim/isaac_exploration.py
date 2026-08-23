"""Isaac runtime for seeded JEPA-WM domain exploration recordings."""

from __future__ import annotations

from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DROID_FPS
from sim.demo_sequence import Phase
from sim.exploration import (
    DatasetSplit,
    ExplorationPlan,
    SegmentOutcome,
    build_exploration_plan,
    validate_sample_times,
)
from sim.isaac_demo_camera import (
    JEPA_WM_CAMERA_SPECS,
    DemoRecorder,
    configure_wrist_camera,
)
from sim.isaac_demo_kinematics import solve_waypoints
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    move_joint_command,
    prepare_plug,
    recording_snapshot,
)
from sim.isaac_demo_scene import PLUG_PATH, ROBOT_PATH, SOCKET_PATH
from sim.recording import RecordingLabel, RecordingMoment


def _apply_variant(stage: Any, plan: ExplorationPlan) -> None:
    """Author seeded camera, task-geometry, and lighting changes in-session."""

    from pxr import Gf, UsdGeom

    configure_wrist_camera(plan.camera_offset_m)
    scene_offset = np.asarray(plan.scene_offset_m, dtype=np.float64)
    for path in (PLUG_PATH, SOCKET_PATH):
        prim = stage.GetPrimAtPath(path)
        translation = prim.GetAttribute("xformOp:translate")
        if not translation.IsValid():
            raise RuntimeError(f"exploration prim has no translation: {path}")
        translation.Set(Gf.Vec3d(*(np.asarray(translation.Get()) + scene_offset)))

    socket = stage.GetPrimAtPath(SOCKET_PATH)
    xformable = UsdGeom.Xformable(socket)
    scale_op = next(
        (
            operation
            for operation in xformable.GetOrderedXformOps()
            if operation.GetOpType() == UsdGeom.XformOp.TypeScale
        ),
        None,
    )
    if scale_op is None:
        scale_op = xformable.AddScaleOp(
            UsdGeom.XformOp.PrecisionDouble,
            "domainVariant",
        )
        scale_op.Set(Gf.Vec3d(plan.socket_scale))
    else:
        current_scale = scale_op.Get()
        scaled = np.asarray(current_scale, dtype=np.float64) * plan.socket_scale
        scale_op.Set(current_scale.__class__(*scaled))

    for prim in stage.Traverse():
        exposure = prim.GetAttribute("inputs:exposure")
        current = exposure.Get() if exposure.IsValid() else None
        if isinstance(current, (int, float)):
            exposure.Set(float(current) + plan.light_exposure_delta)


def _recording_label(outcome: SegmentOutcome) -> RecordingLabel:
    moment = (
        RecordingMoment.SETTLE
        if outcome == SegmentOutcome.STATIONARY
        else RecordingMoment.MOTION
    )
    return RecordingLabel(moment, Phase.READY)


async def record_exploration_trajectory(
    recording_id: str,
    seed: int,
    split: DatasetSplit,
) -> dict[str, Any]:
    """Capture a seeded, true-4-FPS wrist rollout for domain adaptation."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager
    from sim.isaac_demo_runtime import reset_stage

    plan = build_exploration_plan(seed, split)
    await reset_stage()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    _apply_variant(stage, plan)
    recorder = DemoRecorder(
        recording_id,
        fps=DROID_FPS,
        minimum_stage_frames=0,
        camera_specs=JEPA_WM_CAMERA_SPECS,
        metadata=plan.metadata(),
    )
    timeline = omni.timeline.get_timeline_interface()
    original_rendering_dt = RenderingManager.get_dt()
    completed = False
    sample_times = []
    try:
        await recorder.initialize()
        # Isaac keeps the physics scene at its existing high-frequency dt and
        # advances the timeline by this render interval, yielding one rendered
        # observation for each DROID 4 FPS sample instead of rendering every
        # intermediate physics tick. Configure it before creating the physics
        # tensor view because timeline timing changes invalidate that view.
        RenderingManager.set_dt(plan.sample_period_seconds)
        attachment = prepare_plug(stage)
        await omni.kit.app.get_app().next_update_async()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        ready = solve_waypoints()[0]
        origin = JointCommand(
            ready.arm_positions + np.asarray(plan.initial_arm_offset_radians),
            ready.waypoint.gripper_width_m,
        )
        current = origin
        observation_stage = ObservationStage.APPROACHING_CABLE
        timeline.play()
        await omni.kit.app.get_app().next_update_async()
        actuators.apply(origin)
        for _ in range(16):
            await omni.kit.app.get_app().next_update_async()
        initial = recording_snapshot(
            RecordingLabel(RecordingMoment.INITIAL),
            observation_stage,
            current,
            attachment,
        )
        await recorder.capture(initial, advance=False)
        if initial.simulation_time_seconds is not None:
            sample_times.append(initial.simulation_time_seconds)

        for target in plan.targets:
            command = JointCommand(
                origin.arm_positions + np.asarray(target.arm_offset_radians),
                target.gripper_width_m,
            )
            sample_times.extend(
                await move_joint_command(
                    actuators,
                    current,
                    command,
                    attachment,
                    frame_count=target.frames,
                    phase=_recording_label(target.outcome),
                    stage=observation_stage,
                    recorder=recorder,
                    sample_period_seconds=plan.sample_period_seconds,
                )
            )
            current = command
        validate_sample_times(tuple(sample_times), plan.sample_period_seconds)
        completed = True
    except Exception:
        recorder.abort()
        raise
    finally:
        RenderingManager.set_dt(original_rendering_dt)
        if completed:
            timeline.pause()
        else:
            timeline.stop()

    output_dir = recorder.finish()
    return {
        "status": "complete",
        "recording_id": recording_id,
        "output_directory": str(output_dir),
        "frames": recorder.frame_count,
        "metadata": plan.metadata(),
    }
