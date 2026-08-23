"""Live execution facade for the deterministic cable plug-in demo."""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DROID_FPS
from sim.demo_sequence import Phase, PlugAction
from sim.exploration import DatasetSplit
from sim.isaac_demo_camera import (
    DEMO_FPS,
    RECORDING_JOB_ROOT,
    DemoRecorder,
    capture_cameras,
)
from sim.isaac_demo_kinematics import preflight_report, solve_waypoints
from sim.isaac_demo_runtime import (
    Actuators,
    JointCommand,
    PlugAttachment,
    create_actuators,
    move_joint_command,
    prepare_plug,
    recording_snapshot,
    reset_stage,
)
from sim.isaac_demo_scene import (
    PLUG_PATH,
    ROBOT_PATH,
    SOCKET_PATH,
    STAGE_PATH,
    world_pose,
)
from sim.recording import (
    RecordingLabel,
    RecordingMoment,
    RecordingSnapshot,
)
from sim.recording_jobs import RecordingJobManager


_RECORDING_JOBS = RecordingJobManager(RECORDING_JOB_ROOT)


async def reset_demo() -> dict[str, Any]:
    """Stop physics and reopen the saved reusable starting stage."""

    return await reset_stage()


async def _settle_at_target(
    target_hand_position: np.ndarray,
    attachment: PlugAttachment,
    command: JointCommand,
    *,
    phase: RecordingLabel,
    stage: ObservationStage,
    recorder: DemoRecorder | None,
    max_updates: int = 4,
    tolerance_m: float = 0.012,
) -> float:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    hand = omni.usd.get_context().get_stage().GetPrimAtPath(f"{ROBOT_PATH}/panda_hand")
    error = float("inf")
    for _ in range(max_updates):
        await app.next_update_async()
        hand_position, _ = world_pose(hand)
        error = float(np.linalg.norm(hand_position - target_hand_position))
        attachment.follow(hand_position)
        if recorder is not None:
            await recorder.capture(
                recording_snapshot(phase, stage, command, attachment),
                advance=False,
            )
        if error <= tolerance_m:
            return error
    return error


async def _fill_stage_observations(
    recorder: DemoRecorder | None,
    snapshot: RecordingSnapshot,
) -> None:
    if recorder is None:
        return
    while recorder.stage_frame_count(snapshot.stage) < recorder.minimum_stage_frames:
        await recorder.capture(snapshot)


async def run_demo(recorder: DemoRecorder | None = None) -> dict[str, Any]:
    """Execute the preflighted sequence and export its final visual state."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager

    stage = omni.usd.get_context().get_stage()
    solved = solve_waypoints()
    stage.SetEditTarget(stage.GetSessionLayer())
    attachment = prepare_plug(stage)

    await omni.kit.app.get_app().next_update_async()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    actuators = create_actuators(stage, Articulation(ROBOT_PATH))
    current = actuators.current_command()
    observation_stage = ObservationStage.APPROACHING_CABLE

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    phase_reports = []
    completed = False
    try:
        await omni.kit.app.get_app().next_update_async()
        if recorder is not None:
            await recorder.capture(
                recording_snapshot(
                    RecordingLabel(RecordingMoment.INITIAL),
                    observation_stage,
                    current,
                    attachment,
                )
            )
        for result in solved:
            waypoint = result.waypoint
            motion_width = (
                current.gripper_width_m
                if waypoint.phase == Phase.GRASP
                else waypoint.gripper_width_m
            )
            target = JointCommand(result.arm_positions, motion_width)
            max_delta = float(
                np.max(np.abs(np.rad2deg(target.arm_positions - current.arm_positions)))
            )
            duration = min(4.0, max(0.8, max_delta / 45.0))
            authored_fps = recorder.fps if recorder is not None else 8
            await move_joint_command(
                actuators,
                current,
                target,
                attachment,
                frame_count=max(1, ceil(duration * authored_fps)),
                phase=RecordingLabel(RecordingMoment.MOTION, waypoint.phase),
                stage=observation_stage,
                recorder=recorder,
            )
            current = target

            settle_error = await _settle_at_target(
                result.hand_position,
                attachment,
                current,
                phase=RecordingLabel(RecordingMoment.SETTLE, waypoint.phase),
                stage=observation_stage,
                recorder=recorder,
            )
            if settle_error > 0.012:
                raise RuntimeError(
                    f"arm failed to settle at {waypoint.phase.value}: "
                    f"{settle_error:.4f} m hand error"
                )

            if waypoint.phase == Phase.GRASP:
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.GRASP),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
            elif waypoint.phase == Phase.PRE_INSERTION:
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.PRE_INSERTION),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
                observation_stage = ObservationStage.ALIGNED_WITH_SOCKET
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.PRE_INSERTION),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
            elif waypoint.phase == Phase.INSERT:
                observation_stage = ObservationStage.PLUG_SEATED
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.INSERT),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )

            if waypoint.plug_action == PlugAction.ATTACH:
                closed = JointCommand(current.arm_positions, waypoint.gripper_width_m)
                authored_fps = recorder.fps if recorder is not None else 8
                await move_joint_command(
                    actuators,
                    current,
                    closed,
                    attachment,
                    frame_count=max(1, ceil(0.6 * authored_fps)),
                    phase=RecordingLabel(RecordingMoment.CLOSE, Phase.GRASP),
                    stage=observation_stage,
                    recorder=recorder,
                )
                current = closed
                hand = stage.GetPrimAtPath(f"{ROBOT_PATH}/panda_hand")
                attachment.attach(world_pose(hand)[0])
                observation_stage = ObservationStage.CABLE_GRASPED
                if recorder is not None:
                    await recorder.capture(
                        recording_snapshot(
                            RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP),
                            observation_stage,
                            current,
                            attachment,
                        )
                    )
            elif waypoint.plug_action == PlugAction.DETACH:
                socket_position, _ = world_pose(stage.GetPrimAtPath(SOCKET_PATH))
                attachment.detach_at(socket_position)
                if recorder is not None:
                    await recorder.capture(
                        recording_snapshot(
                            RecordingLabel(RecordingMoment.COMPLETE, Phase.RELEASE),
                            observation_stage,
                            current,
                            attachment,
                        )
                    )

            phase_reports.append(
                {
                    "phase": waypoint.phase.value,
                    "duration_seconds": duration,
                    "target_degrees": np.rad2deg(current.arm_positions)
                    .round(3)
                    .tolist(),
                    "gripper_width_m": current.gripper_width_m,
                    "settle_error_m": settle_error,
                }
            )
        completed = True
    finally:
        if completed:
            timeline.pause()
        else:
            timeline.stop()

    result_stage = STAGE_PATH.replace(".usda", "_sequence_result.usda")
    stage.Export(result_stage)
    return {
        "status": "complete",
        "result_stage": result_stage,
        "phases": phase_reports,
        "plug_position": world_pose(attachment.prim)[0].round(6).tolist(),
    }


async def _record_demo(
    recording_id: str,
    *,
    fps: int,
    minimum_stage_frames: int,
) -> dict[str, Any]:
    await reset_demo()
    recorder = DemoRecorder(
        recording_id,
        fps=fps,
        minimum_stage_frames=minimum_stage_frames,
    )
    try:
        await recorder.initialize()
        result = await run_demo(recorder=recorder)
    except Exception:
        recorder.abort()
        raise
    output_dir = recorder.finish()
    return {
        **result,
        "recording_id": recording_id,
        "output_directory": str(output_dir),
        "videos": {camera: str(path) for camera, path in recorder.video_paths.items()},
    }


async def record_demo(recording_id: str) -> dict[str, Any]:
    """Capture the presentation recording with complete stage windows."""

    from jepa.contract import DEFAULT_FRAMES

    return await _record_demo(
        recording_id,
        fps=DEMO_FPS,
        minimum_stage_frames=DEFAULT_FRAMES,
    )


async def record_action_trajectory(recording_id: str) -> dict[str, Any]:
    """Capture motion-only frames at JEPA-WM's four-frame-per-second rate."""

    return await _record_demo(
        recording_id,
        fps=DROID_FPS,
        minimum_stage_frames=0,
    )


def start_recording(recording_id: str) -> dict[str, Any]:
    """Start a long recording without holding the Python server connection."""

    return _RECORDING_JOBS.start(recording_id, record_demo)


def start_action_recording(recording_id: str) -> dict[str, Any]:
    """Start a DROID-action trajectory capture for offline world-model evaluation."""

    return _RECORDING_JOBS.start(recording_id, record_action_trajectory)


def start_exploration_recording(
    recording_id: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    """Start a seeded domain-exploration capture in the simulator background."""

    from sim.isaac_exploration import record_exploration_trajectory

    dataset_split = DatasetSplit(split)
    return _RECORDING_JOBS.start(
        recording_id,
        lambda job_recording_id: record_exploration_trajectory(
            job_recording_id,
            seed,
            dataset_split,
        ),
    )
