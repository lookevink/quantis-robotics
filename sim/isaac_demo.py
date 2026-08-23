"""Live execution facade for the deterministic cable plug-in demo."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DROID_FPS, DroidPose
from sim.demo_sequence import Phase, PlugAction
from sim.isaac_demo_camera import (
    DEMO_FPS,
    RECORDING_JOB_ROOT,
    DemoRecorder,
    capture_cameras,
    configure_wrist_camera,
)
from sim.isaac_demo_kinematics import preflight_report, solve_waypoints
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


@dataclass(frozen=True)
class JointCommand:
    arm_positions: np.ndarray
    gripper_width_m: float


@dataclass
class Actuators:
    articulation: Any
    arm_attributes: list[Any]
    finger_attributes: list[Any]

    def current_command(self) -> JointCommand:
        arm = np.deg2rad(
            np.array(
                [attribute.Get() for attribute in self.arm_attributes], dtype=np.float64
            )
        )
        width = float(self.finger_attributes[0].Get()) * 2.0
        return JointCommand(arm, width)

    def apply(self, command: JointCommand) -> None:
        for attribute, target_degrees in zip(
            self.arm_attributes, np.rad2deg(command.arm_positions)
        ):
            attribute.Set(float(target_degrees))
        finger_position = command.gripper_width_m / 2.0
        for attribute in self.finger_attributes:
            attribute.Set(finger_position)

        self.articulation.set_dof_positions(
            positions=command.arm_positions, dof_indices=np.arange(7)
        )
        self.articulation.set_dof_positions(
            positions=np.array([finger_position, finger_position]),
            dof_indices=np.array([7, 8]),
        )


@dataclass
class PlugAttachment:
    prim: Any
    hand_prim: Any
    collision_attributes: list[Any]
    hand_to_plug_offset: np.ndarray | None = None

    @property
    def attached(self) -> bool:
        return self.hand_to_plug_offset is not None

    def follow(self, hand_position: np.ndarray) -> None:
        if not self.attached:
            return
        from pxr import Gf

        translate = self.prim.GetAttribute("xformOp:translate")
        if not translate.IsValid():
            raise RuntimeError(f"{PLUG_PATH} has no xformOp:translate attribute")
        translate.Set(Gf.Vec3d(*(hand_position + self.hand_to_plug_offset)))

    def attach(self, hand_position: np.ndarray) -> None:
        for attribute in self.collision_attributes:
            attribute.Set(False)
        plug_position, _ = world_pose(self.prim)
        self.hand_to_plug_offset = plug_position - hand_position

    def detach_at(self, position: np.ndarray) -> None:
        from pxr import Gf

        self.hand_to_plug_offset = None
        self.prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))


async def reset_demo() -> dict[str, Any]:
    """Stop physics and reopen the saved reusable starting stage."""

    import omni.timeline
    import omni.usd

    omni.timeline.get_timeline_interface().stop()
    success, error = await omni.usd.get_context().open_stage_async(STAGE_PATH)
    if not success:
        raise RuntimeError(f"failed to open {STAGE_PATH}: {error}")
    stage = omni.usd.get_context().get_stage()
    # The Sdf layer registry can retain unsaved edits when reopening the same
    # identifier. Reload explicitly so reset always means the on-disk source.
    stage.GetRootLayer().Reload()
    stage.SetEditTarget(stage.GetSessionLayer())
    return {
        "status": "ready",
        "stage": STAGE_PATH,
        "wrist_camera": configure_wrist_camera(),
    }


def _find_joint(stage: Any, name: str) -> Any:
    for prim in stage.Traverse():
        if prim.GetName() == name:
            return prim
    raise RuntimeError(f"joint prim is missing: {name}")


def _create_actuators(stage: Any, articulation: Any) -> Actuators:
    from pxr import UsdPhysics

    arm_attributes = []
    for index in range(1, 8):
        joint = _find_joint(stage, f"panda_joint{index}")
        arm_attributes.append(
            UsdPhysics.DriveAPI.Get(joint, "angular").GetTargetPositionAttr()
        )

    # Finger joint 2 is a PhysX mimic joint in Isaac Sim 6; only joint 1 owns
    # a drive. The articulation state still exposes both finger DOFs.
    finger_joint = _find_joint(stage, "panda_finger_joint1")
    finger_attributes = [
        UsdPhysics.DriveAPI.Get(finger_joint, "linear").GetTargetPositionAttr()
    ]
    return Actuators(articulation, arm_attributes, finger_attributes)


def _prepare_plug(stage: Any) -> PlugAttachment:
    from pxr import UsdPhysics

    plug = stage.GetPrimAtPath(PLUG_PATH)
    if not plug.IsValid():
        raise RuntimeError(f"plug prim is missing: {PLUG_PATH}")
    hand = stage.GetPrimAtPath(f"{ROBOT_PATH}/panda_hand")
    if not hand.IsValid():
        raise RuntimeError(f"robot hand prim is missing: {ROBOT_PATH}/panda_hand")
    UsdPhysics.RigidBodyAPI(plug).CreateKinematicEnabledAttr().Set(True)
    collision_attributes = [
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr()
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(plug.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    return PlugAttachment(plug, hand, collision_attributes)


def _smoothstep(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def _recording_snapshot(
    phase: RecordingLabel,
    stage: ObservationStage,
    command: JointCommand,
    attachment: PlugAttachment,
) -> RecordingSnapshot:
    hand_position, hand_orientation = world_pose(attachment.hand_prim)
    robot = attachment.hand_prim.GetStage().GetPrimAtPath(ROBOT_PATH)
    base_position, base_orientation = world_pose(robot)
    return RecordingSnapshot(
        phase=phase,
        stage=stage,
        arm_positions=command.arm_positions,
        gripper_width_m=command.gripper_width_m,
        plug_position=world_pose(attachment.prim)[0],
        plug_attached=attachment.attached,
        end_effector_pose=DroidPose.from_world_poses(
            base_position,
            base_orientation,
            hand_position,
            hand_orientation,
            command.gripper_width_m,
        ),
    )


async def _move_targets(
    actuators: Actuators,
    start: JointCommand,
    end: JointCommand,
    duration_seconds: float,
    attachment: PlugAttachment,
    *,
    phase: RecordingLabel,
    stage: ObservationStage,
    recorder: DemoRecorder | None,
) -> None:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    hand = omni.usd.get_context().get_stage().GetPrimAtPath(f"{ROBOT_PATH}/panda_hand")
    # Keep recorded motion samples aligned with the video manifest. Non-recorded
    # interactive runs retain the original lightweight eight authored targets.
    authored_fps = recorder.fps if recorder is not None else 8
    frames = max(1, ceil(duration_seconds * authored_fps))

    for frame in range(1, frames + 1):
        blend = _smoothstep(frame / frames)
        command = JointCommand(
            start.arm_positions + (end.arm_positions - start.arm_positions) * blend,
            start.gripper_width_m
            + (end.gripper_width_m - start.gripper_width_m) * blend,
        )
        actuators.apply(command)
        await app.next_update_async()
        attachment.follow(world_pose(hand)[0])
        if recorder is not None:
            await recorder.capture(
                _recording_snapshot(phase, stage, command, attachment),
                advance=False,
            )


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
                _recording_snapshot(phase, stage, command, attachment),
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
    attachment = _prepare_plug(stage)

    await omni.kit.app.get_app().next_update_async()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    actuators = _create_actuators(stage, Articulation(ROBOT_PATH))
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
                _recording_snapshot(
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
            await _move_targets(
                actuators,
                current,
                target,
                duration,
                attachment,
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
                    _recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.GRASP),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
            elif waypoint.phase == Phase.PRE_INSERTION:
                await _fill_stage_observations(
                    recorder,
                    _recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.PRE_INSERTION),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
                observation_stage = ObservationStage.ALIGNED_WITH_SOCKET
                await _fill_stage_observations(
                    recorder,
                    _recording_snapshot(
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
                    _recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.INSERT),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )

            if waypoint.plug_action == PlugAction.ATTACH:
                closed = JointCommand(current.arm_positions, waypoint.gripper_width_m)
                await _move_targets(
                    actuators,
                    current,
                    closed,
                    0.6,
                    attachment,
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
                        _recording_snapshot(
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
                        _recording_snapshot(
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
