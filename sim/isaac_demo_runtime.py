"""Shared articulation and synchronized-capture mechanics for Isaac tasks."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DroidPose
from sim.isaac_demo_scene import (
    PLUG_PATH,
    RIGHT_GRIPPER_OFFSET_IN_HAND_METERS,
    ROBOT_PATH,
    STAGE_PATH,
    world_pose,
)
from sim.recording import RecordingLabel, RecordingSnapshot


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
                [attribute.Get() for attribute in self.arm_attributes],
                dtype=np.float64,
            )
        )
        width = float(self.finger_attributes[0].Get()) * 2.0
        return JointCommand(arm, width)

    def actual_command(self) -> JointCommand:
        positions = self.articulation.get_dof_positions()
        if hasattr(positions, "cpu"):
            positions = positions.cpu()
        if hasattr(positions, "numpy"):
            positions = positions.numpy()
        values = np.asarray(positions, dtype=np.float64)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values[0]
        if values.shape != (9,) or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"articulation returned invalid Franka DOF positions: {values.shape}"
            )
        return JointCommand(values[:7].copy(), float(values[7] + values[8]))

    def apply(self, command: JointCommand) -> None:
        for attribute, target_degrees in zip(
            self.arm_attributes, np.rad2deg(command.arm_positions)
        ):
            attribute.Set(float(target_degrees))
        finger_position = command.gripper_width_m / 2.0
        for attribute in self.finger_attributes:
            attribute.Set(finger_position)

        self.articulation.set_dof_positions(
            positions=command.arm_positions,
            dof_indices=np.arange(7),
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


async def reset_stage() -> dict[str, Any]:
    """Stop physics and reopen the saved reusable starting stage."""

    import omni.timeline
    import omni.usd
    from sim.isaac_demo_camera import configure_wrist_camera

    omni.timeline.get_timeline_interface().stop()
    success, error = await omni.usd.get_context().open_stage_async(STAGE_PATH)
    if not success:
        raise RuntimeError(f"failed to open {STAGE_PATH}: {error}")
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Reload()
    session_layer = stage.GetSessionLayer()
    session_layer.Clear()
    stage.SetEditTarget(session_layer)
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


def create_actuators(stage: Any, articulation: Any) -> Actuators:
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


def prepare_plug(stage: Any) -> PlugAttachment:
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


def recording_snapshot(
    phase: RecordingLabel,
    stage: ObservationStage,
    command: JointCommand,
    attachment: PlugAttachment,
) -> RecordingSnapshot:
    import omni.timeline

    hand_position, hand_orientation = world_pose(attachment.hand_prim)
    robot = attachment.hand_prim.GetStage().GetPrimAtPath(ROBOT_PATH)
    base_position, base_orientation = world_pose(robot)
    plug_position, plug_orientation = world_pose(attachment.prim)
    from scipy.spatial.transform import Rotation

    hand_xyzw = np.asarray(
        (
            hand_orientation[1],
            hand_orientation[2],
            hand_orientation[3],
            hand_orientation[0],
        )
    )
    gripper_frame_position = hand_position + Rotation.from_quat(hand_xyzw).apply(
        RIGHT_GRIPPER_OFFSET_IN_HAND_METERS
    )
    return RecordingSnapshot(
        phase=phase,
        stage=stage,
        arm_positions=command.arm_positions,
        gripper_width_m=command.gripper_width_m,
        plug_position=plug_position,
        plug_orientation_wxyz=plug_orientation,
        plug_attached=attachment.attached,
        end_effector_pose=DroidPose.from_world_poses(
            base_position,
            base_orientation,
            hand_position,
            hand_orientation,
            command.gripper_width_m,
        ),
        end_effector_world_position=hand_position,
        gripper_frame_world_position=gripper_frame_position,
        simulation_time_seconds=omni.timeline.get_timeline_interface().get_current_time(),
    )


def _smoothstep(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


async def _advance_sample(sample_period_seconds: float | None) -> None:
    import omni.kit.app
    import omni.timeline

    app = omni.kit.app.get_app()
    if sample_period_seconds is None:
        await app.next_update_async()
        return
    timeline = omni.timeline.get_timeline_interface()
    started_at = timeline.get_current_time()
    maximum_updates = max(120, ceil(sample_period_seconds * 1000))
    for _ in range(maximum_updates):
        await app.next_update_async()
        if timeline.get_current_time() - started_at >= sample_period_seconds - 1e-6:
            return
    raise RuntimeError(
        f"simulation did not advance {sample_period_seconds:.3f}s "
        f"within {maximum_updates} updates"
    )


async def move_joint_command(
    actuators: Actuators,
    start: JointCommand,
    end: JointCommand,
    attachment: PlugAttachment,
    *,
    frame_count: int,
    phase: RecordingLabel,
    stage: ObservationStage,
    recorder: Any | None,
    sample_period_seconds: float | None = None,
) -> tuple[float, ...]:
    """Interpolate one command and capture only after its required sim interval."""

    if frame_count <= 0:
        raise ValueError("frame count must be positive")
    sample_times = []
    for frame in range(1, frame_count + 1):
        blend = _smoothstep(frame / frame_count)
        command = JointCommand(
            start.arm_positions + (end.arm_positions - start.arm_positions) * blend,
            start.gripper_width_m
            + (end.gripper_width_m - start.gripper_width_m) * blend,
        )
        actuators.apply(command)
        await _advance_sample(sample_period_seconds)
        attachment.follow(world_pose(attachment.hand_prim)[0])
        snapshot = recording_snapshot(phase, stage, command, attachment)
        if snapshot.simulation_time_seconds is not None:
            sample_times.append(snapshot.simulation_time_seconds)
        if recorder is not None:
            await recorder.capture(
                snapshot,
                advance=False,
            )
    return tuple(sample_times)
