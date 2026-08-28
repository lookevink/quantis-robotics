"""Shared articulation and synchronized-capture mechanics for Isaac tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any, Protocol

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DroidPose
from jepa_wm.insertion_contract import COMPLIANT_COLLISION_PARTS
from sim.isaac_demo_scene import (
    PLUG_PATH,
    RIGHT_GRIPPER_OFFSET_IN_HAND_METERS,
    ROBOT_PATH,
    STAGE_PATH,
    world_pose,
)
from sim.recording import (
    RecordingLabel,
    RecordingSafetyTelemetry,
    RecordingSnapshot,
)


@dataclass(frozen=True)
class JointCommand:
    arm_positions: np.ndarray
    gripper_width_m: float


@dataclass(frozen=True)
class ContactReading:
    collision_detected: bool = False
    force_newtons: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.collision_detected, bool)
            or isinstance(self.force_newtons, bool)
            or not isfinite(self.force_newtons)
            or self.force_newtons < 0.0
        ):
            raise ValueError("contact reading is invalid")

    def peak(self, other: ContactReading) -> ContactReading:
        return ContactReading(
            self.collision_detected or other.collision_detected,
            max(self.force_newtons, other.force_newtons),
        )


def recording_safety_telemetry(
    commanded: JointCommand,
    actual: JointCommand,
    contact: ContactReading,
) -> RecordingSafetyTelemetry:
    """Bind live tracking and contact evidence to one recorded command."""

    return RecordingSafetyTelemetry(
        collision_detected=contact.collision_detected,
        contact_force_newtons=contact.force_newtons,
        arm_tracking_error_rad=float(
            np.max(np.abs(actual.arm_positions - commanded.arm_positions))
        ),
        gripper_tracking_error_m=abs(
            actual.gripper_width_m - commanded.gripper_width_m
        ),
    )


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

    def _set_drive_targets(self, command: JointCommand) -> float:
        for attribute, target_degrees in zip(
            self.arm_attributes, np.rad2deg(command.arm_positions)
        ):
            attribute.Set(float(target_degrees))
        finger_position = command.gripper_width_m / 2.0
        for attribute in self.finger_attributes:
            attribute.Set(finger_position)
        return finger_position

    def apply_drive_command(self, command: JointCommand) -> None:
        """Set drive targets without directly changing articulation state."""

        finger_position = self._set_drive_targets(command)
        self.articulation.set_dof_position_targets(
            np.asarray(command.arm_positions, dtype=np.float64),
            dof_indices=np.arange(7),
        )
        self.articulation.set_dof_position_targets(
            np.asarray([finger_position], dtype=np.float64),
            dof_indices=np.asarray([7]),
        )

    def set_reset_state(self, command: JointCommand) -> None:
        """Set targets and DOF state for explicit reset or initialization only."""

        finger_position = self._set_drive_targets(command)

        self.articulation.set_dof_positions(
            positions=command.arm_positions,
            dof_indices=np.arange(7),
        )
        self.articulation.set_dof_positions(
            positions=np.array([finger_position, finger_position]),
            dof_indices=np.array([7, 8]),
        )


class PlugMotion(Protocol):
    prim: Any
    hand_prim: Any

    @property
    def attached(self) -> bool: ...

    def world_pose(self) -> tuple[np.ndarray, np.ndarray]: ...

    def follow(self, hand_position: np.ndarray) -> None: ...

    def attach(self, hand_position: np.ndarray) -> None: ...

    def detach_at(self, position: np.ndarray) -> None: ...

    def remove_load_for_diagnostic(self) -> None: ...


@dataclass
class KinematicPlugMotion:
    prim: Any
    hand_prim: Any
    hand_to_plug_offset: np.ndarray | None = None

    @property
    def attached(self) -> bool:
        return self.hand_to_plug_offset is not None

    def world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return world_pose(self.prim)

    def follow(self, hand_position: np.ndarray) -> None:
        if not self.attached:
            return
        position = hand_position + self.hand_to_plug_offset
        from pxr import Gf

        translate = self.prim.GetAttribute("xformOp:translate")
        if not translate.IsValid():
            raise RuntimeError(f"{PLUG_PATH} has no xformOp:translate attribute")
        translate.Set(Gf.Vec3d(*position))

    def attach(self, hand_position: np.ndarray) -> None:
        plug_position, _ = self.world_pose()
        self.hand_to_plug_offset = plug_position - hand_position

    def detach_at(self, position: np.ndarray) -> None:
        from pxr import Gf

        self.hand_to_plug_offset = None
        self.prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*position))

    def remove_load_for_diagnostic(self) -> None:
        self.hand_to_plug_offset = None


@dataclass
class FixedJointPlugMotion:
    prim: Any
    hand_prim: Any
    rigid_prim: Any
    fixed_joint: Any
    hand_to_plug_offset: np.ndarray | None = None

    @property
    def attached(self) -> bool:
        return self.hand_to_plug_offset is not None

    def world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations = self.rigid_prim.get_world_poses()
        if hasattr(positions, "numpy"):
            positions = positions.numpy()
        if hasattr(orientations, "numpy"):
            orientations = orientations.numpy()
        position_values = np.asarray(positions, dtype=np.float64)
        orientation_values = np.asarray(orientations, dtype=np.float64)
        if position_values.shape != (1, 3) or orientation_values.shape != (1, 4):
            raise RuntimeError("plug rigid body returned invalid world poses")
        return position_values[0], orientation_values[0]

    def follow(self, hand_position: np.ndarray) -> None:
        del hand_position

    def attach(self, hand_position: np.ndarray) -> None:
        plug_position, plug_orientation = self.world_pose()
        self.hand_to_plug_offset = plug_position - hand_position
        self._enable_fixed_joint(hand_position, plug_position, plug_orientation)

    def detach_at(self, position: np.ndarray) -> None:
        self.hand_to_plug_offset = None
        self.fixed_joint.CreateJointEnabledAttr().Set(False)
        self.rigid_prim.set_world_poses(positions=[position])

    def remove_load_for_diagnostic(self) -> None:
        """Disable the hand load while leaving the plug fixed and non-colliding."""

        from pxr import UsdPhysics

        self.hand_to_plug_offset = None
        self.fixed_joint.CreateJointEnabledAttr().Set(False)
        UsdPhysics.RigidBodyAPI(self.prim).CreateKinematicEnabledAttr().Set(True)

    def _enable_fixed_joint(
        self,
        hand_position: np.ndarray,
        plug_position: np.ndarray,
        plug_orientation_wxyz: np.ndarray,
    ) -> None:
        from pxr import Gf, UsdPhysics
        from scipy.spatial.transform import Rotation

        _, hand_orientation_wxyz = world_pose(self.hand_prim)
        hand_rotation = Rotation.from_quat(
            [
                hand_orientation_wxyz[1],
                hand_orientation_wxyz[2],
                hand_orientation_wxyz[3],
                hand_orientation_wxyz[0],
            ]
        )
        plug_rotation = Rotation.from_quat(
            [
                plug_orientation_wxyz[1],
                plug_orientation_wxyz[2],
                plug_orientation_wxyz[3],
                plug_orientation_wxyz[0],
            ]
        )
        local_position = hand_rotation.inv().apply(plug_position - hand_position)
        local_xyzw = (hand_rotation.inv() * plug_rotation).as_quat()
        self.fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_position))
        self.fixed_joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(
                float(local_xyzw[3]),
                Gf.Vec3f(*[float(value) for value in local_xyzw[:3]]),
            )
        )
        self.fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        self.fixed_joint.CreateLocalRot1Attr().Set(
            Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
        )
        UsdPhysics.RigidBodyAPI(self.prim).CreateKinematicEnabledAttr().Set(False)
        self.fixed_joint.CreateJointEnabledAttr().Set(True)


@dataclass
class PlugCollisionPolicy:
    collision_attributes: list[Any]
    excluded_collision_paths: frozenset[str] = frozenset()

    @property
    def compliant_collision_parts(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    str(attribute.GetPath().GetPrimPath()).rsplit("/", 1)[-1]
                    for attribute in self.collision_attributes
                    if str(attribute.GetPath().GetPrimPath())
                    in self.excluded_collision_paths
                }
            )
        )

    def set_collisions(self, enabled: bool) -> None:
        for attribute in self.collision_attributes:
            prim_path = str(attribute.GetPath().GetPrimPath())
            attribute.Set(enabled and prim_path not in self.excluded_collision_paths)


@dataclass
class PlugAttachment:
    motion: PlugMotion
    collisions: PlugCollisionPolicy

    @property
    def prim(self) -> Any:
        return self.motion.prim

    @property
    def hand_prim(self) -> Any:
        return self.motion.hand_prim

    @property
    def attached(self) -> bool:
        return self.motion.attached

    @property
    def compliant_collision_parts(self) -> tuple[str, ...]:
        return self.collisions.compliant_collision_parts

    def world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.motion.world_pose()

    def follow(self, hand_position: np.ndarray) -> None:
        self.motion.follow(hand_position)

    def attach(self, hand_position: np.ndarray) -> None:
        self.motion.attach(hand_position)
        if isinstance(self.motion, KinematicPlugMotion):
            self.set_collisions(False)

    def detach_at(self, position: np.ndarray) -> None:
        self.motion.detach_at(position)

    def remove_load_for_diagnostic(self) -> None:
        self.motion.remove_load_for_diagnostic()
        self.set_collisions(False)

    def set_collisions(self, enabled: bool) -> None:
        self.collisions.set_collisions(enabled)

    def with_refreshed_physics(self, rigid_prim: Any) -> PlugAttachment:
        """Rebind tensor-backed plug motion while preserving attachment state."""

        if not isinstance(self.motion, FixedJointPlugMotion):
            return self
        return type(self).from_prior_generation(self, rigid_prim)

    @classmethod
    def from_prior_generation(
        cls,
        prior: Any,
        rigid_prim: Any,
    ) -> PlugAttachment:
        """Rebuild a fixed-joint attachment without stale class dispatch."""

        try:
            motion = prior.motion
            collisions = prior.collisions
            offset = motion.hand_to_plug_offset
            return cls(
                FixedJointPlugMotion(
                    motion.prim,
                    motion.hand_prim,
                    rigid_prim,
                    motion.fixed_joint,
                    None if offset is None else offset.copy(),
                ),
                PlugCollisionPolicy(
                    list(collisions.collision_attributes),
                    frozenset(collisions.excluded_collision_paths),
                ),
            )
        except (AttributeError, TypeError, ValueError) as error:
            raise RuntimeError(
                "live insertion attachment cannot be refreshed"
            ) from error


@dataclass(frozen=True)
class FixedJointPlugPreparation:
    prim: Any
    hand_prim: Any
    fixed_joint: Any
    collisions: PlugCollisionPolicy

    @property
    def compliant_collision_parts(self) -> tuple[str, ...]:
        return self.collisions.compliant_collision_parts

    def bind_physics(self, rigid_prim: Any) -> PlugAttachment:
        return PlugAttachment(
            FixedJointPlugMotion(
                self.prim,
                self.hand_prim,
                rigid_prim,
                self.fixed_joint,
            ),
            self.collisions,
        )


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


def _plug_prims(stage: Any) -> tuple[Any, Any, list[Any]]:
    from pxr import UsdPhysics

    plug = stage.GetPrimAtPath(PLUG_PATH)
    if not plug.IsValid():
        raise RuntimeError(f"plug prim is missing: {PLUG_PATH}")
    hand = stage.GetPrimAtPath(f"{ROBOT_PATH}/panda_hand")
    if not hand.IsValid():
        raise RuntimeError(f"robot hand prim is missing: {ROBOT_PATH}/panda_hand")
    UsdPhysics.RigidBodyAPI(plug).CreateKinematicEnabledAttr().Set(True)
    collision_prims = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(plug.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    return plug, hand, collision_prims


def prepare_plug(stage: Any) -> PlugAttachment:
    from pxr import UsdPhysics

    plug, hand, collision_prims = _plug_prims(stage)
    collisions = PlugCollisionPolicy(
        [
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr()
            for prim in collision_prims
        ]
    )
    return PlugAttachment(KinematicPlugMotion(plug, hand), collisions)


def prepare_fixed_joint_plug(stage: Any) -> FixedJointPlugPreparation:
    from pxr import UsdPhysics

    plug, hand, collision_prims = _plug_prims(stage)
    excluded_collision_paths = frozenset(
        str(prim.GetPath())
        for prim in collision_prims
        if prim.GetName() in COMPLIANT_COLLISION_PARTS
    )
    found_parts = frozenset(
        prim.GetName()
        for prim in collision_prims
        if str(prim.GetPath()) in excluded_collision_paths
    )
    if found_parts != frozenset(COMPLIANT_COLLISION_PARTS):
        missing = sorted(set(COMPLIANT_COLLISION_PARTS) - found_parts)
        raise RuntimeError(f"plug compliant collision prims are missing: {missing}")
    fixed_joint = UsdPhysics.FixedJoint.Define(
        stage,
        f"{PLUG_PATH}/QuantisGraspJoint",
    )
    fixed_joint.CreateBody0Rel().SetTargets([hand.GetPath()])
    fixed_joint.CreateBody1Rel().SetTargets([plug.GetPath()])
    fixed_joint.CreateCollisionEnabledAttr().Set(False)
    fixed_joint.CreateJointEnabledAttr().Set(False)
    collisions = PlugCollisionPolicy(
        [
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr()
            for prim in collision_prims
        ],
        excluded_collision_paths,
    )
    collisions.set_collisions(True)
    return FixedJointPlugPreparation(plug, hand, fixed_joint, collisions)


def recording_snapshot(
    phase: RecordingLabel,
    stage: ObservationStage,
    command: JointCommand,
    attachment: PlugAttachment,
    *,
    safety: RecordingSafetyTelemetry = RecordingSafetyTelemetry(),
) -> RecordingSnapshot:
    hand_position, hand_orientation = world_pose(attachment.hand_prim)
    robot = attachment.hand_prim.GetStage().GetPrimAtPath(ROBOT_PATH)
    base_position, base_orientation = world_pose(robot)
    plug_position, plug_orientation = attachment.world_pose()
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
        simulation_time_seconds=physics_simulation_time_seconds(),
        safety=safety,
    )


def _smoothstep(progress: float) -> float:
    return progress * progress * (3.0 - 2.0 * progress)


def physics_simulation_time_seconds() -> float:
    """Return the monotonic PhysX clock used by observed motion intervals."""

    from isaacsim.core.simulation_manager import SimulationManager

    return float(SimulationManager.get_simulation_time())


def resume_live_simulation(timeline: Any) -> bool:
    """Enable app-driven physics and report whether playback was resumed."""

    timeline.set_auto_update(True)
    resumed = not timeline.is_playing()
    if resumed:
        timeline.play()
    return resumed


async def _advance_sample(
    sample_period_seconds: float | None,
    observe_safety: Callable[[], ContactReading] | None = None,
) -> ContactReading:
    import omni.kit.app

    app = omni.kit.app.get_app()
    started_at = physics_simulation_time_seconds()
    latest_safety = ContactReading()
    if sample_period_seconds is None:
        for _ in range(120):
            await app.next_update_async()
            if observe_safety is not None:
                latest_safety = latest_safety.peak(observe_safety())
            if physics_simulation_time_seconds() > started_at:
                return latest_safety
        raise RuntimeError("simulation did not advance within 120 updates")
    maximum_updates = max(120, ceil(sample_period_seconds * 1000))
    for _ in range(maximum_updates):
        await app.next_update_async()
        if observe_safety is not None:
            latest_safety = latest_safety.peak(observe_safety())
        if (
            physics_simulation_time_seconds() - started_at
            >= sample_period_seconds - 1e-6
        ):
            return latest_safety
    raise RuntimeError(
        f"simulation did not advance {sample_period_seconds:.3f}s "
        f"within {maximum_updates} updates"
    )


async def advance_physics_updates(
    update_count: int,
    observe_safety: Callable[[], ContactReading] | None = None,
) -> ContactReading:
    """Advance exact render/physics updates while polling a live interlock."""

    if update_count <= 0:
        raise ValueError("update_count must be positive")
    latest = ContactReading()
    for _ in range(update_count):
        latest = latest.peak(await _advance_sample(None, observe_safety))
    return latest


async def advance_simulation_period(
    period_seconds: float,
    observe_safety: Callable[[], ContactReading] | None = None,
) -> ContactReading:
    """Advance one measured simulation interval through the live interlock."""

    if not isfinite(period_seconds) or period_seconds <= 0.0:
        raise ValueError("simulation period must be finite and positive")
    return await _advance_sample(period_seconds, observe_safety)


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
    observe_safety: Callable[[], ContactReading] | None = None,
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
        actuators.apply_drive_command(command)
        contact_reading = await _advance_sample(
            sample_period_seconds,
            observe_safety,
        )
        attachment.follow(world_pose(attachment.hand_prim)[0])
        actual = actuators.actual_command()
        safety = recording_safety_telemetry(command, actual, contact_reading)
        snapshot = recording_snapshot(
            phase,
            stage,
            command,
            attachment,
            safety=safety,
        )
        if snapshot.simulation_time_seconds is not None:
            sample_times.append(snapshot.simulation_time_seconds)
        if recorder is not None:
            await recorder.capture_current(snapshot)
    return tuple(sample_times)
