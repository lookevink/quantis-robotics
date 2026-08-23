"""Kinematics and preflight checks for the live Isaac plug-in demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jepa_wm.action import DroidPose, MAX_GRIPPER_WIDTH_M
from sim.demo_sequence import DemoGeometry, Waypoint, build_demo_sequence
from sim.isaac_demo_scene import (
    ARM_JOINT_TARGETS_DEGREES,
    PLUG_PATH,
    ROBOT_PATH,
    SOCKET_PATH,
    STAGE_PATH,
    matrix_to_wxyz,
    world_pose,
)


@dataclass(frozen=True)
class SolvedWaypoint:
    waypoint: Waypoint
    arm_positions: np.ndarray
    hand_position: np.ndarray
    position_error_m: float
    orientation_error_rad: float


@dataclass(frozen=True)
class SolvedPose:
    target_pose: DroidPose
    arm_positions: np.ndarray
    hand_position: np.ndarray
    gripper_width_m: float
    position_error_m: float
    orientation_error_rad: float


def _rotation_error(left: np.ndarray, right: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    delta = Rotation.from_matrix(left).inv() * Rotation.from_matrix(right)
    return float(delta.magnitude())


def _task_geometry(stage: Any) -> DemoGeometry:
    plug_position, _ = world_pose(stage.GetPrimAtPath(PLUG_PATH))
    socket_position, _ = world_pose(stage.GetPrimAtPath(SOCKET_PATH))
    ready_position = np.array([0.25, plug_position[1], plug_position[2] + 0.16])
    return DemoGeometry(
        plug_position=tuple(plug_position),
        socket_position=tuple(socket_position),
        ready_position=tuple(ready_position),
    )


def _solver_for_stage(stage: Any) -> tuple[Any, np.ndarray, np.ndarray]:
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    from isaacsim.robot_motion.motion_generation.interface_config_loader import (
        load_supported_lula_kinematics_solver_config,
    )

    robot = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot.IsValid():
        raise RuntimeError(f"robot prim is missing: {ROBOT_PATH}")
    solver = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config("Franka"))
    base_position, base_orientation = world_pose(robot)
    solver.set_robot_base_pose(base_position, base_orientation)
    return solver, base_position, base_orientation


def solve_waypoints() -> tuple[SolvedWaypoint, ...]:
    """Solve every task waypoint without advancing simulation physics."""

    import omni.usd
    stage = omni.usd.get_context().get_stage()
    solver, _, _ = _solver_for_stage(stage)

    warm_start = np.deg2rad(ARM_JOINT_TARGETS_DEGREES)
    hand_rotation = solver.compute_forward_kinematics("panda_hand", warm_start)[1]
    gripper_rotation = solver.compute_forward_kinematics("right_gripper", warm_start)[1]
    hand_to_gripper = hand_rotation.T @ gripper_rotation

    # Local Z points toward the socket (-world X), local X is vertical, and
    # local Y runs across the rack face.
    desired_hand_rotation = np.array(
        [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    desired_gripper_rotation = desired_hand_rotation @ hand_to_gripper
    desired_gripper_quaternion = matrix_to_wxyz(desired_gripper_rotation)

    solved: list[SolvedWaypoint] = []
    for waypoint in build_demo_sequence(_task_geometry(stage)):
        arm_positions, success = solver.compute_inverse_kinematics(
            "right_gripper",
            np.array(waypoint.target_position, dtype=np.float64),
            desired_gripper_quaternion,
            warm_start=warm_start,
            position_tolerance=0.003,
            orientation_tolerance=0.05,
        )
        if not success:
            raise RuntimeError(f"IK failed for {waypoint.phase.value}")

        achieved_position, achieved_rotation = solver.compute_forward_kinematics(
            "right_gripper", arm_positions
        )
        hand_position = solver.compute_forward_kinematics("panda_hand", arm_positions)[0]
        position_error = float(
            np.linalg.norm(achieved_position - np.array(waypoint.target_position))
        )
        solved.append(
            SolvedWaypoint(
                waypoint,
                arm_positions,
                hand_position,
                position_error,
                _rotation_error(achieved_rotation, desired_gripper_rotation),
            )
        )
        warm_start = arm_positions

    return tuple(solved)


def solve_droid_pose(
    target_pose: DroidPose,
    warm_start: np.ndarray,
) -> SolvedPose:
    """Solve one base-frame DROID hand pose for a bounded control step."""

    import omni.usd
    from scipy.spatial.transform import Rotation

    joints = np.asarray(warm_start, dtype=np.float64)
    if joints.shape != (7,) or not np.all(np.isfinite(joints)):
        raise ValueError("IK warm start must contain seven finite joint positions")
    stage = omni.usd.get_context().get_stage()
    solver, base_position, base_orientation = _solver_for_stage(stage)
    target_position, target_orientation = target_pose.to_world_pose(
        base_position, base_orientation
    )
    arm_positions, success = solver.compute_inverse_kinematics(
        "panda_hand",
        target_position,
        target_orientation,
        warm_start=joints,
        position_tolerance=0.0001,
        orientation_tolerance=0.001,
    )
    if not success:
        raise RuntimeError("IK failed for proposed DROID pose")
    achieved_position, achieved_rotation = solver.compute_forward_kinematics(
        "panda_hand", arm_positions
    )
    target_xyzw = np.asarray(
        (
            target_orientation[1],
            target_orientation[2],
            target_orientation[3],
            target_orientation[0],
        )
    )
    return SolvedPose(
        target_pose=target_pose,
        arm_positions=arm_positions,
        hand_position=achieved_position,
        gripper_width_m=(1.0 - target_pose.values[6]) * MAX_GRIPPER_WIDTH_M,
        position_error_m=float(np.linalg.norm(achieved_position - target_position)),
        orientation_error_rad=_rotation_error(
            achieved_rotation, Rotation.from_quat(target_xyzw).as_matrix()
        ),
    )


def preflight_report() -> dict[str, Any]:
    """Return JSON-friendly IK results for remote inspection."""

    return {
        "stage": STAGE_PATH,
        "waypoints": [
            {
                "phase": result.waypoint.phase.value,
                "target_position": list(result.waypoint.target_position),
                "arm_positions_degrees": np.rad2deg(result.arm_positions).round(4).tolist(),
                "gripper_width_m": result.waypoint.gripper_width_m,
                "plug_action": result.waypoint.plug_action.value,
                "position_error_m": result.position_error_m,
                "orientation_error_rad": result.orientation_error_rad,
            }
            for result in solve_waypoints()
        ],
    }
