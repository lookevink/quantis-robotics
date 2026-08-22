"""Kinematics and preflight checks for the live Isaac plug-in demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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


def solve_waypoints() -> tuple[SolvedWaypoint, ...]:
    """Solve every task waypoint without advancing simulation physics."""

    import omni.usd
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    from isaacsim.robot_motion.motion_generation.interface_config_loader import (
        load_supported_lula_kinematics_solver_config,
    )

    stage = omni.usd.get_context().get_stage()
    robot = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot.IsValid():
        raise RuntimeError(f"robot prim is missing: {ROBOT_PATH}")

    solver = LulaKinematicsSolver(**load_supported_lula_kinematics_solver_config("Franka"))
    base_position, base_orientation = world_pose(robot)
    solver.set_robot_base_pose(base_position, base_orientation)

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
