"""Shared scene paths and transform helpers for the Isaac demo."""

from __future__ import annotations

from typing import Any

import numpy as np


ROBOT_PATH = "/World/Franka_R"
PLUG_PATH = "/World/RJ45_Plug"
SOCKET_PATH = "/World/RJ45_Socket"
PRESENTATION_CAMERA_PATH = "/World/ShotCam"
WRIST_CAMERA_PATH = f"{ROBOT_PATH}/panda_hand/WristCamera"
STAGE_PATH = "/isaac-sim/.local/share/ov/data/quantis/scenes/datacenter_demo.usda"
ARM_JOINT_TARGETS_DEGREES = np.array(
    [0.687549354, -32.658638, 0.0, -161.00102, 0.0, 174.00696, 42.456184],
    dtype=np.float64,
)


def world_pose(prim: Any) -> tuple[np.ndarray, np.ndarray]:
    import omni.usd

    matrix = omni.usd.get_world_transform_matrix(prim)
    translation = matrix.ExtractTranslation()
    rotation = matrix.ExtractRotationQuat()
    position = np.array(translation, dtype=np.float64)
    quaternion = np.array([rotation.GetReal(), *rotation.GetImaginary()], dtype=np.float64)
    return position, quaternion


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
