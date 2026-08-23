"""DROID-compatible Cartesian pose and delta-action contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


ACTION_DIMENSIONS = 7
ACTION_FORMAT = "droid_delta_pose_v1"
ACTION_FIELD = "action_from_previous"
POSE_FIELD = "end_effector_pose"
DROID_FPS = 4
MAX_GRIPPER_WIDTH_M = 0.08


@dataclass(frozen=True)
class ActionRecordingContract:
    """Stable manifest contract shared by trajectory writers and readers."""

    format: str = ACTION_FORMAT
    dimensions: int = ACTION_DIMENSIONS
    action_field: str = ACTION_FIELD
    pose_field: str = POSE_FIELD

    def to_dict(self) -> dict[str, str | int]:
        return {
            "format": self.format,
            "dimensions": self.dimensions,
            "field": self.action_field,
            "pose_field": self.pose_field,
        }

    @classmethod
    def from_mapping(cls, payload: Any) -> ActionRecordingContract:
        if not isinstance(payload, Mapping):
            raise ValueError("recording does not contain DROID-compatible actions")
        contract = cls(
            format=payload.get("format"),
            dimensions=payload.get("dimensions"),
            action_field=payload.get("field"),
            pose_field=payload.get("pose_field"),
        )
        if contract != ACTION_RECORDING_CONTRACT:
            raise ValueError("recording does not contain DROID-compatible actions")
        return contract


ACTION_RECORDING_CONTRACT = ActionRecordingContract()


def _validated_values(name: str, values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != ACTION_DIMENSIONS or not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain seven finite values")
    return result


@dataclass(frozen=True)
class DroidPose:
    """Absolute XYZ, XYZ Euler orientation, and gripper closedness."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _validated_values("pose", self.values))
        if not 0.0 <= self.values[-1] <= 1.0:
            raise ValueError("pose gripper closedness must be between zero and one")

    @classmethod
    def from_world_pose(
        cls,
        position: Sequence[float],
        orientation_wxyz: Sequence[float],
        gripper_width_m: float,
    ) -> DroidPose:
        position_values = tuple(float(value) for value in position)
        quaternion = tuple(float(value) for value in orientation_wxyz)
        if len(position_values) != 3 or len(quaternion) != 4:
            raise ValueError("world pose requires XYZ position and WXYZ orientation")
        if not 0.0 <= gripper_width_m <= MAX_GRIPPER_WIDTH_M:
            raise ValueError("gripper width is outside the Franka range")
        xyzw = (quaternion[1], quaternion[2], quaternion[3], quaternion[0])
        euler = Rotation.from_quat(xyzw).as_euler("xyz", degrees=False)
        closedness = 1.0 - gripper_width_m / MAX_GRIPPER_WIDTH_M
        return cls((*position_values, *euler.tolist(), closedness))


@dataclass(frozen=True)
class DroidAction:
    """Delta XYZ, relative XYZ Euler rotation, and gripper delta."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _validated_values("action", self.values))


@dataclass(frozen=True)
class ActionSelectionBounds:
    """Bounds for selecting model-valid recorded actions."""

    minimum_action_norm: float = 1e-6
    maximum_pose_action_norm: float = 0.1
    maximum_gripper_action: float = 0.75

    def __post_init__(self) -> None:
        values = (
            self.minimum_action_norm,
            self.maximum_pose_action_norm,
            self.maximum_gripper_action,
        )
        if not all(isfinite(value) for value in values):
            raise ValueError("action selection bounds must be finite")
        if self.minimum_action_norm < 0:
            raise ValueError("minimum action norm must be non-negative")
        if self.maximum_pose_action_norm <= 0 or self.maximum_gripper_action <= 0:
            raise ValueError("maximum action bounds must be positive")

    def accepts(self, action: DroidAction) -> bool:
        action_norm = sqrt(sum(value * value for value in action.values))
        pose_action_norm = sqrt(sum(value * value for value in action.values[:6]))
        return (
            action_norm >= self.minimum_action_norm
            and pose_action_norm <= self.maximum_pose_action_norm
            and abs(action.values[6]) <= self.maximum_gripper_action
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_action_norm": self.minimum_action_norm,
            "maximum_pose_action_norm": self.maximum_pose_action_norm,
            "maximum_gripper_action": self.maximum_gripper_action,
        }


DEFAULT_ACTION_SELECTION_BOUNDS = ActionSelectionBounds()


def action_between(previous: DroidPose, current: DroidPose) -> DroidAction:
    previous_values = np.asarray(previous.values, dtype=np.float64)
    current_values = np.asarray(current.values, dtype=np.float64)
    translation = current_values[:3] - previous_values[:3]
    previous_rotation = Rotation.from_euler("xyz", previous_values[3:6])
    current_rotation = Rotation.from_euler("xyz", current_values[3:6])
    relative_rotation = current_rotation * previous_rotation.inv()
    gripper_delta = current_values[6] - previous_values[6]
    return DroidAction(
        (
            *translation.tolist(),
            *relative_rotation.as_euler("xyz").tolist(),
            gripper_delta,
        )
    )
