"""DROID-compatible Cartesian pose and delta-action contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


ACTION_DIMENSIONS = 7
ACTION_FORMAT = "droid_base_delta_pose_v2"
ACTION_FIELD = "action_from_previous"
POSE_FIELD = "end_effector_pose"
COORDINATE_FRAME = "robot_base"
DROID_FPS = 4
MAX_GRIPPER_WIDTH_M = 0.08


@dataclass(frozen=True)
class ActionRecordingContract:
    """Stable manifest contract shared by trajectory writers and readers."""

    format: str = ACTION_FORMAT
    dimensions: int = ACTION_DIMENSIONS
    action_field: str = ACTION_FIELD
    pose_field: str = POSE_FIELD
    coordinate_frame: str = COORDINATE_FRAME

    def to_dict(self) -> dict[str, str | int]:
        return {
            "format": self.format,
            "dimensions": self.dimensions,
            "field": self.action_field,
            "pose_field": self.pose_field,
            "coordinate_frame": self.coordinate_frame,
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
            coordinate_frame=payload.get("coordinate_frame"),
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
    def from_world_poses(
        cls,
        base_position: Sequence[float],
        base_orientation_wxyz: Sequence[float],
        end_effector_position: Sequence[float],
        end_effector_orientation_wxyz: Sequence[float],
        gripper_width_m: float,
    ) -> DroidPose:
        base_position_values = np.asarray(base_position, dtype=np.float64)
        end_effector_position_values = np.asarray(
            end_effector_position, dtype=np.float64
        )
        base_quaternion = tuple(float(value) for value in base_orientation_wxyz)
        end_effector_quaternion = tuple(
            float(value) for value in end_effector_orientation_wxyz
        )
        if (
            base_position_values.shape != (3,)
            or end_effector_position_values.shape != (3,)
            or len(base_quaternion) != 4
            or len(end_effector_quaternion) != 4
        ):
            raise ValueError("base and end-effector poses require XYZ and WXYZ")
        if not 0.0 <= gripper_width_m <= MAX_GRIPPER_WIDTH_M:
            raise ValueError("gripper width is outside the Franka range")
        base_rotation = Rotation.from_quat(
            (
                base_quaternion[1],
                base_quaternion[2],
                base_quaternion[3],
                base_quaternion[0],
            )
        )
        end_effector_rotation = Rotation.from_quat(
            (
                end_effector_quaternion[1],
                end_effector_quaternion[2],
                end_effector_quaternion[3],
                end_effector_quaternion[0],
            )
        )
        relative_position = base_rotation.inv().apply(
            end_effector_position_values - base_position_values
        )
        relative_rotation = base_rotation.inv() * end_effector_rotation
        euler = relative_rotation.as_euler("xyz", degrees=False)
        closedness = 1.0 - gripper_width_m / MAX_GRIPPER_WIDTH_M
        return cls((*relative_position.tolist(), *euler.tolist(), closedness))

    def to_world_pose(
        self,
        base_position: Sequence[float],
        base_orientation_wxyz: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        base_position_values = np.asarray(base_position, dtype=np.float64)
        base_quaternion = tuple(float(value) for value in base_orientation_wxyz)
        if base_position_values.shape != (3,) or len(base_quaternion) != 4:
            raise ValueError("base pose requires XYZ and WXYZ")
        base_rotation = Rotation.from_quat(
            (
                base_quaternion[1],
                base_quaternion[2],
                base_quaternion[3],
                base_quaternion[0],
            )
        )
        position = base_position_values + base_rotation.apply(self.values[:3])
        rotation = base_rotation * Rotation.from_euler("xyz", self.values[3:6])
        xyzw = rotation.as_quat()
        return position, np.asarray((xyzw[3], xyzw[0], xyzw[1], xyzw[2]))

    def applied(self, action: DroidAction) -> DroidPose:
        """Apply one base-frame delta action to this absolute pose."""

        previous_rotation = Rotation.from_euler("xyz", self.values[3:6])
        relative_rotation = Rotation.from_euler("xyz", action.values[3:6])
        current_rotation = relative_rotation * previous_rotation
        return DroidPose(
            (
                *(np.asarray(self.values[:3]) + np.asarray(action.values[:3])),
                *current_rotation.as_euler("xyz"),
                self.values[6] + action.values[6],
            )
        )


@dataclass(frozen=True)
class DroidAction:
    """Delta XYZ, relative XYZ Euler rotation, and gripper delta."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _validated_values("action", self.values))

    def scaled(self, scale: float) -> DroidAction:
        """Scale every action dimension while preserving its direction."""

        return DroidActionScale.uniform(scale).apply(self)


def compose_actions(actions: Sequence[DroidAction]) -> DroidAction:
    """Compose a non-empty base-frame action horizon into one delta action."""

    sequence = tuple(actions)
    if not sequence or any(not isinstance(action, DroidAction) for action in sequence):
        raise ValueError("action composition requires a non-empty action sequence")
    translation = np.sum(
        np.asarray([action.values[:3] for action in sequence], dtype=np.float64),
        axis=0,
    )
    rotation = Rotation.identity()
    for action in sequence:
        rotation = Rotation.from_euler("xyz", action.values[3:6]) * rotation
    return DroidAction(
        (
            *translation.tolist(),
            *rotation.as_euler("xyz").tolist(),
            sum(action.values[6] for action in sequence),
        )
    )


@dataclass(frozen=True)
class DroidActionScale:
    """Independent safety scales for translation, rotation, and gripper."""

    translation: float
    rotation: float
    gripper: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.translation)
            or not 0.0 < self.translation <= 1.0
            or not isfinite(self.rotation)
            or not 0.0 <= self.rotation <= 1.0
            or not isfinite(self.gripper)
            or not 0.0 <= self.gripper <= 1.0
        ):
            raise ValueError(
                "action translation scale must be positive and rotation/gripper "
                "scales must be nonnegative, with every scale at most one"
            )

    @classmethod
    def uniform(cls, scale: float) -> DroidActionScale:
        return cls(scale, scale, scale)

    @classmethod
    def from_payload(cls, payload: Any) -> DroidActionScale:
        if isinstance(payload, bool):
            raise ValueError("action scale is invalid")
        if isinstance(payload, (int, float)):
            return cls.uniform(float(payload))
        if not isinstance(payload, Mapping):
            raise ValueError("action scale is invalid")
        try:
            return cls(
                float(payload["translation"]),
                float(payload["rotation"]),
                float(payload["gripper"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("action scale is incomplete") from error

    def apply(self, action: DroidAction) -> DroidAction:
        scales = (
            (self.translation,) * 3
            + (self.rotation,) * 3
            + (self.gripper,)
        )
        return DroidAction(
            tuple(value * scale for value, scale in zip(action.values, scales))
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "translation": self.translation,
            "rotation": self.rotation,
            "gripper": self.gripper,
        }


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
        return self.accepts_rollout((action,))

    def accepts_rollout(self, actions: Sequence[DroidAction]) -> bool:
        if not actions:
            return False
        action_norm = sqrt(
            sum(value * value for action in actions for value in action.values)
        )
        return action_norm >= self.minimum_action_norm and all(
            sqrt(sum(value * value for value in action.values[:6]))
            <= self.maximum_pose_action_norm
            and abs(action.values[6]) <= self.maximum_gripper_action
            for action in actions
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_action_norm": self.minimum_action_norm,
            "maximum_pose_action_norm": self.maximum_pose_action_norm,
            "maximum_gripper_action": self.maximum_gripper_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionSelectionBounds:
        try:
            return cls(
                minimum_action_norm=float(payload["minimum_action_norm"]),
                maximum_pose_action_norm=float(
                    payload["maximum_pose_action_norm"]
                ),
                maximum_gripper_action=float(payload["maximum_gripper_action"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("action selection bounds are incomplete") from error


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
