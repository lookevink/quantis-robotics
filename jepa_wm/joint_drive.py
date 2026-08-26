"""Dependency-light joint-drive target and bounded load-bias compensation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from jepa_wm.control_safety import SimulatorSafetyLimits


def validate_joint_positions(values: Sequence[float]) -> tuple[float, ...]:
    positions = tuple(float(value) for value in values)
    if len(positions) != 7 or not all(isfinite(value) for value in positions):
        raise ValueError("joint drive requires finite seven-axis positions")
    return positions


@dataclass(frozen=True)
class JointDriveTarget:
    joint_positions: tuple[float, ...]
    gripper_width_m: float

    def __post_init__(self) -> None:
        if (
            len(self.joint_positions) != 7
            or not all(isfinite(value) for value in self.joint_positions)
            or not isfinite(self.gripper_width_m)
            or not 0.0 <= self.gripper_width_m <= 0.08
        ):
            raise ValueError("joint drive target is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_positions": list(self.joint_positions),
            "gripper_width_m": self.gripper_width_m,
        }

    @classmethod
    def for_command(
        cls,
        joint_positions: tuple[float, ...],
        gripper_width_m: float,
    ) -> JointDriveTarget:
        """Canonicalize one command to Isaac's USD float drive attributes."""

        stored_degrees = np.asarray(
            np.rad2deg(np.asarray(joint_positions, dtype=np.float64)),
            dtype=np.float32,
        ).astype(np.float64)
        return cls(
            tuple(float(value) for value in np.deg2rad(stored_degrees)),
            float(np.float32(gripper_width_m / 2.0)) * 2.0,
        )

    def validate_active(
        self,
        joint_positions: tuple[float, ...],
        gripper_width_m: float,
    ) -> None:
        if (
            len(joint_positions) != 7
            or not all(
                isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    joint_positions,
                    self.joint_positions,
                )
            )
            or not isclose(
                gripper_width_m,
                self.gripper_width_m,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("active drive target changed")

    @classmethod
    def from_dict(cls, payload: Any) -> JointDriveTarget:
        if not isinstance(payload, Mapping):
            raise ValueError("joint drive target must be an object")
        try:
            return cls(
                tuple(float(value) for value in payload["joint_positions"]),
                float(payload["gripper_width_m"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("joint drive target is incomplete") from error


@dataclass(frozen=True)
class JointDriveBiasCompensation:
    """Pre-compensate a desired joint target by one measured stable drive bias."""

    maximum_bias_radians: float = 0.002

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_bias_radians, bool)
            or not isinstance(self.maximum_bias_radians, (int, float))
            or not isfinite(self.maximum_bias_radians)
            or self.maximum_bias_radians <= 0.0
        ):
            raise ValueError("drive bias compensation bound is invalid")

    def compensated_joint_target(
        self,
        desired_joint_positions: Sequence[float],
        measured_drive_target: JointDriveTarget,
        realized_joint_positions: Sequence[float],
        safety_limits: SimulatorSafetyLimits,
    ) -> tuple[float, ...]:
        desired = validate_joint_positions(desired_joint_positions)
        realized = validate_joint_positions(realized_joint_positions)
        bias = tuple(
            drive - realized
            for drive, realized in zip(
                measured_drive_target.joint_positions,
                realized,
            )
        )
        if max(abs(value) for value in bias) > self.maximum_bias_radians:
            raise ValueError("measured drive bias exceeds its compensation bound")
        applied = tuple(
            desired_value + bias_value
            for desired_value, bias_value in zip(desired, bias)
        )
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                applied,
                safety_limits.lower_joint_limits,
                safety_limits.upper_joint_limits,
            )
        ):
            raise ValueError("compensated drive target exceeds joint limits")
        return applied

    def to_dict(self) -> dict[str, float]:
        return {"maximum_bias_radians": self.maximum_bias_radians}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> JointDriveBiasCompensation:
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"maximum_bias_radians"}
            or isinstance(payload.get("maximum_bias_radians"), bool)
            or not isinstance(payload.get("maximum_bias_radians"), (int, float))
        ):
            raise ValueError("drive bias compensation must be an object")
        try:
            return cls(float(payload["maximum_bias_radians"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("drive bias compensation is incomplete") from error
