"""Bounded drive-target compensation for control-resolution probes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from jepa_wm.control_resolution_baseline import ControlResolutionDriveTarget
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.joint_drive import (
    JointDriveBiasCompensation,
    validate_joint_positions,
)


@dataclass(frozen=True)
class ControlResolutionDriveBiasCompensation:
    """Add resolution-only rollback feedback to shared forward compensation."""

    maximum_bias_radians: float = 0.002
    maximum_feedback_correction_radians: float = 0.002
    path_dependent_rollback: bool = True

    def __post_init__(self) -> None:
        if (
            not isfinite(self.maximum_feedback_correction_radians)
            or self.maximum_feedback_correction_radians <= 0.0
            or not isinstance(self.path_dependent_rollback, bool)
        ):
            raise ValueError("drive bias compensation bound is invalid")
        JointDriveBiasCompensation(self.maximum_bias_radians)

    def compensated_joint_target(
        self,
        desired_joint_positions: Sequence[float],
        measured_drive_target: ControlResolutionDriveTarget,
        realized_joint_positions: Sequence[float],
        safety_limits: SimulatorSafetyLimits,
    ) -> tuple[float, ...]:
        return JointDriveBiasCompensation(
            self.maximum_bias_radians
        ).compensated_joint_target(
            desired_joint_positions,
            measured_drive_target,
            realized_joint_positions,
            safety_limits,
        )

    def feedback_corrected_joint_target(
        self,
        desired_joint_positions: Sequence[float],
        applied_drive_target: ControlResolutionDriveTarget,
        realized_joint_positions: Sequence[float],
        safety_limits: SimulatorSafetyLimits,
    ) -> tuple[float, ...]:
        desired = validate_joint_positions(desired_joint_positions)
        realized = validate_joint_positions(realized_joint_positions)
        correction = tuple(
            target - actual for target, actual in zip(desired, realized)
        )
        if max(abs(value) for value in correction) > (
            self.maximum_feedback_correction_radians
        ):
            raise ValueError("rollback feedback correction exceeds its bound")
        corrected = tuple(
            drive + residual
            for drive, residual in zip(
                applied_drive_target.joint_positions,
                correction,
            )
        )
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                corrected,
                safety_limits.lower_joint_limits,
                safety_limits.upper_joint_limits,
            )
        ):
            raise ValueError("corrected rollback target exceeds joint limits")
        return corrected

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "maximum_bias_radians": self.maximum_bias_radians,
            "maximum_feedback_correction_radians": (
                self.maximum_feedback_correction_radians
            ),
            "path_dependent_rollback": self.path_dependent_rollback,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> ControlResolutionDriveBiasCompensation:
        if not isinstance(payload, Mapping):
            raise ValueError("drive bias compensation must be an object")
        try:
            path_dependent_rollback = payload.get(
                "path_dependent_rollback",
                False,
            )
            if not isinstance(path_dependent_rollback, bool):
                raise ValueError("drive rollback policy must be boolean")
            return cls(
                float(payload["maximum_bias_radians"]),
                float(
                    payload.get(
                        "maximum_feedback_correction_radians",
                        0.002,
                    )
                ),
                path_dependent_rollback,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("drive bias compensation is incomplete") from error
