"""Persisted realized target-progress policy and evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import DroidPose, action_between


def _strict_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number")
    return float(value)


class RealizedTargetProgressReason(str, Enum):
    TRANSLATION_PROGRESS = "translation_progress_insufficient"
    ORIENTATION_REGRESSION = "orientation_regression"


@dataclass(frozen=True)
class RealizedTargetProgressDecision:
    initial_translation_error_meters: float
    realized_translation_error_meters: float
    translation_error_reduction_fraction: float
    initial_orientation_error_radians: float
    realized_orientation_error_radians: float
    close_enough: bool
    reasons: tuple[RealizedTargetProgressReason, ...]

    def __post_init__(self) -> None:
        if (
            not all(
                isfinite(value) and value >= 0.0
                for value in (
                    self.initial_translation_error_meters,
                    self.realized_translation_error_meters,
                    self.initial_orientation_error_radians,
                    self.realized_orientation_error_radians,
                )
            )
            or not isfinite(self.translation_error_reduction_fraction)
            or not isinstance(self.close_enough, bool)
        ):
            raise ValueError("realized target-progress decision is invalid")

    @property
    def passed(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "close_enough": self.close_enough,
            "initial_translation_error_meters": self.initial_translation_error_meters,
            "realized_translation_error_meters": self.realized_translation_error_meters,
            "translation_error_reduction_fraction": (
                self.translation_error_reduction_fraction
            ),
            "initial_orientation_error_radians": self.initial_orientation_error_radians,
            "realized_orientation_error_radians": self.realized_orientation_error_radians,
            "reasons": [reason.value for reason in self.reasons],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RealizedTargetProgressDecision:
        if not isinstance(payload, dict):
            raise ValueError("realized target-progress decision must be an object")
        try:
            decision = cls(
                initial_translation_error_meters=_strict_number(
                    payload, "initial_translation_error_meters"
                ),
                realized_translation_error_meters=_strict_number(
                    payload, "realized_translation_error_meters"
                ),
                translation_error_reduction_fraction=_strict_number(
                    payload, "translation_error_reduction_fraction"
                ),
                initial_orientation_error_radians=_strict_number(
                    payload, "initial_orientation_error_radians"
                ),
                realized_orientation_error_radians=_strict_number(
                    payload, "realized_orientation_error_radians"
                ),
                close_enough=payload["close_enough"],
                reasons=tuple(
                    RealizedTargetProgressReason(reason)
                    for reason in payload["reasons"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("realized target-progress decision is incomplete") from error
        if payload.get("passed") is not decision.passed:
            raise ValueError("realized target-progress pass claim is inconsistent")
        return decision


@dataclass(frozen=True)
class RealizedTargetProgressPolicy:
    minimum_translation_error_reduction_fraction: float = 0.25
    close_enough_translation_meters: float = 1e-4
    maximum_orientation_error_increase_radians: float = 1.25e-3

    def __post_init__(self) -> None:
        if (
            not isfinite(self.minimum_translation_error_reduction_fraction)
            or not 0.0 < self.minimum_translation_error_reduction_fraction <= 1.0
            or not isfinite(self.close_enough_translation_meters)
            or self.close_enough_translation_meters <= 0.0
            or not isfinite(self.maximum_orientation_error_increase_radians)
            or self.maximum_orientation_error_increase_radians <= 0.0
        ):
            raise ValueError("realized target-progress policy is invalid")

    @staticmethod
    def _errors(pose: DroidPose, target: DroidPose) -> tuple[float, float]:
        delta = action_between(pose, target)
        return (
            float(np.linalg.norm(delta.values[:3])),
            float(Rotation.from_euler("xyz", delta.values[3:6]).magnitude()),
        )

    def evaluate(
        self,
        initial: DroidPose,
        target: DroidPose,
        realized: DroidPose,
    ) -> RealizedTargetProgressDecision:
        initial_translation, initial_orientation = self._errors(initial, target)
        realized_translation, realized_orientation = self._errors(realized, target)
        reduction_fraction = (
            (initial_translation - realized_translation) / initial_translation
            if initial_translation > 1e-12
            else 0.0
        )
        close_enough = realized_translation <= self.close_enough_translation_meters
        reasons = []
        if (
            not close_enough
            and reduction_fraction < self.minimum_translation_error_reduction_fraction
        ):
            reasons.append(RealizedTargetProgressReason.TRANSLATION_PROGRESS)
        if (
            realized_orientation - initial_orientation
            > self.maximum_orientation_error_increase_radians
        ):
            reasons.append(RealizedTargetProgressReason.ORIENTATION_REGRESSION)
        return RealizedTargetProgressDecision(
            initial_translation,
            realized_translation,
            reduction_fraction,
            initial_orientation,
            realized_orientation,
            close_enough,
            tuple(reasons),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_translation_error_reduction_fraction": (
                self.minimum_translation_error_reduction_fraction
            ),
            "close_enough_translation_meters": self.close_enough_translation_meters,
            "maximum_orientation_error_increase_radians": (
                self.maximum_orientation_error_increase_radians
            ),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RealizedTargetProgressPolicy:
        if not isinstance(payload, dict):
            raise ValueError("realized target-progress policy must be an object")
        try:
            return cls(
                minimum_translation_error_reduction_fraction=_strict_number(
                    payload, "minimum_translation_error_reduction_fraction"
                ),
                close_enough_translation_meters=_strict_number(
                    payload, "close_enough_translation_meters"
                ),
                maximum_orientation_error_increase_radians=_strict_number(
                    payload, "maximum_orientation_error_increase_radians"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("realized target-progress policy is incomplete") from error
