"""Measured Cartesian tracking evidence for one simulator control action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import DroidAction


class ActionTrackingReason(str, Enum):
    TRANSLATION_DIRECTION = "translation_direction"
    TRANSLATION_ERROR = "translation_error"
    ROTATION_DIRECTION = "rotation_direction"
    ROTATION_ERROR = "rotation_error"
    GRIPPER_ERROR = "gripper_error"


@dataclass(frozen=True)
class ActionTrackingLimits:
    translation_activity_meters: float = 1e-4
    rotation_activity_radians: float = 1e-3
    minimum_direction_cosine: float = 0.5
    maximum_translation_error_meters: float = 5e-4
    maximum_rotation_error_radians: float = 3e-3
    maximum_gripper_error: float = 0.01

    def __post_init__(self) -> None:
        positive = (
            self.translation_activity_meters,
            self.rotation_activity_radians,
            self.maximum_translation_error_meters,
            self.maximum_rotation_error_radians,
            self.maximum_gripper_error,
        )
        if not all(isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("action tracking limits must be finite and positive")
        if not -1.0 <= self.minimum_direction_cosine <= 1.0:
            raise ValueError("tracking cosine must be between negative and positive one")


@dataclass(frozen=True)
class ActionTrackingDecision:
    translation_cosine: float
    rotation_cosine: float
    translation_error_meters: float
    rotation_error_radians: float
    gripper_error: float
    reasons: tuple[ActionTrackingReason, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons

    @classmethod
    def from_dict(cls, payload: Any) -> ActionTrackingDecision:
        if not isinstance(payload, dict):
            raise ValueError("action tracking decision must be an object")
        try:
            decision = cls(
                translation_cosine=float(payload["translation_cosine"]),
                rotation_cosine=float(payload["rotation_cosine"]),
                translation_error_meters=float(payload["translation_error_meters"]),
                rotation_error_radians=float(payload["rotation_error_radians"]),
                gripper_error=float(payload["gripper_error"]),
                reasons=tuple(
                    ActionTrackingReason(reason) for reason in payload["reasons"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("action tracking decision is incomplete") from error
        if not all(
            isfinite(value)
            for value in (
                decision.translation_cosine,
                decision.rotation_cosine,
                decision.translation_error_meters,
                decision.rotation_error_radians,
                decision.gripper_error,
            )
        ) or payload.get("passed") is not decision.passed:
            raise ValueError("action tracking decision is inconsistent")
        return decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "translation_cosine": self.translation_cosine,
            "rotation_cosine": self.rotation_cosine,
            "translation_error_meters": self.translation_error_meters,
            "rotation_error_radians": self.rotation_error_radians,
            "gripper_error": self.gripper_error,
            "reasons": [reason.value for reason in self.reasons],
        }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0


def evaluate_action_tracking(
    commanded: DroidAction,
    actual: DroidAction,
    limits: ActionTrackingLimits = ActionTrackingLimits(),
) -> ActionTrackingDecision:
    commanded_translation = np.asarray(commanded.values[:3])
    actual_translation = np.asarray(actual.values[:3])
    commanded_rotation = np.asarray(commanded.values[3:6])
    actual_rotation = np.asarray(actual.values[3:6])
    translation_cosine = _cosine(commanded_translation, actual_translation)
    rotation_cosine = _cosine(commanded_rotation, actual_rotation)
    translation_error = float(
        np.linalg.norm(commanded_translation - actual_translation)
    )
    rotation_error = float(
        (
            Rotation.from_euler("xyz", actual_rotation)
            * Rotation.from_euler("xyz", commanded_rotation).inv()
        ).magnitude()
    )
    gripper_error = abs(commanded.values[6] - actual.values[6])
    reasons = []
    if (
        np.linalg.norm(commanded_translation) >= limits.translation_activity_meters
        and translation_cosine < limits.minimum_direction_cosine
    ):
        reasons.append(ActionTrackingReason.TRANSLATION_DIRECTION)
    if translation_error > limits.maximum_translation_error_meters:
        reasons.append(ActionTrackingReason.TRANSLATION_ERROR)
    if (
        np.linalg.norm(commanded_rotation) >= limits.rotation_activity_radians
        and rotation_cosine < limits.minimum_direction_cosine
    ):
        reasons.append(ActionTrackingReason.ROTATION_DIRECTION)
    if rotation_error > limits.maximum_rotation_error_radians:
        reasons.append(ActionTrackingReason.ROTATION_ERROR)
    if gripper_error > limits.maximum_gripper_error:
        reasons.append(ActionTrackingReason.GRIPPER_ERROR)
    return ActionTrackingDecision(
        translation_cosine,
        rotation_cosine,
        translation_error,
        rotation_error,
        gripper_error,
        tuple(reasons),
    )
