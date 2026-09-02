"""Measured Cartesian tracking evidence for one simulator control action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, sqrt
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import DroidAction
from jepa_wm.control_policy import ControlExecutionPolicy


class ActionTrackingReason(str, Enum):
    TRANSLATION_DIRECTION = "translation_direction"
    TRANSLATION_ERROR = "translation_error"
    ROTATION_DIRECTION = "rotation_direction"
    ROTATION_ERROR = "rotation_error"
    GRIPPER_ERROR = "gripper_error"


class CommandRealizationReason(str, Enum):
    """Why a safe command has not yet completed its requested motion."""

    TRANSLATION_UNDERREALIZED = "translation_underrealized"
    ROTATION_UNDERREALIZED = "rotation_underrealized"


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


EXPERIMENTAL_CANDIDATE_TRACKING_LIMITS = ActionTrackingLimits(
    rotation_activity_radians=2e-4,
)


@dataclass(frozen=True)
class CommandRealizationLimits:
    """Completion thresholds, deliberately separate from safety tracking."""

    translation_activity_meters: float = 1e-4
    rotation_activity_radians: float = 1e-3
    minimum_translation_fraction: float = 0.75
    minimum_rotation_fraction: float = 0.75
    required_consecutive_samples: int = 2
    maximum_plateau_samples: int = 32
    minimum_progress_increment: float = 0.01

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                self.translation_activity_meters,
                self.rotation_activity_radians,
                self.minimum_progress_increment,
            )
        ) or not all(
            isfinite(value) and 0.0 < value <= 1.0
            for value in (
                self.minimum_translation_fraction,
                self.minimum_rotation_fraction,
            )
        ):
            raise ValueError("command realization limits are invalid")
        if (
            isinstance(self.required_consecutive_samples, bool)
            or not isinstance(self.required_consecutive_samples, int)
            or self.required_consecutive_samples < 1
            or isinstance(self.maximum_plateau_samples, bool)
            or not isinstance(self.maximum_plateau_samples, int)
            or self.maximum_plateau_samples < 1
        ):
            raise ValueError("command realization sample count must be positive")


@dataclass(frozen=True)
class CommandRealizationDecision:
    translation_fraction: float
    rotation_fraction: float
    active_progress_fraction: float
    reasons: tuple[CommandRealizationReason, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "translation_fraction": self.translation_fraction,
            "rotation_fraction": self.rotation_fraction,
            "active_progress_fraction": self.active_progress_fraction,
            "reasons": [reason.value for reason in self.reasons],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> CommandRealizationDecision:
        if not isinstance(payload, dict):
            raise ValueError("command realization decision must be an object")
        try:
            decision = cls(
                float(payload["translation_fraction"]),
                float(payload["rotation_fraction"]),
                float(payload["active_progress_fraction"]),
                tuple(
                    CommandRealizationReason(reason) for reason in payload["reasons"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("command realization decision is incomplete") from error
        if (
            not all(
                isfinite(value) and value >= 0.0
                for value in (
                    decision.translation_fraction,
                    decision.rotation_fraction,
                    decision.active_progress_fraction,
                )
            )
            or payload.get("passed") is not decision.passed
        ):
            raise ValueError("command realization decision is inconsistent")
        return decision


def tracking_limits_for_policy(
    policy: ControlExecutionPolicy,
) -> ActionTrackingLimits:
    if policy in (
        ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
        ControlExecutionPolicy.INSERTION_RESET_TRIAL,
        ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
    ):
        return EXPERIMENTAL_CANDIDATE_TRACKING_LIMITS
    return ActionTrackingLimits()


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


def _minimum_realized_axis_fraction(
    commanded: np.ndarray,
    actual: np.ndarray,
    activity_magnitude: float,
) -> float:
    """Return the least realization across physically meaningful axes.

    A component smaller than the modality activity threshold divided across
    all axes cannot make that modality active by itself. Treating smaller
    components as mandatory motion would turn floating-point/model residue
    into a sub-resolution drive request.
    """

    axis_activity = activity_magnitude / sqrt(float(commanded.size))
    active = np.abs(commanded) >= axis_activity
    if not np.any(active):
        return 1.0
    fractions = actual[active] / commanded[active]
    return max(0.0, float(np.min(fractions)))


def evaluate_command_realization(
    commanded: DroidAction,
    actual: DroidAction,
    limits: CommandRealizationLimits = CommandRealizationLimits(),
) -> CommandRealizationDecision:
    """Decide whether active Cartesian axes realized enough of a command.

    Tracking answers whether observed motion is safe and acceptably close.
    This decision answers the different question of whether the command is
    complete, so a small command cannot pass merely by remaining inside a
    larger absolute tracking tolerance.
    """

    commanded_translation = np.asarray(commanded.values[:3])
    actual_translation = np.asarray(actual.values[:3])
    commanded_rotation = np.asarray(commanded.values[3:6])
    actual_rotation = np.asarray(actual.values[3:6])
    translation_fraction = _minimum_realized_axis_fraction(
        commanded_translation,
        actual_translation,
        limits.translation_activity_meters,
    )
    rotation_fraction = _minimum_realized_axis_fraction(
        commanded_rotation,
        actual_rotation,
        limits.rotation_activity_radians,
    )
    active_fractions = []
    reasons = []
    if (
        np.linalg.norm(commanded_translation) >= limits.translation_activity_meters
    ):
        active_fractions.append(translation_fraction)
        if translation_fraction < limits.minimum_translation_fraction:
            reasons.append(CommandRealizationReason.TRANSLATION_UNDERREALIZED)
    if (
        np.linalg.norm(commanded_rotation) >= limits.rotation_activity_radians
    ):
        active_fractions.append(rotation_fraction)
        if rotation_fraction < limits.minimum_rotation_fraction:
            reasons.append(CommandRealizationReason.ROTATION_UNDERREALIZED)
    return CommandRealizationDecision(
        translation_fraction,
        rotation_fraction,
        min(active_fractions, default=1.0),
        tuple(reasons),
    )


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


@dataclass(frozen=True)
class CommandCompletionDecision:
    """Joint safety tracking and independent task-space completion."""

    tracking: ActionTrackingDecision
    realization: CommandRealizationDecision

    @property
    def passed(self) -> bool:
        return self.tracking.passed and self.realization.passed


def evaluate_command_completion(
    commanded: DroidAction,
    actual: DroidAction,
    execution_policy: ControlExecutionPolicy,
) -> CommandCompletionDecision:
    return CommandCompletionDecision(
        evaluate_action_tracking(
            commanded,
            actual,
            tracking_limits_for_policy(execution_policy),
        ),
        evaluate_command_realization(commanded, actual),
    )
