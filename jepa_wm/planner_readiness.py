"""Offline first-action gate before any JEPA-WM command may reach Isaac."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite, sqrt
from typing import Any, Sequence

from jepa_wm.action import DroidAction


class FirstActionReason(str, Enum):
    DIRECTION_MISMATCH = "direction_mismatch"
    UNNECESSARY_TRANSLATION = "unnecessary_translation"
    UNNECESSARY_ROTATION = "unnecessary_rotation"
    UNNECESSARY_GRIPPER = "unnecessary_gripper"


@dataclass(frozen=True)
class FirstActionThresholds:
    recorded_translation_activity: float = 0.001
    recorded_rotation_activity: float = 0.005
    recorded_gripper_activity: float = 0.02
    maximum_stationary_translation: float = 0.002
    maximum_stationary_rotation: float = 0.01
    maximum_stationary_gripper: float = 0.02
    minimum_active_cosine: float = 0.5

    def __post_init__(self) -> None:
        activity = (
            self.recorded_translation_activity,
            self.recorded_rotation_activity,
            self.recorded_gripper_activity,
        )
        stationary = (
            self.maximum_stationary_translation,
            self.maximum_stationary_rotation,
            self.maximum_stationary_gripper,
        )
        if not all(isfinite(value) and value >= 0.0 for value in activity + stationary):
            raise ValueError("first-action thresholds must be finite and non-negative")
        if (
            not isfinite(self.minimum_active_cosine)
            or not -1.0 <= self.minimum_active_cosine <= 1.0
        ):
            raise ValueError(
                "minimum active cosine must be between negative and positive one"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "recorded_translation_activity": self.recorded_translation_activity,
            "recorded_rotation_activity": self.recorded_rotation_activity,
            "recorded_gripper_activity": self.recorded_gripper_activity,
            "maximum_stationary_translation": self.maximum_stationary_translation,
            "maximum_stationary_rotation": self.maximum_stationary_rotation,
            "maximum_stationary_gripper": self.maximum_stationary_gripper,
            "minimum_active_cosine": self.minimum_active_cosine,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> FirstActionThresholds:
        if not isinstance(payload, dict):
            raise ValueError("first-action thresholds must be an object")
        try:
            return cls(**{key: float(payload[key]) for key in cls.__dataclass_fields__})
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("first-action thresholds are incomplete") from error


@dataclass(frozen=True)
class FirstActionDecision:
    recorded_action_is_active: bool
    cosine: float
    reasons: tuple[FirstActionReason, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "recorded_action_is_active": self.recorded_action_is_active,
            "cosine": self.cosine,
            "reasons": [reason.value for reason in self.reasons],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> FirstActionDecision:
        if not isinstance(payload, dict):
            raise ValueError("first-action decision must be an object")
        try:
            decision = cls(
                recorded_action_is_active=bool(payload["recorded_action_is_active"]),
                cosine=float(payload["cosine"]),
                reasons=tuple(
                    FirstActionReason(reason) for reason in payload["reasons"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("first-action decision is incomplete") from error
        if payload.get("passed") is not decision.passed:
            raise ValueError("first-action decision pass result is inconsistent")
        return decision


@dataclass(frozen=True)
class FirstActionSummary:
    decisions: tuple[FirstActionDecision, ...]

    def __post_init__(self) -> None:
        if not self.decisions:
            raise ValueError("first-action summary requires at least one decision")

    @staticmethod
    def _rate(decisions: tuple[FirstActionDecision, ...]) -> float | None:
        if not decisions:
            return None
        return sum(decision.passed for decision in decisions) / len(decisions)

    @classmethod
    def aggregate(cls, summaries: Sequence[FirstActionSummary]) -> FirstActionSummary:
        if not summaries:
            raise ValueError("at least one first-action summary is required")
        return cls(
            tuple(
                decision
                for summary in summaries
                for decision in summary.decisions
            )
        )

    @property
    def count(self) -> int:
        return len(self.decisions)

    @property
    def mean_cosine(self) -> float:
        return fsum(decision.cosine for decision in self.decisions) / self.count

    @property
    def mean_active_cosine(self) -> float | None:
        active = tuple(
            decision.cosine
            for decision in self.decisions
            if decision.recorded_action_is_active
        )
        return fsum(active) / len(active) if active else None

    @property
    def gate_passes(self) -> int:
        return sum(decision.passed for decision in self.decisions)

    @property
    def active_count(self) -> int:
        return sum(decision.recorded_action_is_active for decision in self.decisions)

    @property
    def active_gate_passes(self) -> int:
        return sum(
            decision.recorded_action_is_active and decision.passed
            for decision in self.decisions
        )

    @property
    def pass_rate(self) -> float:
        return self.gate_passes / self.count

    @property
    def active_direction_pass_rate(self) -> float | None:
        return self._rate(
            tuple(
                decision
                for decision in self.decisions
                if decision.recorded_action_is_active
            )
        )

    @property
    def stationary_hold_rate(self) -> float | None:
        return self._rate(
            tuple(
                decision
                for decision in self.decisions
                if not decision.recorded_action_is_active
            )
        )

    def to_dict(self) -> dict[str, float | None]:
        return {
            "mean_active_first_action_cosine": self.mean_active_cosine,
            "first_action_gate_pass_rate": self.pass_rate,
            "active_first_action_direction_pass_rate": (
                self.active_direction_pass_rate
            ),
            "stationary_first_action_hold_rate": self.stationary_hold_rate,
        }


def _norm(values: tuple[float, ...]) -> float:
    return sqrt(sum(value * value for value in values))


class FirstActionGate:
    def __init__(self, thresholds: FirstActionThresholds = FirstActionThresholds()):
        self.thresholds = thresholds

    def evaluate(
        self,
        recorded: DroidAction,
        planned: DroidAction,
    ) -> FirstActionDecision:
        active = self.is_active(recorded)
        denominator = _norm(recorded.values) * _norm(planned.values)
        cosine = (
            sum(left * right for left, right in zip(recorded.values, planned.values))
            / denominator
            if denominator > 1e-12
            else 0.0
        )
        reasons = []
        if active:
            if cosine < self.thresholds.minimum_active_cosine:
                reasons.append(FirstActionReason.DIRECTION_MISMATCH)
        else:
            if (
                _norm(planned.values[:3])
                > self.thresholds.maximum_stationary_translation
            ):
                reasons.append(FirstActionReason.UNNECESSARY_TRANSLATION)
            if _norm(planned.values[3:6]) > self.thresholds.maximum_stationary_rotation:
                reasons.append(FirstActionReason.UNNECESSARY_ROTATION)
            if abs(planned.values[6]) > self.thresholds.maximum_stationary_gripper:
                reasons.append(FirstActionReason.UNNECESSARY_GRIPPER)
        return FirstActionDecision(active, cosine, tuple(reasons))

    def is_active(self, action: DroidAction) -> bool:
        return (
            _norm(action.values[:3]) > self.thresholds.recorded_translation_activity
            or _norm(action.values[3:6]) > self.thresholds.recorded_rotation_activity
            or abs(action.values[6]) > self.thresholds.recorded_gripper_activity
        )


def evaluate_first_actions(
    recorded: Sequence[DroidAction],
    planned: Sequence[DroidAction],
    gate: FirstActionGate | None = None,
) -> FirstActionSummary:
    if len(recorded) != len(planned):
        raise ValueError("recorded and planned actions must have the same length")
    evaluator = gate or FirstActionGate()
    return FirstActionSummary(
        tuple(
            evaluator.evaluate(recorded_action, planned_action)
            for recorded_action, planned_action in zip(recorded, planned)
        )
    )
