"""Explicit acceptance gate between offline JEPA-WM evaluation and control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ActionControlReason(str, Enum):
    NON_FINITE_METRICS = "non_finite_metrics"
    NON_POSITIVE_MEAN_IMPROVEMENT = "non_positive_mean_improvement"
    INSUFFICIENT_WIN_RATE = "insufficient_win_rate"


@dataclass(frozen=True)
class ActionControlDecision:
    passed: bool
    minimum_win_rate: float
    reasons: tuple[ActionControlReason, ...]

    def to_dict(self) -> dict[str, bool | float | list[str]]:
        return {
            "passed": self.passed,
            "minimum_win_rate": self.minimum_win_rate,
            "requires_positive_mean_improvement": True,
            "reasons": [reason.value for reason in self.reasons],
        }


@dataclass(frozen=True)
class ActionControlGate:
    minimum_win_rate: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_win_rate <= 1.0:
            raise ValueError("minimum win rate must be between zero and one")

    def evaluate(
        self,
        *,
        mean_improvement_over_zero: float,
        recorded_action_win_rate: float,
    ) -> ActionControlDecision:
        reasons = []
        metrics_are_finite = isfinite(mean_improvement_over_zero) and isfinite(
            recorded_action_win_rate
        )
        if not metrics_are_finite:
            reasons.append(ActionControlReason.NON_FINITE_METRICS)
        elif mean_improvement_over_zero <= 0.0:
            reasons.append(ActionControlReason.NON_POSITIVE_MEAN_IMPROVEMENT)
        if metrics_are_finite and recorded_action_win_rate < self.minimum_win_rate:
            reasons.append(ActionControlReason.INSUFFICIENT_WIN_RATE)
        return ActionControlDecision(
            passed=not reasons,
            minimum_win_rate=self.minimum_win_rate,
            reasons=tuple(reasons),
        )
