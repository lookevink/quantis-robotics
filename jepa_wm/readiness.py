"""Explicit acceptance gate between offline JEPA-WM evaluation and control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping


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


class ResidualTrainReason(str, Enum):
    ACTION_CONTROL_GATE = "action_control_gate"
    NON_FINITE_GATE_METRIC = "non_finite_gate_metric"
    INSUFFICIENT_OVERALL_WIN_RATE = "insufficient_overall_win_rate"
    INSUFFICIENT_RETAINED_WIN_RATE = "insufficient_retained_win_rate"
    INSUFFICIENT_POST_WIN_RATE = "insufficient_post_win_rate"
    NON_POSITIVE_SEGMENT_IMPROVEMENT = "non_positive_segment_improvement"
    INSUFFICIENT_SIGNED_ORDER = "insufficient_signed_order"
    NON_FINITE_RESIDUAL_RATIO = "non_finite_residual_ratio"
    EXCESS_RESIDUAL_RATIO = "excess_residual_ratio"


@dataclass(frozen=True)
class ResidualTrainDecision:
    passed: bool
    reasons: tuple[ResidualTrainReason, ...]
    maximum_residual_ratio: float
    residual_ratio_limit: float
    residual_ratio_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": [reason.value for reason in self.reasons],
            "maximum_residual_ratio": self.maximum_residual_ratio,
            "residual_ratio_limit": self.residual_ratio_limit,
            "residual_ratio_tolerance": self.residual_ratio_tolerance,
        }


@dataclass(frozen=True)
class ResidualTrainGate:
    minimum_overall_win_rate: float = 0.90
    minimum_retained_win_rate: float = 0.85
    minimum_post_win_rate: float = 0.95
    minimum_signed_order_fraction: float = 0.75
    maximum_residual_ratio: float = 0.15
    residual_ratio_tolerance: float = 1e-6
    required_signed_segments: tuple[str, ...] = ("retreat", "align", "insert")

    def __post_init__(self) -> None:
        fractions = (
            self.minimum_overall_win_rate,
            self.minimum_retained_win_rate,
            self.minimum_post_win_rate,
            self.minimum_signed_order_fraction,
            self.maximum_residual_ratio,
            self.residual_ratio_tolerance,
        )
        if (
            any(not isfinite(value) or value < 0.0 for value in fractions)
            or any(
                value > 1.0
                for value in (
                    self.minimum_overall_win_rate,
                    self.minimum_retained_win_rate,
                    self.minimum_post_win_rate,
                    self.minimum_signed_order_fraction,
                )
            )
            or not self.required_signed_segments
            or len(set(self.required_signed_segments))
            != len(self.required_signed_segments)
        ):
            raise ValueError("residual TRAIN gate configuration is invalid")

    def evaluate(
        self,
        *,
        aggregate: Mapping[str, float],
        retained: Mapping[str, float],
        post: Mapping[str, float],
        by_segment: Mapping[str, Mapping[str, float]],
        maximum_residual_ratio: float,
    ) -> ResidualTrainDecision:
        control_gate = ActionControlGate().evaluate(
            mean_improvement_over_zero=aggregate["mean_improvement_over_zero"],
            recorded_action_win_rate=aggregate["recorded_action_win_rate"],
        )
        reasons = []
        if not control_gate.passed:
            reasons.append(ResidualTrainReason.ACTION_CONTROL_GATE)
        remaining_metrics = (
            retained["recorded_action_win_rate"],
            post["recorded_action_win_rate"],
            *(
                value
                for metrics in by_segment.values()
                for value in (
                    metrics["mean_improvement_over_zero"],
                    metrics["signed_order_fraction"],
                )
            ),
        )
        if any(not isfinite(value) for value in remaining_metrics):
            reasons.append(ResidualTrainReason.NON_FINITE_GATE_METRIC)
        if aggregate["recorded_action_win_rate"] < self.minimum_overall_win_rate:
            reasons.append(ResidualTrainReason.INSUFFICIENT_OVERALL_WIN_RATE)
        if retained["recorded_action_win_rate"] < self.minimum_retained_win_rate:
            reasons.append(ResidualTrainReason.INSUFFICIENT_RETAINED_WIN_RATE)
        if post["recorded_action_win_rate"] < self.minimum_post_win_rate:
            reasons.append(ResidualTrainReason.INSUFFICIENT_POST_WIN_RATE)
        if any(
            metrics["mean_improvement_over_zero"] <= 0.0
            for metrics in by_segment.values()
        ):
            reasons.append(ResidualTrainReason.NON_POSITIVE_SEGMENT_IMPROVEMENT)
        if any(
            segment not in by_segment
            or by_segment[segment]["signed_order_fraction"]
            < self.minimum_signed_order_fraction
            for segment in self.required_signed_segments
        ):
            reasons.append(ResidualTrainReason.INSUFFICIENT_SIGNED_ORDER)
        if not isfinite(maximum_residual_ratio):
            reasons.append(ResidualTrainReason.NON_FINITE_RESIDUAL_RATIO)
        elif maximum_residual_ratio > (
            self.maximum_residual_ratio + self.residual_ratio_tolerance
        ):
            reasons.append(ResidualTrainReason.EXCESS_RESIDUAL_RATIO)
        return ResidualTrainDecision(
            passed=not reasons,
            reasons=tuple(reasons),
            maximum_residual_ratio=maximum_residual_ratio,
            residual_ratio_limit=self.maximum_residual_ratio,
            residual_ratio_tolerance=self.residual_ratio_tolerance,
        )
