"""Promotion requirements scoped to the routes an adapter actually owns."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, AbstractSet


@dataclass(frozen=True)
class SliceEvaluation:
    recorded_win_rate: float
    mean_improvement_over_zero: float
    signed_order_fraction: float
    matches_baseline: bool | None = None

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (
                self.recorded_win_rate,
                self.mean_improvement_over_zero,
                self.signed_order_fraction,
            )
        ):
            raise ValueError("slice evaluation values must be finite")
        if (
            not 0.0 <= self.recorded_win_rate <= 1.0
            or not 0.0 <= self.signed_order_fraction <= 1.0
        ):
            raise ValueError("slice evaluation fractions must be in [0, 1]")


@dataclass(frozen=True)
class SliceRequirement:
    mode: str
    minimum_win_rate: float | None = None
    require_positive_mean: bool = False
    minimum_signed_order_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"owned", "passthrough"}:
            raise ValueError("slice requirement mode must be owned or passthrough")
        for value in (self.minimum_win_rate, self.minimum_signed_order_fraction):
            if value is not None and (not isfinite(value) or not 0.0 <= value <= 1.0):
                raise ValueError("slice thresholds must be in [0, 1]")
        if self.mode == "passthrough" and (
            self.minimum_win_rate is not None
            or self.require_positive_mean
            or self.minimum_signed_order_fraction is not None
        ):
            raise ValueError("passthrough slices require only baseline equivalence")

    @classmethod
    def owned(
        cls,
        *,
        minimum_win_rate: float,
        require_positive_mean: bool,
        minimum_signed_order_fraction: float | None = None,
    ) -> SliceRequirement:
        return cls(
            "owned",
            minimum_win_rate,
            require_positive_mean,
            minimum_signed_order_fraction,
        )

    @classmethod
    def passthrough(cls) -> SliceRequirement:
        return cls("passthrough")


@dataclass(frozen=True)
class SliceGateDecision:
    passed: bool
    reasons: tuple[str, ...]


def _owned_reasons(
    name: str,
    requirement: SliceRequirement,
    evaluation: SliceEvaluation,
) -> list[str]:
    reasons = []
    assert requirement.minimum_win_rate is not None
    if evaluation.recorded_win_rate < requirement.minimum_win_rate:
        reasons.append(f"{name} recorded win rate is below its owned threshold")
    if (
        requirement.require_positive_mean
        and evaluation.mean_improvement_over_zero <= 0.0
    ):
        reasons.append(f"{name} mean improvement is not positive")
    if (
        requirement.minimum_signed_order_fraction is not None
        and evaluation.signed_order_fraction < requirement.minimum_signed_order_fraction
    ):
        reasons.append(f"{name} signed order is below its owned threshold")
    return reasons


class OwnedSliceGate:
    def __init__(self, requirements: Mapping[str, SliceRequirement]) -> None:
        if not requirements:
            raise ValueError("owned-slice gate requires at least one slice")
        self.requirements = dict(requirements)

    def evaluate(
        self,
        evaluations: Mapping[str, SliceEvaluation],
    ) -> SliceGateDecision:
        reasons = []
        for name, requirement in self.requirements.items():
            evaluation = evaluations.get(name)
            if evaluation is None:
                reasons.append(f"{name} evaluation is missing")
                continue
            if requirement.mode == "passthrough":
                if evaluation.matches_baseline is not True:
                    reasons.append(f"{name} does not preserve baseline equivalence")
            else:
                reasons.extend(_owned_reasons(name, requirement, evaluation))
        return SliceGateDecision(not reasons, tuple(reasons))


@dataclass(frozen=True)
class TrainingFeasibility:
    feasible: bool
    reasons: tuple[str, ...]

    @classmethod
    def evaluate(
        cls,
        *,
        requirements: Mapping[str, SliceRequirement],
        route_counts: Mapping[str, Mapping[str, int]],
        trainable_routes: AbstractSet[str],
    ) -> TrainingFeasibility:
        """Reject structurally untrainable owned slices before model encoding."""

        reasons = []
        for name, requirement in requirements.items():
            if requirement.mode != "owned":
                continue
            counts = route_counts.get(name)
            if counts is None:
                reasons.append(f"{name} route audit is missing")
                continue
            has_trainable_route = any(
                route in trainable_routes and count > 0
                for route, count in counts.items()
            )
            if has_trainable_route:
                continue
            reasons.append(f"{name} has no trainable route in its future horizon")
        return cls(not reasons, tuple(reasons))
