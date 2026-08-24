"""Task semantics shared by planner search identity and readiness reporting."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np

from jepa_wm.action import ACTION_DIMENSIONS, DroidAction
from jepa_wm.planner import CandidateTrustRegion
from jepa_wm.planner_readiness import FirstActionThresholds


@dataclass(frozen=True)
class GoalAlignmentDecision:
    cosine: float
    passed: bool

    def __post_init__(self) -> None:
        if not isfinite(self.cosine) or not -1.0 <= self.cosine <= 1.0:
            raise ValueError("goal-alignment decision cosine is invalid")

    def to_dict(self) -> dict[str, float | bool]:
        return {"cosine": self.cosine, "passed": self.passed}


@dataclass(frozen=True)
class GoalActionAlignment:
    """Keep the first searched DROID action aligned with the observable goal."""

    minimum_cosine: float = 0.95
    failure_penalty: float = 0.01

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_cosine <= 1.0:
            raise ValueError("goal-alignment cosine must be between 0 and 1")
        if not isfinite(self.failure_penalty) or self.failure_penalty <= 0.0:
            raise ValueError("goal-alignment failure penalty must be positive")

    @staticmethod
    def _cosines(actions: np.ndarray, goal: DroidAction) -> np.ndarray:
        values = np.asarray(actions, dtype=np.float64)
        goal_values = np.asarray(goal.values, dtype=np.float64)
        goal_norm = float(np.linalg.norm(goal_values))
        if goal_norm <= 1e-12:
            raise ValueError("goal alignment requires a nonzero action goal")
        norms = np.linalg.norm(values, axis=-1)
        denominator = np.maximum(norms * goal_norm, 1e-12)
        return np.sum(values * goal_values, axis=-1) / denominator

    def evaluate(
        self,
        action: DroidAction,
        goal: DroidAction,
    ) -> GoalAlignmentDecision:
        cosine = float(self._cosines(np.asarray(action.values)[None, :], goal)[0])
        return GoalAlignmentDecision(
            cosine,
            cosine + 1e-12 >= self.minimum_cosine,
        )

    def penalty(self, candidates: np.ndarray, goal: DroidAction) -> np.ndarray:
        values = np.asarray(candidates, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[1] == 0
            or values.shape[2] != ACTION_DIMENSIONS
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "goal alignment requires finite [batch, horizon, 7] actions"
            )
        deficit = np.maximum(
            self.minimum_cosine - self._cosines(values[:, 0, :], goal),
            0.0,
        )
        return np.where(
            deficit > 0.0,
            self.failure_penalty * (1.0 + deficit),
            0.0,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_cosine": self.minimum_cosine,
            "failure_penalty": self.failure_penalty,
        }


class RefinementRejectionReason(str, Enum):
    GOAL_MISALIGNED = "goal_misaligned"
    INSUFFICIENT_LATENT_IMPROVEMENT = "insufficient_latent_improvement"


@dataclass(frozen=True)
class RefinementAcceptanceDecision:
    latent_improvement: float
    accepted: bool
    reasons: tuple[RefinementRejectionReason, ...]

    def __post_init__(self) -> None:
        if not isfinite(self.latent_improvement):
            raise ValueError("refinement latent improvement must be finite")
        if self.accepted == bool(self.reasons):
            raise ValueError("refinement acceptance reasons are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "latent_improvement": self.latent_improvement,
            "reasons": [reason.value for reason in self.reasons],
        }


@dataclass(frozen=True)
class RefinementAcceptancePolicy:
    """Accept search only when it preserves task direction and improves JEPA."""

    minimum_latent_improvement: float = 1e-6

    def __post_init__(self) -> None:
        if (
            not isfinite(self.minimum_latent_improvement)
            or self.minimum_latent_improvement <= 0.0
        ):
            raise ValueError("minimum latent improvement must be finite and positive")

    def evaluate(
        self,
        initial_latent_energy: float,
        searched_latent_energy: float,
        goal_alignment: GoalAlignmentDecision,
    ) -> RefinementAcceptanceDecision:
        if not isfinite(initial_latent_energy) or not isfinite(searched_latent_energy):
            raise ValueError("refinement energies must be finite")
        latent_improvement = initial_latent_energy - searched_latent_energy
        reasons = []
        if not goal_alignment.passed:
            reasons.append(RefinementRejectionReason.GOAL_MISALIGNED)
        if latent_improvement < self.minimum_latent_improvement:
            reasons.append(
                RefinementRejectionReason.INSUFFICIENT_LATENT_IMPROVEMENT
            )
        return RefinementAcceptanceDecision(
            latent_improvement,
            not reasons,
            tuple(reasons),
        )

    def to_dict(self) -> dict[str, float]:
        return {"minimum_latent_improvement": self.minimum_latent_improvement}


@dataclass(frozen=True)
class PlannerTaskPolicy:
    proposal_trust_region: CandidateTrustRegion | None = None
    first_action_thresholds: FirstActionThresholds = FirstActionThresholds()
    goal_action_alignment: GoalActionAlignment | None = None
    refinement_acceptance: RefinementAcceptancePolicy | None = None

    def __post_init__(self) -> None:
        if (
            self.refinement_acceptance is not None
            and self.goal_action_alignment is None
        ):
            raise ValueError("refinement acceptance requires goal alignment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_trust_region": (
                self.proposal_trust_region.to_dict()
                if self.proposal_trust_region is not None
                else None
            ),
            "first_action_thresholds": self.first_action_thresholds.to_dict(),
            "goal_action_alignment": (
                self.goal_action_alignment.to_dict()
                if self.goal_action_alignment is not None
                else None
            ),
            "refinement_acceptance": (
                self.refinement_acceptance.to_dict()
                if self.refinement_acceptance is not None
                else None
            ),
        }
