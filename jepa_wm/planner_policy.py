"""Task semantics shared by planner search identity and readiness reporting."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class PlannerTaskPolicy:
    proposal_trust_region: CandidateTrustRegion | None = None
    first_action_thresholds: FirstActionThresholds = FirstActionThresholds()
    goal_action_alignment: GoalActionAlignment | None = None

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
        }
