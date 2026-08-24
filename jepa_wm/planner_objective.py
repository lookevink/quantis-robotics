"""Typed latent, prior, and task objective composition for action search."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

import numpy as np

from jepa_wm.action_prior import EmpiricalActionPrior


CandidateScorer = Callable[[np.ndarray], np.ndarray]
CandidateTaskPenalty = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CandidateObjective:
    latent_energy: float
    prior_penalty: float
    task_penalty: float

    def __post_init__(self) -> None:
        if (
            not all(
                isfinite(value)
                for value in (
                    self.latent_energy,
                    self.prior_penalty,
                    self.task_penalty,
                )
            )
            or self.prior_penalty < 0.0
            or self.task_penalty < 0.0
        ):
            raise ValueError("candidate objective components are invalid")

    @property
    def total(self) -> float:
        return self.latent_energy + self.prior_penalty + self.task_penalty


@dataclass(frozen=True)
class PlannerObjectiveComponents:
    latent_energy: np.ndarray
    prior_penalty: np.ndarray
    task_penalty: np.ndarray

    def __post_init__(self) -> None:
        components = (
            self.latent_energy,
            self.prior_penalty,
            self.task_penalty,
        )
        shapes = {np.asarray(values).shape for values in components}
        if (
            len(shapes) != 1
            or len(next(iter(shapes))) != 1
            or not all(np.all(np.isfinite(values)) for values in components)
            or np.any(self.prior_penalty < 0.0)
            or np.any(self.task_penalty < 0.0)
        ):
            raise ValueError("planner objective components are invalid")

    @property
    def total(self) -> np.ndarray:
        return self.latent_energy + self.prior_penalty + self.task_penalty

    def candidate(self, index: int) -> CandidateObjective:
        return CandidateObjective(
            float(self.latent_energy[index]),
            float(self.prior_penalty[index]),
            float(self.task_penalty[index]),
        )


def evaluate_planner_objective(
    candidates: np.ndarray,
    score: CandidateScorer,
    prior: EmpiricalActionPrior,
    task_penalty: CandidateTaskPenalty | None = None,
    *,
    latent_energy: np.ndarray | None = None,
) -> PlannerObjectiveComponents:
    values = np.asarray(candidates, dtype=np.float64)
    latent = np.asarray(
        score(values) if latent_energy is None else latent_energy,
        dtype=np.float64,
    )
    prior_values = prior.penalty(values)
    task_values = (
        np.asarray(task_penalty(values), dtype=np.float64)
        if task_penalty is not None
        else np.zeros(len(values), dtype=np.float64)
    )
    return PlannerObjectiveComponents(latent, prior_values, task_values)
