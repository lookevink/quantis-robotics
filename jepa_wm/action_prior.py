"""Empirical action support learned only from adapter training recordings."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import numpy as np

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.planner import PlannerDistribution


@dataclass(frozen=True)
class ActionPriorConfig:
    minimum_translation_std: float = 0.001
    minimum_rotation_std: float = 0.005
    minimum_gripper_std: float = 0.01
    penalty_weight: float = 0.002

    def __post_init__(self) -> None:
        values = (
            self.minimum_translation_std,
            self.minimum_rotation_std,
            self.minimum_gripper_std,
            self.penalty_weight,
        )
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError(
                "action prior scales and penalty must be finite and positive"
            )

    @property
    def standard_deviation_floors(self) -> np.ndarray:
        return np.asarray(
            (
                *(self.minimum_translation_std,) * 3,
                *(self.minimum_rotation_std,) * 3,
                self.minimum_gripper_std,
            ),
            dtype=np.float64,
        )

    def distribution_for(self, sequences: np.ndarray) -> PlannerDistribution:
        actions = _validated_sequences(sequences)
        return PlannerDistribution(
            actions.mean(axis=0),
            np.maximum(
                actions.std(axis=0),
                self.standard_deviation_floors[None, :],
            ),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum_translation_std": self.minimum_translation_std,
            "minimum_rotation_std": self.minimum_rotation_std,
            "minimum_gripper_std": self.minimum_gripper_std,
            "penalty_weight": self.penalty_weight,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionPriorConfig:
        return cls(
            minimum_translation_std=float(payload["minimum_translation_std"]),
            minimum_rotation_std=float(payload["minimum_rotation_std"]),
            minimum_gripper_std=float(payload["minimum_gripper_std"]),
            penalty_weight=float(payload["penalty_weight"]),
        )


def _validated_sequences(sequences: np.ndarray) -> np.ndarray:
    actions = np.asarray(sequences, dtype=np.float64)
    if (
        actions.ndim != 3
        or actions.shape[0] == 0
        or actions.shape[1] == 0
        or actions.shape[2] != ACTION_DIMENSIONS
        or not np.all(np.isfinite(actions))
    ):
        raise ValueError(
            "action sequences must have finite shape [samples, horizon, 7]"
        )
    return actions


@dataclass(frozen=True)
class ActionLibrary:
    sequences: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequences", _validated_sequences(self.sequences).copy()
        )

    def goal_conditioned_prior(
        self,
        energies: np.ndarray,
        *,
        elites: int,
        config: ActionPriorConfig,
    ) -> EmpiricalActionPrior:
        values = np.asarray(energies, dtype=np.float64)
        if values.shape != (len(self.sequences),) or not np.all(np.isfinite(values)):
            raise ValueError("action library requires one finite energy per sequence")
        if not 1 < elites <= len(self.sequences):
            raise ValueError("action library elites must be between two and its size")
        indices = np.argsort(values, kind="stable")[:elites]
        return EmpiricalActionPrior.fit(self.sequences[indices], config)


@dataclass(frozen=True)
class EmpiricalActionPrior:
    distribution: PlannerDistribution
    penalty_weight: float

    @classmethod
    def fit(
        cls,
        sequences: np.ndarray,
        config: ActionPriorConfig,
    ) -> EmpiricalActionPrior:
        actions = _validated_sequences(sequences)
        return cls(
            distribution=config.distribution_for(actions),
            penalty_weight=config.penalty_weight,
        )

    def penalty(self, candidates: np.ndarray) -> np.ndarray:
        actions = np.asarray(candidates, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[1:] != self.distribution.mean.shape:
            raise ValueError("candidate actions do not match the action prior horizon")
        standardized = (
            actions - self.distribution.mean[None, :, :]
        ) / self.distribution.standard_deviation[None, :, :]
        return self.penalty_weight * np.square(standardized).mean(axis=(1, 2))
