"""Bounded cross-entropy action search for the DROID JEPA-WM horizon."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jepa_wm.action import ACTION_DIMENSIONS, DroidAction


CandidateScorer = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class PlannerDistribution:
    mean: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        standard_deviation = np.asarray(self.standard_deviation, dtype=np.float64)
        if (
            mean.ndim != 2
            or mean.shape[1] != ACTION_DIMENSIONS
            or standard_deviation.shape != mean.shape
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(standard_deviation))
            or np.any(standard_deviation <= 0.0)
        ):
            raise ValueError(
                "planner distribution requires finite [horizon, 7] mean and positive standard deviation"
            )
        object.__setattr__(self, "mean", mean.copy())
        object.__setattr__(self, "standard_deviation", standard_deviation.copy())


@dataclass(frozen=True)
class PlannerActionBounds:
    """Conservative per-step Cartesian limits for simulator planning."""

    maximum_translation_norm: float = 0.02
    maximum_rotation_norm: float = 0.08
    maximum_gripper_delta: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.maximum_translation_norm,
            self.maximum_rotation_norm,
            self.maximum_gripper_delta,
        )
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError("planner action bounds must be finite and positive")

    @property
    def initial_standard_deviation(self) -> np.ndarray:
        return np.asarray(
            (
                *(self.maximum_translation_norm,) * 3,
                *(self.maximum_rotation_norm,) * 3,
                self.maximum_gripper_delta,
            ),
            dtype=np.float64,
        )

    def clip(self, candidates: np.ndarray) -> np.ndarray:
        bounded = np.asarray(candidates, dtype=np.float64).copy()
        if bounded.ndim != 3 or bounded.shape[2] != ACTION_DIMENSIONS:
            raise ValueError("candidate actions must have shape [batch, horizon, 7]")
        self._clip_vector_norm(bounded[:, :, :3], self.maximum_translation_norm)
        self._clip_vector_norm(bounded[:, :, 3:6], self.maximum_rotation_norm)
        np.clip(
            bounded[:, :, 6],
            -self.maximum_gripper_delta,
            self.maximum_gripper_delta,
            out=bounded[:, :, 6],
        )
        return bounded

    def clip_tensor(self, candidates):
        """Apply the same planner bounds to a Torch candidate tensor."""

        import torch

        if candidates.ndim < 2 or candidates.shape[-1] != ACTION_DIMENSIONS:
            raise ValueError("candidate actions must end with the seven DROID values")
        bounded = candidates.clone()
        bounded[..., :3] = self._clip_tensor_vector_norm(
            bounded[..., :3], self.maximum_translation_norm, torch
        )
        bounded[..., 3:6] = self._clip_tensor_vector_norm(
            bounded[..., 3:6], self.maximum_rotation_norm, torch
        )
        bounded[..., 6].clamp_(
            -self.maximum_gripper_delta, self.maximum_gripper_delta
        )
        return bounded

    def accepts(self, actions: Sequence[DroidAction]) -> bool:
        return bool(actions) and all(
            np.linalg.norm(action.values[:3])
            <= self.maximum_translation_norm + 1e-12
            and np.linalg.norm(action.values[3:6])
            <= self.maximum_rotation_norm + 1e-12
            and abs(action.values[6]) <= self.maximum_gripper_delta + 1e-12
            for action in actions
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "maximum_translation_norm": self.maximum_translation_norm,
            "maximum_rotation_norm": self.maximum_rotation_norm,
            "maximum_gripper_delta": self.maximum_gripper_delta,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannerActionBounds:
        return cls(
            maximum_translation_norm=float(payload["maximum_translation_norm"]),
            maximum_rotation_norm=float(payload["maximum_rotation_norm"]),
            maximum_gripper_delta=float(payload["maximum_gripper_delta"]),
        )

    @staticmethod
    def _clip_vector_norm(vectors: np.ndarray, maximum_norm: float) -> None:
        norms = np.linalg.norm(vectors, axis=2, keepdims=True)
        scales = np.minimum(1.0, maximum_norm / np.maximum(norms, 1e-12))
        vectors *= scales

    @staticmethod
    def _clip_tensor_vector_norm(vectors, maximum_norm: float, torch):
        norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        scales = torch.clamp(
            maximum_norm / torch.clamp_min(norms, 1e-12), max=1.0
        )
        return vectors * scales


@dataclass(frozen=True)
class CandidateTrustRegion:
    """Maximum per-step departure from an initial action proposal."""

    maximum_translation_deviation: float = 0.001
    maximum_rotation_deviation: float = 0.004
    maximum_gripper_deviation: float = 0.02

    def __post_init__(self) -> None:
        values = (
            self.maximum_translation_deviation,
            self.maximum_rotation_deviation,
            self.maximum_gripper_deviation,
        )
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError("candidate trust-region limits must be finite and positive")

    @property
    def standard_deviation_ceiling(self) -> np.ndarray:
        return np.asarray(
            (
                *(self.maximum_translation_deviation,) * 3,
                *(self.maximum_rotation_deviation,) * 3,
                self.maximum_gripper_deviation,
            ),
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "maximum_translation_deviation": self.maximum_translation_deviation,
            "maximum_rotation_deviation": self.maximum_rotation_deviation,
            "maximum_gripper_deviation": self.maximum_gripper_deviation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateTrustRegion:
        return cls(
            maximum_translation_deviation=float(
                payload["maximum_translation_deviation"]
            ),
            maximum_rotation_deviation=float(payload["maximum_rotation_deviation"]),
            maximum_gripper_deviation=float(payload["maximum_gripper_deviation"]),
        )


class ProposalCenteredBounds:
    """Intersection of global action limits and a proposal trust region."""

    def __init__(
        self,
        center: np.ndarray,
        global_bounds: PlannerActionBounds,
        trust_region: CandidateTrustRegion,
    ) -> None:
        values = np.asarray(center, dtype=np.float64)
        if (
            values.ndim != 2
            or values.shape[1] != ACTION_DIMENSIONS
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("candidate center must have finite shape [horizon, 7]")
        center_actions = tuple(DroidAction(tuple(row)) for row in values)
        if not global_bounds.accepts(center_actions):
            raise ValueError("direct proposal is outside global planner bounds")
        self.center = values.copy()
        self.global_bounds = global_bounds
        self.trust_region = trust_region

    @property
    def initial_standard_deviation(self) -> np.ndarray:
        return np.minimum(
            self.global_bounds.initial_standard_deviation,
            self.trust_region.standard_deviation_ceiling,
        )

    def clip(self, candidates: np.ndarray) -> np.ndarray:
        bounded = self.global_bounds.clip(candidates)
        if bounded.shape[1:] != self.center.shape:
            raise ValueError("candidate actions do not match the proposal horizon")
        translation_delta = bounded[:, :, :3] - self.center[None, :, :3]
        self._clip_vector_norm(
            translation_delta,
            self.trust_region.maximum_translation_deviation,
        )
        bounded[:, :, :3] = self.center[None, :, :3] + translation_delta

        rotation_delta = bounded[:, :, 3:6] - self.center[None, :, 3:6]
        self._clip_vector_norm(
            rotation_delta,
            self.trust_region.maximum_rotation_deviation,
        )
        bounded[:, :, 3:6] = self.center[None, :, 3:6] + rotation_delta

        np.clip(
            bounded[:, :, 6],
            self.center[None, :, 6]
            - self.trust_region.maximum_gripper_deviation,
            self.center[None, :, 6]
            + self.trust_region.maximum_gripper_deviation,
            out=bounded[:, :, 6],
        )
        return self.global_bounds.clip(bounded)

    @staticmethod
    def _clip_vector_norm(vectors: np.ndarray, maximum_norm: float) -> None:
        norms = np.linalg.norm(vectors, axis=2, keepdims=True)
        vectors *= np.minimum(1.0, maximum_norm / np.maximum(norms, 1e-12))


@dataclass(frozen=True)
class CEMConfig:
    horizon: int = 3
    iterations: int = 6
    samples: int = 300
    elites: int = 10
    seed: int = 234
    minimum_standard_deviation: float = 1e-4

    def __post_init__(self) -> None:
        if self.horizon <= 0 or self.iterations <= 0 or self.samples <= 1:
            raise ValueError("CEM horizon, iterations, and samples must be positive")
        if not 1 < self.elites <= self.samples:
            raise ValueError("CEM elites must be between two and the sample count")
        if self.seed < 0:
            raise ValueError("CEM seed must be non-negative")
        if (
            not isfinite(self.minimum_standard_deviation)
            or self.minimum_standard_deviation <= 0
        ):
            raise ValueError("CEM minimum standard deviation must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "horizon": self.horizon,
            "iterations": self.iterations,
            "samples": self.samples,
            "elites": self.elites,
            "seed": self.seed,
            "minimum_standard_deviation": self.minimum_standard_deviation,
        }


@dataclass(frozen=True)
class PlanResult:
    actions: np.ndarray
    energy: float
    candidates_scored: int
    iteration_best_energies: tuple[float, ...]


class CEMPlanner:
    """Refine a diagonal Gaussian over bounded three-action sequences."""

    def __init__(self, config: CEMConfig, bounds: PlannerActionBounds) -> None:
        self.config = config
        self.bounds = bounds

    def plan(
        self,
        score: CandidateScorer,
        *,
        initial_distribution: PlannerDistribution | None = None,
    ) -> PlanResult:
        generator = np.random.default_rng(self.config.seed)
        if initial_distribution is None:
            mean = np.zeros((self.config.horizon, ACTION_DIMENSIONS), dtype=np.float64)
            standard_deviation = np.broadcast_to(
                self.bounds.initial_standard_deviation,
                mean.shape,
            ).copy()
        else:
            if initial_distribution.mean.shape != (
                self.config.horizon,
                ACTION_DIMENSIONS,
            ):
                raise ValueError("initial distribution does not match the CEM horizon")
            mean = initial_distribution.mean.copy()
            standard_deviation = initial_distribution.standard_deviation.copy()
        best_actions: np.ndarray | None = None
        best_energy = float("inf")
        iteration_best_energies = []

        for _ in range(self.config.iterations):
            candidates = generator.normal(
                loc=mean,
                scale=standard_deviation,
                size=(
                    self.config.samples,
                    self.config.horizon,
                    ACTION_DIMENSIONS,
                ),
            )
            candidates[0] = mean
            candidates = self.bounds.clip(candidates)
            energies = np.asarray(score(candidates), dtype=np.float64)
            if energies.shape != (self.config.samples,) or not np.all(
                np.isfinite(energies)
            ):
                raise ValueError("scorer must return one finite energy per candidate")

            order = np.argsort(energies, kind="stable")
            iteration_best = int(order[0])
            iteration_energy = float(energies[iteration_best])
            iteration_best_energies.append(iteration_energy)
            if iteration_energy < best_energy:
                best_energy = iteration_energy
                best_actions = candidates[iteration_best].copy()

            elites = candidates[order[: self.config.elites]]
            mean = elites.mean(axis=0)
            standard_deviation = np.maximum(
                elites.std(axis=0),
                self.config.minimum_standard_deviation,
            )

        if best_actions is None:
            raise RuntimeError("CEM planner did not evaluate any candidates")
        return PlanResult(
            actions=best_actions,
            energy=best_energy,
            candidates_scored=self.config.iterations * self.config.samples,
            iteration_best_energies=tuple(iteration_best_energies),
        )
