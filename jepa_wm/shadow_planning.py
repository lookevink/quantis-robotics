"""Proposal-centered candidate search that cannot command the simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jepa_wm.action import ACTION_DIMENSIONS, DroidAction
from jepa_wm.action_prior import ActionPriorConfig, EmpiricalActionPrior
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.planner import (
    CEMConfig,
    CEMPlanner,
    PlannerActionBounds,
    PlannerDistribution,
)
from jepa_wm.planner_readiness import (
    FirstActionDecision,
    FirstActionGate,
    FirstActionThresholds,
)


CandidateScorer = Callable[[np.ndarray], np.ndarray]
SHADOW_REQUEST_SCHEMA = "quantis.jepa_wm_shadow_request.v1"


class CandidateAuthority(str, Enum):
    """The candidate is evidence only and can never become a command implicitly."""

    SHADOW_ONLY = "shadow_only"


@dataclass(frozen=True)
class ShadowPlanningRequest:
    """Bind a shadow search to the exact proposal returned for one observation."""

    observation: ControlObservation
    direct_control: ProposedControl
    expected_adapter: Path

    def __post_init__(self) -> None:
        if self.observation.observation_id != self.direct_control.observation_id:
            raise ValueError("shadow request observation identity does not match")
        if self.observation.expected_proposal != self.direct_control.proposal:
            raise ValueError("shadow request proposal checkpoint does not match")
        if not self.expected_adapter.is_absolute():
            raise ValueError("shadow request adapter path must be absolute")
        if (
            self.direct_control.created_at_unix_seconds
            < self.observation.captured_at_unix_seconds
        ):
            raise ValueError("shadow request predates its observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": SHADOW_REQUEST_SCHEMA,
            "observation": self.observation.to_dict(),
            "direct_control": self.direct_control.to_dict(),
            "expected_adapter": str(self.expected_adapter),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ShadowPlanningRequest:
        if not isinstance(payload, Mapping) or payload.get("schema") != SHADOW_REQUEST_SCHEMA:
            raise ValueError("shadow planning request schema is unsupported")
        try:
            return cls(
                ControlObservation.from_dict(payload["observation"]),
                ProposedControl.from_dict(payload["direct_control"]),
                Path(payload["expected_adapter"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("shadow planning request is incomplete") from error


@dataclass(frozen=True)
class CandidateTrustRegion:
    """Maximum per-step departure from the direct proposal."""

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
    """Intersection of global action limits and a direct-proposal trust region."""

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
class ShadowSearchConfig:
    planner: CEMConfig = CEMConfig(iterations=4, samples=64, elites=8)
    # JEPA terminal energies are ~1e-2 in the live domain. Keep the prior as a
    # tie-breaker inside the strict trust region instead of overwhelming them.
    prior: ActionPriorConfig = ActionPriorConfig(penalty_weight=1e-5)
    global_bounds: PlannerActionBounds = PlannerActionBounds()
    trust_region: CandidateTrustRegion = CandidateTrustRegion()
    first_action_thresholds: FirstActionThresholds = FirstActionThresholds(
        minimum_active_cosine=0.9
    )
    minimum_energy_improvement: float = 0.0

    def __post_init__(self) -> None:
        if self.planner.horizon != 3:
            raise ValueError("shadow search requires the native three-action horizon")
        if (
            not isfinite(self.minimum_energy_improvement)
            or self.minimum_energy_improvement < 0.0
        ):
            raise ValueError("minimum energy improvement must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner": self.planner.to_dict(),
            "prior": self.prior.to_dict(),
            "global_bounds": self.global_bounds.to_dict(),
            "trust_region": self.trust_region.to_dict(),
            "first_action_thresholds": self.first_action_thresholds.to_dict(),
            "minimum_energy_improvement": self.minimum_energy_improvement,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShadowSearchConfig:
        return cls(
            planner=CEMConfig(**payload["planner"]),
            prior=ActionPriorConfig.from_dict(payload["prior"]),
            global_bounds=PlannerActionBounds(**payload["global_bounds"]),
            trust_region=CandidateTrustRegion.from_dict(payload["trust_region"]),
            first_action_thresholds=FirstActionThresholds.from_dict(
                payload["first_action_thresholds"]
            ),
            minimum_energy_improvement=float(payload["minimum_energy_improvement"]),
        )


@dataclass(frozen=True)
class ShadowCandidate:
    actions: tuple[DroidAction, ...]
    latent_energy: float
    prior_penalty: float
    objective: float

    def __post_init__(self) -> None:
        if len(self.actions) != 3:
            raise ValueError("shadow candidate requires exactly three actions")
        if not all(
            isfinite(value)
            for value in (self.latent_energy, self.prior_penalty, self.objective)
        ):
            raise ValueError("shadow candidate metrics must be finite")
        if self.prior_penalty < 0.0 or not np.isclose(
            self.objective,
            self.latent_energy + self.prior_penalty,
            rtol=0.0,
            atol=1e-9,
        ):
            raise ValueError("shadow candidate objective is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [list(action.values) for action in self.actions],
            "latent_energy": self.latent_energy,
            "prior_penalty": self.prior_penalty,
            "objective": self.objective,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShadowCandidate:
        return cls(
            actions=tuple(DroidAction(tuple(values)) for values in payload["actions"]),
            latent_energy=float(payload["latent_energy"]),
            prior_penalty=float(payload["prior_penalty"]),
            objective=float(payload["objective"]),
        )


@dataclass(frozen=True)
class ShadowSearchEvidence:
    observation_id: int
    proposal: Path
    adapter: Path
    direct: ShadowCandidate
    planned: ShadowCandidate
    config: ShadowSearchConfig
    first_action_gate: FirstActionDecision
    candidates_scored: int
    iteration_best_objectives: tuple[float, ...]
    planning_seconds: float
    authority: CandidateAuthority = CandidateAuthority.SHADOW_ONLY

    def __post_init__(self) -> None:
        if self.observation_id <= 0 or self.candidates_scored <= 0:
            raise ValueError("shadow evidence requires positive observation and sample counts")
        if not isfinite(self.planning_seconds) or self.planning_seconds < 0.0:
            raise ValueError("shadow planning time must be finite and non-negative")
        if not self.iteration_best_objectives or not all(
            isfinite(value) for value in self.iteration_best_objectives
        ):
            raise ValueError("shadow evidence requires finite CEM iteration objectives")
        if self.authority is not CandidateAuthority.SHADOW_ONLY:
            raise ValueError("shadow candidates cannot have command authority")
        direct_values = np.asarray(
            [action.values for action in self.direct.actions], dtype=np.float64
        )
        planned_values = np.asarray(
            [action.values for action in self.planned.actions], dtype=np.float64
        )
        bounds = ProposalCenteredBounds(
            direct_values,
            self.config.global_bounds,
            self.config.trust_region,
        )
        if not np.allclose(
            bounds.clip(planned_values[None, :, :])[0],
            planned_values,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("planned shadow actions violate their saved bounds")
        expected_gate = FirstActionGate(
            self.config.first_action_thresholds
        ).evaluate(self.direct.actions[0], self.planned.actions[0])
        if self.first_action_gate != expected_gate:
            raise ValueError("shadow first-action decision is inconsistent")
        if (
            self.candidates_scored
            != self.config.planner.iterations * self.config.planner.samples
            or len(self.iteration_best_objectives) != self.config.planner.iterations
        ):
            raise ValueError("shadow CEM evidence does not match its saved budget")

    @property
    def energy_improvement(self) -> float:
        return self.direct.latent_energy - self.planned.latent_energy

    @property
    def objective_improvement(self) -> float:
        return self.direct.objective - self.planned.objective

    @property
    def passes_shadow_gate(self) -> bool:
        return (
            self.energy_improvement > self.config.minimum_energy_improvement
            and self.objective_improvement > 0.0
            and self.first_action_gate.passed
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "quantis_jepa_wm_shadow_search_v1",
            "observation_id": self.observation_id,
            "proposal": str(self.proposal),
            "adapter": str(self.adapter),
            "direct": self.direct.to_dict(),
            "planned": self.planned.to_dict(),
            "config": self.config.to_dict(),
            "first_action_gate": self.first_action_gate.to_dict(),
            "candidates_scored": self.candidates_scored,
            "iteration_best_objectives": list(self.iteration_best_objectives),
            "planning_seconds": self.planning_seconds,
            "energy_improvement": self.energy_improvement,
            "objective_improvement": self.objective_improvement,
            "passes_shadow_gate": self.passes_shadow_gate,
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ShadowSearchEvidence:
        if payload.get("schema") != "quantis_jepa_wm_shadow_search_v1":
            raise ValueError("shadow-search evidence schema is invalid")
        return cls(
            observation_id=int(payload["observation_id"]),
            proposal=Path(payload["proposal"]),
            adapter=Path(payload["adapter"]),
            direct=ShadowCandidate.from_dict(payload["direct"]),
            planned=ShadowCandidate.from_dict(payload["planned"]),
            config=ShadowSearchConfig.from_dict(payload["config"]),
            first_action_gate=FirstActionDecision.from_dict(
                payload["first_action_gate"]
            ),
            candidates_scored=int(payload["candidates_scored"]),
            iteration_best_objectives=tuple(
                float(value) for value in payload["iteration_best_objectives"]
            ),
            planning_seconds=float(payload["planning_seconds"]),
            authority=CandidateAuthority(payload["authority"]),
        )


def _candidate(
    actions: np.ndarray,
    score: CandidateScorer,
    prior: EmpiricalActionPrior,
) -> ShadowCandidate:
    batch = np.asarray(actions, dtype=np.float64)[None, :, :]
    latent_energy = float(np.asarray(score(batch), dtype=np.float64)[0])
    prior_penalty = float(prior.penalty(batch)[0])
    return ShadowCandidate(
        actions=tuple(DroidAction(tuple(row)) for row in batch[0]),
        latent_energy=latent_energy,
        prior_penalty=prior_penalty,
        objective=latent_energy + prior_penalty,
    )


def plan_shadow_candidates(
    *,
    observation_id: int,
    direct_actions: Sequence[DroidAction],
    score: CandidateScorer,
    proposal: Path,
    adapter: Path,
    config: ShadowSearchConfig = ShadowSearchConfig(),
) -> ShadowSearchEvidence:
    """Refine a direct proposal while preserving it as the command authority."""

    direct = tuple(direct_actions)
    if len(direct) != config.planner.horizon:
        raise ValueError("direct proposal does not match the CEM horizon")
    center = np.asarray([action.values for action in direct], dtype=np.float64)
    bounds = ProposalCenteredBounds(center, config.global_bounds, config.trust_region)
    prior = EmpiricalActionPrior.fit(center[None, :, :], config.prior)
    initial = PlannerDistribution(
        mean=center,
        standard_deviation=np.minimum(
            prior.distribution.standard_deviation,
            bounds.initial_standard_deviation[None, :],
        ),
    )

    def objective(candidates: np.ndarray) -> np.ndarray:
        latent = np.asarray(score(candidates), dtype=np.float64)
        return latent + prior.penalty(candidates)

    planning_started = monotonic()
    result = CEMPlanner(config.planner, bounds).plan(
        objective,
        initial_distribution=initial,
    )
    measured_planning_seconds = monotonic() - planning_started
    direct_candidate = _candidate(center, score, prior)
    planned_candidate = _candidate(result.actions, score, prior)
    first_action_gate = FirstActionGate(config.first_action_thresholds).evaluate(
        direct_candidate.actions[0],
        planned_candidate.actions[0],
    )
    return ShadowSearchEvidence(
        observation_id=observation_id,
        proposal=proposal,
        adapter=adapter,
        direct=direct_candidate,
        planned=planned_candidate,
        config=config,
        first_action_gate=first_action_gate,
        candidates_scored=result.candidates_scored,
        iteration_best_objectives=result.iteration_best_energies,
        planning_seconds=measured_planning_seconds,
    )
