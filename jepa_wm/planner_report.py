"""Typed aggregation and persistence for offline planner benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from jepa_wm.action import ActionSelectionBounds, DroidAction
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.planner import CEMConfig, PlannerActionBounds
from jepa_wm.planner_policy import PlannerTaskPolicy
from jepa_wm.planner_readiness import (
    FirstActionDecision,
    FirstActionGate,
    FirstActionThresholds,
    evaluate_first_actions,
)
from jepa_wm.trajectory import RolloutWindow
from jepa_wm.training_artifact import ArtifactIdentity, TrainingArtifactIdentity


REPORT_SCHEMA = "quantis.jepa_wm_planner_benchmark.v1"


class PlannerInitialization(str, Enum):
    LIBRARY = "library"
    PROPOSAL = "proposal"


@dataclass(frozen=True)
class CandidateEvaluation:
    actions: np.ndarray
    energy: float
    objective: float

    def __post_init__(self) -> None:
        if (
            self.actions.ndim != 2
            or self.actions.shape[1] != 7
            or not np.all(np.isfinite(self.actions))
            or not isfinite(self.energy)
            or not isfinite(self.objective)
        ):
            raise ValueError("candidate evaluation must contain finite actions and scores")


@dataclass(frozen=True)
class PlannerRolloutEvaluation:
    context_index: int
    target_index: int
    recorded_actions: np.ndarray
    recorded_energy: float
    zero_energy: float
    initialization: PlannerInitialization
    initial_candidate: CandidateEvaluation
    planned_candidate: CandidateEvaluation

    def __post_init__(self) -> None:
        if (
            self.recorded_actions.shape != self.planned_candidate.actions.shape
            or not np.all(np.isfinite(self.recorded_actions))
            or not isfinite(self.recorded_energy)
            or not isfinite(self.zero_energy)
        ):
            raise ValueError("recorded and planned action horizons must match")

    @property
    def planned_prior_penalty(self) -> float:
        return self.planned_candidate.objective - self.planned_candidate.energy

    @property
    def refinement_improvement(self) -> float:
        return self.initial_candidate.objective - self.planned_candidate.objective

    @property
    def improvement_over_zero(self) -> float:
        return self.zero_energy - self.planned_candidate.energy

    @property
    def improvement_over_recorded(self) -> float:
        return self.recorded_energy - self.planned_candidate.energy

    def first_action_decision(
        self,
        gate: FirstActionGate | None = None,
    ) -> FirstActionDecision:
        return evaluate_first_actions(
            [DroidAction(tuple(self.recorded_actions[0]))],
            [DroidAction(tuple(self.planned_candidate.actions[0]))],
            gate,
        ).decisions[0]

    @property
    def first_action_cosine(self) -> float:
        return self.first_action_decision().cosine

    def to_dict(self, gate: FirstActionGate | None = None) -> dict[str, Any]:
        first_action = self.first_action_decision(gate)
        library = (
            self.initial_candidate
            if self.initialization is PlannerInitialization.LIBRARY
            else None
        )
        proposal = (
            self.initial_candidate
            if self.initialization is PlannerInitialization.PROPOSAL
            else None
        )
        return {
            "context_index": self.context_index,
            "target_index": self.target_index,
            "recorded_actions": self.recorded_actions.tolist(),
            "library_action": library.actions.tolist() if library else None,
            "library_energy": library.energy if library else None,
            "library_objective": library.objective if library else None,
            "proposal_action": proposal.actions.tolist() if proposal else None,
            "proposal_energy": proposal.energy if proposal else None,
            "proposal_objective": proposal.objective if proposal else None,
            "planned_actions": self.planned_candidate.actions.tolist(),
            "recorded_energy": self.recorded_energy,
            "zero_energy": self.zero_energy,
            "planned_energy": self.planned_candidate.energy,
            "planned_objective": self.planned_candidate.objective,
            "planned_prior_penalty": self.planned_prior_penalty,
            "refinement_improvement": self.refinement_improvement,
            "improvement_over_zero": self.improvement_over_zero,
            "improvement_over_recorded": self.improvement_over_recorded,
            "first_action_cosine": first_action.cosine,
            "first_action_gate": first_action.to_dict(),
        }


def _mean(evaluations: Sequence[PlannerRolloutEvaluation], name: str) -> float:
    return float(np.mean([getattr(evaluation, name) for evaluation in evaluations]))


def _win_rate(evaluations: Sequence[PlannerRolloutEvaluation], name: str) -> float:
    return sum(getattr(evaluation, name) > 0.0 for evaluation in evaluations) / len(
        evaluations
    )


@dataclass(frozen=True)
class PlannerBenchmarkProvenance:
    model: str
    source_revision: str
    adapter: TrainingArtifactIdentity
    proposal: TrainingArtifactIdentity | None
    base_checkpoint: ArtifactIdentity
    recording: DomainRecording
    camera: str
    window: RolloutWindow
    selection_bounds: ActionSelectionBounds
    scoring_batch_size: int

    def __post_init__(self) -> None:
        if (
            not self.model
            or not self.source_revision
            or not self.camera
            or self.scoring_batch_size <= 0
        ):
            raise ValueError("planner benchmark provenance is incomplete")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "source_revision": self.source_revision,
            "adapter": str(self.adapter.path),
            "adapter_fingerprint": self.adapter.fingerprint,
            "proposal": str(self.proposal.path) if self.proposal else None,
            "proposal_fingerprint": (
                self.proposal.fingerprint if self.proposal else None
            ),
            "base_checkpoint": str(self.base_checkpoint.path),
            "base_checkpoint_fingerprint": self.base_checkpoint.fingerprint,
            "recording": str(self.recording.path),
            "recording_split": self.recording.split.value,
            "recording_seed": self.recording.seed,
            "camera": self.camera,
            "window": self.window.to_dict(),
            "selection_bounds": self.selection_bounds.to_dict(),
            "scoring_batch_size": self.scoring_batch_size,
        }


@dataclass(frozen=True)
class PlannerRunSummary:
    config: CEMConfig
    training_action_library: int
    prior_penalty_weight: float
    initialization: PlannerInitialization
    task_policy: PlannerTaskPolicy = PlannerTaskPolicy()

    def __post_init__(self) -> None:
        if self.training_action_library <= 0:
            raise ValueError("planner action library must be non-empty")
        if not isfinite(self.prior_penalty_weight) or self.prior_penalty_weight <= 0:
            raise ValueError("planner prior penalty must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "candidates_per_rollout": self.config.iterations * self.config.samples,
            "training_action_library": self.training_action_library,
            "prior_penalty_weight": self.prior_penalty_weight,
            "initialization": self.initialization.value,
            "task_policy": self.task_policy.to_dict(),
        }


@dataclass(frozen=True)
class PlannerTimings:
    load_seconds: float
    encoding_seconds: float
    planning_seconds: float
    peak_allocated_gib: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value >= 0.0
            for value in (
                self.load_seconds,
                self.encoding_seconds,
                self.planning_seconds,
                self.peak_allocated_gib,
            )
        ):
            raise ValueError("planner timings must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "load_seconds": round(self.load_seconds, 3),
            "encoding_seconds": round(self.encoding_seconds, 3),
            "planning_seconds": round(self.planning_seconds, 3),
            "peak_allocated_gib": round(self.peak_allocated_gib, 3),
        }


@dataclass(frozen=True)
class PlannerBenchmarkReport:
    provenance: PlannerBenchmarkProvenance
    planner: PlannerRunSummary
    bounds: PlannerActionBounds
    timings: PlannerTimings
    evaluations: tuple[PlannerRolloutEvaluation, ...]

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise ValueError("planner benchmark requires at least one evaluation")

    def to_dict(self) -> dict[str, Any]:
        first_action_gate = FirstActionGate(
            self.planner.task_policy.first_action_thresholds
        )
        summary = evaluate_first_actions(
            [
                DroidAction(tuple(evaluation.recorded_actions[0]))
                for evaluation in self.evaluations
            ],
            [
                DroidAction(tuple(evaluation.planned_candidate.actions[0]))
                for evaluation in self.evaluations
            ],
            first_action_gate,
        )
        return {
            "schema": REPORT_SCHEMA,
            "status": "benchmarked",
            **self.provenance.to_dict(),
            "rollouts": len(self.evaluations),
            "planner": self.planner.to_dict(),
            "bounds": self.bounds.to_dict(),
            "mean_refinement_improvement": _mean(
                self.evaluations, "refinement_improvement"
            ),
            "refinement_win_rate": _win_rate(
                self.evaluations, "refinement_improvement"
            ),
            "mean_improvement_over_zero": _mean(
                self.evaluations, "improvement_over_zero"
            ),
            "planned_action_win_rate_over_zero": _win_rate(
                self.evaluations, "improvement_over_zero"
            ),
            "mean_improvement_over_recorded": _mean(
                self.evaluations, "improvement_over_recorded"
            ),
            "planned_action_win_rate_over_recorded": _win_rate(
                self.evaluations, "improvement_over_recorded"
            ),
            "mean_first_action_cosine": summary.mean_cosine,
            **summary.to_dict(),
            **self.timings.to_dict(),
            "results": [
                evaluation.to_dict(first_action_gate)
                for evaluation in self.evaluations
            ],
        }

    def write(self, output: Path) -> dict[str, Any]:
        payload = self.to_dict()
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload["output_path"] = str(output)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        return payload
