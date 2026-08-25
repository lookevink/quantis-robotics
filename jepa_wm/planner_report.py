"""Typed aggregation and persistence for offline planner benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np

from jepa_wm.action import ActionSelectionBounds, DroidAction
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.planner import CEMConfig, PlannerActionBounds
from jepa_wm.planner_policy import (
    GoalAlignmentDecision,
    GoalActionAlignment,
    PlannerTaskPolicy,
    RefinementAcceptanceDecision,
)
from jepa_wm.planner_objective import CandidateObjective
from jepa_wm.planner_readiness import (
    FirstActionDecision,
    FirstActionGate,
    FirstActionSummary,
    evaluate_first_actions,
)
from jepa_wm.trajectory import RolloutWindow
from jepa_wm.training_artifact import ArtifactIdentity, TrainingArtifactIdentity


REPORT_SCHEMA = "quantis.jepa_wm_planner_benchmark.v3"


class PlannerInitialization(str, Enum):
    LIBRARY = "library"
    PROPOSAL = "proposal"


class PlannerSelectedSource(str, Enum):
    INITIAL = "initial"
    SEARCHED = "searched"


@dataclass(frozen=True)
class CandidateEvaluation:
    actions: np.ndarray
    scores: CandidateObjective

    def __post_init__(self) -> None:
        if (
            self.actions.ndim != 2
            or self.actions.shape[1] != 7
            or not np.all(np.isfinite(self.actions))
        ):
            raise ValueError("candidate evaluation must contain finite actions and scores")


@dataclass(frozen=True)
class SelectedRefinement:
    source: PlannerSelectedSource
    candidate: CandidateEvaluation
    searched_goal_alignment: GoalAlignmentDecision | None
    initial_goal_alignment: GoalAlignmentDecision | None
    acceptance: RefinementAcceptanceDecision | None
    selected_goal_alignment: GoalAlignmentDecision | None
    first_action: FirstActionDecision
    improvement_over_zero: float
    improvement_over_recorded: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value)
            for value in (self.improvement_over_zero, self.improvement_over_recorded)
        ):
            raise ValueError("selected candidate improvements must be finite")
        if self.source is PlannerSelectedSource.SEARCHED and (
            self.acceptance is not None and not self.acceptance.accepted
        ):
            raise ValueError("refinement acceptance and selected source disagree")
        if self.source is PlannerSelectedSource.INITIAL and (
            self.acceptance is None
            or self.acceptance.accepted
            or self.initial_goal_alignment is None
            or not self.initial_goal_alignment.passed
        ):
            raise ValueError("initial fallback requires rejected search and alignment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_status": "selected",
            "refinement_acceptance": (
                self.acceptance.to_dict() if self.acceptance is not None else None
            ),
            "initial_goal_action_alignment": (
                self.initial_goal_alignment.to_dict()
                if self.initial_goal_alignment is not None
                else None
            ),
            "selected_source": self.source.value,
            "selected_actions": self.candidate.actions.tolist(),
            "selected_energy": self.candidate.scores.latent_energy,
            "selected_objective": self.candidate.scores.total,
            "selected_prior_penalty": self.candidate.scores.prior_penalty,
            "selected_task_penalty": self.candidate.scores.task_penalty,
            "selected_first_action_cosine": self.first_action.cosine,
            "selected_first_action_gate": self.first_action.to_dict(),
            "selected_goal_action_alignment": (
                self.selected_goal_alignment.to_dict()
                if self.selected_goal_alignment is not None
                else None
            ),
            "selected_improvement_over_zero": self.improvement_over_zero,
            "selected_improvement_over_recorded": self.improvement_over_recorded,
        }


@dataclass(frozen=True)
class BlockedRefinement:
    searched_goal_alignment: GoalAlignmentDecision
    initial_goal_alignment: GoalAlignmentDecision
    acceptance: RefinementAcceptanceDecision

    def __post_init__(self) -> None:
        if (
            self.acceptance.accepted
            or self.initial_goal_alignment.passed
            or self.acceptance.reasons == ()
        ):
            raise ValueError("blocked refinement evidence is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_status": "blocked",
            "refinement_acceptance": self.acceptance.to_dict(),
            "initial_goal_action_alignment": self.initial_goal_alignment.to_dict(),
            "selected_source": None,
            "selected_actions": None,
            "selected_energy": None,
            "selected_objective": None,
            "selected_prior_penalty": None,
            "selected_task_penalty": None,
            "selected_first_action_cosine": None,
            "selected_first_action_gate": None,
            "selected_goal_action_alignment": None,
            "selected_improvement_over_zero": None,
            "selected_improvement_over_recorded": None,
        }


PlannerRefinementSelection = Union[SelectedRefinement, BlockedRefinement]


@dataclass(frozen=True)
class PlannerRolloutEvaluation:
    context_index: int
    target_index: int
    recorded_actions: np.ndarray
    recorded_energy: float
    zero_energy: float
    initialization: PlannerInitialization
    initial_candidate: CandidateEvaluation
    searched_candidate: CandidateEvaluation
    goal_action: DroidAction | None = None

    def __post_init__(self) -> None:
        if (
            self.recorded_actions.shape != self.searched_candidate.actions.shape
            or not np.all(np.isfinite(self.recorded_actions))
            or not isfinite(self.recorded_energy)
            or not isfinite(self.zero_energy)
        ):
            raise ValueError("recorded and planned action horizons must match")

    @property
    def searched_prior_penalty(self) -> float:
        return self.searched_candidate.scores.prior_penalty

    @property
    def search_total_improvement(self) -> float:
        return (
            self.initial_candidate.scores.total
            - self.searched_candidate.scores.total
        )

    @property
    def searched_improvement_over_zero(self) -> float:
        return self.zero_energy - self.searched_candidate.scores.latent_energy

    @property
    def searched_improvement_over_recorded(self) -> float:
        return self.recorded_energy - self.searched_candidate.scores.latent_energy

    def first_action_decision(
        self,
        gate: FirstActionGate | None = None,
        candidate: CandidateEvaluation | None = None,
    ) -> FirstActionDecision:
        evaluated = candidate or self.searched_candidate
        return evaluate_first_actions(
            [DroidAction(tuple(self.recorded_actions[0]))],
            [DroidAction(tuple(evaluated.actions[0]))],
            gate,
        ).decisions[0]

    @property
    def first_action_cosine(self) -> float:
        return self.first_action_decision().cosine

    def goal_alignment_decision(
        self,
        policy: GoalActionAlignment | None,
    ) -> GoalAlignmentDecision | None:
        if policy is None:
            return None
        if self.goal_action is None:
            raise ValueError("planner rollout is missing its goal action")
        return policy.evaluate(
            DroidAction(tuple(self.searched_candidate.actions[0])),
            self.goal_action,
        )

    def refinement_selection(
        self,
        task_policy: PlannerTaskPolicy,
    ) -> PlannerRefinementSelection:
        goal_decision = self.goal_alignment_decision(
            task_policy.goal_action_alignment
        )
        acceptance = None
        if task_policy.refinement_acceptance is not None:
            if goal_decision is None:
                raise ValueError(
                    "refinement acceptance requires goal alignment evidence"
                )
            acceptance = task_policy.refinement_acceptance.evaluate(
                self.initial_candidate.scores.latent_energy,
                self.searched_candidate.scores.latent_energy,
                goal_decision,
            )
        initial_goal_alignment = (
            task_policy.goal_action_alignment.evaluate(
                DroidAction(tuple(self.initial_candidate.actions[0])),
                self.goal_action,
            )
            if (
                task_policy.goal_action_alignment is not None
                and self.goal_action is not None
            )
            else None
        )
        if acceptance is None or acceptance.accepted:
            source = PlannerSelectedSource.SEARCHED
            candidate = self.searched_candidate
        elif initial_goal_alignment is None or (
            task_policy.refinement_acceptance.allows_initial(
                initial_goal_alignment
            )
        ):
            source = PlannerSelectedSource.INITIAL
            candidate = self.initial_candidate
        else:
            return BlockedRefinement(
                searched_goal_alignment=goal_decision,
                initial_goal_alignment=initial_goal_alignment,
                acceptance=acceptance,
            )
        selected_goal_alignment = (
            task_policy.goal_action_alignment.evaluate(
                DroidAction(tuple(candidate.actions[0])),
                self.goal_action,
            )
            if (
                task_policy.goal_action_alignment is not None
                and self.goal_action is not None
            )
            else None
        )
        return SelectedRefinement(
            source=source,
            candidate=candidate,
            searched_goal_alignment=goal_decision,
            initial_goal_alignment=initial_goal_alignment,
            acceptance=acceptance,
            selected_goal_alignment=selected_goal_alignment,
            first_action=self.first_action_decision(
                FirstActionGate(task_policy.first_action_thresholds),
                candidate,
            ),
            improvement_over_zero=(
                self.zero_energy - candidate.scores.latent_energy
            ),
            improvement_over_recorded=(
                self.recorded_energy - candidate.scores.latent_energy
            ),
        )

    def to_dict(
        self,
        task_policy: PlannerTaskPolicy = PlannerTaskPolicy(),
        selection: PlannerRefinementSelection | None = None,
    ) -> dict[str, Any]:
        gate = FirstActionGate(task_policy.first_action_thresholds)
        first_action = self.first_action_decision(gate)
        selected = selection or self.refinement_selection(task_policy)
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
            "goal_action": (
                list(self.goal_action.values) if self.goal_action is not None else None
            ),
            "recorded_actions": self.recorded_actions.tolist(),
            "library_action": library.actions.tolist() if library else None,
            "library_energy": library.scores.latent_energy if library else None,
            "library_objective": library.scores.total if library else None,
            "library_prior_penalty": (
                library.scores.prior_penalty if library else None
            ),
            "library_task_penalty": library.scores.task_penalty if library else None,
            "proposal_action": proposal.actions.tolist() if proposal else None,
            "proposal_energy": proposal.scores.latent_energy if proposal else None,
            "proposal_objective": proposal.scores.total if proposal else None,
            "proposal_prior_penalty": (
                proposal.scores.prior_penalty if proposal else None
            ),
            "proposal_task_penalty": proposal.scores.task_penalty if proposal else None,
            "searched_actions": self.searched_candidate.actions.tolist(),
            "recorded_energy": self.recorded_energy,
            "zero_energy": self.zero_energy,
            "searched_energy": self.searched_candidate.scores.latent_energy,
            "searched_objective": self.searched_candidate.scores.total,
            "searched_prior_penalty": self.searched_prior_penalty,
            "searched_task_penalty": self.searched_candidate.scores.task_penalty,
            "search_total_improvement": self.search_total_improvement,
            "searched_improvement_over_zero": self.searched_improvement_over_zero,
            "searched_improvement_over_recorded": (
                self.searched_improvement_over_recorded
            ),
            "searched_first_action_cosine": first_action.cosine,
            "searched_first_action_gate": first_action.to_dict(),
            "searched_goal_action_alignment": (
                selected.searched_goal_alignment.to_dict()
                if selected.searched_goal_alignment is not None
                else None
            ),
            **selected.to_dict(),
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
        contextual_candidates = self.task_policy.context_matched_candidates
        return {
            **self.config.to_dict(),
            "candidates_per_rollout": (
                self.config.iterations * self.config.samples
                + (
                    contextual_candidates.candidates_per_context
                    if contextual_candidates is not None
                    else 0
                )
            ),
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
        task_policy = self.planner.task_policy
        first_action_gate = FirstActionGate(task_policy.first_action_thresholds)
        searched_summary = evaluate_first_actions(
            [
                DroidAction(tuple(evaluation.recorded_actions[0]))
                for evaluation in self.evaluations
            ],
            [
                DroidAction(tuple(evaluation.searched_candidate.actions[0]))
                for evaluation in self.evaluations
            ],
            first_action_gate,
        )
        selections = tuple(
            evaluation.refinement_selection(task_policy)
            for evaluation in self.evaluations
        )
        selected_selections = tuple(
            selection
            for selection in selections
            if isinstance(selection, SelectedRefinement)
        )
        selected_summary = (
            FirstActionSummary(
                tuple(
                    selection.first_action
                    for selection in selected_selections
                    if selection.first_action is not None
                )
            )
            if selected_selections
            else None
        )
        goal_alignment_decisions = tuple(
            selection.searched_goal_alignment
            for selection in selections
            if selection.searched_goal_alignment is not None
        )
        selected_goal_alignment_decisions = tuple(
            selection.selected_goal_alignment
            for selection in selected_selections
            if selection.selected_goal_alignment is not None
        )
        acceptance_decisions = tuple(
            selection.acceptance
            for selection in selections
            if selection.acceptance is not None
        )
        selected_improvement_over_zero = tuple(
            selection.improvement_over_zero
            for selection in selected_selections
        )
        selected_improvement_over_recorded = tuple(
            selection.improvement_over_recorded
            for selection in selected_selections
        )
        return {
            "schema": REPORT_SCHEMA,
            "status": "benchmarked",
            **self.provenance.to_dict(),
            "rollouts": len(self.evaluations),
            "planner": self.planner.to_dict(),
            "bounds": self.bounds.to_dict(),
            "mean_search_total_improvement": _mean(
                self.evaluations, "search_total_improvement"
            ),
            "search_total_improvement_win_rate": _win_rate(
                self.evaluations, "search_total_improvement"
            ),
            "mean_searched_improvement_over_zero": _mean(
                self.evaluations, "searched_improvement_over_zero"
            ),
            "searched_action_win_rate_over_zero": _win_rate(
                self.evaluations, "searched_improvement_over_zero"
            ),
            "mean_searched_improvement_over_recorded": _mean(
                self.evaluations, "searched_improvement_over_recorded"
            ),
            "searched_action_win_rate_over_recorded": _win_rate(
                self.evaluations, "searched_improvement_over_recorded"
            ),
            "mean_searched_first_action_cosine": searched_summary.mean_cosine,
            "searched_first_action": searched_summary.to_dict(),
            "searched_goal_action_alignment_pass_rate": (
                sum(decision.passed for decision in goal_alignment_decisions)
                / len(goal_alignment_decisions)
                if goal_alignment_decisions
                else None
            ),
            "refinement_acceptance_rate": (
                sum(decision.accepted for decision in acceptance_decisions)
                / len(acceptance_decisions)
                if acceptance_decisions
                else None
            ),
            "selection_rate": len(selected_selections) / len(selections),
            "blocked_rate": (
                len(selections) - len(selected_selections)
            )
            / len(selections),
            "mean_selected_improvement_over_zero": (
                float(np.mean(selected_improvement_over_zero))
                if selected_improvement_over_zero
                else None
            ),
            "selected_action_win_rate_over_zero": (
                sum(value > 0.0 for value in selected_improvement_over_zero)
                / len(selected_improvement_over_zero)
                if selected_improvement_over_zero
                else None
            ),
            "mean_selected_improvement_over_recorded": (
                float(np.mean(selected_improvement_over_recorded))
                if selected_improvement_over_recorded
                else None
            ),
            "selected_action_win_rate_over_recorded": (
                sum(value > 0.0 for value in selected_improvement_over_recorded)
                / len(selected_improvement_over_recorded)
                if selected_improvement_over_recorded
                else None
            ),
            "mean_selected_first_action_cosine": (
                selected_summary.mean_cosine if selected_summary is not None else None
            ),
            "selected_first_action": (
                selected_summary.to_dict() if selected_summary is not None else None
            ),
            "selected_goal_action_alignment_pass_rate": (
                sum(
                    decision.passed
                    for decision in selected_goal_alignment_decisions
                )
                / len(selected_goal_alignment_decisions)
                if selected_goal_alignment_decisions
                else None
            ),
            **self.timings.to_dict(),
            "results": [
                evaluation.to_dict(task_policy, selection)
                for evaluation, selection in zip(self.evaluations, selections)
            ],
        }

    def write(self, output: Path) -> dict[str, Any]:
        payload = self.to_dict()
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload["output_path"] = str(output)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        return payload
