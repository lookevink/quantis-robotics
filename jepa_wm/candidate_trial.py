"""Strict realized comparison for one reset-only shadow-candidate trial."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jepa_wm.control_baselines import (
    ControlPolicy,
    PoseAxisDecision,
    RealizedBaselineReport,
    RealizedPolicyOutcome,
    ScriptedOutcomeTolerances,
)
from jepa_wm.control_rollout import ControlStepSummary
from sim.control_session import ControlSession
from sim.recording import validate_recording_id


CANDIDATE_TRIAL_REPORT_SCHEMA = "quantis.jepa_wm_candidate_trial.v1"


@dataclass(frozen=True)
class RealizedCandidateComparison:
    zero: RealizedPolicyOutcome
    direct: RealizedPolicyOutcome
    scripted: RealizedPolicyOutcome
    candidate: RealizedPolicyOutcome
    tolerances: ScriptedOutcomeTolerances = ScriptedOutcomeTolerances()

    @property
    def candidate_improves_over_zero(self) -> PoseAxisDecision:
        return self.candidate.improvement_over(self.zero)

    @property
    def candidate_improves_over_direct(self) -> PoseAxisDecision:
        return self.candidate.improvement_over(self.direct)

    @property
    def candidate_reaches_scripted_tolerance(self) -> PoseAxisDecision:
        return self.candidate.reaches_within(self.scripted, self.tolerances)

    @property
    def candidate_trial_gate_passed(self) -> bool:
        return (
            self.candidate_improves_over_zero.passed
            and self.candidate_improves_over_direct.passed
            and self.candidate_reaches_scripted_tolerance.passed
        )

    @property
    def production_authority_granted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcomes": {
                outcome.policy.value: outcome.to_dict()
                for outcome in (
                    self.zero,
                    self.direct,
                    self.scripted,
                    self.candidate,
                )
            },
            "candidate_improves_over_zero": (
                self.candidate_improves_over_zero.to_dict()
            ),
            "candidate_improves_over_direct": (
                self.candidate_improves_over_direct.to_dict()
            ),
            "candidate_reaches_scripted_tolerance": (
                self.candidate_reaches_scripted_tolerance.to_dict()
            ),
            "candidate_trial_gate_passed": self.candidate_trial_gate_passed,
            "production_authority_granted": self.production_authority_granted,
        }


@dataclass(frozen=True)
class CandidateTrialReport:
    experiment_id: str
    baseline_experiment_id: str
    candidate_session_id: str
    source_session_id: str
    comparison: RealizedCandidateComparison

    @classmethod
    def from_sessions(
        cls,
        data_root: Path,
        experiment_id: str,
        baseline_experiment_id: str,
        candidate_session_id: str,
        source_session_id: str,
    ) -> CandidateTrialReport:
        validate_recording_id(experiment_id)
        baseline = RealizedBaselineReport.load_persisted(
            data_root, baseline_experiment_id
        )
        first_step = baseline.first_step()
        if first_step.direct_session_id != source_session_id:
            raise ValueError("candidate source is not the direct baseline's first step")
        candidate_session = ControlSession.at(
            data_root / "control_sessions", candidate_session_id
        )
        binding = candidate_session.load_candidate_binding()
        if binding.source_session_id != source_session_id:
            raise ValueError("candidate binding does not match the requested source")
        candidate = ControlStepSummary.from_session(candidate_session)
        if candidate.observation.target_frame != first_step.target_frame:
            raise ValueError("candidate and baseline do not share one target")
        comparison = RealizedCandidateComparison(
            zero=first_step.zero,
            direct=first_step.direct,
            scripted=first_step.scripted,
            candidate=RealizedPolicyOutcome.from_step(
                ControlPolicy.EXPERIMENTAL_CANDIDATE,
                candidate,
                first_step.target_pose,
            ),
        )
        return cls(
            experiment_id,
            baseline_experiment_id,
            candidate_session_id,
            source_session_id,
            comparison,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_TRIAL_REPORT_SCHEMA,
            "experiment_id": self.experiment_id,
            "baseline_experiment_id": self.baseline_experiment_id,
            "candidate_session_id": self.candidate_session_id,
            "source_session_id": self.source_session_id,
            **self.comparison.to_dict(),
        }
