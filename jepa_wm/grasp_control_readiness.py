"""Two-seed live readiness gate for the reach-and-grasp task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.control_baselines import RealizedBaselineReport
from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.grasp_task import ReachAndGraspDecision
from jepa_wm.training_artifact import ArtifactIdentity, TrainingArtifactIdentity
from jepa_wm.whole_seed_readiness import WholeSeedReadiness


GRASP_CONTROL_READINESS_SCHEMA = "quantis.jepa_wm_grasp_control_readiness.v1"


@dataclass(frozen=True)
class GraspControlReadinessEvidence:
    report: RealizedBaselineReport
    proposal: ArtifactIdentity

    def __post_init__(self) -> None:
        if any(decision is None for decision in self.task_outcomes):
            raise ValueError("baseline trial lacks reach-and-grasp telemetry")
        if any(
            rollout.reference_task != GRASP_TASK_ID
            for rollout in (self.report.direct, self.report.zero, self.report.scripted)
        ):
            raise ValueError("baseline trial is not the canonical grasp task")
        fingerprints = {
            step.response.proposal_fingerprint
            for step in self.report.direct.complete_steps
        }
        if fingerprints != {self.proposal.fingerprint}:
            raise ValueError(
                "direct rollout is not bound to the proposal fingerprint"
            )

    @classmethod
    def from_persisted(
        cls,
        data_root: Path,
        baseline_experiment_id: str,
    ) -> GraspControlReadinessEvidence:
        report = RealizedBaselineReport.load_persisted(
            data_root,
            baseline_experiment_id,
        )
        proposal = TrainingArtifactIdentity.from_artifact(
            report.direct.proposal,
            fingerprint_field="proposal_fingerprint",
        )
        return cls(report, ArtifactIdentity(proposal.path, proposal.fingerprint))

    @property
    def task_outcomes(
        self,
    ) -> tuple[
        ReachAndGraspDecision | None,
        ReachAndGraspDecision | None,
        ReachAndGraspDecision | None,
    ]:
        return (
            self.report.direct.reach_and_grasp,
            self.report.zero.reach_and_grasp,
            self.report.scripted.reach_and_grasp,
        )

    @property
    def seed(self) -> int:
        return self.report.seed

    @property
    def direct(self) -> ReachAndGraspDecision:
        decision = self.report.direct.reach_and_grasp
        assert decision is not None
        return decision

    @property
    def zero(self) -> ReachAndGraspDecision:
        decision = self.report.zero.reach_and_grasp
        assert decision is not None
        return decision

    @property
    def scripted(self) -> ReachAndGraspDecision:
        decision = self.report.scripted.reach_and_grasp
        assert decision is not None
        return decision

    @property
    def task_comparison_passed(self) -> bool:
        return self.direct.passed and not self.zero.passed and self.scripted.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_experiment_id": self.report.experiment_id,
            "reference_recording": self.report.reference_recording,
            "seed": self.seed,
            "proposal": self.proposal.to_dict(),
            "rollouts": {
                "direct": self.report.direct.rollout_id,
                "zero": self.report.zero.rollout_id,
                "scripted": self.report.scripted.rollout_id,
            },
            "task_outcomes": {
                "direct": self.direct.to_dict(),
                "zero": self.zero.to_dict(),
                "scripted": self.scripted.to_dict(),
            },
            "task_comparison_passed": self.task_comparison_passed,
            "generic_pose_baseline_gate_passed": (
                self.report.comparison.direct_baseline_gate_passed
            ),
        }


@dataclass(frozen=True)
class GraspControlReadinessSummary:
    evidence: tuple[GraspControlReadinessEvidence, ...]

    def __post_init__(self) -> None:
        if len({item.seed for item in self.evidence}) != len(self.evidence):
            raise ValueError("grasp readiness requires unique evaluation seeds")
        if len({item.report.experiment_id for item in self.evidence}) != len(
            self.evidence
        ):
            raise ValueError("grasp readiness baseline experiments must be unique")
        if len({item.report.reference_recording for item in self.evidence}) != len(
            self.evidence
        ):
            raise ValueError("grasp readiness references must be unique")
        if len({item.proposal for item in self.evidence}) > 1:
            raise ValueError("grasp readiness requires one identical proposal artifact")

    @classmethod
    def from_evidence(
        cls,
        evidence: Sequence[GraspControlReadinessEvidence],
    ) -> GraspControlReadinessSummary:
        return cls(tuple(evidence))

    @classmethod
    def from_persisted(
        cls,
        data_root: Path,
        baseline_experiment_ids: Sequence[str],
    ) -> GraspControlReadinessSummary:
        return cls.from_evidence(
            tuple(
                GraspControlReadinessEvidence.from_persisted(
                    data_root,
                    experiment_id,
                )
                for experiment_id in baseline_experiment_ids
            )
        )

    @property
    def whole_seed_count(self) -> int:
        return self.readiness.whole_seed_count

    @property
    def task_pass_count(self) -> int:
        return self.readiness.pass_count

    @property
    def readiness(self) -> WholeSeedReadiness:
        return WholeSeedReadiness.from_passes(
            item.task_comparison_passed for item in self.evidence
        )

    @property
    def filming_readiness_passed(self) -> bool:
        return self.readiness.passed

    @property
    def production_authority_granted(self) -> bool:
        return self.readiness.production_authority_granted

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRASP_CONTROL_READINESS_SCHEMA,
            "minimum_whole_seeds": self.readiness.minimum_whole_seeds,
            "whole_seed_count": self.whole_seed_count,
            "task_pass_count": self.task_pass_count,
            "filming_readiness_passed": self.filming_readiness_passed,
            "production_authority_granted": self.production_authority_granted,
            "trials": [item.to_dict() for item in self.evidence],
        }
