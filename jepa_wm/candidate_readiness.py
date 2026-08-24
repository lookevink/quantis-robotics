"""Cross-seed readiness evidence for realized shadow-candidate trials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.candidate_trial import (
    CandidateReadinessProvenance,
    CandidateTrialReport,
    RealizedCandidateComparison,
)
from sim.recording import validate_recording_id


CANDIDATE_READINESS_SCHEMA = "quantis.jepa_wm_candidate_readiness.v1"
MINIMUM_WHOLE_SEEDS = 2


@dataclass(frozen=True)
class CandidateReadinessEvidence:
    report: CandidateTrialReport
    provenance: CandidateReadinessProvenance

    def __post_init__(self) -> None:
        validate_recording_id(self.report.experiment_id)

    @classmethod
    def from_persisted(
        cls,
        data_root: Path,
        experiment_id: str,
    ) -> CandidateReadinessEvidence:
        report = CandidateTrialReport.load_persisted(data_root, experiment_id)
        return cls(report, report.readiness_provenance(data_root))

    @property
    def experiment_id(self) -> str:
        return self.report.experiment_id

    @property
    def seed(self) -> int:
        return self.provenance.seed

    @property
    def comparison(self) -> RealizedCandidateComparison:
        return self.report.comparison

    @property
    def strict_gate_passed(self) -> bool:
        return self.comparison.candidate_trial_gate_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.report.experiment_id,
            "candidate_session_id": self.report.candidate_session_id,
            "source_session_id": self.report.source_session_id,
            **self.provenance.to_dict(),
            "strict_gate_passed": self.strict_gate_passed,
            **self.comparison.to_dict(),
        }


@dataclass(frozen=True)
class CandidateReadinessSummary:
    evidence: tuple[CandidateReadinessEvidence, ...]

    def __post_init__(self) -> None:
        if len({item.experiment_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("candidate readiness experiments must be unique")
        if len({item.seed for item in self.evidence}) != len(self.evidence):
            raise ValueError("candidate readiness requires unique evaluation seeds")
        evaluation_seeds = {item.seed for item in self.evidence}
        calibration_seeds = {
            seed
            for item in self.evidence
            for seed in item.provenance.calibration_seeds
        }
        if evaluation_seeds & calibration_seeds:
            raise ValueError(
                "candidate readiness requires globally disjoint calibration "
                "and evaluation seeds"
            )
        if len({item.provenance.worker for item in self.evidence}) != 1:
            raise ValueError(
                "candidate readiness requires one identical worker policy"
            )

    @classmethod
    def from_evidence(
        cls,
        evidence: Sequence[CandidateReadinessEvidence],
    ) -> CandidateReadinessSummary:
        return cls(tuple(evidence))

    @property
    def whole_seed_count(self) -> int:
        return len(self.evidence)

    @property
    def strict_pass_count(self) -> int:
        return sum(item.strict_gate_passed for item in self.evidence)

    @property
    def candidate_readiness_passed(self) -> bool:
        return (
            self.whole_seed_count >= MINIMUM_WHOLE_SEEDS
            and self.strict_pass_count == self.whole_seed_count
        )

    @property
    def production_authority_granted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_READINESS_SCHEMA,
            "minimum_whole_seeds": MINIMUM_WHOLE_SEEDS,
            "whole_seed_count": self.whole_seed_count,
            "strict_pass_count": self.strict_pass_count,
            "candidate_readiness_passed": self.candidate_readiness_passed,
            "production_authority_granted": self.production_authority_granted,
            "trials": [item.to_dict() for item in self.evidence],
        }
