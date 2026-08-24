from __future__ import annotations

import unittest
from pathlib import Path

from jepa_wm.candidate_readiness import (
    CandidateReadinessEvidence,
    CandidateReadinessSummary,
)
from jepa_wm.candidate_trial import (
    CandidateReadinessProvenance,
    CandidateWorkerIdentity,
    CandidateTrialReport,
    RealizedCandidateComparison,
)
from jepa_wm.objective_calibration import TaskProgressMargins
from jepa_wm.planner import CEMConfig
from jepa_wm.shadow_planning import ShadowSearchConfig
from jepa_wm.control_baselines import (
    ControlPolicy,
    PoseError,
    RealizedPolicyOutcome,
)


class CandidateReadinessSummaryTest(unittest.TestCase):
    @staticmethod
    def _evidence(
        experiment_id: str,
        seed: int,
        *,
        strict_pass: bool,
        calibration_seed: int = 11300,
        planner_seed: int = 234,
    ) -> CandidateReadinessEvidence:
        initial = PoseError(0.03, 0.03, 0.3)

        def outcome(
            policy: ControlPolicy,
            translation: float,
            rotation: float,
            gripper: float,
        ) -> RealizedPolicyOutcome:
            return RealizedPolicyOutcome(
                policy,
                initial,
                PoseError(translation, rotation, gripper),
            )

        comparison = RealizedCandidateComparison(
            zero=outcome(ControlPolicy.ZERO, 0.029, 0.029, 0.29),
            direct=outcome(ControlPolicy.DIRECT, 0.028, 0.028, 0.20),
            scripted=outcome(ControlPolicy.SCRIPTED, 0.005, 0.005, 0.10),
            candidate=outcome(
                ControlPolicy.EXPERIMENTAL_CANDIDATE,
                0.004 if strict_pass else 0.006,
                0.004,
                0.09,
            ),
        )
        return CandidateReadinessEvidence(
            CandidateTrialReport(
                experiment_id,
                f"baseline-{seed}",
                f"candidate-{seed}",
                f"source-{seed}",
                comparison,
            ),
            CandidateReadinessProvenance(
                seed,
                (calibration_seed,),
                CandidateWorkerIdentity(
                    Path("/tmp/proposal.pth"),
                    Path("/tmp/adapter.pth"),
                    "a" * 64,
                    TaskProgressMargins(0.0005, 0.001, 0.005),
                    ShadowSearchConfig(
                        planner=CEMConfig(
                            iterations=5,
                            samples=128,
                            elites=12,
                            seed=planner_seed,
                        )
                    ),
                ),
            ),
        )

    def test_requires_every_whole_seed_trial_to_pass(self) -> None:
        summary = CandidateReadinessSummary.from_evidence(
            (
                self._evidence("candidate-a", 11400, strict_pass=True),
                self._evidence("candidate-b", 11401, strict_pass=False),
            )
        )

        self.assertEqual(summary.whole_seed_count, 2)
        self.assertEqual(summary.strict_pass_count, 1)
        self.assertFalse(summary.candidate_readiness_passed)
        self.assertFalse(summary.production_authority_granted)

    def test_rejects_duplicate_evaluation_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique evaluation seeds"):
            CandidateReadinessSummary.from_evidence(
                (
                    self._evidence("candidate-a", 11400, strict_pass=True),
                    self._evidence("candidate-b", 11400, strict_pass=True),
                )
            )

    def test_readiness_does_not_grant_production_authority(self) -> None:
        summary = CandidateReadinessSummary.from_evidence(
            (
                self._evidence("candidate-a", 11400, strict_pass=True),
                self._evidence("candidate-b", 11401, strict_pass=True),
            )
        )

        self.assertTrue(summary.candidate_readiness_passed)
        self.assertFalse(summary.production_authority_granted)
        self.assertFalse(summary.to_dict()["production_authority_granted"])

    def test_rejects_calibration_leakage_into_evaluation_seed(self) -> None:
        evidence = self._evidence("candidate-a", 11400, strict_pass=True)

        with self.assertRaisesRegex(ValueError, "calibration seed leakage"):
            CandidateReadinessProvenance(
                evidence.seed,
                (11400,),
                evidence.provenance.worker,
            )

    def test_rejects_per_seed_planner_tuning(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical worker policy"):
            CandidateReadinessSummary.from_evidence(
                (
                    self._evidence(
                        "candidate-a", 11400, strict_pass=True, planner_seed=235
                    ),
                    self._evidence(
                        "candidate-b", 11401, strict_pass=True, planner_seed=234
                    ),
                )
            )

    def test_rejects_reciprocal_cross_validation_as_held_out_readiness(self) -> None:
        with self.assertRaisesRegex(ValueError, "globally disjoint"):
            CandidateReadinessSummary.from_evidence(
                (
                    self._evidence(
                        "candidate-a",
                        11400,
                        strict_pass=True,
                        calibration_seed=11401,
                    ),
                    self._evidence(
                        "candidate-b",
                        11401,
                        strict_pass=True,
                        calibration_seed=11400,
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
