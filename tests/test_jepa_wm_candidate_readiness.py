from __future__ import annotations

import unittest

from jepa_wm.candidate_readiness import (
    CandidateReadinessEvidence,
    CandidateReadinessSummary,
)
from jepa_wm.candidate_trial import (
    CandidateSeedProvenance,
    CandidateTrialReport,
    RealizedCandidateComparison,
)
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
            CandidateSeedProvenance(seed, (calibration_seed,)),
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
            CandidateSeedProvenance(
                evidence.seed,
                (11400,),
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
