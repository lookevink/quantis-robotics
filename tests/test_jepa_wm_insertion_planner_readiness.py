from pathlib import Path
import unittest

from jepa_wm.insertion_planner import INSERTION_PLANNER_PROFILE
from jepa_wm.insertion_planner_readiness import (
    InsertionPlannerSeedEvidence,
    _assert_same,
)
from jepa_wm.training_artifact import ArtifactIdentity


class InsertionPlannerReadinessTest(unittest.TestCase):
    @staticmethod
    def _evidence(**overrides) -> InsertionPlannerSeedEvidence:
        values = {
            "report": Path("/report.json"),
            "recording": "fresh-held-00",
            "seed": 22600,
            "base_checkpoint": ArtifactIdentity(Path("/base.pth"), "a" * 64),
            "training_action_library": 1308,
            "selection_rate": 1.0,
            "selected_goal_alignment_pass_rate": 1.0,
            "selected_first_action_pass_rate": 1.0,
            "mean_selected_improvement_over_zero": 1e-5,
            "selected_win_rate_over_zero": 0.75,
            "mean_selected_improvement_over_recorded": 1e-5,
            "selected_win_rate_over_recorded": 0.75,
        }
        values.update(overrides)
        return InsertionPlannerSeedEvidence(**values)

    def test_requires_every_context_and_both_energy_comparisons(self) -> None:
        evidence = self._evidence(
            selection_rate=0.875,
            selected_win_rate_over_recorded=0.5714285714285714,
        )

        self.assertFalse(evidence.passed)
        self.assertEqual(
            evidence.reasons,
            ("blocked_context", "recorded_action_energy"),
        )

    def test_accepts_only_a_complete_aligned_positive_seed(self) -> None:
        evidence = self._evidence()

        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.reasons, ())
        self.assertTrue(evidence.to_dict()["passed"])

    def test_reconstructed_evidence_rejects_derived_metric_tampering(self) -> None:
        expected = {"selection_rate": 1.0, "reasons": []}

        with self.assertRaisesRegex(ValueError, "reconstructed evidence"):
            _assert_same(
                {"selection_rate": 0.875, "reasons": []},
                expected,
            )

    def test_profile_owns_the_bounded_memory_policy(self) -> None:
        self.assertEqual(INSERTION_PLANNER_PROFILE.scoring_batch_size, 64)


if __name__ == "__main__":
    unittest.main()
