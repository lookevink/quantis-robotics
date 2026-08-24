from __future__ import annotations

import unittest

from jepa_wm.action import DroidAction, DroidActionScale
from jepa_wm.candidate_demo import CandidateDemoMetadata
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.planner import CEMConfig
from jepa_wm.replay_verification import ReplayVerification


class CandidateDemoMetadataTest(unittest.TestCase):
    def test_round_trips_the_complete_replay_evidence(self) -> None:
        metadata = CandidateDemoMetadata(
            report_id="candidate-proof-11401",
            candidate_session="candidate-11401",
            source_session="source-11401",
            seed=11401,
            policy=ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
            selected_action_scale=DroidActionScale(1.0, 0.25, 0.25),
            candidates_scored=640,
            planner=CEMConfig(iterations=5, samples=128, elites=12, seed=237),
            energy_improvement=0.00014,
            actual_action=DroidAction((0.002, 0.0, 0.0, 0.001, 0.0, 0.0, 0.03)),
            replay=ReplayVerification(0.001, 0.0002, 0.0, False),
        )

        restored = CandidateDemoMetadata.from_dict(metadata.to_dict())

        self.assertEqual(restored, metadata)
        self.assertEqual(restored.action_scale_label, "PROJECTED")

    def test_rejects_a_budget_that_does_not_match_candidates_scored(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata is invalid"):
            CandidateDemoMetadata(
                report_id="candidate-proof-11401",
                candidate_session="candidate-11401",
                source_session="source-11401",
                seed=11401,
                policy=ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
                selected_action_scale=DroidActionScale.uniform(1.0),
                candidates_scored=639,
                planner=CEMConfig(iterations=5, samples=128, elites=12, seed=237),
                energy_improvement=0.00014,
                actual_action=DroidAction((0.0,) * 7),
                replay=ReplayVerification(0.001, 0.0002, 0.0, False),
            )


if __name__ == "__main__":
    unittest.main()
