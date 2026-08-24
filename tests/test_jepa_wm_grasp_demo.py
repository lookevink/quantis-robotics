from __future__ import annotations

import unittest
from pathlib import Path

from jepa_wm.grasp_demo import GraspDemoMetadata
from jepa_wm.grasp_task import ReachAndGraspDecision
from jepa_wm.replay_verification import ReplayVerification
from jepa_wm.training_artifact import ArtifactIdentity


class GraspDemoMetadataTest(unittest.TestCase):
    @staticmethod
    def _metadata() -> GraspDemoMetadata:
        return GraspDemoMetadata(
            readiness_id="grasp-readiness-v2",
            baseline_experiment_id="grasp-baseline-12401-v2",
            rollout_id="rollout-12401",
            seed=12401,
            proposal=ArtifactIdentity(Path("/tmp/proposal.pth"), "a" * 64),
            source_steps=8,
            task_outcome=ReachAndGraspDecision(1, 8, 0.0585, ()),
            replay=ReplayVerification(0.001, 0.0002, 0.0, False),
        )

    def test_round_trips_visualization_provenance_and_derived_claims(self) -> None:
        metadata = self._metadata()

        self.assertEqual(GraspDemoMetadata.from_dict(metadata.to_dict()), metadata)
        self.assertTrue(metadata.replay.tracking_passed)
        self.assertTrue(metadata.replay.safety_passed)

    def test_rejects_tampered_replay_claim(self) -> None:
        payload = self._metadata().to_dict()
        payload["replay_tracking_passed"] = False

        with self.assertRaisesRegex(ValueError, "incomplete"):
            GraspDemoMetadata.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
