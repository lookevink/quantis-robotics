from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.grasp_proposal_readiness import (
    validate_grasp_evaluation_window,
    validate_grasp_training_selection,
    validate_grasp_training_window,
)
from jepa_wm.training_artifact import TrainingArtifactMetadata


class GraspProposalReadinessTest(unittest.TestCase):
    def test_rejects_an_evaluation_that_mixes_exploration_with_the_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "recording": "/tmp/recording",
                        "window": {
                            "start_index": 49,
                            "count": 50,
                            "stride": 1,
                        },
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "complete task window"):
                validate_grasp_evaluation_window(report)

    def test_requires_stationary_attached_hold_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "recording": "/tmp/recording",
                        "window": {
                            "start_index": 69,
                            "count": 30,
                            "stride": 1,
                        },
                        "selection_bounds": {
                            "minimum_action_norm": 1e-6,
                            "maximum_pose_action_norm": 0.1,
                            "maximum_gripper_action": 0.75,
                        },
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "stationary hold"):
                validate_grasp_evaluation_window(report)

    def test_requires_task_window_weighted_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal = Path(temp_dir) / "proposal.pth"
            proposal.with_suffix(".pth.json").write_text(
                json.dumps({"window": None})
            )

            with self.assertRaisesRegex(ValueError, "task window"):
                validate_grasp_training_window(proposal)

    def test_requires_exact_stationary_inclusive_training_contexts(self) -> None:
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid",
            "revision",
            "wrist",
            ("train-00",),
            500,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal = Path(temp_dir) / "proposal.pth"
            proposal.with_suffix(".pth.json").write_text(
                json.dumps(
                    {
                        "window": {
                            "start_index": 69,
                            "count": 30,
                            "stride": 1,
                        },
                        "selection_bounds": {
                            "minimum_action_norm": 0.0,
                            "maximum_pose_action_norm": 0.1,
                            "maximum_gripper_action": 0.75,
                        },
                        "rollouts": 29,
                        "recording_selections": [
                            {
                                "recording": "train-00",
                                "context_indices": list(range(69, 98)),
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "selection evidence"):
                validate_grasp_training_selection(proposal, metadata)


if __name__ == "__main__":
    unittest.main()
