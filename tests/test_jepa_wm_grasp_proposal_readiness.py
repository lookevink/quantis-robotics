from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from jepa_wm.action import DroidAction
from jepa_wm.grasp_proposal_readiness import (
    validate_grasp_evaluation_window,
    validate_grasp_goal_deltas,
    validate_grasp_proposal_identity,
    validate_grasp_training_selection,
    validate_grasp_training_window,
)
from jepa_wm.training_artifact import (
    TrainingArtifactMetadata,
    artifact_fingerprint,
)


class GraspProposalReadinessTest(unittest.TestCase):
    def test_binds_conditioning_to_the_checkpoint_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proposal = Path(temp_dir) / "proposal.pth"
            proposal.write_bytes(b"goal-conditioned-checkpoint")
            proposal.with_suffix(".pth.json").write_text(
                json.dumps(
                    {
                        "proposal_fingerprint": artifact_fingerprint(proposal),
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": True,
                            "task_progress": True,
                        },
                    }
                )
            )

            self.assertEqual(
                validate_grasp_proposal_identity(proposal).fingerprint,
                artifact_fingerprint(proposal),
            )
            proposal.write_bytes(b"replaced-checkpoint")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_grasp_proposal_identity(proposal)

    def test_rejects_tampered_held_out_goal_deltas(self) -> None:
        rollout = SimpleNamespace(
            context=(SimpleNamespace(index=69),),
            target=SimpleNamespace(index=72),
            goal_action=DroidAction((0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        )
        results = [
            {
                "context_index": 69 + offset,
                "target_index": 72 + offset,
                "goal_delta": list(rollout.goal_action.values),
            }
            for offset in range(30)
        ]
        with patch(
            "jepa_wm.grasp_proposal_readiness.load_rollout_at",
            side_effect=lambda *args, context_index, **kwargs: SimpleNamespace(
                context=(SimpleNamespace(index=context_index),),
                target=SimpleNamespace(index=context_index + 3),
                goal_action=rollout.goal_action,
            ),
        ):
            validate_grasp_goal_deltas(Path("/tmp/held-out"), results)
            results[12]["goal_delta"][0] = -0.03
            with self.assertRaisesRegex(ValueError, "does not match telemetry"):
                validate_grasp_goal_deltas(Path("/tmp/held-out"), results)

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
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": True,
                            "task_progress": True,
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

    def test_requires_goal_delta_conditioning(self) -> None:
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
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": False,
                            "task_progress": True,
                        },
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "goal-delta"):
                validate_grasp_training_selection(proposal, metadata)

    def test_requires_task_progress_conditioning(self) -> None:
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 500
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
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": True,
                            "task_progress": False,
                        },
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "task-progress"):
                validate_grasp_training_selection(proposal, metadata)


if __name__ == "__main__":
    unittest.main()
