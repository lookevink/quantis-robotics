from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # The lightweight local test venv omits CUDA/PyTorch.
    torch = None

if torch is not None:
    from jepa_wm.proposal import (
        ActionProposalNetwork,
        ProposalConditioning,
        ProposalInputs,
        load_action_proposal,
        save_action_proposal,
    )
    from jepa_wm.proprioception import DroidValueNormalization, ScalarNormalization
    from jepa_wm.training_artifact import TrainingArtifactMetadata


def _normalization() -> DroidValueNormalization:
    return DroidValueNormalization(
        np.zeros(7, dtype=np.float32),
        np.ones(7, dtype=np.float32),
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class ActionProposalTest(unittest.TestCase):
    def test_task_progress_conditioning_changes_the_prediction(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=1,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(
                task_progress=ScalarNormalization(80.0, 10.0)
            ),
        )
        with torch.no_grad():
            proposal.network[0].weight.zero_()
            proposal.network[0].bias.zero_()
            proposal.network[0].weight[0, -1] = 1.0
            proposal.network[2].weight.fill_(1.0)
            proposal.network[2].bias.zero_()
        context = torch.zeros((1, 1, 2))
        target = torch.zeros((1, 1, 2))

        with self.assertRaisesRegex(ValueError, "task-conditioned proposal"):
            proposal(context, target)
        early = proposal(
            context,
            target,
            ProposalInputs(task_progress=torch.tensor(((80.0,),))),
        )
        late = proposal(
            context,
            target,
            ProposalInputs(task_progress=torch.tensor(((90.0,),))),
        )

        self.assertTrue(proposal.uses_task_progress)
        self.assertFalse(torch.equal(early, late))

    def test_goal_conditioning_is_required_and_changes_the_prediction(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=1,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(
                pose=_normalization(),
                previous_action=_normalization(),
                goal_delta=_normalization(),
            ),
        )
        first = proposal.network[0]
        second = proposal.network[2]
        with torch.no_grad():
            first.weight.zero_()
            first.bias.zero_()
            first.weight[0, -7] = 1.0
            second.weight.fill_(1.0)
            second.bias.zero_()
        context = torch.zeros((1, 1, 2))
        target = torch.zeros((1, 1, 2))
        pose = torch.zeros((1, 7))
        previous_action = torch.zeros((1, 7))

        with self.assertRaisesRegex(ValueError, "goal-conditioned proposal"):
            proposal(context, target, ProposalInputs(pose, previous_action))

        zero_goal = proposal(
            context,
            target,
            ProposalInputs(pose, previous_action, torch.zeros((1, 7))),
        )
        translated_goal = proposal(
            context,
            target,
            ProposalInputs(
                pose,
                previous_action,
                torch.tensor(((1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),)),
            ),
        )

        self.assertTrue(proposal.uses_goal_delta)
        self.assertFalse(torch.equal(zero_goal, translated_goal))

    def test_goal_conditioning_round_trips_through_the_checkpoint(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=4,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(
                pose=_normalization(),
                previous_action=_normalization(),
                goal_delta=_normalization(),
                task_progress=ScalarNormalization(80.0, 10.0),
            ),
        )
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 10
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "proposal.pth"
            save_action_proposal(proposal, checkpoint, metadata)

            loaded, loaded_metadata = load_action_proposal(
                checkpoint, device=torch.device("cpu")
            )

        self.assertTrue(loaded.uses_goal_delta)
        self.assertTrue(loaded.uses_task_progress)
        self.assertEqual(loaded_metadata, metadata)
        self.assertTrue(
            torch.equal(loaded.goal_delta_mean, proposal.goal_delta_mean)
        )

    def test_conditioning_residual_round_trips_through_the_checkpoint(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=4,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(
                goal_delta=_normalization(),
                task_progress=ScalarNormalization(80.0, 10.0),
            ),
            conditioning_residual=True,
        )
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 10
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "residual.pth"
            save_action_proposal(proposal, checkpoint, metadata)
            loaded, _ = load_action_proposal(
                checkpoint, device=torch.device("cpu")
            )

        self.assertTrue(loaded.conditioning_residual)
        self.assertIsNotNone(loaded.conditioning_network)

    def test_conditioned_gripper_head_is_independent_of_visual_features(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=1,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(
                task_progress=ScalarNormalization(80.0, 10.0)
            ),
            conditioned_gripper_head=True,
        )
        with torch.no_grad():
            proposal.network[0].weight.fill_(1.0)
            proposal.network[0].bias.zero_()
            proposal.network[2].weight.fill_(1.0)
            proposal.network[2].bias.zero_()
            proposal.gripper_network[0].weight.zero_()
            proposal.gripper_network[0].bias.zero_()
            proposal.gripper_network[2].weight.zero_()
            proposal.gripper_network[2].bias.fill_(0.25)
        inputs = ProposalInputs(task_progress=torch.tensor(((80.0,),)))
        first = proposal(
            torch.zeros((1, 1, 2)),
            torch.zeros((1, 1, 2)),
            inputs,
        )
        second = proposal(
            torch.ones((1, 1, 2)),
            torch.zeros((1, 1, 2)),
            inputs,
        )

        self.assertFalse(torch.equal(first[:, :, :6], second[:, :, :6]))
        self.assertTrue(torch.equal(first[:, :, 6], second[:, :, 6]))
        self.assertTrue(torch.all(first[:, :, 6] == 0.25))
        self.assertEqual(proposal.network[2].out_features, 18)

    def test_v2_conditioned_gripper_checkpoint_migrates_dead_output_rows(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=4,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(
                goal_delta=_normalization(),
                task_progress=ScalarNormalization(80.0, 10.0),
            ),
            conditioning_residual=True,
            conditioned_gripper_head=True,
        )
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 10
        )
        inputs = ProposalInputs(
            goal_delta=torch.zeros((1, 7)),
            task_progress=torch.tensor(((83.0,),)),
        )
        context = torch.zeros((1, 1, 2))
        target = torch.ones((1, 1, 2))
        expected = proposal(context, target, inputs)

        def expanded_rows(values: torch.Tensor) -> torch.Tensor:
            shape = (21, *values.shape[1:])
            expanded = torch.full(shape, 99.0, dtype=values.dtype)
            keep = [step * 7 + axis for step in range(3) for axis in range(6)]
            expanded[keep] = values
            return expanded

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "v2-gripper-head.pth"
            save_action_proposal(proposal, checkpoint, metadata)
            payload = torch.load(checkpoint, weights_only=True)
            payload["schema"] = "quantis.jepa_wm_action_proposal.v2"
            for prefix in ("network.2", "conditioning_network.2"):
                payload["state_dict"][f"{prefix}.weight"] = expanded_rows(
                    payload["state_dict"][f"{prefix}.weight"]
                )
                payload["state_dict"][f"{prefix}.bias"] = expanded_rows(
                    payload["state_dict"][f"{prefix}.bias"]
                )
            torch.save(payload, checkpoint)
            loaded, _ = load_action_proposal(checkpoint, device=torch.device("cpu"))

        self.assertTrue(torch.allclose(loaded(context, target, inputs), expected))

    def test_legacy_pose_history_checkpoint_is_strict_except_for_goal_buffers(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=4,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(
                pose=_normalization(),
                previous_action=_normalization(),
            ),
        )
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 10
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "legacy.pth"
            save_action_proposal(proposal, checkpoint, metadata)
            payload = torch.load(checkpoint, weights_only=True)
            payload["schema"] = "quantis.jepa_wm_action_proposal.v1"
            payload.pop("conditioning")
            payload["uses_proprioception"] = True
            payload["uses_action_history"] = True
            payload["state_dict"].pop("goal_delta_mean")
            payload["state_dict"].pop("goal_delta_standard_deviation")
            payload["state_dict"].pop("task_progress_mean")
            payload["state_dict"].pop("task_progress_standard_deviation")
            torch.save(payload, checkpoint)

            loaded, _ = load_action_proposal(
                checkpoint, device=torch.device("cpu")
            )
            self.assertFalse(loaded.uses_goal_delta)

            payload["state_dict"].pop("network.0.weight")
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(RuntimeError, "network.0.weight"):
                load_action_proposal(checkpoint, device=torch.device("cpu"))

    def test_legacy_checkpoint_rejects_partial_goal_delta_state(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=4,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(pose=_normalization()),
        )
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 10
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "partial-legacy.pth"
            save_action_proposal(proposal, checkpoint, metadata)
            payload = torch.load(checkpoint, weights_only=True)
            payload["schema"] = "quantis.jepa_wm_action_proposal.v1"
            payload.pop("conditioning")
            payload["uses_proprioception"] = True
            payload["state_dict"].pop("goal_delta_standard_deviation")
            payload["state_dict"].pop("task_progress_mean")
            payload["state_dict"].pop("task_progress_standard_deviation")
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(ValueError, "goal-delta state is incomplete"):
                load_action_proposal(checkpoint, device=torch.device("cpu"))

    def test_current_checkpoint_rejects_missing_task_progress_state(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=4,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(pose=_normalization()),
        )
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 10
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "corrupt-task-progress.pth"
            save_action_proposal(proposal, checkpoint, metadata)
            payload = torch.load(checkpoint, weights_only=True)
            payload["state_dict"].pop("task_progress_mean")
            payload["state_dict"].pop("task_progress_standard_deviation")
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(ValueError, "task-progress state is incomplete"):
                load_action_proposal(checkpoint, device=torch.device("cpu"))

    def test_current_checkpoint_rejects_missing_goal_delta_state(self) -> None:
        proposal = ActionProposalNetwork(
            feature_dimension=2,
            horizon=3,
            hidden_dimension=4,
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            conditioning=ProposalConditioning(pose=_normalization()),
        )
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 10
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "corrupt-current.pth"
            save_action_proposal(proposal, checkpoint, metadata)
            payload = torch.load(checkpoint, weights_only=True)
            payload["state_dict"].pop("goal_delta_mean")
            payload["state_dict"].pop("goal_delta_standard_deviation")
            torch.save(payload, checkpoint)

            with self.assertRaisesRegex(ValueError, "goal-delta state is incomplete"):
                load_action_proposal(checkpoint, device=torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
