from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.train_proposal import (
        ProposalLossWeights,
        ProposalTrainingConfig,
        proposal_loss,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class ProposalTrainingLossTest(unittest.TestCase):
    def test_goal_consistency_is_zero_when_sequence_reaches_goal(self) -> None:
        predicted = torch.zeros((2, 3, 7))
        action_mean = torch.full((3, 7), 0.25)
        components = proposal_loss(
            predicted,
            torch.zeros_like(predicted),
            torch.full((2, 7), 0.75),
            action_mean=action_mean,
            action_standard_deviation=torch.ones((3, 7)),
            goal_standard_deviation=torch.ones(7),
        )

        self.assertEqual(float(components.action_mse), 0.0)
        self.assertEqual(float(components.goal_consistency_mse), 0.0)

    def test_spurious_gripper_sequence_increases_goal_loss(self) -> None:
        predicted = torch.zeros((1, 3, 7))
        predicted[:, :, -1] = 1.0
        components = proposal_loss(
            predicted,
            torch.zeros_like(predicted),
            torch.zeros((1, 7)),
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            goal_standard_deviation=torch.ones(7),
        )

        self.assertGreater(float(components.goal_consistency_mse), 1.0)
        self.assertGreater(float(components.inactive_gripper_loss), 1.0)
        self.assertGreater(
            float(components.total(ProposalTrainingConfig().loss_weights)),
            float(components.action_mse),
        )

    def test_negative_goal_weight_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss weights"):
            ProposalLossWeights(goal_consistency=-1.0)

    def test_active_first_action_direction_is_supervised(self) -> None:
        target = torch.zeros((1, 3, 7))
        target[:, 0, 0] = 1.0
        predicted = target.clone()
        predicted[:, 0, 0] = -1.0
        components = proposal_loss(
            predicted,
            target,
            torch.ones((1, 7)),
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            goal_standard_deviation=torch.ones(7),
        )

        self.assertAlmostEqual(float(components.active_direction_loss), 2.0)
        self.assertGreater(float(components.first_action_mse), 0.0)

    def test_first_gripper_timing_is_supervised_separately(self) -> None:
        target = torch.zeros((1, 3, 7))
        predicted = target.clone()
        predicted[:, 0, 6] = 0.5
        components = proposal_loss(
            predicted,
            target,
            torch.zeros((1, 7)),
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            goal_standard_deviation=torch.ones(7),
        )

        self.assertAlmostEqual(float(components.first_gripper_mse), 0.25)
        config = ProposalTrainingConfig(
            loss_weights=ProposalLossWeights(
                action_mse=0.0,
                goal_consistency=0.0,
                first_action_mse=0.0,
                active_direction=0.0,
                inactive_gripper=0.0,
                first_gripper_mse=2.0,
            )
        )
        self.assertAlmostEqual(float(components.total(config.loss_weights)), 0.5)

    def test_goal_consistency_does_not_add_euler_rotation_deltas(self) -> None:
        predicted = torch.zeros((1, 3, 7))
        predicted[0, 0, 3] = 0.5
        predicted[0, 1, 4] = 0.5
        components = proposal_loss(
            predicted,
            torch.zeros_like(predicted),
            torch.zeros((1, 7)),
            action_mean=torch.zeros((3, 7)),
            action_standard_deviation=torch.ones((3, 7)),
            goal_standard_deviation=torch.ones(7),
        )

        self.assertEqual(float(components.goal_consistency_mse), 0.0)


if __name__ == "__main__":
    unittest.main()
