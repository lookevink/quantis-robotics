from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_conditioning_training import (
        AlternatingStratumSampler,
        signed_x_margin_loss,
        signed_x_negatives,
    )


@unittest.skipIf(torch is None, "PyTorch is required for action conditioning")
class ActionConditioningTrainingTest(unittest.TestCase):
    def test_signed_x_negatives_preserve_every_other_dimension(self) -> None:
        actions = torch.tensor(
            [
                [[-0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                [[0.3, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]],
            ]
        )

        zero, opposed = signed_x_negatives(actions)

        torch.testing.assert_close(zero[..., 0], torch.zeros((2, 1)))
        torch.testing.assert_close(opposed[..., 0], -actions[..., 0])
        torch.testing.assert_close(zero[..., 1:], actions[..., 1:])
        torch.testing.assert_close(opposed[..., 1:], actions[..., 1:])

    def test_sampler_alternates_and_covers_each_stratum_before_reuse(self) -> None:
        regimes = torch.tensor((0, 0, 1, 1, 1))
        first = AlternatingStratumSampler(regimes, seed=17)
        second = AlternatingStratumSampler(regimes, seed=17)

        indices = [first.next_index() for _ in range(6)]
        replay = [second.next_index() for _ in range(6)]

        self.assertEqual(replay, indices)
        self.assertEqual([int(regimes[index]) for index in indices], [0, 1] * 3)
        self.assertEqual(set(indices[0::2][:2]), {0, 1})
        self.assertEqual(set(indices[1::2]), {2, 3, 4})
        self.assertEqual(
            first.to_dict(),
            {
                "strategy": "alternating_seeded_shuffled_strata",
                "seed": 17,
                "samples_drawn": 6,
                "samples_by_regime": {"retained": 3, "post": 3},
                "rollouts_by_regime": {"retained": 2, "post": 3},
            },
        )

    def test_sampler_requires_both_known_regimes(self) -> None:
        with self.assertRaisesRegex(ValueError, "both regimes"):
            AlternatingStratumSampler(torch.tensor((0, 0)), seed=1)
        with self.assertRaisesRegex(ValueError, "zero or one"):
            AlternatingStratumSampler(torch.tensor((0, 2)), seed=1)

    def test_signed_margin_ignores_inactive_x_without_constant_loss(self) -> None:
        actions = torch.zeros((3, 2, 7))
        actions[:, 1, 0] = 0.002
        recorded = torch.tensor((1.0, 1.0))
        negative = torch.tensor((1.0, 1.0005))

        loss = signed_x_margin_loss(
            recorded,
            negative,
            actions,
            weight=1.0,
            margin=0.001,
            minimum_activity=0.001,
        )

        torch.testing.assert_close(loss, torch.tensor(0.0005))


if __name__ == "__main__":
    unittest.main()
