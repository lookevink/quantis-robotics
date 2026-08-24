from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.rollout_scoring import score_recorded_against_mismatched


class _CumulativeActionModel:
    @staticmethod
    def unroll(context, actions):
        return actions.cumsum(dim=0)


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class ContrastiveRolloutScoringTest(unittest.TestCase):
    def test_scores_zero_and_plausible_wrong_actions_as_separate_negatives(self) -> None:
        recorded = torch.tensor([[[1.0]]])
        mismatched = torch.tensor([[[-1.0]]])
        target = torch.tensor([[[1.0]]])

        energies = score_recorded_against_mismatched(
            _CumulativeActionModel(),
            context=torch.zeros_like(target),
            target=target,
            actions=recorded,
            mismatched_actions=mismatched,
        )

        torch.testing.assert_close(energies.recorded, torch.tensor([0.0]))
        torch.testing.assert_close(energies.zero, torch.tensor([1.0]))
        torch.testing.assert_close(energies.mismatched_negative, torch.tensor([4.0]))

    def test_rejects_a_mismatched_negative_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "must match"):
            score_recorded_against_mismatched(
                _CumulativeActionModel(),
                context=torch.zeros((1, 1, 1)),
                target=torch.zeros((1, 1, 1)),
                actions=torch.zeros((1, 1, 1)),
                mismatched_actions=torch.zeros((2, 1, 1)),
            )


if __name__ == "__main__":
    unittest.main()
