from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.evaluate_recording import (
        EvaluationResourceBatches,
        score_evaluation_batches,
    )


class _TrackingEnergyModel:
    def __init__(self) -> None:
        self.batch_sizes = []

    def unroll(self, context, actions):
        self.batch_sizes.append(actions.shape[1])
        return context + actions.sum(dim=0, keepdim=False).unsqueeze(1)


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class EvaluateRecordingBatchTest(unittest.TestCase):
    def test_resource_batch_contract_owns_validation_and_serialization(self) -> None:
        batches = EvaluationResourceBatches(encoding=4, evaluation=2)

        self.assertEqual(batches.to_dict(), {"encoding": 4, "evaluation": 2})
        with self.assertRaisesRegex(ValueError, "positive"):
            EvaluationResourceBatches(encoding=0)

    def test_scores_all_rollouts_in_bounded_batches(self) -> None:
        model = _TrackingEnergyModel()
        context = torch.zeros((5, 1, 7))
        target = torch.ones((5, 1, 7))
        actions = torch.full((3, 5, 7), 0.1)

        energies = score_evaluation_batches(
            model,
            context,
            target,
            actions,
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(energies.recorded.shape, (5,))
        self.assertEqual(energies.zero.shape, (5,))
        self.assertEqual(model.batch_sizes, [2, 2, 2, 2, 1, 1])

    def test_rejects_invalid_batch_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            score_evaluation_batches(
                _TrackingEnergyModel(),
                torch.zeros((1, 1, 7)),
                torch.zeros((1, 1, 7)),
                torch.zeros((3, 1, 7)),
                batch_size=0,
                device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
