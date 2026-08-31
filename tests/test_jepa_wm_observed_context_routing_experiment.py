from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_conditioning_experiment import EvaluationEnergies
    from jepa_wm.observed_context_routing_experiment import (
        _gate_for_context_indices,
        observed_route_roster,
        previous_action_tensor,
    )


@unittest.skipIf(torch is None, "PyTorch is required for routing experiments")
class ObservedContextRoutingExperimentTest(unittest.TestCase):
    def test_previous_action_roster_uses_only_observed_actions(self) -> None:
        rollouts = tuple(
            SimpleNamespace(
                previous_action=SimpleNamespace(values=values),
            )
            for values in (
                (-0.0003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        )

        previous = previous_action_tensor(rollouts)
        roster = observed_route_roster(previous)

        self.assertEqual(previous.shape, (3, 7))
        self.assertEqual(
            roster,
            {"negative_x": 1, "positive_x": 1, "base": 1, "total": 3},
        )

    def test_gate_requires_signed_insertion_order(self) -> None:
        contexts = (114, 166, 216)
        energies = EvaluationEnergies(
            recorded=torch.tensor((0.0, 0.0, 0.0)),
            zero=torch.tensor((1.0, 1.0, 1.0)),
            x_zero=torch.tensor((0.2, 0.2, 0.2)),
            x_opposed=torch.tensor((0.3, 0.3, 0.1)),
        )

        _, _, _, by_segment, passed = _gate_for_context_indices(
            energies,
            contexts,
            maximum_residual_ratio=0.1,
        )

        self.assertEqual(by_segment["retreat"]["signed_order_fraction"], 1.0)
        self.assertEqual(by_segment["align"]["signed_order_fraction"], 1.0)
        self.assertEqual(by_segment["insert"]["signed_order_fraction"], 0.0)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
