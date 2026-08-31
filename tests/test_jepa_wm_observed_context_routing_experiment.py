from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_conditioning import (
        ObservedContextResidualActionEncoder,
        ObservedContextRoutingSpec,
    )
    from jepa_wm.action_conditioning_experiment import EvaluationEnergies
    from jepa_wm.observed_context_routing_experiment import (
        _gate_for_context_indices,
        _residual_ratio_report,
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

    def test_residual_gate_covers_every_frozen_evaluation_candidate(self) -> None:
        base = torch.nn.Linear(7, 4, bias=False)
        encoder = ObservedContextResidualActionEncoder(
            base,
            ObservedContextRoutingSpec(1e-4, 1e-4),
        )
        with torch.no_grad():
            base.weight.fill_(1.0)
            encoder.residuals[0].weight.fill_(0.5)
            encoder.residuals[1].weight.fill_(-0.5)
        actions = torch.zeros((3, 2, 7))
        actions[:, 0, 0] = -0.001
        actions[:, 1, 0] = 0.001
        previous = torch.tensor(
            (
                (-0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )
        )

        report = _residual_ratio_report(encoder, actions, previous)

        expected_candidates = {"recorded", "zero", "x_zero", "x_opposed"}
        self.assertEqual(
            set(report["negative_x"]["candidates"]),
            expected_candidates,
        )
        self.assertEqual(
            set(report["positive_x"]["candidates"]),
            expected_candidates,
        )


if __name__ == "__main__":
    unittest.main()
