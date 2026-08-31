from __future__ import annotations

import unittest

from jepa_wm.owned_slice_gate import (
    OwnedSliceGate,
    SliceEvaluation,
    SliceRequirement,
    TrainingFeasibility,
)


class OwnedSliceGateTest(unittest.TestCase):
    def test_preflight_blocks_failed_slice_with_no_trainable_route(self) -> None:
        feasibility = TrainingFeasibility.evaluate(
            requirements={
                "grasp_attach": SliceRequirement.owned(
                    minimum_win_rate=0.75,
                    require_positive_mean=True,
                )
            },
            route_counts={"grasp_attach": {"base": 12}},
            trainable_routes={"retreat", "advance"},
        )

        self.assertFalse(feasibility.feasible)
        self.assertIn("no trainable route", feasibility.reasons[0])

    def test_passthrough_slice_requires_baseline_equivalence_not_improvement(
        self,
    ) -> None:
        requirements = {
            "retreat": SliceRequirement.owned(
                minimum_win_rate=0.90,
                require_positive_mean=True,
                minimum_signed_order_fraction=0.75,
            ),
            "retreat_hold": SliceRequirement.passthrough(),
        }
        baseline = {
            "retreat_hold": SliceEvaluation(
                recorded_win_rate=0.25,
                mean_improvement_over_zero=-0.1,
                signed_order_fraction=0.25,
                matches_baseline=True,
            )
        }
        candidate = {
            "retreat": SliceEvaluation(
                recorded_win_rate=1.0,
                mean_improvement_over_zero=0.1,
                signed_order_fraction=1.0,
            ),
            "retreat_hold": baseline["retreat_hold"],
        }

        decision = OwnedSliceGate(requirements).evaluate(candidate)

        self.assertTrue(decision.passed)

    def test_passthrough_slice_rejects_changed_output(self) -> None:
        gate = OwnedSliceGate({"hold": SliceRequirement.passthrough()})

        decision = gate.evaluate(
            {
                "hold": SliceEvaluation(
                    recorded_win_rate=1.0,
                    mean_improvement_over_zero=1.0,
                    signed_order_fraction=1.0,
                    matches_baseline=False,
                )
            }
        )

        self.assertFalse(decision.passed)
        self.assertIn("baseline equivalence", decision.reasons[0])


if __name__ == "__main__":
    unittest.main()
