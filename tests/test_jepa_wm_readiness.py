from __future__ import annotations

import unittest

from jepa_wm.readiness import ActionControlGate, ActionControlReason


class ActionControlGateTest(unittest.TestCase):
    def test_fails_closed_for_non_finite_metrics(self) -> None:
        decision = ActionControlGate().evaluate(
            mean_improvement_over_zero=float("nan"),
            recorded_action_win_rate=float("nan"),
        )

        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.reasons,
            (ActionControlReason.NON_FINITE_METRICS,),
        )

    def test_passes_positive_mean_and_required_win_rate(self) -> None:
        decision = ActionControlGate().evaluate(
            mean_improvement_over_zero=0.001,
            recorded_action_win_rate=0.8,
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.reasons, ())

    def test_reports_every_failed_requirement(self) -> None:
        decision = ActionControlGate().evaluate(
            mean_improvement_over_zero=-0.001,
            recorded_action_win_rate=0.5,
        )

        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.reasons,
            (
                ActionControlReason.NON_POSITIVE_MEAN_IMPROVEMENT,
                ActionControlReason.INSUFFICIENT_WIN_RATE,
            ),
        )
        self.assertEqual(
            decision.to_dict(),
            {
                "passed": False,
                "minimum_win_rate": 0.75,
                "requires_positive_mean_improvement": True,
                "reasons": [
                    "non_positive_mean_improvement",
                    "insufficient_win_rate",
                ],
            },
        )


if __name__ == "__main__":
    unittest.main()
