import unittest

from jepa_wm.action import DroidAction
from jepa_wm.planner_readiness import (
    FirstActionGate,
    FirstActionReason,
    evaluate_first_actions,
)


class FirstActionGateTest(unittest.TestCase):
    def test_accepts_an_aligned_active_action(self) -> None:
        decision = FirstActionGate().evaluate(
            DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)),
            DroidAction((0.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.08)),
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.reasons, ())
        self.assertGreater(decision.cosine, 0.99)

    def test_rejects_motion_when_the_recorded_first_action_is_stationary(self) -> None:
        decision = FirstActionGate().evaluate(
            DroidAction((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.004, 0.0, 0.0, 0.02, 0.0, 0.0, 0.1)),
        )

        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.reasons,
            (
                FirstActionReason.UNNECESSARY_TRANSLATION,
                FirstActionReason.UNNECESSARY_ROTATION,
                FirstActionReason.UNNECESSARY_GRIPPER,
            ),
        )

    def test_rejects_an_opposite_active_action(self) -> None:
        decision = FirstActionGate().evaluate(
            DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((-0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )

        self.assertFalse(decision.passed)
        self.assertEqual(decision.reasons, (FirstActionReason.DIRECTION_MISMATCH,))

    def test_summarizes_active_and_stationary_actions_once(self) -> None:
        summary = evaluate_first_actions(
            (
                DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                DroidAction((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ),
            (
                DroidAction((0.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                DroidAction((0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ),
        )

        self.assertEqual(summary.pass_rate, 0.5)
        self.assertEqual(summary.mean_cosine, 0.5)
        self.assertEqual(summary.mean_active_cosine, 1.0)
        self.assertEqual(summary.active_direction_pass_rate, 1.0)
        self.assertEqual(summary.stationary_hold_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
