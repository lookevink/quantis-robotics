from __future__ import annotations

import unittest

from jepa_wm.action import DroidPose
from jepa_wm.joint_settlement import (
    JointSettlementEvidence,
    TrackedJointSettlementPolicy,
)
from jepa_wm.target_progress import (
    RealizedTargetProgressDecision,
    RealizedTargetProgressPolicy,
    RealizedTargetProgressReason,
)


def _pose(x: float, rotation_x: float = 0.0) -> DroidPose:
    return DroidPose((x, 0.0, 0.5, rotation_x, 0.0, 0.0, 0.5))


class RealizedTargetProgressPolicyTest(unittest.TestCase):
    def test_requires_quarter_translation_progress_outside_deadband(self) -> None:
        policy = RealizedTargetProgressPolicy()
        initial = _pose(0.0)
        target = _pose(0.001)

        passing = policy.evaluate(initial, target, _pose(0.0003))
        insufficient = policy.evaluate(initial, target, _pose(0.0001))

        self.assertTrue(passing.passed)
        self.assertAlmostEqual(passing.translation_error_reduction_fraction, 0.3)
        self.assertFalse(insufficient.passed)
        self.assertEqual(
            insufficient.reasons,
            (RealizedTargetProgressReason.TRANSLATION_PROGRESS,),
        )

    def test_close_enough_deadband_replaces_unstable_relative_progress(self) -> None:
        decision = RealizedTargetProgressPolicy().evaluate(
            _pose(0.0),
            _pose(0.00011),
            _pose(0.00002),
        )

        self.assertTrue(decision.close_enough)
        self.assertLess(decision.translation_error_reduction_fraction, 0.25)
        self.assertTrue(decision.passed)

    def test_orientation_regression_still_fails_inside_translation_deadband(self) -> None:
        decision = RealizedTargetProgressPolicy().evaluate(
            _pose(0.0),
            _pose(0.00011),
            _pose(0.00002, 0.002),
        )

        self.assertTrue(decision.close_enough)
        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.reasons,
            (RealizedTargetProgressReason.ORIENTATION_REGRESSION,),
        )

    def test_progress_decision_round_trip_rejects_typed_claim_tampering(self) -> None:
        decision = RealizedTargetProgressPolicy().evaluate(
            _pose(0.0),
            _pose(0.001),
            _pose(0.0003),
        )
        payload = decision.to_dict()

        self.assertEqual(RealizedTargetProgressDecision.from_dict(payload), decision)

        payload["passed"] = False
        with self.assertRaisesRegex(ValueError, "pass claim"):
            RealizedTargetProgressDecision.from_dict(payload)

        payload = decision.to_dict()
        payload["close_enough"] = 0
        with self.assertRaisesRegex(ValueError, "incomplete"):
            RealizedTargetProgressDecision.from_dict(payload)

        payload = decision.to_dict()
        payload["realized_translation_error_meters"] = "0.0007"
        with self.assertRaises(ValueError):
            RealizedTargetProgressDecision.from_dict(payload)


class TrackedJointSettlementPolicyTest(unittest.TestCase):
    def test_settlement_evidence_round_trip_rejects_numeric_strings(self) -> None:
        policy = TrackedJointSettlementPolicy()
        evidence = JointSettlementEvidence(
            requested_joint_motion_radians=0.002,
            required_tracking_error_radians=0.0005,
            updates_used=3,
            passing_tracking_errors_radians=(0.0004, 0.0003),
        )
        evidence.validate(policy)

        self.assertEqual(JointSettlementEvidence.from_dict(evidence.to_dict()), evidence)

        payload = evidence.to_dict()
        payload["required_tracking_error_radians"] = "0.0005"
        with self.assertRaises(ValueError):
            JointSettlementEvidence.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
