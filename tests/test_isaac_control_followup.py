from __future__ import annotations

import unittest

import numpy as np

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_tracking import ActionTrackingDecision
from sim.control_session import PostActionEvidence
from sim.isaac_control_followup import validate_followup_continuity
from sim.isaac_demo_runtime import JointCommand


class FollowupContinuityTest(unittest.TestCase):
    def _previous(self) -> PostActionEvidence:
        action = DroidAction((0.0,) * 7)
        return PostActionEvidence(
            action,
            action,
            action,
            ActionTrackingDecision(0.0, 0.0, 0.0, 0.0, 0.0, ()),
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
            0.0,
            0.0,
            False,
            {"path": "/tmp/post.png", "shape": [512, 512, 4]},
        )

    def test_accepts_the_same_live_articulation(self) -> None:
        previous = self._previous()
        validate_followup_continuity(
            previous,
            JointCommand(np.asarray(previous.joint_positions), 0.04),
            previous.pose,
        )

    def test_rejects_a_reset_or_changed_live_articulation(self) -> None:
        previous = self._previous()
        with self.subTest("joint drift"):
            joints = np.asarray(previous.joint_positions)
            joints[0] += 0.01
            with self.assertRaisesRegex(ValueError, "live stage"):
                validate_followup_continuity(
                    previous,
                    JointCommand(joints, 0.04),
                    previous.pose,
                )
        with self.subTest("Cartesian drift"):
            with self.assertRaisesRegex(ValueError, "live stage"):
                validate_followup_continuity(
                    previous,
                    JointCommand(np.asarray(previous.joint_positions), 0.04),
                    DroidPose((0.41, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                )


if __name__ == "__main__":
    unittest.main()
