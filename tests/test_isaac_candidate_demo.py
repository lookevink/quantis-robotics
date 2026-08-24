from __future__ import annotations

import unittest

import numpy as np

from sim.isaac_candidate_demo import (
    ReplaySafetyMonitor,
    ReplayTrackingMonitor,
    command_errors,
)
from sim.isaac_demo_runtime import JointCommand


class IsaacCandidateDemoTest(unittest.TestCase):
    def test_command_errors_include_arm_and_gripper_state(self) -> None:
        actual = JointCommand(np.asarray([0.01] * 7), 0.071)
        expected = JointCommand(np.asarray([0.0] * 7), 0.070)

        arm_error, gripper_error = command_errors(actual, expected)

        self.assertAlmostEqual(arm_error, 0.01)
        self.assertAlmostEqual(gripper_error, 0.001)

    def test_monitor_rejects_transient_contact_during_replay(self) -> None:
        monitor = ReplaySafetyMonitor()
        monitor.observe(False, 0.4)

        with self.assertRaisesRegex(RuntimeError, "unsafe contact"):
            monitor.observe(True, 0.2)

        self.assertTrue(monitor.collision_detected)
        self.assertEqual(monitor.maximum_contact_force_newtons, 0.4)

    def test_tracking_monitor_accumulates_every_frame_and_rejects_lag(self) -> None:
        monitor = ReplayTrackingMonitor()
        expected = JointCommand(np.asarray([0.0] * 7), 0.07)
        monitor.observe(JointCommand(np.asarray([0.005] * 7), 0.071), expected)

        with self.assertRaisesRegex(RuntimeError, "replay tracking"):
            monitor.observe(JointCommand(np.asarray([0.02] * 7), 0.07), expected)

        self.assertAlmostEqual(monitor.maximum_arm_error_rad, 0.02)
        self.assertAlmostEqual(monitor.maximum_gripper_error_m, 0.001)


if __name__ == "__main__":
    unittest.main()
