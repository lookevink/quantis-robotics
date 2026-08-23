from __future__ import annotations

from math import pi
import unittest

import numpy as np

from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    ActionRecordingContract,
    ActionSelectionBounds,
    DroidAction,
    DroidPose,
    action_between,
)


class DroidActionTest(unittest.TestCase):
    def test_round_trips_the_recording_contract(self) -> None:
        payload = ACTION_RECORDING_CONTRACT.to_dict()

        self.assertEqual(
            ActionRecordingContract.from_mapping(payload),
            ACTION_RECORDING_CONTRACT,
        )

    def test_action_selection_bounds_filter_motion(self) -> None:
        bounds = ActionSelectionBounds()

        self.assertTrue(bounds.accepts(DroidAction((0.01, 0, 0, 0, 0, 0, 0))))
        self.assertFalse(bounds.accepts(DroidAction((0, 0, 0, 0, 0, 0, 0))))
        self.assertFalse(bounds.accepts(DroidAction((0.2, 0, 0, 0, 0, 0, 0))))
        self.assertFalse(bounds.accepts(DroidAction((0, 0, 0, 0, 0, 0, 0.8))))

    def test_converts_world_pose_and_gripper_width_to_droid_pose(self) -> None:
        pose = DroidPose.from_world_pose(
            position=[0.1, -0.2, 0.3],
            orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            gripper_width_m=0.02,
        )

        np.testing.assert_allclose(
            pose.values,
            [0.1, -0.2, 0.3, 0.0, 0.0, 0.0, 0.75],
            atol=1e-7,
        )

    def test_computes_relative_translation_rotation_and_gripper_action(self) -> None:
        previous = DroidPose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25))
        current = DroidPose((0.01, -0.02, 0.03, 0.0, 0.0, pi / 2, 0.75))

        action = action_between(previous, current)

        np.testing.assert_allclose(
            action.values,
            [0.01, -0.02, 0.03, 0.0, 0.0, pi / 2, 0.5],
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
