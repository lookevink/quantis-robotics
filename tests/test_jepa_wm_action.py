from __future__ import annotations

from math import pi
import unittest

import numpy as np

from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    ActionRecordingContract,
    ActionSelectionBounds,
    DroidAction,
    DroidActionScale,
    DroidPose,
    action_between,
    compose_actions,
)


class DroidActionTest(unittest.TestCase):
    def test_composes_a_native_action_horizon(self) -> None:
        actions = (
            DroidAction((0.001, 0.0, 0.0, 0.01, 0.0, 0.0, 0.1)),
            DroidAction((0.0, 0.002, 0.0, 0.0, 0.02, 0.0, -0.05)),
            DroidAction((0.0, 0.0, 0.003, 0.0, 0.0, 0.03, 0.02)),
        )

        composed = compose_actions(actions)
        expected = DroidPose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
        for action in actions:
            expected = expected.applied(action)

        np.testing.assert_allclose(
            DroidPose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
            .applied(composed)
            .values,
            expected.values,
            atol=1e-12,
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            compose_actions(())

    def test_scales_translation_rotation_and_gripper_independently(self) -> None:
        action = DroidAction((1.0,) * 7)

        scaled = DroidActionScale(1.0, 0.25, 0.125).apply(action)

        self.assertEqual(scaled.values, (1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.125))

    def test_allows_an_explicit_rotation_hold_scale(self) -> None:
        action = DroidAction((1.0,) * 7)

        scaled = DroidActionScale(1.0, 0.0, 1.0).apply(action)

        self.assertEqual(scaled.values, (1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            DroidActionScale(0.0, 0.25, 1.0)
        self.assertEqual(
            DroidActionScale(1.0, 0.25, 0.0).apply(action).values[-1],
            0.0,
        )

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

    def test_rollout_bounds_allow_zero_steps_if_the_rollout_moves(self) -> None:
        bounds = ActionSelectionBounds()

        self.assertTrue(
            bounds.accepts_rollout(
                (
                    DroidAction((0, 0, 0, 0, 0, 0, 0)),
                    DroidAction((0.01, 0, 0, 0, 0, 0, 0)),
                    DroidAction((0, 0, 0, 0, 0, 0, 0)),
                )
            )
        )
        self.assertFalse(
            bounds.accepts_rollout((DroidAction((0, 0, 0, 0, 0, 0, 0)),) * 3)
        )

    def test_converts_world_pose_and_gripper_width_to_droid_pose(self) -> None:
        pose = DroidPose.from_world_poses(
            base_position=[0.0, 0.0, 0.0],
            base_orientation_wxyz=[0.0, 0.0, 0.0, 1.0],
            end_effector_position=[0.1, -0.2, 0.3],
            end_effector_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
            gripper_width_m=0.02,
        )

        np.testing.assert_allclose(
            pose.values,
            [-0.1, 0.2, 0.3, 0.0, 0.0, -pi, 0.75],
            atol=1e-7,
        )

    def test_round_trips_a_base_relative_pose_to_world(self) -> None:
        base_position = (1.0, 2.0, 0.5)
        base_orientation = (0.70710678, 0.0, 0.0, 0.70710678)
        world_position = (1.0, 2.4, 0.8)
        world_orientation = (0.5, 0.5, 0.5, 0.5)
        pose = DroidPose.from_world_poses(
            base_position,
            base_orientation,
            world_position,
            world_orientation,
            0.04,
        )

        position, orientation = pose.to_world_pose(
            base_position, base_orientation
        )

        np.testing.assert_allclose(position, world_position, atol=1e-7)
        self.assertAlmostEqual(abs(float(np.dot(orientation, world_orientation))), 1.0)

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
