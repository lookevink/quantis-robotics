from __future__ import annotations

import unittest

from jepa_wm.action import DroidAction
from jepa_wm.physical_observation import (
    PHYSICAL_ROUTING_FEATURE_NAMES,
    PhysicalRoutingObservation,
)


def _target(offset: tuple[float, float, float]) -> dict[str, object]:
    return {
        "socket_position": [offset[0] - 0.071, offset[1] - 0.25, offset[2] + 1.32],
        "socket_orientation_wxyz": [0.0, 0.0, 0.0, 1.0],
        "insertion_axis": [-1.0, 0.0, 0.0],
    }


def _step(offset: tuple[float, float, float]) -> dict[str, object]:
    return {
        "plug_position": [offset[0] - 0.02, offset[1] - 0.25, offset[2] + 1.32],
        "plug_orientation_wxyz": [0.0, 0.0, 0.0, 1.0],
        "end_effector_world_position": [
            offset[0] + 0.12,
            offset[1] - 0.25,
            offset[2] + 1.32,
        ],
        "gripper_frame_world_position": [
            offset[0] + 0.02,
            offset[1] - 0.25,
            offset[2] + 1.32,
        ],
        "gripper_width_m": 0.018,
        "arm_tracking_error_rad": 0.002,
        "gripper_tracking_error_m": 0.0005,
        "contact_force_newtons": 0.0,
        "plug_attached": True,
        "phase": "must_not_be_a_feature",
        "index": 113,
    }


class PhysicalRoutingObservationTest(unittest.TestCase):
    def test_task_relative_features_ignore_shared_scene_translation(self) -> None:
        previous = DroidAction((0.001, -0.002, 0.003, 0.0, 0.0, 0.0, 0.01))

        original = PhysicalRoutingObservation.from_recorded_step(
            _step((0.0, 0.0, 0.0)),
            _target((0.0, 0.0, 0.0)),
            previous,
        )
        translated = PhysicalRoutingObservation.from_recorded_step(
            _step((0.3, -0.2, 0.1)),
            _target((0.3, -0.2, 0.1)),
            previous,
        )

        self.assertEqual(len(PHYSICAL_ROUTING_FEATURE_NAMES), 26)
        for original_value, translated_value in zip(
            original.values,
            translated.values,
        ):
            self.assertAlmostEqual(original_value, translated_value, places=12)
        self.assertNotIn("phase", PHYSICAL_ROUTING_FEATURE_NAMES)
        self.assertNotIn("context_index", PHYSICAL_ROUTING_FEATURE_NAMES)
        self.assertEqual(original.values[-7:], previous.values)

    def test_relative_orientation_is_invariant_to_quaternion_sign(self) -> None:
        previous = DroidAction((0.0,) * 7)
        target = _target((0.0, 0.0, 0.0))
        target["socket_orientation_wxyz"] = [0.5, 0.5, 0.5, 0.5]
        positive = _step((0.0, 0.0, 0.0))
        positive["plug_orientation_wxyz"] = [0.5, 0.5, 0.5, 0.5]
        negative = dict(positive)
        negative["plug_orientation_wxyz"] = [-0.5, -0.5, -0.5, -0.5]

        positive_observation = PhysicalRoutingObservation.from_recorded_step(
            positive,
            target,
            previous,
        )
        negative_observation = PhysicalRoutingObservation.from_recorded_step(
            negative,
            target,
            previous,
        )

        self.assertEqual(positive_observation.values, negative_observation.values)
        self.assertEqual(
            positive_observation.values[10:14],
            (1.0, 0.0, 0.0, 0.0),
        )


if __name__ == "__main__":
    unittest.main()
