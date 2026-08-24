from __future__ import annotations

import unittest

from sim.exploration import (
    DatasetSplit,
    SegmentOutcome,
    build_exploration_plan,
    exploration_prefix,
    validate_sample_times,
)


class ExplorationPlanTest(unittest.TestCase):
    def test_selects_only_complete_segments_for_a_control_context(self) -> None:
        plan = build_exploration_plan(11401, DatasetSplit.HELD_OUT)

        prefix = exploration_prefix(plan, 44)

        self.assertEqual(sum(target.frames for target in prefix), 44)
        self.assertEqual(prefix, plan.targets[:11])
        with self.assertRaisesRegex(ValueError, "segment boundary"):
            exploration_prefix(plan, 45)

    def test_rejects_samples_outside_the_droid_cadence(self) -> None:
        validate_sample_times((1.0, 1.25, 1.501), 0.25)

        with self.assertRaisesRegex(ValueError, "cadence is invalid"):
            validate_sample_times((1.0, 1.20), 0.25)
        with self.assertRaisesRegex(ValueError, "cadence is invalid"):
            validate_sample_times((1.0, 1.30), 0.25)

    def test_seeded_plan_excites_every_joint_and_returns_to_origin(self) -> None:
        plan = build_exploration_plan(1200, DatasetSplit.TRAIN)

        self.assertEqual(plan, build_exploration_plan(1200, DatasetSplit.TRAIN))
        self.assertNotEqual(plan, build_exploration_plan(1201, DatasetSplit.TRAIN))
        self.assertEqual(len(plan.targets), 17)
        self.assertEqual(plan.targets[0].outcome, SegmentOutcome.STATIONARY)
        self.assertEqual(plan.targets[-2].outcome, SegmentOutcome.FAILED_GRASP)
        self.assertEqual(plan.targets[-1].outcome, SegmentOutcome.RECOVERY)
        self.assertEqual(plan.targets[-1].arm_offset_radians, (0.0,) * 7)
        for joint_index in range(7):
            self.assertTrue(
                any(
                    abs(target.arm_offset_radians[joint_index]) >= 0.04
                    for target in plan.targets
                ),
                f"joint {joint_index + 1} was not excited",
            )
        self.assertEqual(
            {target.gripper_width_m for target in plan.targets[:-1]},
            {0.025, 0.07},
        )
        self.assertTrue(
            all(
                abs(value) <= 0.10
                for target in plan.targets
                for value in target.arm_offset_radians
            )
        )
        self.assertEqual(plan.sample_period_seconds, 0.25)
        self.assertTrue(
            all(abs(value) <= 0.01 for value in plan.initial_arm_offset_radians)
        )
        self.assertGreaterEqual(plan.socket_scale, 0.98)
        self.assertLessEqual(plan.socket_scale, 1.02)

    def test_manifest_metadata_records_the_whole_seeded_variant(self) -> None:
        plan = build_exploration_plan(2200, DatasetSplit.HELD_OUT)

        self.assertEqual(
            plan.metadata(),
            {
                "dataset": "jepa_wm_domain_v1",
                "split": "held_out",
                "seed": 2200,
                "segments": 17,
                "sample_period_seconds": 0.25,
                "initial_arm_offset_radians": list(plan.initial_arm_offset_radians),
                "camera_offset_m": list(plan.camera_offset_m),
                "scene_offset_m": list(plan.scene_offset_m),
                "socket_scale": plan.socket_scale,
                "light_exposure_delta": plan.light_exposure_delta,
                "segment_outcomes": [target.outcome.value for target in plan.targets],
            },
        )


if __name__ == "__main__":
    unittest.main()
