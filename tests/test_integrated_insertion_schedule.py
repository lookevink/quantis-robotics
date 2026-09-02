from __future__ import annotations

import unittest

from jepa_wm.integrated_insertion import INTEGRATED_INSERTION_SCHEDULE
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    INSERTION_CONTROL_TARGET_POLICY,
)
from jepa_wm.insertion_layout import ContactInsertionSegment


class IntegratedInsertionScheduleTest(unittest.TestCase):
    def test_terminal_targets_are_reachable_before_the_expensive_run(self) -> None:
        schedule = INTEGRATED_INSERTION_SCHEDULE
        seated_start = CONTACT_INSERTION_RECORDING.start_index(
            ContactInsertionSegment.SEATED_HOLD
        )

        self.assertEqual(len(schedule.context_indices), schedule.action_count)
        self.assertEqual(len(schedule.terminal_context_indices), 4)
        self.assertEqual(
            schedule.initial_context_index,
            CONTACT_INSERTION_RECORDING.start_index(
                ContactInsertionSegment.GRASP_ATTACH
            ),
        )
        self.assertEqual(
            schedule.context_indices,
            tuple(range(113, 281)),
        )
        self.assertGreaterEqual(
            schedule.terminal_context_indices[0]
            + INSERTION_CONTROL_TARGET_POLICY.minimum_action_horizon,
            seated_start,
        )
        self.assertTrue(
            all(
                INSERTION_CONTROL_TARGET_POLICY.minimum_action_horizon
                <= CONTACT_INSERTION_RECORDING.frame_count - 1 - context
                <= INSERTION_CONTROL_TARGET_POLICY.maximum_action_horizon
                for context in schedule.terminal_context_indices
            )
        )


if __name__ == "__main__":
    unittest.main()
