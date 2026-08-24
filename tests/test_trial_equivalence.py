from __future__ import annotations

import unittest

from jepa_wm.action import DroidPose
from jepa_wm.trial_equivalence import TrialResetState, validate_reset_equivalence


class TrialResetEquivalenceTest(unittest.TestCase):
    def test_rejects_a_different_connector_reset(self) -> None:
        pose = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2))
        reference = TrialResetState(
            pose,
            (0.0,) * 7,
            False,
            0.0,
            (0.5, 0.0, 0.4),
            False,
        )
        moved = TrialResetState(
            pose,
            (0.0,) * 7,
            False,
            0.0,
            (0.51, 0.0, 0.4),
            False,
        )

        with self.assertRaisesRegex(ValueError, "same reset"):
            validate_reset_equivalence(reference, moved)

    def test_rejects_an_already_attached_candidate_reset(self) -> None:
        pose = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2))
        reference = TrialResetState(
            pose, (0.0,) * 7, False, 0.0, (0.5, 0.0, 0.4), False
        )
        attached = TrialResetState(
            pose, (0.0,) * 7, False, 0.0, (0.5, 0.0, 0.4), True
        )

        with self.assertRaisesRegex(ValueError, "same reset"):
            validate_reset_equivalence(reference, attached)


if __name__ == "__main__":
    unittest.main()
