from __future__ import annotations

import unittest

from sim.grasp_task import (
    GraspAcquisitionFailure,
    evaluate_grasp_acquisition,
)


class GraspAcquisitionTest(unittest.TestCase):
    def test_requires_alignment_and_a_closed_gripper(self) -> None:
        decision = evaluate_grasp_acquisition(
            (0.01, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.018,
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.failures, ())

    def test_rejects_free_space_gripper_closure(self) -> None:
        decision = evaluate_grasp_acquisition(
            (0.10, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.018,
        )

        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.failures,
            (GraspAcquisitionFailure.OUTSIDE_GRASP_REGION,),
        )

    def test_rejects_an_open_gripper_at_the_connector(self) -> None:
        decision = evaluate_grasp_acquisition(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            0.07,
        )

        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.failures,
            (GraspAcquisitionFailure.GRIPPER_OPEN,),
        )


if __name__ == "__main__":
    unittest.main()
