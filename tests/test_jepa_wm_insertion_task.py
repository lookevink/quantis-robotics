from __future__ import annotations

import unittest

from jepa_wm.insertion_task import (
    InsertionFailure,
    InsertionTarget,
    InsertionTaskStep,
    evaluate_insertion,
)


def _step(
    tip_x: float,
    *,
    attached: bool,
    orientation_error: float = 0.0,
) -> InsertionTaskStep:
    return InsertionTaskStep(
        plug_tip_position=(tip_x, 0.0, 0.0),
        gripper_frame_position=(tip_x + 0.04, 0.0, 0.0),
        plug_attached=attached,
        orientation_error_rad=orientation_error,
        tracking_passed=True,
        collision_detected=False,
        contact_force_newtons=0.5,
    )


class InsertionTaskTest(unittest.TestCase):
    def test_passes_a_rearward_grasp_retained_until_the_tip_is_seated(self) -> None:
        decision = evaluate_insertion(
            (
                _step(0.0, attached=False),
                _step(0.0, attached=True),
                _step(-0.08, attached=True),
                _step(-0.10, attached=True, orientation_error=0.01),
                _step(-0.10, attached=False, orientation_error=0.01),
            ),
            InsertionTarget(
                socket_position=(-0.10, 0.0, 0.0),
                insertion_axis=(-1.0, 0.0, 0.0),
            ),
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.acquisition_index, 1)
        self.assertEqual(decision.seated_index, 3)
        self.assertAlmostEqual(decision.grasp_clearance_meters, 0.04)
        self.assertAlmostEqual(decision.seating_depth_error_meters, 0.0)
        self.assertAlmostEqual(decision.seating_lateral_error_meters, 0.0)

    def test_rejects_a_tip_grasp_even_when_the_tip_reaches_the_socket(self) -> None:
        steps = tuple(
            InsertionTaskStep(
                plug_tip_position=step.plug_tip_position,
                gripper_frame_position=(
                    step.plug_tip_position[0] + 0.01,
                    0.0,
                    0.0,
                ),
                plug_attached=step.plug_attached,
                orientation_error_rad=step.orientation_error_rad,
                tracking_passed=step.tracking_passed,
                collision_detected=step.collision_detected,
                contact_force_newtons=step.contact_force_newtons,
            )
            for step in (
                _step(0.0, attached=False),
                _step(0.0, attached=True),
                _step(-0.10, attached=True),
            )
        )

        decision = evaluate_insertion(
            steps,
            InsertionTarget((-0.10, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            InsertionFailure.INSUFFICIENT_GRASP_CLEARANCE,
            decision.failures,
        )


if __name__ == "__main__":
    unittest.main()
