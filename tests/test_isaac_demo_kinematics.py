from __future__ import annotations

import unittest

import numpy as np

from jepa_wm.action import DroidAction
from sim.isaac_demo_kinematics import (
    IK_ACTIVE_ROTATION_TOLERANCE_RADIANS,
    IK_ACTIVE_ROTATION_TOLERANCES_RADIANS,
    IK_ORIENTATION_HOLD_TOLERANCE_RADIANS,
    _closest_inverse_kinematics,
    _orientation_tolerance_for_delta,
    orientation_tolerance_for_action,
    orientation_tolerances_for_action,
)


class _BranchingSolver:
    def __init__(self) -> None:
        self.calls = 0

    def compute_inverse_kinematics(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        if self.calls == 1:
            return np.full(7, 3.0), True
        if self.calls == 2:
            return np.full(7, 0.002), True
        return np.full(7, 0.01), True

    def compute_forward_kinematics(self, *args, **kwargs):
        del args, kwargs
        return np.zeros(3), np.eye(3)


class _IndependentJointBranchSolver:
    def __init__(self) -> None:
        self.calls = 0

    def compute_inverse_kinematics(self, *args, **kwargs):
        del args
        self.calls += 1
        warm_start = kwargs["warm_start"]
        if np.allclose(
            warm_start,
            np.asarray((0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0)),
            rtol=0.0,
            atol=1e-12,
        ):
            return np.full(7, 0.0065), True
        return np.full(7, 1.6), True

    def compute_forward_kinematics(self, *args, **kwargs):
        del args, kwargs
        return np.zeros(3), np.eye(3)


class _ToleranceCapturingSolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def compute_inverse_kinematics(self, *args, **kwargs):
        del args
        self.calls.append(kwargs)
        return np.zeros(7), True

    def compute_forward_kinematics(self, *args, **kwargs):
        del args, kwargs
        return np.zeros(3), np.eye(3)


class _InaccurateSolver(_ToleranceCapturingSolver):
    def compute_forward_kinematics(self, *args, **kwargs):
        del args, kwargs
        return np.asarray((0.0002, 0.0, 0.0)), np.eye(3)


class _OrientationInaccurateSolver(_ToleranceCapturingSolver):
    def compute_forward_kinematics(self, *args, **kwargs):
        del args, kwargs
        rotation = np.asarray(
            (
                (1.0, 0.0, 0.0),
                (0.0, np.cos(0.002), -np.sin(0.002)),
                (0.0, np.sin(0.002), np.cos(0.002)),
            )
        )
        return np.zeros(3), rotation


class ClosestInverseKinematicsTest(unittest.TestCase):
    def test_selects_successful_solution_closest_to_captured_joints(self) -> None:
        solved, success = _closest_inverse_kinematics(
            _BranchingSolver(),
            "panda_hand",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
        )

        self.assertTrue(success)
        np.testing.assert_allclose(solved, np.full(7, 0.002))

    def test_fails_when_no_branch_solves(self) -> None:
        solver = _BranchingSolver()
        solver.compute_inverse_kinematics = lambda *args, **kwargs: (np.ones(7), False)

        solved, success = _closest_inverse_kinematics(
            solver,
            "panda_hand",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
        )

        self.assertFalse(success)
        np.testing.assert_allclose(solved, np.zeros(7))

    def test_searches_independent_bounded_joint_seeds_for_a_local_branch(self) -> None:
        solver = _IndependentJointBranchSolver()

        solved, success = _closest_inverse_kinematics(
            solver,
            "panda_hand",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
        )

        self.assertTrue(success)
        self.assertEqual(solver.calls, 45)
        np.testing.assert_allclose(solved, np.full(7, 0.0065))

    def test_requires_submillimeter_waypoint_accuracy(self) -> None:
        solver = _ToleranceCapturingSolver()

        _closest_inverse_kinematics(
            solver,
            "right_gripper",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
        )

        self.assertEqual(len(solver.calls), 45)
        self.assertTrue(
            all(
                call["position_tolerance"] == 0.0001
                and call["orientation_tolerance"] == 0.001
                for call in solver.calls
            )
        )

    def test_uses_strict_orientation_tolerance_only_for_active_rotation(self) -> None:
        self.assertEqual(
            _orientation_tolerance_for_delta(0.0),
            IK_ORIENTATION_HOLD_TOLERANCE_RADIANS,
        )
        self.assertEqual(
            _orientation_tolerance_for_delta(0.000999),
            IK_ORIENTATION_HOLD_TOLERANCE_RADIANS,
        )
        self.assertEqual(
            _orientation_tolerance_for_delta(0.001),
            IK_ACTIVE_ROTATION_TOLERANCE_RADIANS,
        )
        self.assertEqual(
            orientation_tolerance_for_action(
                DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
            IK_ORIENTATION_HOLD_TOLERANCE_RADIANS,
        )
        self.assertEqual(
            orientation_tolerance_for_action(
                DroidAction((0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0))
            ),
            IK_ACTIVE_ROTATION_TOLERANCE_RADIANS,
        )
        self.assertEqual(
            orientation_tolerances_for_action(
                DroidAction((0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0))
            ),
            IK_ACTIVE_ROTATION_TOLERANCES_RADIANS,
        )
        self.assertEqual(
            orientation_tolerances_for_action(
                DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
            (IK_ORIENTATION_HOLD_TOLERANCE_RADIANS,),
        )

    def test_accepts_a_stricter_diagnostic_orientation_tolerance(self) -> None:
        solver = _ToleranceCapturingSolver()

        _closest_inverse_kinematics(
            solver,
            "panda_hand",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
            orientation_tolerance_radians=0.0005,
        )

        self.assertEqual(len(solver.calls), 45)
        self.assertTrue(
            all(
                call["orientation_tolerance"] == 0.0005
                for call in solver.calls
            )
        )

    def test_rejects_success_claim_when_forward_error_exceeds_tolerance(self) -> None:
        solved, success = _closest_inverse_kinematics(
            _InaccurateSolver(),
            "right_gripper",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
        )

        self.assertFalse(success)
        np.testing.assert_allclose(solved, np.zeros(7))

    def test_rejects_success_claim_when_orientation_error_exceeds_tolerance(self) -> None:
        solved, success = _closest_inverse_kinematics(
            _OrientationInaccurateSolver(),
            "right_gripper",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
        )

        self.assertFalse(success)
        np.testing.assert_allclose(solved, np.zeros(7))


if __name__ == "__main__":
    unittest.main()
