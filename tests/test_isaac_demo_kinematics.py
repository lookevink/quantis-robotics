from __future__ import annotations

import unittest

import numpy as np

from sim.isaac_demo_kinematics import _closest_inverse_kinematics


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


class _LateLocalBranchSolver:
    def __init__(self) -> None:
        self.calls = 0

    def compute_inverse_kinematics(self, *args, **kwargs):
        del args, kwargs
        self.calls += 1
        if self.calls < 9:
            return np.full(7, 1.6), True
        return np.full(7, 0.0065), True


class _ToleranceCapturingSolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, float]] = []

    def compute_inverse_kinematics(self, *args, **kwargs):
        del args
        self.calls.append(kwargs)
        return np.zeros(7), True


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

    def test_searches_bounded_milliradian_seeds_for_a_local_branch(self) -> None:
        solver = _LateLocalBranchSolver()

        solved, success = _closest_inverse_kinematics(
            solver,
            "panda_hand",
            np.zeros(3),
            np.asarray((1.0, 0.0, 0.0, 0.0)),
            np.zeros(7),
        )

        self.assertTrue(success)
        self.assertEqual(solver.calls, 9)
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

        self.assertEqual(len(solver.calls), 9)
        self.assertTrue(
            all(
                call["position_tolerance"] == 0.0001
                and call["orientation_tolerance"] == 0.001
                for call in solver.calls
            )
        )


if __name__ == "__main__":
    unittest.main()
