import unittest

import numpy as np

from jepa_wm.planner import (
    CEMConfig,
    CEMPlanner,
    PlannerActionBounds,
    PlannerDistribution,
)


class CEMPlannerTest(unittest.TestCase):
    def test_recovers_a_known_bounded_three_action_goal(self) -> None:
        target = np.asarray(
            [
                [0.012, -0.006, 0.003, 0.02, 0.0, -0.01, 0.2],
                [0.006, 0.002, -0.004, 0.0, 0.03, 0.01, -0.1],
                [-0.004, 0.003, 0.002, -0.01, 0.0, 0.02, 0.05],
            ],
            dtype=np.float64,
        )

        def squared_distance(candidates: np.ndarray) -> np.ndarray:
            return np.square(candidates - target).sum(axis=(1, 2))

        result = CEMPlanner(
            CEMConfig(iterations=8, samples=300, elites=20, seed=17),
            PlannerActionBounds(),
        ).plan(squared_distance)

        np.testing.assert_allclose(result.actions, target, atol=0.004)
        self.assertLess(result.energy, 0.0001)
        self.assertEqual(result.candidates_scored, 2400)

    def test_every_scored_candidate_obeys_the_action_bounds(self) -> None:
        bounds = PlannerActionBounds(
            maximum_translation_norm=0.01,
            maximum_rotation_norm=0.04,
            maximum_gripper_delta=0.2,
        )
        observed = []

        def record_candidates(candidates: np.ndarray) -> np.ndarray:
            observed.append(candidates.copy())
            return np.square(candidates).sum(axis=(1, 2))

        CEMPlanner(CEMConfig(iterations=3, samples=40, elites=5, seed=3), bounds).plan(
            record_candidates
        )

        candidates = np.concatenate(observed, axis=0)
        self.assertTrue(
            np.all(np.linalg.norm(candidates[:, :, :3], axis=2) <= 0.0100001)
        )
        self.assertTrue(
            np.all(np.linalg.norm(candidates[:, :, 3:6], axis=2) <= 0.0400001)
        )
        self.assertTrue(np.all(np.abs(candidates[:, :, 6]) <= 0.2000001))

    def test_rejects_non_finite_or_miscounted_energy_batches(self) -> None:
        planner = CEMPlanner(
            CEMConfig(iterations=1, samples=8, elites=2), PlannerActionBounds()
        )

        with self.assertRaisesRegex(ValueError, "one finite energy per candidate"):
            planner.plan(lambda candidates: np.zeros(7))
        with self.assertRaisesRegex(ValueError, "one finite energy per candidate"):
            planner.plan(lambda candidates: np.full(len(candidates), np.nan))

    def test_starts_from_a_supplied_action_distribution(self) -> None:
        mean = np.zeros((3, 7), dtype=np.float64)
        mean[:, 0] = 0.008
        distribution = PlannerDistribution(mean, np.full((3, 7), 1e-6))
        seen_first_iteration = []

        def observe(candidates: np.ndarray) -> np.ndarray:
            seen_first_iteration.append(candidates.copy())
            return np.square(candidates - mean).sum(axis=(1, 2))

        result = CEMPlanner(
            CEMConfig(iterations=1, samples=8, elites=2), PlannerActionBounds()
        ).plan(observe, initial_distribution=distribution)

        np.testing.assert_allclose(seen_first_iteration[0][0], mean)
        np.testing.assert_allclose(result.actions, mean)


if __name__ == "__main__":
    unittest.main()
