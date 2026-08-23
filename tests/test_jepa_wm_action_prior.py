import unittest

import numpy as np

from jepa_wm.action_prior import ActionLibrary, ActionPriorConfig, EmpiricalActionPrior


class EmpiricalActionPriorTest(unittest.TestCase):
    def test_fits_each_horizon_step_and_penalizes_out_of_distribution_actions(
        self,
    ) -> None:
        sequences = np.asarray(
            [
                [
                    [0.010, 0.0, 0.0, 0.02, 0.0, 0.0, 0.10],
                    [0.005, 0.0, 0.0, 0.01, 0.0, 0.0, 0.05],
                    [0.000, 0.0, 0.0, 0.00, 0.0, 0.0, 0.00],
                ],
                [
                    [0.012, 0.0, 0.0, 0.03, 0.0, 0.0, 0.12],
                    [0.007, 0.0, 0.0, 0.02, 0.0, 0.0, 0.07],
                    [0.002, 0.0, 0.0, 0.01, 0.0, 0.0, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        prior = EmpiricalActionPrior.fit(
            sequences,
            ActionPriorConfig(
                minimum_translation_std=0.001,
                minimum_rotation_std=0.005,
                minimum_gripper_std=0.01,
                penalty_weight=0.002,
            ),
        )

        np.testing.assert_allclose(prior.distribution.mean[:, 0], [0.011, 0.006, 0.001])
        self.assertEqual(prior.penalty(prior.distribution.mean[None, :, :])[0], 0.0)
        out_of_distribution = prior.distribution.mean.copy()
        out_of_distribution[:, 6] += 0.2
        self.assertGreater(prior.penalty(out_of_distribution[None, :, :])[0], 0.1)

    def test_rejects_an_invalid_action_sequence_batch(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape.*7"):
            EmpiricalActionPrior.fit(np.zeros((3, 7)), ActionPriorConfig())

    def test_builds_a_goal_conditioned_prior_from_the_best_library_actions(
        self,
    ) -> None:
        sequences = np.zeros((4, 3, 7), dtype=np.float64)
        sequences[:, :, 0] = np.asarray((0.001, 0.010, 0.012, -0.008))[:, None]
        library = ActionLibrary(sequences)

        prior = library.goal_conditioned_prior(
            np.asarray((0.8, 0.2, 0.1, 0.9)),
            elites=2,
            config=ActionPriorConfig(),
        )

        np.testing.assert_allclose(prior.distribution.mean[:, 0], [0.011] * 3)


if __name__ == "__main__":
    unittest.main()
