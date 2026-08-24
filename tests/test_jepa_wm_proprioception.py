import unittest

import numpy as np

from jepa_wm.proprioception import DroidValueNormalization, ScalarNormalization


class DroidValueNormalizationTest(unittest.TestCase):
    def test_standardizes_pose_and_round_trips(self) -> None:
        poses = np.asarray(
            [
                [0.4, -0.1, 0.2, 1.0, 0.0, -1.0, 0.25],
                [0.6, 0.1, 0.4, 1.2, 0.2, -0.8, 0.75],
            ],
            dtype=np.float32,
        )

        normalization = DroidValueNormalization.from_samples(poses)
        standardized = normalization.standardize(poses)

        np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(
            normalization.restore(standardized), poses, atol=1e-6
        )

    def test_floors_constant_dimensions(self) -> None:
        poses = np.ones((3, 7), dtype=np.float32)

        normalization = DroidValueNormalization.from_samples(poses)

        self.assertTrue(np.all(normalization.standard_deviation > 0))
        np.testing.assert_allclose(normalization.standardize(poses), 0.0)

    def test_rejects_an_invalid_pose_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            DroidValueNormalization.from_samples(np.zeros((2, 6), dtype=np.float32))

    def test_normalizes_task_progress_with_a_deviation_floor(self) -> None:
        normalization = ScalarNormalization.from_samples(
            np.asarray((72.0, 72.0), dtype=np.float32)
        )

        self.assertEqual(normalization.mean, 72.0)
        self.assertEqual(normalization.standard_deviation, 1e-3)


if __name__ == "__main__":
    unittest.main()
