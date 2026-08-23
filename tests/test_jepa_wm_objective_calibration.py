from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
    ActionResponseTrial,
    CalibrationIdentity,
    TaskProgressObjective,
)


class ObjectiveCalibrationTest(unittest.TestCase):
    @staticmethod
    def _calibration() -> ActionResponseCalibration:
        return ActionResponseCalibration.fit(
            tuple(
                ActionResponseTrial(
                    f"trial-{index}",
                    index + 1,
                    DroidAction(
                        (
                            *(0.002 if axis == index else 0.0 for axis in range(3)),
                            *(0.004 if axis == index else 0.0 for axis in range(3)),
                            0.2,
                        )
                    ),
                    DroidAction(
                        (
                            *(0.001 if axis == index else 0.0 for axis in range(3)),
                            *(0.0005 if axis == index else 0.0 for axis in range(3)),
                            0.025,
                        )
                    ),
                )
                for index in range(3)
            )
        )

    def test_fits_axis_response_from_realized_actions(self) -> None:
        trials = tuple(
            ActionResponseTrial(
                f"trial-{index}",
                1100 + index,
                DroidAction(
                    (
                        *(0.002 if axis == index else 0.0 for axis in range(3)),
                        *(0.004 if axis == index else 0.0 for axis in range(3)),
                        0.2,
                    )
                ),
                DroidAction(
                    (
                        *(0.001 if axis == index else 0.0 for axis in range(3)),
                        *(0.0005 if axis == index else 0.0 for axis in range(3)),
                        0.025,
                    )
                ),
            )
            for index in range(3)
        )

        calibration = ActionResponseCalibration.fit(trials)

        self.assertEqual(calibration.trial_count, 3)
        self.assertAlmostEqual(calibration.translation_scale, 0.5)
        self.assertAlmostEqual(calibration.rotation_scale, 0.125)
        self.assertAlmostEqual(calibration.gripper_scale, 0.125)
        self.assertEqual(
            ActionResponseCalibration.from_dict(calibration.to_dict()),
            calibration,
        )
        self.assertEqual(len(calibration.fingerprint), 64)
        self.assertEqual(
            CalibrationIdentity.from_calibration(
                Path("/tmp/calibration.json"), calibration
            ).fingerprint,
            calibration.fingerprint,
        )
        self.assertTrue(calibration.ready_for_reranking)

    def test_requires_directionally_diverse_trials(self) -> None:
        raw = DroidAction((0.002, 0.0, 0.0, 0.0, 0.004, 0.0, 0.2))
        trials = tuple(
            ActionResponseTrial(
                f"trial-{index}",
                1100,
                raw,
                DroidAction((0.001, 0.0, 0.0, 0.0, 0.0005, 0.0, 0.025)),
            )
            for index in range(3)
        )

        calibration = ActionResponseCalibration.fit(trials)

        self.assertEqual(calibration.distinct_action_directions, 1)
        self.assertFalse(calibration.ready_for_reranking)

    def test_requires_directional_coverage_on_each_vector_axis(self) -> None:
        trials = tuple(
            ActionResponseTrial(
                f"trial-{index}",
                index,
                DroidAction((0.002, 0.0, 0.0, 0.004, index * 0.001, 0.0, 0.2)),
                DroidAction((0.001, 0.0, 0.0, 0.001, index * 0.00025, 0.0, 0.05)),
            )
            for index in range(3)
        )

        calibration = ActionResponseCalibration.fit(trials)

        self.assertEqual(calibration.translation_direction_count, 1)
        self.assertGreaterEqual(calibration.rotation_direction_count, 2)
        self.assertFalse(calibration.ready_for_reranking)

    def test_refits_raw_trials_and_rejects_tampered_scale_claims(self) -> None:
        calibration = self._calibration()
        payload = calibration.to_dict()
        payload["translation_scale"] = calibration.translation_scale * 2.0

        with self.assertRaisesRegex(ValueError, "claims"):
            ActionResponseCalibration.from_dict(payload)

    def test_refuses_reranking_when_realized_rotation_reverses(self) -> None:
        raw = DroidAction((0.002, 0.0, 0.0, 0.0, 0.004, 0.0, 0.2))
        trials = tuple(
            ActionResponseTrial(
                f"trial-{index}",
                1100 + index,
                raw,
                DroidAction((0.001, 0.0, 0.0, 0.0, -0.0005, 0.0, 0.025)),
            )
            for index in range(3)
        )

        calibration = ActionResponseCalibration.fit(trials)

        self.assertFalse(calibration.ready_for_reranking)
        with self.assertRaisesRegex(ValueError, "not ready"):
            TaskProgressObjective(
                DroidPose((0.0,) * 7),
                DroidPose((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25)),
                calibration,
            )

    def test_reranks_latent_winner_that_moves_away_from_the_goal(self) -> None:
        calibration = self._calibration()
        start = DroidPose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25))
        target = DroidPose((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5))
        moves_away = np.asarray(
            [[[-0.002, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1]] * 3],
            dtype=np.float64,
        )
        moves_toward = np.asarray(
            [[[0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]] * 3],
            dtype=np.float64,
        )
        candidates = np.concatenate((moves_away, moves_toward), axis=0)
        latent_energy = np.asarray((0.005, 0.006), dtype=np.float64)

        scores = latent_energy + TaskProgressObjective(
            start, target, calibration
        ).penalty(candidates)

        self.assertLess(scores[1], scores[0])
        self.assertAlmostEqual(scores[1], latent_energy[1])

        farther_away = moves_away.copy()
        farther_away[0, 0, 0] *= 2.0
        farther_penalty = TaskProgressObjective(
            start,
            target,
            calibration,
        ).penalty(farther_away)[0]
        self.assertGreater(farther_penalty, scores[0] - latent_energy[0])


if __name__ == "__main__":
    unittest.main()
