from __future__ import annotations

import unittest

from jepa.contract import ConfidenceThresholds, ObservationStage, StagePrediction
from jepa.stage_gate import GateAction, GateReason, StageGate, StageObservation


class StageGateTest(unittest.TestCase):
    def test_advances_only_after_consecutive_confident_observations(self) -> None:
        gate = StageGate(
            confirmations=2, thresholds=ConfidenceThresholds(0.9, 0.005)
        )

        first = gate.observe(
            StageObservation(
                observation_id=1,
                prediction=StagePrediction(
                    ObservationStage.APPROACHING_CABLE, 0.99, 0.02
                ),
            )
        )
        second = gate.observe(
            StageObservation(
                observation_id=2,
                prediction=StagePrediction(
                    ObservationStage.APPROACHING_CABLE, 0.99, 0.02
                ),
            )
        )

        self.assertEqual(first.action, GateAction.HOLD)
        self.assertEqual(second.action, GateAction.ADVANCE)
        self.assertEqual(second.next_stage, ObservationStage.CABLE_GRASPED)

    def test_pauses_and_resets_confirmation_on_an_unknown_prediction(self) -> None:
        gate = StageGate(
            confirmations=2, thresholds=ConfidenceThresholds(0.9, 0.01)
        )
        gate.observe(
            StageObservation(
                observation_id=1,
                prediction=StagePrediction(
                    ObservationStage.APPROACHING_CABLE, 0.99, 0.02
                ),
            )
        )

        decision = gate.observe(
            StageObservation(
                observation_id=2,
                prediction=StagePrediction(None, 0.99, 0.0),
            )
        )

        self.assertEqual(decision.action, GateAction.PAUSE)
        self.assertEqual(decision.reason, GateReason.UNKNOWN_STAGE)
        self.assertEqual(decision.confirmations, 0)

    def test_rejects_a_stale_observation_id(self) -> None:
        gate = StageGate(confirmations=1)
        observation = StageObservation(
            observation_id=4,
            prediction=StagePrediction(
                ObservationStage.APPROACHING_CABLE, 1.0, 1.0
            ),
        )
        gate.observe(observation)

        with self.assertRaisesRegex(ValueError, "stale observation"):
            gate.observe(observation)


if __name__ == "__main__":
    unittest.main()
