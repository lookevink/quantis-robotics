"""Safety gate between JEPA stage predictions and high-level robot subgoals."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from jepa.contract import (
    ONLINE_CONFIDENCE_THRESHOLDS,
    ConfidenceThresholds,
    ObservationStage,
    StagePrediction,
)


class GateAction(str, Enum):
    HOLD = "hold"
    ADVANCE = "advance"
    PAUSE = "pause"
    COMPLETE = "complete"


class GateReason(str, Enum):
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    STAGE_CONFIRMED = "stage_confirmed"
    SEQUENCE_CONFIRMED = "sequence_confirmed"
    UNKNOWN_STAGE = "unknown_stage"
    UNEXPECTED_STAGE = "unexpected_stage"
    LOW_CONFIDENCE = "low_confidence"


@dataclass(frozen=True)
class StageObservation:
    observation_id: int
    prediction: StagePrediction

    def to_dict(self) -> dict[str, object]:
        return {"observation_id": self.observation_id, **self.prediction.to_dict()}


@dataclass(frozen=True)
class GateDecision:
    action: GateAction
    expected_stage: ObservationStage
    next_stage: ObservationStage | None
    confirmations: int
    reason: GateReason


class StageGate:
    """Require fresh, repeated visual confirmation before phase advancement."""

    def __init__(
        self,
        *,
        confirmations: int = 2,
        thresholds: ConfidenceThresholds = ONLINE_CONFIDENCE_THRESHOLDS,
    ) -> None:
        if confirmations <= 0:
            raise ValueError("confirmations must be positive")
        self._stages = tuple(ObservationStage)
        self._stage_index = 0
        self._required_confirmations = confirmations
        self._thresholds = thresholds
        self._confirmations = 0
        self._last_observation_id = -1

    def observe(self, observation: StageObservation) -> GateDecision:
        if observation.observation_id <= self._last_observation_id:
            raise ValueError(
                f"stale observation {observation.observation_id}; "
                f"last accepted ID is {self._last_observation_id}"
            )
        self._last_observation_id = observation.observation_id
        expected = self._stages[self._stage_index]

        if observation.prediction.stage is None:
            return self._pause(expected, GateReason.UNKNOWN_STAGE)
        if observation.prediction.stage != expected:
            return self._pause(expected, GateReason.UNEXPECTED_STAGE)
        if not self._thresholds.accepts(observation.prediction):
            return self._pause(expected, GateReason.LOW_CONFIDENCE)

        self._confirmations += 1
        if self._confirmations < self._required_confirmations:
            return GateDecision(
                GateAction.HOLD,
                expected,
                expected,
                self._confirmations,
                GateReason.AWAITING_CONFIRMATION,
            )

        self._confirmations = 0
        if self._stage_index == len(self._stages) - 1:
            return GateDecision(
                GateAction.COMPLETE,
                expected,
                None,
                self._required_confirmations,
                GateReason.SEQUENCE_CONFIRMED,
            )
        self._stage_index += 1
        return GateDecision(
            GateAction.ADVANCE,
            expected,
            self._stages[self._stage_index],
            self._required_confirmations,
            GateReason.STAGE_CONFIRMED,
        )

    def _pause(
        self, expected: ObservationStage, reason: GateReason
    ) -> GateDecision:
        self._confirmations = 0
        return GateDecision(GateAction.PAUSE, expected, expected, 0, reason)
