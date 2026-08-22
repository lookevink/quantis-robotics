"""Shared observation contract between Isaac capture and V-JEPA inference."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

DEFAULT_FRAMES = 64


class ObservationStage(str, Enum):
    APPROACHING_CABLE = "approaching_cable"
    CABLE_GRASPED = "cable_grasped"
    ALIGNED_WITH_SOCKET = "aligned_with_socket"
    PLUG_SEATED = "plug_seated"


@dataclass(frozen=True)
class StagePrediction:
    stage: ObservationStage | None
    similarity: float
    margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value if self.stage is not None else "unknown",
            "similarity": self.similarity,
            "margin": self.margin,
        }


@dataclass(frozen=True)
class ConfidenceThresholds:
    min_similarity: float
    min_margin: float

    def accepts(self, prediction: StagePrediction) -> bool:
        return (
            prediction.similarity >= self.min_similarity
            and prediction.margin >= self.min_margin
        )


ONLINE_CONFIDENCE_THRESHOLDS = ConfidenceThresholds(0.9, 0.005)
OFFLINE_CONFIDENCE_THRESHOLDS = ConfidenceThresholds(-1.0, 0.0)
