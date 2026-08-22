"""Fresh-ID prediction producer for rolling online observation windows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from jepa.contract import (
    DEFAULT_FRAMES,
    ONLINE_CONFIDENCE_THRESHOLDS,
    ConfidenceThresholds,
)
from jepa.stage_gate import StageObservation
from jepa.stage_scoring import StageClassifier


class WindowEncoder(Protocol):
    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        ...


class OnlineStagePredictor:
    def __init__(
        self,
        references: list[Path],
        *,
        camera: str,
        encoder: WindowEncoder,
        model_id: str | None = None,
        frame_count: int = DEFAULT_FRAMES,
        thresholds: ConfidenceThresholds = ONLINE_CONFIDENCE_THRESHOLDS,
    ) -> None:
        self.classifier = StageClassifier(
            references,
            camera=camera,
            thresholds=thresholds,
        )
        if model_id is not None and self.classifier.model != model_id:
            raise ValueError("online encoder and reference cache use different models")
        self.encoder = encoder
        self.frame_count = frame_count
        self._next_observation_id = 1

    def predict(self, frames: list[Path]) -> StageObservation:
        if len(frames) != self.frame_count:
            raise ValueError(
                f"online window has {len(frames)} frames; "
                f"expected {self.frame_count}"
            )
        score = self.classifier.predict(self.encoder.embed_paths(frames))
        observation = StageObservation(
            observation_id=self._next_observation_id,
            prediction=score.prediction,
        )
        self._next_observation_id += 1
        return observation
