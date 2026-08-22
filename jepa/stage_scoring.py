"""Score held-out stage embeddings against frozen reference centroids."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from jepa.contract import (
    OFFLINE_CONFIDENCE_THRESHOLDS,
    ConfidenceThresholds,
    ObservationStage,
    StagePrediction,
)
from jepa.stage_embeddings import CACHE_SCHEMA


@dataclass(frozen=True)
class StageScore:
    prediction: StagePrediction
    scores: dict[ObservationStage, float]


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("stage embedding cannot be the zero vector")
    return vector / norm


def _load_stage_embeddings(
    recording: Path, camera: str
) -> tuple[str, dict[ObservationStage, np.ndarray]]:
    root = recording / "jepa" / camera
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != CACHE_SCHEMA:
        raise ValueError(f"unsupported stage embedding cache: {root}")
    if manifest.get("camera") != camera:
        raise ValueError(f"stage embedding cache is not for camera {camera!r}")

    entries = manifest.get("stages", {})
    embeddings = {}
    for stage in ObservationStage:
        try:
            relative_path = entries[stage.value]["embedding"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"stage embedding is missing: {stage.value}") from error
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"unsafe stage embedding path: {relative_path}") from error
        embeddings[stage] = _unit(np.load(path))
    return str(manifest.get("model", "")), embeddings


class StageClassifier:
    def __init__(
        self,
        references: list[Path],
        *,
        camera: str,
        thresholds: ConfidenceThresholds = OFFLINE_CONFIDENCE_THRESHOLDS,
    ) -> None:
        if not references:
            raise ValueError("at least one reference recording is required")
        loaded = [_load_stage_embeddings(reference, camera) for reference in references]
        self.model = loaded[0][0]
        if any(reference_model != self.model for reference_model, _ in loaded):
            raise ValueError("reference recordings use different JEPA models")
        self.centroids = {
            stage: _unit(
                np.mean([embeddings[stage] for _, embeddings in loaded], axis=0)
            )
            for stage in ObservationStage
        }
        self.thresholds = thresholds

    def predict(self, embedding: np.ndarray) -> StageScore:
        embedding = _unit(embedding)
        scores = {
            stage: float(np.dot(embedding, centroid))
            for stage, centroid in self.centroids.items()
        }
        ranked = sorted(scores, key=lambda stage: scores[stage], reverse=True)
        best, second = ranked[:2]
        margin = scores[best] - scores[second]
        prediction = StagePrediction(best, scores[best], margin)
        if not self.thresholds.accepts(prediction):
            prediction = StagePrediction(None, scores[best], margin)
        return StageScore(
            prediction,
            scores,
        )


def score_stage_recording(
    references: list[Path],
    query: Path,
    *,
    camera: str,
    thresholds: ConfidenceThresholds = OFFLINE_CONFIDENCE_THRESHOLDS,
) -> dict[str, object]:
    """Classify each held-out stage using reference cosine centroids."""

    query_path = query.resolve()
    if query_path in {reference.resolve() for reference in references}:
        raise ValueError("query must be a separate held-out run from every reference")
    classifier = StageClassifier(
        references,
        camera=camera,
        thresholds=thresholds,
    )
    query_model, query_embeddings = _load_stage_embeddings(query, camera)
    if query_model != classifier.model:
        raise ValueError("query and reference recordings use different JEPA models")

    predictions = []
    correct = 0
    confusion = {stage.value: {} for stage in ObservationStage}
    for actual in ObservationStage:
        score = classifier.predict(query_embeddings[actual])
        prediction = score.prediction.to_dict()
        predicted = prediction["stage"]
        if predicted == actual.value:
            correct += 1
        confusion[actual.value][predicted] = 1
        predictions.append(
            {
                "actual": actual.value,
                **prediction,
                "scores": {
                    stage.value: score.scores[stage] for stage in ObservationStage
                },
            }
        )

    return {
        "model": classifier.model,
        "camera": camera,
        "references": [str(reference) for reference in references],
        "query": str(query),
        "correct": correct,
        "total": len(ObservationStage),
        "accuracy": correct / len(ObservationStage),
        "confusion": confusion,
        "predictions": predictions,
    }
