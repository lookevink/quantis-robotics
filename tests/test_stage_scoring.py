from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jepa.contract import ConfidenceThresholds, ObservationStage
from jepa.online_predictor import OnlineStagePredictor
from jepa.stage_scoring import score_stage_recording


def write_embeddings(
    root: Path,
    name: str,
    vectors: dict[ObservationStage, list[float]],
) -> Path:
    recording = root / name
    output = recording / "jepa" / "wrist"
    output.mkdir(parents=True)
    stages = {}
    for stage, vector in vectors.items():
        path = output / f"{stage.value}.npy"
        np.save(path, np.array(vector, dtype=np.float32))
        stages[stage.value] = {"embedding": path.name}
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "quantis.jepa_stage_embeddings.v1",
                "model": "test/model",
                "camera": "wrist",
                "stages": stages,
            }
        )
    )
    return recording


class StageScoringTest(unittest.TestCase):
    def test_rejects_a_query_that_is_also_a_reference(self) -> None:
        basis = {
            ObservationStage.APPROACHING_CABLE: [1.0, 0.0, 0.0, 0.0],
            ObservationStage.CABLE_GRASPED: [0.0, 1.0, 0.0, 0.0],
            ObservationStage.ALIGNED_WITH_SOCKET: [0.0, 0.0, 1.0, 0.0],
            ObservationStage.PLUG_SEATED: [0.0, 0.0, 0.0, 1.0],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = write_embeddings(Path(temp_dir), "same-run", basis)

            with self.assertRaisesRegex(ValueError, "separate held-out run"):
                score_stage_recording([recording], recording, camera="wrist")

    def test_predicts_held_out_stages_from_reference_centroids(self) -> None:
        basis = {
            ObservationStage.APPROACHING_CABLE: [1.0, 0.0, 0.0, 0.0],
            ObservationStage.CABLE_GRASPED: [0.0, 1.0, 0.0, 0.0],
            ObservationStage.ALIGNED_WITH_SOCKET: [0.0, 0.0, 1.0, 0.0],
            ObservationStage.PLUG_SEATED: [0.0, 0.0, 0.0, 1.0],
        }
        query = {
            ObservationStage.APPROACHING_CABLE: [0.9, 0.1, 0.0, 0.0],
            ObservationStage.CABLE_GRASPED: [0.1, 0.9, 0.0, 0.0],
            ObservationStage.ALIGNED_WITH_SOCKET: [0.0, 0.1, 0.9, 0.0],
            ObservationStage.PLUG_SEATED: [0.0, 0.0, 0.1, 0.9],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = write_embeddings(root, "reference", basis)
            held_out = write_embeddings(root, "held-out", query)

            report = score_stage_recording([reference], held_out, camera="wrist")

            self.assertEqual(report["accuracy"], 1.0)
            self.assertEqual(report["correct"], 4)
            self.assertEqual(report["total"], 4)
            self.assertEqual(
                [prediction["stage"] for prediction in report["predictions"]],
                [stage.value for stage in ObservationStage],
            )
            self.assertGreater(report["predictions"][0]["margin"], 0.8)

    def test_abstains_when_the_best_two_stages_are_too_close(self) -> None:
        basis = {
            ObservationStage.APPROACHING_CABLE: [1.0, 0.0],
            ObservationStage.CABLE_GRASPED: [0.0, 1.0],
            ObservationStage.ALIGNED_WITH_SOCKET: [-1.0, 0.0],
            ObservationStage.PLUG_SEATED: [0.0, -1.0],
        }
        ambiguous = dict(basis)
        ambiguous[ObservationStage.APPROACHING_CABLE] = [1.0, 1.0]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = write_embeddings(root, "reference", basis)
            held_out = write_embeddings(root, "held-out", ambiguous)

            report = score_stage_recording(
                [reference],
                held_out,
                camera="wrist",
                thresholds=ConfidenceThresholds(-1.0, 0.1),
            )

            self.assertEqual(report["predictions"][0]["stage"], "unknown")


class OnlineStagePredictorTest(unittest.TestCase):
    def test_assigns_fresh_ids_to_successive_window_predictions(self) -> None:
        basis = {
            ObservationStage.APPROACHING_CABLE: [1.0, 0.0, 0.0, 0.0],
            ObservationStage.CABLE_GRASPED: [0.0, 1.0, 0.0, 0.0],
            ObservationStage.ALIGNED_WITH_SOCKET: [0.0, 0.0, 1.0, 0.0],
            ObservationStage.PLUG_SEATED: [0.0, 0.0, 0.0, 1.0],
        }

        class Encoder:
            def __init__(self) -> None:
                self.embeddings = iter(
                    [
                        np.array([0.9, 0.1, 0.0, 0.0], dtype=np.float32),
                        np.array([0.1, 0.9, 0.0, 0.0], dtype=np.float32),
                    ]
                )

            def embed_paths(self, paths: list[Path]) -> np.ndarray:
                self.assert_frame_count = len(paths)
                return next(self.embeddings)

        with tempfile.TemporaryDirectory() as temp_dir:
            reference = write_embeddings(Path(temp_dir), "reference", basis)
            predictor = OnlineStagePredictor(
                [reference], camera="wrist", encoder=Encoder()
            )
            window = [Path(f"frame_{index:06d}.png") for index in range(64)]

            first = predictor.predict(window)
            second = predictor.predict(window)

            self.assertEqual(first.observation_id, 1)
            self.assertEqual(second.observation_id, 2)
            self.assertEqual(
                first.prediction.stage, ObservationStage.APPROACHING_CABLE
            )
            self.assertEqual(second.prediction.stage, ObservationStage.CABLE_GRASPED)


if __name__ == "__main__":
    unittest.main()
