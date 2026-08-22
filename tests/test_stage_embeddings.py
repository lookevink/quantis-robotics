from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jepa.contract import DEFAULT_FRAMES, ObservationStage
from jepa.stage_embeddings import embed_recording_stages


def write_recording(root: Path, *, frames_per_stage: int = DEFAULT_FRAMES) -> Path:
    recording = root / "demo-reference"
    wrist = recording / "wrist"
    wrist.mkdir(parents=True)
    steps = []
    index = 0
    for stage in ObservationStage:
        for _ in range(frames_per_stage):
            frame = wrist / f"frame_{index:06d}.png"
            frame.touch()
            steps.append(
                {
                    "index": index,
                    "stage": stage.value,
                    "frames": {"wrist": frame.relative_to(recording).as_posix()},
                }
            )
            index += 1
    (recording / "steps.jsonl").write_text(
        "".join(json.dumps(step) + "\n" for step in steps)
    )
    (recording / "manifest.json").write_text(
        json.dumps({"cameras": ["wrist"], "frames": len(steps)})
    )
    return recording


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[list[Path]] = []

    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        self.calls.append(paths)
        embedding = np.zeros(4, dtype=np.float32)
        embedding[len(self.calls) - 1] = 1.0
        return embedding


class StageEmbeddingTest(unittest.TestCase):
    def test_embeds_each_stage_once_and_reuses_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = write_recording(Path(temp_dir))
            encoder = FakeEncoder()
            factory_calls = 0

            def encoder_factory() -> FakeEncoder:
                nonlocal factory_calls
                factory_calls += 1
                return encoder

            first = embed_recording_stages(
                recording,
                camera="wrist",
                model_id="test/model",
                encoder_factory=encoder_factory,
            )
            second = embed_recording_stages(
                recording,
                camera="wrist",
                model_id="test/model",
                encoder_factory=encoder_factory,
            )

            self.assertEqual(factory_calls, 1)
            self.assertEqual(len(encoder.calls), 4)
            self.assertTrue(all(len(paths) == DEFAULT_FRAMES for paths in encoder.calls))
            self.assertTrue(all(not result.cached for result in first))
            self.assertTrue(all(result.cached for result in second))
            self.assertEqual(
                [result.stage for result in first], list(ObservationStage)
            )
            for result in second:
                self.assertTrue(result.path.is_file())
                self.assertEqual(np.load(result.path).shape, (4,))

    def test_rejects_a_recording_without_a_complete_stage_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = write_recording(Path(temp_dir), frames_per_stage=63)

            with self.assertRaisesRegex(
                ValueError, "approaching_cable has 63 frames but 64 are required"
            ):
                embed_recording_stages(
                    recording,
                    camera="wrist",
                    model_id="test/model",
                    encoder_factory=FakeEncoder,
                )


if __name__ == "__main__":
    unittest.main()
