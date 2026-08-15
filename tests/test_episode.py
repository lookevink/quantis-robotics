from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sim.episode import EpisodeWriter, SCHEMA_VERSION


class EpisodeWriterTest(unittest.TestCase):
    def test_writes_synchronized_manifest_and_steps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir) / "episode-1"
            frame = episode_dir / "rgb" / "frame.png"
            writer = EpisodeWriter(
                episode_dir,
                task="test-task",
                robot="test-robot",
                action_labels=["dx", "gripper"],
                fps=4.0,
            )
            frame.parent.mkdir()
            frame.touch()
            writer.add_step(frame=frame, action=[0.1, 1], state={"progress": 0.5})
            writer.finish(success=True)

            manifest = json.loads((episode_dir / "episode.json").read_text())
            step = json.loads((episode_dir / "steps.jsonl").read_text())
            self.assertEqual(manifest["schema"], SCHEMA_VERSION)
            self.assertEqual(manifest["frames"], 1)
            self.assertEqual(step["frame"], "rgb/frame.png")
            self.assertEqual(step["action"], [0.1, 1.0])

    def test_rejects_action_with_wrong_dimension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            episode_dir = Path(temp_dir) / "episode-1"
            writer = EpisodeWriter(
                episode_dir,
                task="test-task",
                robot="test-robot",
                action_labels=["dx", "gripper"],
                fps=4.0,
            )
            frame = episode_dir / "frame.png"
            frame.touch()
            with self.assertRaisesRegex(ValueError, "expected 2 action values"):
                writer.add_step(frame=frame, action=[0.1], state={})


if __name__ == "__main__":
    unittest.main()
