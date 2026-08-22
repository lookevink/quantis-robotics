from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jepa.contract import ObservationStage
from sim.demo_sequence import Phase
from sim.recording import (
    RECORDING_SCHEMA,
    RecordingLabel,
    RecordingMoment,
    RecordingSnapshot,
    RecordingWriter,
)


class RecordingWriterTest(unittest.TestCase):
    def test_composes_recording_labels_from_task_phases_and_moments(self) -> None:
        labels = (
            (RecordingLabel(RecordingMoment.INITIAL), "initial"),
            (RecordingLabel(RecordingMoment.MOTION, Phase.READY), "ready"),
            (RecordingLabel(RecordingMoment.SETTLE, Phase.READY), "ready_settle"),
            (
                RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP),
                "grasp_attached",
            ),
        )

        for label, expected in labels:
            with self.subTest(expected=expected):
                self.assertEqual(label.value, expected)

        with self.assertRaises(ValueError):
            RecordingLabel(RecordingMoment.SETTLE)

    def test_writes_synchronized_frames_steps_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            writer = RecordingWriter(
                Path(temp_dir),
                recording_id="demo-20260822T031500Z",
                fps=8,
                cameras=("presentation", "wrist"),
            )
            frame_paths = writer.frame_paths()
            for path in frame_paths.values():
                path.touch()

            self.assertEqual(writer.frame_count, 0)
            self.assertEqual(
                writer.stage_frame_count(ObservationStage.APPROACHING_CABLE), 0
            )
            writer.add_step(
                RecordingSnapshot(
                    phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                    stage=ObservationStage.APPROACHING_CABLE,
                    arm_positions=[0.1, 0.2],
                    gripper_width_m=0.07,
                    plug_position=[-0.02, -0.25, 1.32],
                    plug_attached=False,
                )
            )
            self.assertEqual(writer.frame_count, 1)
            self.assertEqual(
                writer.stage_frame_count(ObservationStage.APPROACHING_CABLE), 1
            )
            output = writer.finish()

            manifest = json.loads((output / "manifest.json").read_text())
            step = json.loads((output / "steps.jsonl").read_text())
            self.assertEqual(manifest["schema"], RECORDING_SCHEMA)
            self.assertEqual(manifest["fps"], 8)
            self.assertEqual(manifest["frames"], 1)
            self.assertEqual(
                manifest["stage_frames"], {"approaching_cable": 1}
            )
            self.assertEqual(
                manifest["videos"],
                {
                    "presentation": "presentation.mp4",
                    "wrist": "wrist.mp4",
                },
            )
            self.assertEqual(step["index"], 0)
            self.assertEqual(step["timestamp_seconds"], 0.0)
            self.assertEqual(
                step["frames"],
                {
                    "presentation": "presentation/frame_000000.png",
                    "wrist": "wrist/frame_000000.png",
                },
            )
            self.assertEqual(step["phase"], "ready")
            self.assertEqual(step["stage"], "approaching_cable")
            self.assertFalse(step["plug_attached"])

    def test_rejects_an_unsafe_recording_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "recording_id"):
                RecordingWriter(
                    Path(temp_dir),
                    recording_id="../escape",
                    fps=8,
                    cameras=("presentation", "wrist"),
                )


if __name__ == "__main__":
    unittest.main()
