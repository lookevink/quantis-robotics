from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DroidPose
from sim.demo_sequence import Phase
from sim.recording import (
    RECORDING_SCHEMA,
    RecordingLabel,
    RecordingMoment,
    RecordingSafetyTelemetry,
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
                camera_resolutions={
                    "presentation": (1920, 1080),
                    "wrist": (1920, 1080),
                },
                metadata={
                    "dataset": "jepa_wm_domain",
                    "split": "train",
                    "seed": 1200,
                },
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
                    plug_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
                    plug_attached=False,
                    end_effector_pose=DroidPose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.125)),
                    end_effector_world_position=[0.2, -0.25, 1.4],
                    gripper_frame_world_position=[0.1, -0.25, 1.4],
                    simulation_time_seconds=1.0,
                    safety=RecordingSafetyTelemetry(
                        collision_detected=True,
                        contact_force_newtons=0.8,
                        arm_tracking_error_rad=0.001,
                        gripper_tracking_error_m=0.0002,
                    ),
                )
            )
            for path in writer.frame_paths().values():
                path.touch()
            writer.add_step(
                RecordingSnapshot(
                    phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                    stage=ObservationStage.APPROACHING_CABLE,
                    arm_positions=[0.2, 0.3],
                    gripper_width_m=0.03,
                    plug_position=[-0.02, -0.25, 1.32],
                    plug_orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
                    plug_attached=False,
                    end_effector_pose=DroidPose((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.625)),
                    end_effector_world_position=[0.19, -0.25, 1.4],
                    gripper_frame_world_position=[0.09, -0.25, 1.4],
                    simulation_time_seconds=1.125,
                )
            )
            self.assertEqual(writer.frame_count, 2)
            self.assertEqual(
                writer.stage_frame_count(ObservationStage.APPROACHING_CABLE), 2
            )
            output = writer.finish()

            manifest = json.loads((output / "manifest.json").read_text())
            steps = [
                json.loads(line)
                for line in (output / "steps.jsonl").read_text().splitlines()
            ]
            step = steps[0]
            self.assertEqual(
                [step["simulation_time_seconds"] for step in steps],
                [1.0, 1.125],
            )
            self.assertEqual(manifest["schema"], RECORDING_SCHEMA)
            self.assertTrue(step["collision_detected"])
            self.assertEqual(step["contact_force_newtons"], 0.8)
            self.assertEqual(step["arm_tracking_error_rad"], 0.001)
            self.assertEqual(step["gripper_tracking_error_m"], 0.0002)
            self.assertEqual(manifest["fps"], 8)
            self.assertEqual(manifest["frames"], 2)
            self.assertEqual(
                manifest["metadata"],
                {
                    "dataset": "jepa_wm_domain",
                    "split": "train",
                    "seed": 1200,
                },
            )
            self.assertEqual(
                manifest["resolutions"],
                {
                    "presentation": [1920, 1080],
                    "wrist": [1920, 1080],
                },
            )
            self.assertEqual(manifest["stage_frames"], {"approaching_cable": 2})
            self.assertEqual(
                manifest["action"],
                {
                    "format": "droid_base_delta_pose_v2",
                    "dimensions": 7,
                    "field": "action_from_previous",
                    "pose_field": "end_effector_pose",
                    "coordinate_frame": "robot_base",
                },
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
            self.assertEqual(step["plug_orientation_wxyz"], [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(
                step["end_effector_world_position"],
                [0.2, -0.25, 1.4],
            )
            self.assertEqual(
                step["gripper_frame_world_position"],
                [0.1, -0.25, 1.4],
            )
            self.assertIsNone(step["action_from_previous"])
            np.testing.assert_allclose(
                steps[1]["action_from_previous"],
                [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                atol=1e-7,
            )

    def test_rejects_an_unsafe_recording_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "recording_id"):
                RecordingWriter(
                    Path(temp_dir),
                    recording_id="../escape",
                    fps=8,
                    camera_resolutions={
                        "presentation": (1920, 1080),
                        "wrist": (1920, 1080),
                    },
                )


if __name__ == "__main__":
    unittest.main()
