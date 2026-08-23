from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jepa_wm.trajectory import TransitionWindow, load_transitions


class RecordedTrajectoryTest(unittest.TestCase):
    def test_loads_consecutive_frames_with_the_action_that_connected_them(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir)
            wrist = recording / "wrist"
            wrist.mkdir()
            for index in range(3):
                (wrist / f"frame_{index:06d}.png").touch()
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "quantis.demo_recording.v2",
                        "frames": 3,
                        "cameras": ["wrist"],
                        "action": {
                            "format": "droid_delta_pose_v1",
                            "dimensions": 7,
                            "field": "action_from_previous",
                            "pose_field": "end_effector_pose",
                        },
                    }
                )
            )
            steps = [
                {
                    "index": index,
                    "frames": {"wrist": f"wrist/frame_{index:06d}.png"},
                    "action_from_previous": (
                        None if index == 0 else [index / 100, 0, 0, 0, 0, 0, 0]
                    ),
                    "end_effector_pose": [index / 100, 0, 0, 0, 0, 0, 0.5],
                }
                for index in range(3)
            ]
            (recording / "steps.jsonl").write_text(
                "".join(json.dumps(step) + "\n" for step in steps)
            )

            transitions = load_transitions(
                recording,
                camera="wrist",
                window=TransitionWindow(start_index=0, count=1, stride=2),
            )

            self.assertEqual(len(transitions), 1)
            transition = transitions[0]
            self.assertEqual(transition.current_index, 0)
            self.assertEqual(transition.next_index, 2)
            self.assertEqual(transition.action.values[0], 0.02)
            self.assertEqual(
                transition.current_frame, (wrist / "frame_000000.png").resolve()
            )
            self.assertEqual(
                transition.next_frame, (wrist / "frame_000002.png").resolve()
            )

    def test_rejects_a_recording_without_droid_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir)
            (recording / "manifest.json").write_text(
                json.dumps({"schema": "quantis.demo_recording.v1", "frames": 2})
            )
            (recording / "steps.jsonl").write_text("")

            with self.assertRaisesRegex(ValueError, "DROID-compatible actions"):
                load_transitions(
                    recording,
                    camera="wrist",
                    window=TransitionWindow(start_index=0, count=1, stride=1),
                )


if __name__ == "__main__":
    unittest.main()
