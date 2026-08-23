from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jepa_wm.trajectory import RolloutProtocol, RolloutWindow, load_rollouts


class RecordedTrajectoryTest(unittest.TestCase):
    def test_loads_native_context_actions_and_terminal_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir)
            wrist = recording / "wrist"
            wrist.mkdir()
            for index in range(6):
                (wrist / f"frame_{index:06d}.png").touch()
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "quantis.demo_recording.v3",
                        "frames": 6,
                        "cameras": ["wrist"],
                        "action": {
                            "format": "droid_base_delta_pose_v2",
                            "dimensions": 7,
                            "field": "action_from_previous",
                            "pose_field": "end_effector_pose",
                            "coordinate_frame": "robot_base",
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
                for index in range(6)
            ]
            (recording / "steps.jsonl").write_text(
                "".join(json.dumps(step) + "\n" for step in steps)
            )

            rollouts = load_rollouts(
                recording,
                camera="wrist",
                protocol=RolloutProtocol(context_frames=1, action_horizon=3),
            )

            rollout = RolloutWindow(start_index=0, count=1, stride=1).select(rollouts)[
                0
            ]
            self.assertSequenceEqual(
                tuple(frame.index for frame in rollout.context),
                (0,),
            )
            self.assertEqual(rollout.target.index, 3)
            self.assertEqual(
                rollout.context_pose.values,
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5),
            )
            self.assertEqual(rollout.previous_action.values, (0.0,) * 7)
            next_rollout = RolloutWindow(start_index=1, count=1, stride=1).select(
                rollouts
            )[0]
            self.assertAlmostEqual(next_rollout.previous_action.values[0], 0.01)
            self.assertTrue(
                all(abs(action.values[0] - 0.01) < 1e-12 for action in rollout.actions)
            )
            self.assertEqual(
                tuple(frame.path for frame in rollout.context),
                ((wrist / "frame_000000.png").resolve(),),
            )
            self.assertEqual(
                rollout.target.path, (wrist / "frame_000003.png").resolve()
            )

    def test_rejects_a_recording_without_droid_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir)
            (recording / "manifest.json").write_text(
                json.dumps({"schema": "quantis.demo_recording.v1", "frames": 2})
            )
            (recording / "steps.jsonl").write_text("")

            with self.assertRaisesRegex(ValueError, "DROID-compatible actions"):
                load_rollouts(
                    recording,
                    camera="wrist",
                    protocol=RolloutProtocol(context_frames=1, action_horizon=3),
                )


if __name__ == "__main__":
    unittest.main()
