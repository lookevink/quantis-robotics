from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.grasp_contract import GRASP_TASK_ID
from sim.control_context import load_control_context
from sim.exploration import DatasetSplit, build_exploration_plan


class RecordedControlContextTest(unittest.TestCase):
    def _recording(self, root: Path, *, task: str, frames: int = 102) -> Path:
        recording = root / "recording"
        recording.mkdir()
        (recording / "manifest.json").write_text(
            json.dumps({"metadata": {"task": task}})
        )
        with (recording / "steps.jsonl").open("w") as output:
            for index in range(frames):
                output.write(
                    json.dumps(
                        {
                            "index": index,
                            "arm_positions": [index / 1000.0] * 7,
                            "gripper_width_m": 0.018 if index >= 85 else 0.07,
                            "plug_attached": index >= 89,
                        }
                    )
                    + "\n"
                )
        return recording

    def test_allows_the_three_action_grasp_acquisition_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            steps = load_control_context(
                self._recording(Path(temp_dir), task=GRASP_TASK_ID),
                86,
                build_exploration_plan(11401, DatasetSplit.HELD_OUT),
            )

        self.assertEqual(len(steps), 87)
        self.assertFalse(steps[-1].plug_attached)

    def test_rejects_arbitrary_mid_motion_context_for_exploration(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "segment boundary"):
                load_control_context(
                    self._recording(Path(temp_dir), task="domain_exploration"),
                    5,
                    build_exploration_plan(11401, DatasetSplit.HELD_OUT),
                )

    def test_rejects_a_context_without_a_complete_action_horizon(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "three-action target"):
                load_control_context(
                    self._recording(Path(temp_dir), task=GRASP_TASK_ID, frames=90),
                    88,
                    build_exploration_plan(11401, DatasetSplit.HELD_OUT),
                )


if __name__ == "__main__":
    unittest.main()
