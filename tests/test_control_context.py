from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
    INSERTION_TASK_ID,
)
from jepa_wm.task_windows import CONTACT_GRASP_PROPOSAL_WINDOW
from sim.control_context import ControlContextPurpose, load_control_context
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
                            "plug_position": [-0.1, 0.0, 1.0],
                            "plug_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
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

    def test_allows_exact_contact_insertion_command_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(
                Path(temp_dir),
                task=INSERTION_TASK_ID,
                frames=CONTACT_INSERTION_RECORDING.frame_count,
            )
            plan = build_exploration_plan(52600, DatasetSplit.HELD_OUT)
            contexts = (
                CONTACT_INSERTION_RECORDING.insertion_command_window.context_indices
            )

            first = load_control_context(recording, contexts[0], plan)
            last = load_control_context(recording, contexts[-1], plan)

            self.assertEqual(first[-1].index, contexts[0])
            self.assertEqual(last[-1].index, contexts[-1])

    def test_rejects_contact_insertion_contexts_outside_command_window(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(
                Path(temp_dir),
                task=INSERTION_TASK_ID,
                frames=CONTACT_INSERTION_RECORDING.frame_count,
            )
            plan = build_exploration_plan(52600, DatasetSplit.HELD_OUT)
            contexts = (
                CONTACT_INSERTION_RECORDING.insertion_command_window.context_indices
            )

            for context_index in (contexts[0] - 1, contexts[-1] + 1):
                with self.subTest(context_index=context_index):
                    with self.assertRaisesRegex(ValueError, "insertion command window"):
                        load_control_context(recording, context_index, plan)

    def test_contact_grasp_purpose_admits_only_the_canonical_pre_grasp_context(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(
                Path(temp_dir),
                task=INSERTION_TASK_ID,
                frames=CONTACT_INSERTION_RECORDING.frame_count,
            )
            plan = build_exploration_plan(52601, DatasetSplit.HELD_OUT)
            context_index = CONTACT_GRASP_PROPOSAL_WINDOW.start_index

            context = load_control_context(
                recording,
                context_index,
                plan,
                ControlContextPurpose.CONTACT_GRASP,
            )

            self.assertEqual(context[-1].index, context_index)
            for invalid_context in (
                context_index - 1,
                context_index + 1,
                CONTACT_INSERTION_RECORDING.insertion_command_window.context_indices[0],
            ):
                with self.subTest(context_index=invalid_context):
                    with self.assertRaisesRegex(ValueError, "canonical context"):
                        load_control_context(
                            recording,
                            invalid_context,
                            plan,
                            ControlContextPurpose.CONTACT_GRASP,
                        )

    def test_contact_grasp_purpose_rejects_a_non_insertion_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "requires an insertion recording"):
                load_control_context(
                    self._recording(Path(temp_dir), task=GRASP_TASK_ID),
                    69,
                    build_exploration_plan(12401, DatasetSplit.HELD_OUT),
                    ControlContextPurpose.CONTACT_GRASP,
                )


if __name__ == "__main__":
    unittest.main()
