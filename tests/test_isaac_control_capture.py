from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DROID_FPS
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from sim.control_identity import observation_id_for_session
from sim.isaac_control_capture import (
    capture_and_pause_control_state,
    validated_control_reference,
)


class ControlCaptureContractTest(unittest.TestCase):
    def test_reads_physics_backed_state_before_pausing_timeline(self) -> None:
        timeline = Mock()
        actuators = Mock()
        actuators.actual_command.return_value = SimpleNamespace(
            arm_positions=(0.0,) * 7,
            gripper_width_m=0.018,
        )
        attachment = Mock(attached=True)

        def world_pose():
            if timeline.pause.called:
                raise RuntimeError("physics tensor backend is unavailable after pause")
            return ((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))

        attachment.world_pose.side_effect = world_pose
        with patch(
            "sim.isaac_control_capture.read_control_contact",
            return_value=(False, 0.0),
        ):
            state = capture_and_pause_control_state(
                timeline, actuators, attachment, Mock()
            )

        self.assertEqual(state.plug_position, (0.1, 0.2, 0.3))
        timeline.pause.assert_called_once_with()

    def _recording(
        self,
        root: Path,
        *,
        seed: int,
        split: str = "held_out",
        task: str | None = None,
    ) -> Path:
        recording = root / "held-reference"
        recording.mkdir()
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                    "recording_id": recording.name,
                    "fps": DROID_FPS,
                    "cameras": ["wrist"],
                    "action": ACTION_RECORDING_CONTRACT.to_dict(),
                    "metadata": {
                        "dataset": "jepa_wm_domain_v1",
                        "split": split,
                        "seed": seed,
                        **({"task": task} if task is not None else {}),
                    },
                }
            )
        )
        return recording

    def test_binds_observation_identity_to_the_session(self) -> None:
        self.assertGreater(observation_id_for_session("session-a"), 0)
        self.assertNotEqual(
            observation_id_for_session("session-a"),
            observation_id_for_session("session-b"),
        )

    def test_accepts_only_the_matching_held_out_droid_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=11400)
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                reference = validated_control_reference(
                    recording.name,
                    11400,
                    ControlExecutionPolicy.DIRECT,
                )

                self.assertEqual(reference.seed, 11400)

    def test_rejects_a_reference_from_another_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=11400)
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    validated_control_reference(
                        recording.name,
                        11401,
                        ControlExecutionPolicy.DIRECT,
                    )

    def test_training_references_require_the_calibration_collection_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=1400, split="train")
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                reference = validated_control_reference(
                    recording.name,
                    1400,
                    ControlExecutionPolicy.CALIBRATION_COLLECTION,
                )
                self.assertEqual(reference.split.value, "train")
                with self.assertRaisesRegex(ValueError, "expected 'held_out'"):
                    validated_control_reference(
                        recording.name,
                        1400,
                        ControlExecutionPolicy.DIRECT,
                    )

    def test_contact_insertion_reference_requires_strict_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(
                root, seed=52600, task=INSERTION_TASK_ID
            )
            with (
                patch("sim.isaac_control_capture.RECORDING_ROOT", root),
                patch(
                    "sim.isaac_control_capture.ContactInsertionEvidence.from_recording"
                ) as validate,
            ):
                validated_control_reference(
                    recording.name,
                    52600,
                    ControlExecutionPolicy.DIRECT,
                )

            validate.assert_called_once_with(
                recording.resolve(), expected_split="held_out"
            )


if __name__ == "__main__":
    unittest.main()
