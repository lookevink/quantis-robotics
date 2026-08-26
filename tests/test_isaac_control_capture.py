from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DROID_FPS
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from sim.control_identity import observation_id_for_session
from sim.isaac_control_capture import (
    requires_stable_insertion_capture,
    validated_control_reference,
)


class ControlCaptureContractTest(unittest.TestCase):
    def test_stabilizes_every_insertion_capture_that_can_lead_to_motion(self) -> None:
        for policy in (
            ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            ControlExecutionPolicy.INSERTION_RESET_TRIAL,
            ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT,
        ):
            with self.subTest(policy=policy):
                self.assertTrue(
                    requires_stable_insertion_capture(
                        policy,
                        insertion_control=True,
                        step_index=43,
                        context_index=43,
                    )
                )
        self.assertFalse(
            requires_stable_insertion_capture(
                ControlExecutionPolicy.DIRECT,
                insertion_control=True,
                step_index=43,
                context_index=43,
            )
        )
        self.assertFalse(
            requires_stable_insertion_capture(
                ControlExecutionPolicy.INSERTION_RESET_TRIAL,
                insertion_control=False,
                step_index=43,
                context_index=43,
            )
        )
        self.assertFalse(
            requires_stable_insertion_capture(
                ControlExecutionPolicy.INSERTION_RESET_TRIAL,
                insertion_control=True,
                step_index=42,
                context_index=43,
            )
        )

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
