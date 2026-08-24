from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jepa_wm.calibration_sessions import validated_calibration_recording
from jepa_wm.control_policy import ControlExecutionPolicy
from sim.control_session import ControlSessionState
from sim.exploration import DOMAIN_DATASET_ID, DatasetSplit


class CalibrationSessionEvidenceTest(unittest.TestCase):
    @staticmethod
    def _state(policy: ControlExecutionPolicy) -> ControlSessionState:
        return ControlSessionState(
            "calibration-session",
            "training-recording",
            1400,
            "control-calibration-session",
            (0.0,) * 7,
            False,
            0.0,
            execution_policy=policy,
        )

    @staticmethod
    def _write_recording(root: Path, split: DatasetSplit) -> None:
        recording = root / "recordings" / "training-recording"
        recording.mkdir(parents=True)
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                    "recording_id": "training-recording",
                    "metadata": {
                        "dataset": DOMAIN_DATASET_ID,
                        "split": split.value,
                        "seed": 1400,
                    },
                }
            )
        )

    def test_accepts_only_training_collection_policy_and_split(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, DatasetSplit.TRAIN)

            recording = validated_calibration_recording(
                root,
                self._state(ControlExecutionPolicy.CALIBRATION_COLLECTION),
            )

            self.assertEqual(recording.split, DatasetSplit.TRAIN)

    def test_rejects_held_out_recording_for_calibration_collection(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, DatasetSplit.HELD_OUT)

            with self.assertRaisesRegex(ValueError, "expected 'train'"):
                validated_calibration_recording(
                    root,
                    self._state(ControlExecutionPolicy.CALIBRATION_COLLECTION),
                )

    def test_rejects_normal_direct_policy_even_for_training_recording(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_recording(root, DatasetSplit.TRAIN)

            with self.assertRaisesRegex(ValueError, "calibration_collection"):
                validated_calibration_recording(
                    root,
                    self._state(ControlExecutionPolicy.DIRECT),
                )


if __name__ == "__main__":
    unittest.main()
