import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DROID_FPS
from sim.isaac_control_capture import _observation_id, _validated_reference


class ControlCaptureContractTest(unittest.TestCase):
    def _recording(self, root: Path, *, seed: int, split: str = "held_out") -> Path:
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
                    },
                }
            )
        )
        return recording

    def test_binds_observation_identity_to_the_session(self) -> None:
        self.assertGreater(_observation_id("session-a"), 0)
        self.assertNotEqual(_observation_id("session-a"), _observation_id("session-b"))

    def test_accepts_only_the_matching_held_out_droid_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=11400)
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                reference = _validated_reference(recording.name, 11400)

                self.assertEqual(reference.seed, 11400)

    def test_rejects_a_reference_from_another_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=11400)
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    _validated_reference(recording.name, 11401)


if __name__ == "__main__":
    unittest.main()
