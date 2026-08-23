from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_wm.worker_artifacts import ControlWorkerArtifacts


class ControlWorkerArtifactsTest(unittest.TestCase):
    def test_round_trips_one_portable_worker_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            manifest = root / "calibrated.worker.json"
            artifacts = ControlWorkerArtifacts(
                root / "proposal.pth",
                root / "adapter.pth",
                root / "calibration.json",
            )

            artifacts.write(manifest)

            self.assertEqual(ControlWorkerArtifacts.load(manifest), artifacts)
            self.assertNotIn(str(root), manifest.read_text())

    def test_rejects_a_tampered_calibrated_claim(self) -> None:
        root = Path("/tmp/quantis-worker-artifacts")
        payload = ControlWorkerArtifacts(
            root / "proposal.pth", root / "adapter.pth"
        ).to_dict()
        payload["calibrated"] = True

        with self.assertRaisesRegex(ValueError, "claims"):
            ControlWorkerArtifacts.from_dict(payload, relative_to=root)


if __name__ == "__main__":
    unittest.main()
