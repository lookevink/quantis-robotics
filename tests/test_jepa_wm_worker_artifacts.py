from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from jepa_wm.objective_calibration import TaskProgressMargins
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
                TaskProgressMargins(5e-4, 1e-3, 0.005),
            )

            artifacts.write(manifest)

            self.assertEqual(ControlWorkerArtifacts.load(manifest), artifacts)
            self.assertNotIn(str(root), manifest.read_text())

    def test_legacy_calibrated_manifest_receives_default_progress_margins(self) -> None:
        root = Path("/tmp/quantis-worker-artifacts")
        payload = {
            "schema": "quantis.jepa_wm_control_worker_artifacts.v1",
            "proposal": "proposal.pth",
            "adapter": "adapter.pth",
            "calibration": "calibration.json",
            "calibrated": True,
        }

        artifacts = ControlWorkerArtifacts.from_dict(payload, relative_to=root)

        self.assertEqual(artifacts.progress_margins, TaskProgressMargins())

    def test_rejects_progress_margins_without_calibration(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "uncalibrated workers cannot define progress margins"
        ):
            ControlWorkerArtifacts(
                Path("/tmp/proposal.pth"),
                Path("/tmp/adapter.pth"),
                progress_margins=TaskProgressMargins(),
            )

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
