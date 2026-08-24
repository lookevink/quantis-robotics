from __future__ import annotations

import json
from pathlib import Path
import tempfile
from time import time
import unittest

from jepa_wm.contact_insertion_status_cli import recording_status


class ContactInsertionStatusTest(unittest.TestCase):
    def test_running_job_takes_precedence_over_manifestless_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording = root / "recordings" / "run"
            recording.mkdir(parents=True)
            jobs = root / "recording_jobs"
            jobs.mkdir()
            (jobs / "run.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "run_id": "active",
                        "heartbeat_unix_seconds": time(),
                    }
                )
            )

            status = recording_status(recording, "train", 2600)

        self.assertEqual(status, "running")

    def test_stale_running_job_becomes_quarantinable_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording = root / "recordings" / "run"
            recording.mkdir(parents=True)
            jobs = root / "recording_jobs"
            jobs.mkdir()
            (jobs / "run.json").write_text(
                json.dumps(
                    {
                        "status": "running",
                        "run_id": "orphaned",
                        "heartbeat_unix_seconds": 0.0,
                    }
                )
            )

            status = recording_status(recording, "train", 2600)

        self.assertEqual(status, "partial")

    def test_only_terminal_error_is_quarantinable_partial(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording = root / "recordings" / "run"
            recording.mkdir(parents=True)
            jobs = root / "recording_jobs"
            jobs.mkdir()
            (jobs / "run.json").write_text(json.dumps({"status": "error"}))

            status = recording_status(recording, "train", 2600)

        self.assertEqual(status, "partial")

    def test_manifestless_directory_without_terminal_job_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = Path(temporary_directory) / "recordings" / "run"
            recording.mkdir(parents=True)

            status = recording_status(recording, "train", 2600)

        self.assertEqual(status, "invalid")

    def test_complete_job_without_artifact_is_invalid_not_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            recording = root / "recordings" / "run"
            jobs = root / "recording_jobs"
            jobs.mkdir(parents=True)
            (jobs / "run.json").write_text(json.dumps({"status": "complete"}))

            status = recording_status(recording, "train", 2600)

        self.assertEqual(status, "invalid")


if __name__ == "__main__":
    unittest.main()
