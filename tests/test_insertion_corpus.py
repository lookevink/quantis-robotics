from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_SCRIPT = REPO_ROOT / "ops" / "jepa_wm_insertion_corpus.sh"


class InsertionCorpusWorkflowTest(unittest.TestCase):
    def test_records_exact_split_seeds_and_always_backs_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls"
            fake = root / "aws"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${CORPUS_CALLS}\"\n"
                "if [[ \"$1\" == jepa-wm-contact-insertion-status ]]; then printf 'missing\\n'; exit 0; fi\n"
                "if [[ \"$1\" == jepa-wm-contact-insertion-validate ]]; then exit 1; fi\n"
            )
            fake.chmod(0o755)
            result = subprocess.run(
                [str(CORPUS_SCRIPT), "12", "2", "3000", "insertion-test"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake),
                    "CORPUS_CALLS": str(calls),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            invoked = calls.read_text().splitlines()

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            invoked[:2],
            [
                "jepa-wm-contact-insertion-status insertion-test-train-00 train 3000",
                "demo-record-contact-insertion insertion-test-train-00 3000 train",
            ],
        )
        self.assertEqual(invoked[-1], "backup-state")

    def test_reuses_validated_recordings_and_finishes_with_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls"
            fake = root / "aws"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${CORPUS_CALLS}\"\n"
                "if [[ \"$1\" == jepa-wm-contact-insertion-status ]]; then printf 'valid\\n'; fi\n"
            )
            fake.chmod(0o755)
            result = subprocess.run(
                [str(CORPUS_SCRIPT), "12", "2", "3000", "insertion-test"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake),
                    "CORPUS_CALLS": str(calls),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            invoked = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            sum(call.startswith("jepa-wm-contact-insertion-status") for call in invoked),
            14,
        )
        self.assertFalse(any(call.startswith("demo-record") for call in invoked))
        self.assertEqual(invoked[-1], "backup-state")

    def test_quarantines_only_partial_recordings_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls"
            fake = root / "aws"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${CORPUS_CALLS}\"\n"
                "if [[ \"$1\" == jepa-wm-contact-insertion-status ]]; then printf 'partial\\n'; fi\n"
            )
            fake.chmod(0o755)
            result = subprocess.run(
                [str(CORPUS_SCRIPT), "12", "2", "3000", "insertion-test"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake),
                    "CORPUS_CALLS": str(calls),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            invoked = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            invoked[:4],
            [
                "jepa-wm-contact-insertion-status insertion-test-train-00 train 3000",
                "demo-quarantine-partial-recording insertion-test-train-00",
                "demo-record-contact-insertion insertion-test-train-00 3000 train",
                "jepa-wm-contact-insertion-validate insertion-test-train-00 train 3000",
            ],
        )

    def test_reconnects_to_running_recording_without_quarantine(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls"
            fake = root / "aws"
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${CORPUS_CALLS}\"\n"
                "if [[ \"$1\" == jepa-wm-contact-insertion-status ]]; then printf 'running\\n'; fi\n"
            )
            fake.chmod(0o755)
            result = subprocess.run(
                [str(CORPUS_SCRIPT), "12", "2", "3000", "insertion-test"],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake),
                    "CORPUS_CALLS": str(calls),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            invoked = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            invoked[:3],
            [
                "jepa-wm-contact-insertion-status insertion-test-train-00 train 3000",
                "demo-wait-recording insertion-test-train-00",
                "jepa-wm-contact-insertion-validate insertion-test-train-00 train 3000",
            ],
        )
        self.assertFalse(any("quarantine" in call for call in invoked))


if __name__ == "__main__":
    unittest.main()
