from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MILESTONE = REPO_ROOT / "ops" / "jepa_wm_insertion_wm_fresh_milestone.sh"


class InsertionWorldModelFreshMilestoneTest(unittest.TestCase):
    def test_requires_the_current_frozen_adapter_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls.log"
            fake_aws = root / "aws"
            fake_aws.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'aws %s\\n' \"$*\" >> \"${CALLS}\"\n"
            )
            fake_aws.chmod(0o755)

            result = subprocess.run(
                [str(MILESTONE)],
                cwd=root,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake_aws),
                    "CALLS": str(calls),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            invoked = calls.read_text().splitlines()

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen adapter fingerprint must be supplied", result.stderr)
        self.assertEqual(invoked, ["aws backup-state"])

    def test_early_profile_failure_still_backs_up(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls.log"
            fake_aws = root / "aws"
            fake_aws.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'aws %s\\n' \"$*\" >> \"${CALLS}\"\n"
            )
            fake_aws.chmod(0o755)

            result = subprocess.run(
                [
                    str(MILESTONE),
                    "2600",
                    "contact-insertion-v9-2600",
                    "22600",
                    "contact-insertion-v9-2600-fresh-22600",
                    "adapter",
                    "invalid-profile",
                    "a" * 64,
                ],
                cwd=root,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake_aws),
                    "CALLS": str(calls),
                },
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls.read_text().splitlines(), ["aws backup-state"])

    def test_captures_only_fresh_seeds_and_never_retrains_the_frozen_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls.log"
            fake_capture = root / "capture"
            fake_capture.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'capture %s\\n' \"$*\" >> \"${CALLS}\"\n"
            )
            fake_aws = root / "aws"
            fake_aws.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'aws %s\\n' \"$*\" >> \"${CALLS}\"\n"
                "if [[ \"$1\" == jepa-wm-insertion-wm-fresh-summarize ]]; "
                "then exit 2; fi\n"
            )
            fake_capture.chmod(0o755)
            fake_aws.chmod(0o755)

            result = subprocess.run(
                [str(MILESTONE), "", "", "", "", "", "", "a" * 64],
                cwd=root,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake_aws),
                    "CONTACT_INSERTION_CAPTURE_WORKFLOW": str(fake_capture),
                    "CALLS": str(calls),
                },
                text=True,
                capture_output=True,
                check=False,
            )
            invoked = calls.read_text().splitlines()

        adapter = (
            "contact-insertion-v10-drive-slow-2600_"
            "insertion_adapter_goal_aligned_relative_finetune_s2016"
        )
        evaluation = "contact-insertion-v10-drive-slow-2600-fresh-22600"
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(
            invoked[:2],
            [
                f"capture {evaluation}-held-00 22600 held_out",
                f"capture {evaluation}-held-01 22601 held_out",
            ],
        )
        self.assertIn("aws jepa-wm-control-worker-stop", invoked)
        self.assertIn(
            f"aws jepa-wm-insertion-wm-eval {evaluation}-held-00 {adapter}",
            invoked,
        )
        self.assertIn(
            "aws jepa-wm-insertion-wm-fresh-summarize "
            f"/tmp/{evaluation}_insertion_fresh_evaluation.json "
            "goal_aligned_relative_finetune",
            invoked,
        )
        self.assertFalse(
            any("jepa-wm-insertion-adapt " in call for call in invoked)
        )
        self.assertEqual(invoked[-1], "aws backup-state")


if __name__ == "__main__":
    unittest.main()
