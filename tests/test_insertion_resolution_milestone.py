from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from jepa_wm.control_resolution_profile import (
    CONTROL_RESOLUTION_CONTEXTS,
    CONTROL_RESOLUTION_LOADS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MILESTONE = REPO_ROOT / "ops" / "jepa_wm_insertion_resolution_milestone.sh"


class InsertionResolutionMilestoneTest(unittest.TestCase):
    def test_runs_three_poses_with_and_without_load_and_backs_up(self) -> None:
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
                [str(MILESTONE), "insertion-fresh-held-00", "52600"],
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

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            invoked[:-1],
            [
                "aws jepa-wm-insertion-resolution "
                f"insertion-fresh-held-00 52600 {context} {load}"
                for load in (load.value for load in CONTROL_RESOLUTION_LOADS)
                for context in CONTROL_RESOLUTION_CONTEXTS
            ],
        )
        self.assertEqual(invoked[-1], "aws backup-state")

    def test_early_validation_failure_still_backs_up(self) -> None:
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
                [str(MILESTONE), "../unsafe", "52600"],
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
        self.assertEqual(invoked, ["aws backup-state"])


if __name__ == "__main__":
    unittest.main()
