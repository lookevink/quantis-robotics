from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from jepa_wm.insertion_corpus import (
    FrozenInsertionAdapter,
    InsertionCorpusRoster,
    InsertionFreshEvaluationRoster,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MILESTONE = REPO_ROOT / "ops" / "jepa_wm_insertion_planner_milestone.sh"


class InsertionPlannerMilestoneTest(unittest.TestCase):
    def test_runs_both_fresh_seeds_and_backs_up_the_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls.log"
            roster = root / "fresh.json"
            InsertionFreshEvaluationRoster.create(
                "insertion-fresh-22600",
                22600,
                InsertionCorpusRoster.create("insertion-2600", 2600),
                FrozenInsertionAdapter("frozen-adapter", "a" * 64),
            ).write(roster)
            fake_aws = root / "aws"
            fake_aws.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'aws %s\\n' \"$*\" >> \"${CALLS}\"\n"
                "if [[ \"$1\" == jepa-wm-insertion-plan-summarize ]]; "
                "then exit 2; fi\n"
            )
            fake_aws.chmod(0o755)

            result = subprocess.run(
                [str(MILESTONE), str(roster), "frozen-proposal"],
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

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(
            invoked[:2],
            [
                "aws jepa-wm-insertion-plan-benchmark "
                "insertion-fresh-22600-held-00 frozen-adapter frozen-proposal "
                "sampled_readiness",
                "aws jepa-wm-insertion-plan-benchmark "
                "insertion-fresh-22600-held-01 frozen-adapter frozen-proposal "
                "sampled_readiness",
            ],
        )
        self.assertEqual(
            invoked[2],
            "aws jepa-wm-insertion-plan-summarize "
            f"{roster} frozen-proposal sampled_readiness",
        )
        self.assertEqual(invoked[-1], "aws backup-state")
        self.assertFalse(
            any("jepa-wm-insertion-adapt " in call for call in invoked)
        )

    def test_missing_roster_still_backs_up(self) -> None:
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
                [str(MILESTONE), str(root / "missing.json"), "proposal"],
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

    def test_dense_profile_is_forwarded_to_both_seeds_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls.log"
            roster = root / "fresh.json"
            InsertionFreshEvaluationRoster.create(
                "insertion-fresh-22600",
                22600,
                InsertionCorpusRoster.create("insertion-2600", 2600),
                FrozenInsertionAdapter("frozen-adapter", "a" * 64),
            ).write(roster)
            fake_aws = root / "aws"
            fake_aws.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'aws %s\\n' \"$*\" >> \"${CALLS}\"\n"
            )
            fake_aws.chmod(0o755)

            result = subprocess.run(
                [
                    str(MILESTONE),
                    str(roster),
                    "frozen-proposal",
                    "dense_execution",
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
            invoked = calls.read_text().splitlines()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            invoked[:2],
            [
                "aws jepa-wm-insertion-plan-benchmark "
                "insertion-fresh-22600-held-00 frozen-adapter frozen-proposal "
                "dense_execution",
                "aws jepa-wm-insertion-plan-benchmark "
                "insertion-fresh-22600-held-01 frozen-adapter frozen-proposal "
                "dense_execution",
            ],
        )
        self.assertEqual(
            invoked[2],
            "aws jepa-wm-insertion-plan-summarize "
            f"{roster} frozen-proposal dense_execution",
        )
        self.assertEqual(invoked[-1], "aws backup-state")


if __name__ == "__main__":
    unittest.main()
