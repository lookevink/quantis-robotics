from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MILESTONE = REPO_ROOT / "ops" / "jepa_wm_insertion_wm_milestone.sh"


class InsertionWorldModelMilestoneTest(unittest.TestCase):
    def test_runs_exact_adapter_train_two_seed_gate_and_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls.log"
            fake_corpus = root / "corpus"
            fake_corpus.write_text(
                "#!/usr/bin/env bash\n"
                'printf "corpus %s\\n" "$*" >> "${CALLS}"\n'
                'python3 -m jepa_wm.insertion_corpus create '
                '--experiment-id "$4" --base-seed "$3" '
                '--output "${INSERTION_CORPUS_ROSTER}"\n'
            )
            fake_aws = root / "aws"
            fake_aws.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf 'aws %s\n' "$*" >> "${CALLS}"
                    if [[ "$1" == jepa-wm-insertion-wm-summarize ]]; then
                      exit 2
                    fi
                    """
                )
            )
            fake_corpus.chmod(0o755)
            fake_aws.chmod(0o755)

            result = subprocess.run(
                [
                    str(MILESTONE),
                    "",
                    "2600",
                    "",
                ],
                cwd=root,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake_aws),
                    "CORPUS_WORKFLOW": str(fake_corpus),
                    "CALLS": str(calls),
                },
                text=True,
                capture_output=True,
            )

            invoked = calls.read_text().splitlines()
            training = ",".join(
                f"contact-insertion-v10-drive-slow-2600-train-{index:02d}"
                for index in range(12)
            )
            adapter = (
                "contact-insertion-v10-drive-slow-2600_"
                "insertion_adapter_s2016"
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(
                f"aws jepa-wm-insertion-adapt {training} 2016 {adapter} generic",
                invoked,
            )
            self.assertIn("aws jepa-wm-control-worker-stop", invoked)
            self.assertIn(
                f"aws jepa-wm-insertion-wm-eval "
                f"contact-insertion-v10-drive-slow-2600-held-00 {adapter}",
                invoked,
            )
            self.assertIn(
                f"aws jepa-wm-insertion-wm-summarize "
                f"contact-insertion-v10-drive-slow-2600-held-00,"
                f"contact-insertion-v10-drive-slow-2600-held-01 {adapter} "
                "contact-insertion-v10-drive-slow-2600 2600 generic",
                invoked,
            )
            self.assertEqual(invoked[-1], "aws backup-state")

    def test_goal_aligned_profile_uses_a_distinct_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            calls = root / "calls.log"
            fake_corpus = root / "corpus"
            fake_corpus.write_text(
                "#!/usr/bin/env bash\n"
                'python3 -m jepa_wm.insertion_corpus create '
                '--experiment-id "$4" --base-seed "$3" '
                '--output "${INSERTION_CORPUS_ROSTER}"\n'
            )
            fake_aws = root / "aws"
            fake_aws.write_text(
                "#!/usr/bin/env bash\n"
                "printf 'aws %s\\n' \"$*\" >> \"${CALLS}\"\n"
            )
            fake_corpus.chmod(0o755)
            fake_aws.chmod(0o755)

            result = subprocess.run(
                [
                    str(MILESTONE),
                    "1056",
                    "2600",
                    "contact-insertion-v9-2600",
                    "goal_aligned",
                ],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake_aws),
                    "CORPUS_WORKFLOW": str(fake_corpus),
                    "CALLS": str(calls),
                },
                text=True,
                capture_output=True,
            )
            relative_result = subprocess.run(
                [
                    str(MILESTONE),
                    "1056",
                    "2600",
                    "contact-insertion-v9-2600",
                    "goal_aligned_relative",
                ],
                cwd=REPO_ROOT,
                env={
                    **os.environ,
                    "AWS_WORKFLOW": str(fake_aws),
                    "CORPUS_WORKFLOW": str(fake_corpus),
                    "CALLS": str(calls),
                },
                text=True,
                capture_output=True,
            )
            invoked = calls.read_text().splitlines()

        adapter = (
            "contact-insertion-v9-2600_"
            "insertion_adapter_goal_aligned_s1056"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(relative_result.returncode, 0, relative_result.stderr)
        self.assertTrue(
            any(
                call.startswith("aws jepa-wm-insertion-adapt ")
                and call.endswith(f" 1056 {adapter} goal_aligned")
                for call in invoked
            )
        )
        relative_adapter = (
            "contact-insertion-v9-2600_"
            "insertion_adapter_goal_aligned_relative_s1056"
        )
        self.assertTrue(
            any(
                call.startswith("aws jepa-wm-insertion-adapt ")
                and call.endswith(
                    f" 1056 {relative_adapter} goal_aligned_relative"
                )
                for call in invoked
            )
        )
        self.assertIn(
            "aws jepa-wm-insertion-wm-summarize "
            "contact-insertion-v9-2600-held-00,"
            "contact-insertion-v9-2600-held-01 "
            f"{relative_adapter} contact-insertion-v9-2600 2600 "
            "goal_aligned_relative",
            invoked,
        )
        self.assertIn(
            "aws jepa-wm-insertion-wm-summarize "
            "contact-insertion-v9-2600-held-00,"
            "contact-insertion-v9-2600-held-01 "
            f"{adapter} contact-insertion-v9-2600 2600 goal_aligned",
            invoked,
        )
        self.assertTrue(
            any(
                call == (
                    "aws jepa-wm-insertion-wm-eval "
                    f"contact-insertion-v9-2600-held-00 {adapter}"
                )
                for call in invoked
            )
        )

    def test_finetune_profile_derives_the_generic_parent_path(self) -> None:
        from jepa_wm.insertion_adapter_profile import InsertionAdapterProfile

        output = Path(
            "/tmp/contact-insertion-v10-drive-slow-2600_"
            "insertion_adapter_goal_aligned_relative_finetune_s2016.pth"
        )

        initial = (
            InsertionAdapterProfile.GOAL_ALIGNED_RELATIVE_FINETUNE
            .descriptor.initial_adapter_path(output, 2016)
        )

        self.assertEqual(
            initial,
            Path(
                "/tmp/contact-insertion-v10-drive-slow-2600_"
                "insertion_adapter_s2016.pth"
            ).resolve(),
        )
        with self.assertRaisesRegex(ValueError, "exact training epoch"):
            (
                InsertionAdapterProfile.GOAL_ALIGNED_RELATIVE_FINETUNE
                .descriptor.initial_adapter_path(output, 1056)
            )


if __name__ == "__main__":
    unittest.main()
