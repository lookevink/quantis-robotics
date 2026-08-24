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
                [str(MILESTONE), "500", "2600", "contact-insertion-v9-2600"],
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
            training = ",".join(
                f"contact-insertion-v9-2600-train-{index:02d}"
                for index in range(12)
            )
            adapter = "contact-insertion-v9-2600_insertion_adapter_s500"
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn(
                f"aws jepa-wm-insertion-adapt {training} 500 {adapter}",
                invoked,
            )
            self.assertIn("aws jepa-wm-control-worker-stop", invoked)
            self.assertIn(
                f"aws jepa-wm-insertion-wm-eval "
                f"contact-insertion-v9-2600-held-00 {adapter}",
                invoked,
            )
            self.assertIn(
                f"aws jepa-wm-insertion-wm-summarize "
                f"contact-insertion-v9-2600-held-00,"
                f"contact-insertion-v9-2600-held-01 {adapter} "
                "contact-insertion-v9-2600 2600",
                invoked,
            )
            self.assertEqual(invoked[-1], "aws backup-state")


if __name__ == "__main__":
    unittest.main()
