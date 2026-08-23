from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ControlRolloutShellTest(unittest.TestCase):
    def test_finalizes_a_report_when_the_first_control_step_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_control_rollout.sh", ops)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            (ops / "run_control_step.sh").write_text(
                "#!/usr/bin/env bash\nexit 7\n"
            )
            (ops / "jepa_wm.sh").write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"${ROLLOUT_LOG}\"\n"
            )
            log = home / "report.log"

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_control_rollout.sh"),
                    "rollout-test",
                    "held-reference",
                    "11401",
                    "3",
                    "proposal-test",
                ],
                env={**os.environ, "HOME": str(home), "ROLLOUT_LOG": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 7)
            call = log.read_text()
            self.assertIn("control-rollout-report", call)
            self.assertIn("--requested-steps 3", call)
            self.assertIn("--reference held-reference", call)
            self.assertIn(
                "--orchestration-failure initial_control_step:exit_7",
                call,
            )


if __name__ == "__main__":
    unittest.main()
