from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ControlRolloutShellTest(unittest.TestCase):
    def test_insertion_safety_check_never_calls_the_execution_entrypoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_insertion_safety_check.sh", ops)
            log = home / "calls.log"
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
isaac_server_call() { printf '%s\\n' "$1" >> "${CALLS}"; }
capture_and_respond_control_session() {
  [[ "$6" == insertion_safety_evaluation ]] || return 8
  printf 'capture_control_observation %s\\n' "$2" >> "${CALLS}"
  printf 'respond %s\\n' "$2" >> "${CALLS}"
}
"""
            )

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_insertion_safety_check.sh"),
                    "safety-session",
                    "insertion-held-00",
                    "52600",
                    "worker-test",
                    "43",
                ],
                env={
                    **os.environ,
                    "HOME": str(home),
                    "CALLS": str(log),
                },
                text=True,
                capture_output=True,
                check=False,
            )

            calls = log.read_text()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("capture_control_observation", calls)
            self.assertIn("respond safety-session", calls)
            self.assertIn("evaluate_direct_insertion_candidate", calls)
            self.assertNotIn("apply_control_response", calls)

    def test_control_step_delegates_to_the_shared_capture_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_control_step.sh", ops)
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
capture_and_respond_control_session() {
  [[ "$1" == "${HOME}/quantis-robotics" ]] || return 9
}
isaac_server_call() { return 0; }
capture_shadow_control_evidence() { return 0; }
"""
            )

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_control_step.sh"),
                    "session-test",
                    "held-reference",
                    "11401",
                    "worker-test",
                ],
                env={**os.environ, "HOME": str(home)},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)

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
            venv = home / ".venvs" / "quantis-jepa-wm" / "bin"
            venv.mkdir(parents=True)
            (venv / "python").write_text(
                "#!/usr/bin/env bash\nprintf 'proposal-test\\n'\n"
            )
            (venv / "python").chmod(0o755)
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
