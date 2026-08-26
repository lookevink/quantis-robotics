from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class ControlRolloutShellTest(unittest.TestCase):
    def test_insertion_two_step_requires_action_one_before_followup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_insertion_two_step_trial.sh", ops)
            log = home / "calls.log"
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
is_safe_identifier() { return 0; }
require_nonnegative_integer() { return 0; }
require_positive_integer() { return 0; }
isaac_server_call() {
  printf 'verify %s\n' "$1" >> "${CALLS}"
  if [[ "$1" == *"verify_insertion_followup_source"* ]]; then
    [[ "${VERIFY_FIRST_FAIL:-0}" != "1" ]]
  else
    [[ "${VERIFY_FINAL_FAIL:-0}" != "1" ]]
  fi
}
"""
            )
            for name in (
                "run_insertion_safety_check.sh",
                "run_insertion_reset_trial.sh",
                "run_insertion_followup_trial.sh",
            ):
                (ops / name).write_text(
                    "#!/usr/bin/env bash\nprintf '%s %s\\n' "
                    f"'{name}' \"$*\" >> \"${{CALLS}}\"\n"
                )

            arguments = (
                "two-step-run",
                "insertion-held-00",
                "52600",
                "worker-test",
                "43",
            )
            result = subprocess.run(
                ["bash", str(ops / "run_insertion_two_step_trial.sh"), *arguments],
                env={**os.environ, "HOME": str(home), "CALLS": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            calls = log.read_text().splitlines()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                [line.split()[0] for line in calls],
                [
                    "run_insertion_safety_check.sh",
                    "run_insertion_reset_trial.sh",
                    "verify",
                    "run_insertion_followup_trial.sh",
                    "verify",
                ],
            )
            self.assertIn(
                "verify_insertion_followup_source('two-step-run-action1')",
                calls[2],
            )
            self.assertIn(
                "verify_insertion_two_step_result('two-step-run-action1','two-step-run-action2','insertion-held-00',52600)",
                calls[4],
            )

            log.write_text("")
            failed = subprocess.run(
                ["bash", str(ops / "run_insertion_two_step_trial.sh"), *arguments],
                env={
                    **os.environ,
                    "HOME": str(home),
                    "CALLS": str(log),
                    "VERIFY_FIRST_FAIL": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertNotIn(
                "run_insertion_followup_trial.sh",
                log.read_text(),
            )

            log.write_text("")
            failed_final = subprocess.run(
                ["bash", str(ops / "run_insertion_two_step_trial.sh"), *arguments],
                env={
                    **os.environ,
                    "HOME": str(home),
                    "CALLS": str(log),
                    "VERIFY_FINAL_FAIL": "1",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(failed_final.returncode, 0)
            self.assertIn("run_insertion_followup_trial.sh", log.read_text())
            self.assertIn("verify_insertion_two_step_result", log.read_text())

    def test_insertion_followup_runs_safety_before_one_non_reset_apply(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            log = home / "calls.log"
            (ops / "jepa_wm.sh").write_text(
                "#!/usr/bin/env bash\nprintf 'report %s\\n' \"$*\" >> \"${CALLS}\"\n"
            )
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() { printf '%s|%s\n' "$1" "${3:-false}" >> "${CALLS}"; }
respond_to_control_session() { printf 'respond %s %s\n' "$2" "$3" >> "${CALLS}"; }
run_insertion_followup_trial \
  "${HOME}/quantis-robotics" followup-safety followup-trial previous-trial \
  insertion-held-00 52600 proposal-test
"""
            )

            result = subprocess.run(
                ["bash", str(runner)],
                env={**os.environ, "HOME": str(home), "CALLS": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            calls = log.read_text().splitlines()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("capture_followup_observation", calls[0])
            self.assertTrue(calls[0].endswith("|true"))
            self.assertEqual(calls[1], "respond followup-safety insertion_safety_evaluation")
            self.assertIn("evaluate_direct_insertion_candidate", calls[2])
            self.assertIn("prepare_insertion_trial_source", calls[3])
            self.assertIn("persist_insertion_followup_response", calls[4])
            self.assertIn("apply_control_response", calls[5])
            self.assertNotIn("capture_control_observation", "\n".join(calls))
            self.assertIn("--sessions previous-trial,followup-trial", calls[6])
            self.assertIn("--requested-steps 2", calls[6])

    def test_shared_reset_trial_reports_a_typed_preflight_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            log = home / "calls.log"
            (ops / "jepa_wm.sh").write_text(
                "#!/usr/bin/env bash\nprintf '%s\n' \"$*\" >> \"${CALLS}\"\n"
            )
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() { return 7; }
run_reset_trial_control_session \
  "${HOME}/quantis-robotics" trial-session insertion-held-00 52600 \
  proposal-test insertion_reset_trial safety-session 43 900 \
  prepare_insertion_trial_source persist_insertion_trial_response
"""
            )

            result = subprocess.run(
                ["bash", str(runner)],
                env={**os.environ, "HOME": str(home), "CALLS": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 7, result.stderr)
            self.assertIn(
                "--orchestration-failure reset_trial_source_preflight:exit_7",
                log.read_text(),
            )

    def test_shared_reset_trial_flow_preflights_binds_applies_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            log = home / "calls.log"
            (ops / "jepa_wm.sh").write_text(
                "#!/usr/bin/env bash\nprintf 'report %s\n' \"$*\" >> \"${CALLS}\"\n"
            )
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() { printf '%s\n' "$1" >> "${CALLS}"; }
run_reset_trial_control_session \
  "${HOME}/quantis-robotics" trial-session insertion-held-00 52600 \
  proposal-test insertion_reset_trial safety-session 43 900 \
  prepare_insertion_trial_source persist_insertion_trial_response
"""
            )

            result = subprocess.run(
                ["bash", str(runner)],
                env={**os.environ, "HOME": str(home), "CALLS": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            calls = log.read_text().splitlines()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("prepare_insertion_trial_source", calls[0])
            self.assertIn("capture_control_observation", calls[1])
            self.assertIn("persist_insertion_trial_response", calls[2])
            self.assertIn("apply_control_response", calls[3])
            self.assertIn("--requested-steps 1", calls[4])

    def test_insertion_reset_trial_delegates_to_shared_one_action_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_insertion_reset_trial.sh", ops)
            log = home / "calls.log"
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
is_safe_identifier() { return 0; }
require_nonnegative_integer() { return 0; }
require_positive_integer() { return 0; }
control_proposal_from_identity() { printf 'proposal-test\n'; }
run_reset_trial_control_session() { printf '%s\n' "$*" >> "${CALLS}"; }
"""
            )

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_insertion_reset_trial.sh"),
                    "trial-session",
                    "insertion-held-00",
                    "52600",
                    "worker-test",
                    "safety-session",
                    "43",
                ],
                env={**os.environ, "HOME": str(home), "CALLS": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            call = log.read_text()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("insertion_reset_trial safety-session 43 900", call)
            self.assertIn("prepare_insertion_trial_source", call)
            self.assertIn("persist_insertion_trial_response", call)

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
