from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from jepa_wm.insertion_layout import CONTACT_INSERTION_LAYOUT


REPO_ROOT = Path(__file__).resolve().parents[1]


class ControlRolloutShellTest(unittest.TestCase):
    def test_demo_preflight_accepts_forwarded_revision_in_gitless_deployment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = root / "quantis-robotics"
            ops = repository / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            calls = root / "calls.log"
            fake_python = root / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"${CALLS}\"\n"
            )
            fake_python.chmod(0o755)
            source_revision = "a" * 40
            runner = root / "run.sh"
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source '{ops / 'shell_helpers.sh'}'\n"
                "sudo() { printf 'sha256:%064d\\n' 0; }\n"
                "validate_demo_run_spec \\\n"
                f"  '{source_revision}' '{fake_python}' /tmp/spec {'f' * 64} \\\n"
                "  /tmp/recordings /tmp/stage grasp /tmp/grasp.worker.json \\\n"
                "  insertion /tmp/insertion.worker.json reference 12601 run \\\n"
                "  /tmp/binding.json 52 4\n"
            )

            result = subprocess.run(
                ["bash", str(runner)],
                env={**os.environ, "CALLS": str(calls)},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertFalse((repository / ".git").exists())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"--source-revision {source_revision}", calls.read_text())

    def test_control_capture_timeout_cancels_the_owned_isaac_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            jobs = home / "docker" / "isaac-sim" / "data" / "quantis" / "recording_jobs"
            ops.mkdir(parents=True)
            jobs.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            job = jobs / "control-timeout-session.json"
            job.write_text('{"status":"running"}\n')
            log = home / "calls.log"
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -u
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() {
  printf '%s\n' "$1" >> "${CALLS}"
  (
    sleep 0.2
    printf '{"status":"error","error":"recording task was cancelled"}\n' \
      > "${HOME}/docker/isaac-sim/data/quantis/recording_jobs/control-timeout-session.json"
  ) &
}
wait_control_capture_job control-timeout-session 1
printf '%s\n' "$?" >> "${CALLS}"
exit 0
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
            self.assertIn("cancel_recording_job", calls[0])
            self.assertEqual(calls[-1], "124")
            self.assertIn("recording task was cancelled", result.stderr)

    def test_insertion_context_resolver_uses_the_canonical_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = Path(temp_dir) / "run.sh"
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source '{REPO_ROOT}/ops/shell_helpers.sh'\n"
                f"resolve_insertion_context '' '{REPO_ROOT}' python3\n"
                f"resolve_insertion_context 43 '{REPO_ROOT}' python3\n"
            )

            result = subprocess.run(
                ["bash", str(runner)],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.splitlines(),
            [
                str(CONTACT_INSERTION_LAYOUT.insertion_command_context_indices[0]),
                "43",
            ],
        )

    def test_grasp_transition_milestone_reestablishes_grasp_then_one_action(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_grasp_transition_milestone.sh", ops)
            log = home / "calls.log"
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
is_safe_identifier() { return 0; }
require_nonnegative_integer() { return 0; }
contact_grasp_maximum_actions() { printf '52\n'; }
contact_grasp_initial_context() { printf '110\n'; }
require_control_rollout_reach_and_grasp() { return 0; }
control_rollout_terminal_session() { printf 'milestone-grasp-42\n'; }
isaac_server_call() { printf 'isaac %s\n' "$1" >> "${CALLS}"; }
"""
            )
            (ops / "jepa_wm.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'worker %s\\n\' "$*" >> "${CALLS}"\n'
            )
            (ops / "run_control_rollout.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'grasp %s\\n\' "$*" >> "${CALLS}"\n'
            )
            (ops / "run_grasp_transition_trial.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'transition %s\\n\' "$*" >> "${CALLS}"\n'
            )

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_grasp_transition_milestone.sh"),
                    "milestone",
                    "contact-reference",
                    "42601",
                    "grasp-control",
                    "transition-control",
                ],
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
                    "worker",
                    "worker",
                    "grasp",
                    "isaac",
                    "worker",
                    "worker",
                    "transition",
                ],
            )
            self.assertIn("grasp-control", calls[1])
            self.assertIn("milestone-grasp-42", calls[3])
            self.assertIn("transition-control", calls[5])
            self.assertIn(
                "milestone-transition milestone-grasp-42 contact-reference 42601 transition-control",
                calls[6],
            )

    def test_grasp_transition_resolves_worker_manifest_from_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            repository = home / "quantis-robotics"
            ops = repository / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_grasp_transition_trial.sh", ops)
            log = home / "calls.log"
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
is_safe_identifier() { return 0; }
require_nonnegative_integer() { return 0; }
control_proposal_from_identity() {
  pwd >> "${CALLS}"
  printf 'transition-proposal\n'
}
run_insertion_followup_trial() { printf '%s\n' "$*" >> "${CALLS}"; }
require_control_rollout_applied() { return 0; }
"""
            )

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_grasp_transition_trial.sh"),
                    "transition-run",
                    "grasp-action-42",
                    "contact-reference",
                    "42601",
                    "transition-control",
                ],
                cwd=home,
                env={**os.environ, "HOME": str(home), "CALLS": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            calls = log.read_text().splitlines()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(calls[0], str(repository))
            self.assertIn("transition-proposal", calls[1])

    def test_contact_grasp_bounds_shadow_postmortem_to_rollout_endpoints(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
control_rollout_shadow_session_roster contact_grasp 'grasp-00,grasp-01,grasp-39'
control_rollout_shadow_session_roster standard 'direct-00,direct-01,direct-02'
"""
            )

            result = subprocess.run(
                ["bash", str(runner)],
                env={**os.environ, "HOME": str(home)},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.splitlines(),
                ["grasp-00,grasp-39", "direct-00,direct-01,direct-02"],
            )

    def test_grasp_to_insertion_runs_one_task_terminal_grasp_plus_four_chain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_grasp_to_insertion_milestone.sh", ops)
            log = home / "calls.log"
            log.touch()
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
is_safe_identifier() { return 0; }
require_nonnegative_integer() { return 0; }
insertion_rollout_profile_field() {
  [[ "$3" == demo && "$4" == maximum-steps ]] || return 9
  printf '4\n'
}
contact_grasp_maximum_actions() { printf '52\n'; }
contact_grasp_initial_context() { printf '110\n'; }
isaac_server_call() { printf 'isaac %s|%s\n' "$1" "${3:-false}" >> "${CALLS}"; }
control_proposal_from_identity() {
  printf 'proposal %s %s %s\n' "$1" "$2" "${PWD}" >> "${CALLS}"
  [[ "${PWD}" == "${HOME}/quantis-robotics" ]] || return 91
  printf 'insertion-proposal\n'
}
require_control_rollout_reach_and_grasp() { return 0; }
control_rollout_terminal_session() { printf 'full-chain-grasp-12\n'; }
validate_demo_run_spec() {
  printf 'preflight %s\n' "$*" >> "${CALLS}"
}
run_insertion_followup_trial() {
  printf 'followup %s\n' "$*" >> "${CALLS}"
}
"""
            )
            (ops / "jepa_wm.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'worker %s\\n\' "$*" >> "${CALLS}"\n'
            )
            (ops / "run_control_rollout.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'grasp %s\\n\' "$*" >> "${CALLS}"\n'
            )

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_grasp_to_insertion_milestone.sh"),
                    "full-chain",
                    "contact-reference",
                    "12401",
                    "grasp-worker",
                    "insertion-worker",
                    "demo-spec",
                    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ],
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
                    "proposal",
                    "proposal",
                    "preflight",
                    "worker",
                    "worker",
                    "grasp",
                    "isaac",
                    "worker",
                    "worker",
                    "followup",
                    "isaac",
                    "followup",
                    "isaac",
                    "followup",
                    "isaac",
                    "followup",
                    "isaac",
                ],
            )
            self.assertEqual(
                calls[:2],
                [
                    f"proposal direct grasp-worker {home / 'quantis-robotics'}",
                    (
                        "proposal insertion_followup_trial insertion-worker "
                        f"{home / 'quantis-robotics'}"
                    ),
                ],
            )
            self.assertIn(
                "full-chain-grasp contact-reference 12401 52 grasp-worker direct 110 contact_grasp",
                calls[5],
            )
            self.assertIn(
                "verify_grasp_to_insertion_source('full-chain-grasp-12')",
                calls[6],
            )
            self.assertIn("demo-spec", calls[2])
            self.assertIn("datacenter_demo.usda", calls[2])
            self.assertIn("grasp-worker.worker.json", calls[2])
            self.assertIn("insertion-worker.worker.json", calls[2])
            self.assertIn("contact-reference 12401", calls[2])
            self.assertIn("a" * 40, calls[2])
            self.assertTrue(calls[2].endswith("52 4"))
            followups = [line for line in calls if line.startswith("followup ")]
            self.assertEqual(len(followups), 4)
            self.assertIn(
                "full-chain-action1 full-chain-grasp-12 contact-reference 12401 insertion-proposal full-chain-action1 4 full-chain-grasp-12",
                followups[0],
            )
            self.assertIn(
                "full-chain-action1,full-chain-action2,full-chain-action3,full-chain-action4 4 full-chain-grasp-12",
                followups[-1],
            )
            self.assertIn(
                "verify_grasp_to_insertion_result('full-chain','full-chain-grasp','full-chain-action1,full-chain-action2,full-chain-action3,full-chain-action4','contact-reference',12401)",
                calls[-1],
            )

    def test_insertion_demo_rollout_runs_exactly_four_verified_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_insertion_demo_rollout.sh", ops)
            log = home / "calls.log"
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
is_safe_identifier() { return 0; }
require_nonnegative_integer() { return 0; }
require_positive_integer() { return 0; }
resolve_insertion_context() { printf '%s\n' "$1"; }
insertion_rollout_profile_field() {
  [[ "$3" == demo ]] || return 9
  printf '4\n'
}
isaac_server_call() { printf 'verify %s\n' "$1" >> "${CALLS}"; }
"""
            )
            for name in (
                "run_insertion_safety_check.sh",
                "run_insertion_reset_trial.sh",
                "run_insertion_followup_trial.sh",
            ):
                (ops / name).write_text(
                    "#!/usr/bin/env bash\nprintf '%s %s\\n' "
                    f'\'{name}\' "$*" >> "${{CALLS}}"\n'
                )

            result = subprocess.run(
                [
                    "bash",
                    str(ops / "run_insertion_demo_rollout.sh"),
                    "demo-run",
                    "insertion-held-00",
                    "52600",
                    "worker-test",
                    "43",
                ],
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
                    "run_insertion_followup_trial.sh",
                    "verify",
                    "run_insertion_followup_trial.sh",
                    "verify",
                ],
            )
            self.assertIn(
                "verify_insertion_demo_rollout_result('demo-run-action1,demo-run-action2,demo-run-action3,demo-run-action4'",
                calls[-1],
            )
            self.assertTrue(calls[0].endswith(" demo"), calls[0])
            self.assertTrue(calls[1].endswith(" demo"), calls[1])
            self.assertIn(
                "demo-run-action1,demo-run-action2,demo-run-action3 3",
                calls[5],
            )
            self.assertIn(
                "demo-run-action1,demo-run-action2,demo-run-action3,demo-run-action4 4",
                calls[7],
            )

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
resolve_insertion_context() { printf '%s\n' "$1"; }
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
                    f'\'{name}\' "$*" >> "${{CALLS}}"\n'
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
            self.assertTrue(calls[0].endswith(" two-step"), calls[0])
            self.assertTrue(calls[1].endswith(" two-step"), calls[1])
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
                '#!/usr/bin/env bash\nprintf \'report %s\\n\' "$*" >> "${CALLS}"\n'
            )
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() { printf '%s|%s|%s\n' "$1" "$2" "${3:-false}" >> "${CALLS}"; }
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
            self.assertEqual(
                calls[1], "respond followup-safety insertion_safety_evaluation"
            )
            self.assertIn("evaluate_direct_insertion_candidate", calls[2])
            self.assertIn("prepare_insertion_trial_source", calls[3])
            self.assertIn("persist_insertion_followup_response", calls[4])
            self.assertIn("apply_control_response", calls[5])
            self.assertIn("|600|false", calls[5])
            self.assertNotIn("capture_control_observation", "\n".join(calls))
            self.assertIn("--sessions previous-trial,followup-trial", calls[6])
            self.assertIn("--requested-steps 2", calls[6])

    def test_insertion_followup_reports_one_proposal_handoff_segment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            log = home / "calls.log"
            (ops / "jepa_wm.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "${CALLS}"\n'
            )
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() { printf '%s|%s\n' "$1" "${3:-false}" >> "${CALLS}"; }
respond_to_control_session() { :; }
run_insertion_followup_trial \
  "${HOME}/quantis-robotics" followup-safety parent-trial bridge-trial \
  insertion-held-00 52600 parent-proposal parent-trial 1 bridge-trial true
"""
            )

            result = subprocess.run(
                ["bash", str(runner)],
                env={**os.environ, "HOME": str(home), "CALLS": str(log)},
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            calls = log.read_text().splitlines()
            self.assertIn("insertion-transition-handoff", calls[0])
            self.assertIn("persist_insertion_proposal_handoff", calls[1])
            self.assertTrue(calls[1].endswith("|true"))
            self.assertIn("capture_followup_observation", calls[2])
            self.assertTrue(calls[2].endswith("|false"))
            report = calls[-1]
            self.assertIn("--sessions parent-trial", report)
            self.assertIn("--requested-steps 1", report)
            self.assertIn("--predecessor-session bridge-trial", report)

    def test_insertion_segment_retry_restores_the_settled_rollback_runtime(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            log = home / "calls.log"
            (ops / "jepa_wm.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() { printf '%s|%s\n' "$1" "${3:-false}" >> "${CALLS}"; }
respond_to_control_session() { :; }
run_insertion_followup_trial \
  "${HOME}/quantis-robotics" followup-safety retry-trial previous-trial \
  insertion-held-00 52600 proposal-test retry-trial 1 previous-trial false \
  rolled-back-trial
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
            self.assertIn("restore_insertion_retry", calls[0])
            self.assertTrue(calls[0].endswith("|true"))
            self.assertIn("capture_followup_observation", calls[1])
            self.assertTrue(calls[1].endswith("|false"))

    def test_shared_reset_trial_reports_a_typed_preflight_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "shell_helpers.sh", ops)
            log = home / "calls.log"
            (ops / "jepa_wm.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'%s\n\' "$*" >> "${CALLS}"\n'
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
                '#!/usr/bin/env bash\nprintf \'report %s\n\' "$*" >> "${CALLS}"\n'
            )
            runner = home / "run.sh"
            runner.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
source "${HOME}/quantis-robotics/ops/shell_helpers.sh"
isaac_server_call() { printf '%s|%s\n' "$1" "$2" >> "${CALLS}"; }
wait_control_capture_job() { printf 'wait %s|%s\n' "$1" "$2" >> "${CALLS}"; }
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
            self.assertIn("start_control_capture", calls[1])
            self.assertEqual(calls[2], "wait control-trial-session|900")
            self.assertIn("persist_insertion_trial_response", calls[3])
            self.assertIn("apply_control_response", calls[4])
            self.assertTrue(calls[4].endswith("|180"), calls[4])
            self.assertIn("--requested-steps 1", calls[5])

    def test_insertion_reset_trial_delegates_to_shared_one_action_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            ops = home / "quantis-robotics" / "ops"
            ops.mkdir(parents=True)
            shutil.copy(REPO_ROOT / "ops" / "run_insertion_reset_trial.sh", ops)
            log = home / "calls.log"
            (ops / "shell_helpers.sh").write_text(
                """#!/usr/bin/env bash
isaac_control_capture_timeout_seconds=900
isaac_insertion_trial_apply_timeout_seconds=600
is_safe_identifier() { return 0; }
require_nonnegative_integer() { return 0; }
require_positive_integer() { return 0; }
resolve_insertion_context() { printf '%s\n' "$1"; }
control_proposal_from_identity() { printf 'proposal-test\n'; }
insertion_rollout_profile_field() {
  [[ "$3" == two-step ]] || return 9
  printf '2\n'
}
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
                    "two-step",
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
            self.assertTrue(call.rstrip().endswith("2 600"), call)

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
resolve_insertion_context() { printf '%s\\n' "$1"; }
insertion_rollout_profile_field() {
  [[ "$3" == two-step ]] || return 9
  printf '2\n'
}
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
                    "two-step",
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
            (ops / "run_control_step.sh").write_text("#!/usr/bin/env bash\nexit 7\n")
            (ops / "jepa_wm.sh").write_text(
                '#!/usr/bin/env bash\nprintf \'%s\\n\' "$*" >> "${ROLLOUT_LOG}"\n'
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
