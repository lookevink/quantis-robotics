import json
import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Optional

from jepa_wm.control_resolution_profile import CONTROL_RESOLUTION_CONTEXTS


REPO_ROOT = Path(__file__).resolve().parents[1]
AWS_SCRIPT = REPO_ROOT / "ops" / "aws.sh"
CW_AGENT_CONFIG = REPO_ROOT / "ops" / "cloudwatch-agent.json"
REMOTE_BOOTSTRAP = REPO_ROOT / "ops" / "remote_bootstrap.sh"
ISAAC_CONTAINER = REPO_ROOT / "ops" / "isaac_container.sh"
ENCODE_RECORDING = REPO_ROOT / "ops" / "encode_demo_recording.sh"
BACKUP_STATE = REPO_ROOT / "ops" / "backup_state.sh"


def write_fake_findmnt(path: Path) -> Path:
    fake_findmnt = path / "findmnt"
    fake_findmnt.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            query_path=""
            output_field=""
            while (( $# )); do
              case "$1" in
                -T) query_path="$2"; shift 2 ;;
                -o) output_field="$2"; shift 2 ;;
                *) shift ;;
              esac
            done
            if [[ "${query_path}" == "${FAKE_ASSET_HOME}" ]]; then
              if [[ "${output_field}" == "TARGET" ]]; then
                printf '%s\\n' "${FAKE_ASSET_MOUNT_TARGET}"
              elif [[ "${output_field}" == "MAJ:MIN" ]]; then
                printf '%s\\n' "${FAKE_ASSET_DEVICE_ID:-259:2}"
              else
                printf '%s\\n' "${FAKE_ASSET_SOURCE:-/dev/fake-asset}"
              fi
            elif [[ "${output_field}" == "TARGET" ]]; then
              printf '/\\n'
            elif [[ "${output_field}" == "MAJ:MIN" ]]; then
              printf '%s\\n' "${FAKE_LIVE_DEVICE_ID:-259:1}"
            else
              printf '%s\\n' "${FAKE_LIVE_SOURCE:-/dev/fake-root}"
            fi
            """
        )
    )
    fake_findmnt.chmod(0o755)
    return fake_findmnt


class AwsLifecycleTests(unittest.TestCase):
    def run_isaac_container(
        self,
        command: str,
        arguments: tuple[str, ...] = (),
        *,
        fail_checkpoint_preflight: bool = False,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            log_path = root / "calls.log"
            (root / "docker/jepa-wm/checkpoints").mkdir(parents=True)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_curl = fake_bin / "curl"
            fake_curl.write_text("#!/usr/bin/env bash\nprintf '203.0.113.10'\n")
            fake_curl.chmod(0o755)
            fake_sudo = fake_bin / "sudo"
            fake_sudo.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >> "${FAKE_ISAAC_LOG}"
                    if [[ "${FAKE_CHECKPOINT_PREFLIGHT_FAILURE:-0}" == 1 \
                      && " $* " == *" docker run --rm "* \
                      && " $* " == *" test -r "* ]]; then
                      exit 17
                    fi
                    """
                )
            )
            fake_sudo.chmod(0o755)
            result = subprocess.run(
                [str(ISAAC_CONTAINER), command, *arguments],
                env={
                    **os.environ,
                    "HOME": str(root),
                    "FAKE_ISAAC_LOG": str(log_path),
                    "FAKE_CHECKPOINT_PREFLIGHT_FAILURE": (
                        "1" if fail_checkpoint_preflight else "0"
                    ),
                    "PATH": f"{fake_bin}:{os.environ['PATH']}",
                },
                text=True,
                capture_output=True,
                check=False,
            )
            calls = log_path.read_text() if log_path.exists() else ""
            return result, calls, root

    def run_command(
        self,
        command: str,
        arguments: tuple[str, ...] = (),
        state: str = "running",
        account: str = "686410906008",
        extra_env: Optional[dict[str, str]] = None,
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_path = temp_path / "calls.log"
            private_key = temp_path / "key.pem"
            private_key.touch()
            fake_aws = temp_path / "aws"
            fake_aws.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    printf '%s\\n' "$*" >> "${FAKE_AWS_LOG}"
                    args=" $* "
                    if [[ "${args}" == *" sts get-caller-identity "* ]]; then
                      printf '%s\\n' "${FAKE_AWS_ACCOUNT}"
                    elif [[ "${args}" == *"IamInstanceProfile.Arn"* ]]; then
                      printf 'arn:aws:iam::686410906008:instance-profile/quantis-isaac-sim-ssm\\n'
                    elif [[ "${args}" == *" iam get-instance-profile "* ]]; then
                      printf 'quantis-isaac-sim-ssm\\n'
                    elif [[ "${args}" == *"State.Name"* ]]; then
                      printf '%s\\n' "${FAKE_AWS_STATE}"
                    elif [[ "${args}" == *"PublicIpAddress"* ]]; then
                      printf '198.51.100.42\\n'
                    fi
                    """
                )
            )
            fake_aws.chmod(0o755)
            for command_name in ("rsync", "ssh"):
                fake_command = temp_path / command_name
                fake_command.write_text(
                    "#!/usr/bin/env bash\n"
                    'printf \'%s %s\\n\' "$(basename "$0")" "$*" >> "${FAKE_AWS_LOG}"\n'
                    'if [[ "$(basename "$0")" == ssh && -n "${FAKE_SSH_RESPONSE:-}" ]]; then\n'
                    "  printf '%s\\n' \"${FAKE_SSH_RESPONSE}\"\n"
                    "fi\n"
                    'if [[ "$(basename "$0")" == ssh '
                    '&& -n "${FAKE_SSH_FAIL_MATCH:-}" '
                    '&& "$*" == *"${FAKE_SSH_FAIL_MATCH}"* ]]; then\n'
                    "  exit 7\n"
                    "fi\n"
                )
                fake_command.chmod(0o755)
            fake_git = temp_path / "git"
            fake_git.write_text(
                textwrap.dedent(
                    """\
                    #!/usr/bin/env bash
                    set -euo pipefail
                    if [[ " $* " == *" status --porcelain --untracked-files=all "* ]]; then
                      printf '%s' "${FAKE_GIT_STATUS:-}"
                    elif [[ " $* " == *" rev-parse HEAD "* ]]; then
                      printf '%s\n' "${FAKE_GIT_REVISION}"
                    else
                      exit 9
                    fi
                    """
                )
            )
            fake_git.chmod(0o755)
            fake_curl = temp_path / "curl"
            fake_curl.write_text("#!/usr/bin/env bash\nprintf '203.0.113.10'\n")
            fake_curl.chmod(0o755)
            env = {
                **os.environ,
                "AWS_INSTANCE_ID": "i-0123456789abcdef0",
                "AWS_SECURITY_GROUP_ID": "sg-0123456789abcdef0",
                "AWS_SSH_PRIVATE_KEY": str(private_key),
                "ENV_FILE": str(temp_path / "test.env"),
                "FAKE_AWS_ACCOUNT": account,
                "FAKE_AWS_LOG": str(log_path),
                "FAKE_AWS_STATE": state,
                "FAKE_GIT_REVISION": "1" * 40,
                "PATH": f"{temp_dir}:{os.environ['PATH']}",
                **(extra_env or {}),
            }
            result = subprocess.run(
                [str(AWS_SCRIPT), command, *arguments],
                cwd=REPO_ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            calls = log_path.read_text() if log_path.exists() else ""
            return result, calls

    def test_ensure_running_is_noop_for_running_instance(self):
        result, calls = self.run_command("ensure-running", state="running")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(" start-instances ", f" {calls} ")
        self.assertIn("ec2 wait instance-status-ok", calls)

    def test_ensure_running_starts_stopped_instance(self):
        result, calls = self.run_command("ensure-running", state="stopped")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ec2 start-instances", calls)
        self.assertIn("ec2 wait instance-running", calls)

    def test_refuses_wrong_aws_account(self):
        result, calls = self.run_command("ensure-running", account="111111111111")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("expected 686410906008", result.stderr)
        self.assertNotIn("ec2", calls)

    def test_down_stops_and_waits_for_a_running_instance(self):
        result, calls = self.run_command("down", state="running")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ec2 stop-instances", calls)
        self.assertIn("ec2 wait instance-stopped", calls)

    def test_up_forwards_isaac_configuration_to_remote_host(self):
        result, calls = self.run_command(
            "up",
            extra_env={
                "ISAAC_SIM_VERSION": "5.0.0",
                "ISAAC_SIGNAL_PORT": "50100",
                "ISAAC_STREAM_PORT": "48998",
                "DOWNLOAD_PHYSICALAI_DATASET": "0",
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ISAAC_SIM_VERSION=5.0.0", calls)
        self.assertIn("ISAAC_SIGNAL_PORT=50100", calls)
        self.assertIn("ISAAC_STREAM_PORT=48998", calls)
        self.assertIn("DOWNLOAD_PHYSICALAI_DATASET=0", calls)

    def test_isaac_status_queries_the_remote_container(self):
        result, calls = self.run_command("isaac-status")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/isaac_container.sh status", calls)

    def test_ssh_forwards_an_explicit_remote_command(self):
        result, calls = self.run_command(
            "ssh",
            arguments=("printf", "remote-ok"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ubuntu@198.51.100.42 printf remote-ok", calls)

    def test_isaac_container_mounts_checkpoints_read_only_for_runtime_user(self):
        result, calls, root = self.run_isaac_container("start")

        checkpoint_dir = root / "docker/jepa-wm/checkpoints"
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"--group-add {os.getgid()}", calls)
        self.assertIn(
            f"-v {checkpoint_dir}:{checkpoint_dir}:ro",
            calls,
        )
        self.assertIn(" test -r ", calls)
        self.assertIn("docker run -d", calls)

    def test_isaac_container_refuses_start_when_checkpoints_are_unreadable(self):
        result, calls, _ = self.run_isaac_container(
            "start",
            fail_checkpoint_preflight=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checkpoint directory is unreadable", result.stderr)
        self.assertNotIn("docker run -d", calls)

    def test_isaac_container_checks_exact_proposal_and_metadata_readability(self):
        result, calls, root = self.run_isaac_container(
            "checkpoint-readable",
            ("contact-grasp-v1",),
        )

        checkpoint = root / "docker/jepa-wm/checkpoints/contact-grasp-v1.pth"
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            f"docker exec quantis-isaac-sim test -r {checkpoint}", calls
        )
        self.assertIn(
            f"docker exec quantis-isaac-sim test -r {checkpoint}.json", calls
        )

    def test_backup_state_syncs_and_runs_on_the_remote_host(self):
        result, calls = self.run_command("backup-state")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_cloudwatch_enable_attaches_policy_and_configures_remote_agent(self):
        result, calls = self.run_command("cloudwatch-enable")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("iam attach-role-policy", calls)
        self.assertIn("CloudWatchAgentServerPolicy", calls)
        self.assertIn("rsync ", calls)
        self.assertIn("ops/cloudwatch_agent.sh enable", calls)

    def test_cloudwatch_config_has_only_four_lean_metrics(self):
        config = json.loads(CW_AGENT_CONFIG.read_text())
        collected = config["metrics"]["metrics_collected"]

        self.assertEqual(config["agent"]["metrics_collection_interval"], 60)
        self.assertEqual(collected["mem"]["measurement"], ["mem_used_percent"])
        self.assertEqual(
            collected["nvidia_gpu"]["measurement"],
            ["utilization_gpu", "memory_used", "memory_total"],
        )

    def test_backup_state_copies_persistent_runtime_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            isaac_data = root / "isaac-data"
            checkpoints = root / "checkpoints"
            asset_home = root / "assets"
            asset_home.mkdir()
            for path, contents in (
                (isaac_data / "quantis/scenes/demo.usda", "scene"),
                (
                    isaac_data / "quantis/recordings/trajectory/manifest.json",
                    "recording",
                ),
                (
                    isaac_data / "quantis/control_sessions/session/result.json",
                    "control-session",
                ),
                (
                    isaac_data / "quantis/control_rollouts/rollout/report.json",
                    "control-rollout",
                ),
                (
                    isaac_data / "quantis/control_baselines/proof/report.json",
                    "control-baseline",
                ),
                (
                    isaac_data / "quantis/control_candidates/proof/report.json",
                    "control-candidate",
                ),
                (checkpoints / "model.pth", "checkpoint"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
            fake_findmnt = write_fake_findmnt(root)

            result = subprocess.run(
                [str(BACKUP_STATE)],
                env={
                    **os.environ,
                    "ISAAC_DATA_ROOT": str(isaac_data),
                    "JEPA_WM_CHECKPOINT_DIR": str(checkpoints),
                    "QUANTIS_ASSET_HOME": str(asset_home),
                    "FINDMNT_COMMAND": str(fake_findmnt),
                    "FAKE_ASSET_HOME": str(asset_home),
                    "FAKE_ASSET_MOUNT_TARGET": str(asset_home),
                },
                text=True,
                capture_output=True,
                check=False,
            )

            backup = asset_home / "quantis-state"
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                (backup / "isaac/scenes/demo.usda").read_text(),
                "scene",
            )
            self.assertEqual(
                (backup / "isaac/recordings/trajectory/manifest.json").read_text(),
                "recording",
            )
            self.assertEqual(
                (backup / "isaac/control_sessions/session/result.json").read_text(),
                "control-session",
            )
            self.assertEqual(
                (backup / "isaac/control_rollouts/rollout/report.json").read_text(),
                "control-rollout",
            )
            self.assertEqual(
                (backup / "isaac/control_baselines/proof/report.json").read_text(),
                "control-baseline",
            )
            self.assertEqual(
                (backup / "isaac/control_candidates/proof/report.json").read_text(),
                "control-candidate",
            )
            self.assertEqual(
                (backup / "jepa-wm/checkpoints/model.pth").read_text(),
                "checkpoint",
            )

    def test_backup_state_refuses_an_unmounted_destination(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            isaac_data = root / "isaac-data"
            checkpoints = root / "checkpoints"
            asset_home = root / "unmounted-assets"
            (isaac_data / "quantis/scenes").mkdir(parents=True)
            (isaac_data / "quantis/recordings").mkdir(parents=True)
            checkpoints.mkdir()
            asset_home.mkdir()
            fake_findmnt = write_fake_findmnt(root)

            result = subprocess.run(
                [str(BACKUP_STATE)],
                env={
                    **os.environ,
                    "ISAAC_DATA_ROOT": str(isaac_data),
                    "JEPA_WM_CHECKPOINT_DIR": str(checkpoints),
                    "QUANTIS_ASSET_HOME": str(asset_home),
                    "FINDMNT_COMMAND": str(fake_findmnt),
                    "FAKE_ASSET_HOME": str(asset_home),
                    "FAKE_ASSET_MOUNT_TARGET": "/",
                    "FAKE_ASSET_SOURCE": "/dev/fake-root",
                    "FAKE_LIVE_SOURCE": "/dev/fake-root",
                },
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not a dedicated mount point", result.stderr)
            self.assertFalse((asset_home / "quantis-state").exists())

    def test_backup_state_refuses_a_bind_mount_from_the_live_filesystem(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            isaac_data = root / "isaac-data"
            checkpoints = root / "checkpoints"
            asset_home = root / "bind-mounted-assets"
            (isaac_data / "quantis/scenes").mkdir(parents=True)
            (isaac_data / "quantis/recordings").mkdir(parents=True)
            checkpoints.mkdir()
            asset_home.mkdir()
            fake_findmnt = write_fake_findmnt(root)

            result = subprocess.run(
                [str(BACKUP_STATE)],
                env={
                    **os.environ,
                    "ISAAC_DATA_ROOT": str(isaac_data),
                    "JEPA_WM_CHECKPOINT_DIR": str(checkpoints),
                    "QUANTIS_ASSET_HOME": str(asset_home),
                    "FINDMNT_COMMAND": str(fake_findmnt),
                    "FAKE_ASSET_HOME": str(asset_home),
                    "FAKE_ASSET_MOUNT_TARGET": str(asset_home),
                    "FAKE_ASSET_SOURCE": "/dev/fake-root[/asset-backup]",
                    "FAKE_LIVE_SOURCE": "/dev/fake-root",
                    "FAKE_ASSET_DEVICE_ID": "259:1",
                    "FAKE_LIVE_DEVICE_ID": "259:1",
                },
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("shares a filesystem", result.stderr)
            self.assertFalse((asset_home / "quantis-state").exists())

    def test_demo_run_syncs_and_calls_the_loopback_python_server(self):
        result, calls = self.run_command("demo-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn("127.0.0.1 8226", calls)
        self.assertIn("sim/runtime_loader.py", calls)
        self.assertIn("run_demo", calls)

    def test_demo_run_propagates_python_server_errors(self):
        result, _ = self.run_command(
            "demo-run",
            extra_env={"FAKE_SSH_RESPONSE": '{"status":"error","evalue":"bad motion"}'},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bad motion", result.stdout)

    def test_demo_record_captures_then_encodes_the_same_recording(self):
        result, calls = self.run_command("demo-record")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start_recording", calls)
        self.assertIn("ops/wait_demo_recording.sh", calls)
        self.assertIn("ops/encode_demo_recording.sh", calls)
        self.assertRegex(calls, r"demo-[0-9]{8}T[0-9]{6}Z")

    def test_demo_record_actions_uses_the_short_world_model_capture(self):
        result, calls = self.run_command("demo-record-actions")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start_action_recording", calls)
        self.assertIn("ops/wait_demo_recording.sh", calls)
        self.assertIn("ops/encode_demo_recording.sh", calls)
        self.assertRegex(calls, r"trajectory-[0-9]{8}T[0-9]{6}Z")

    def test_demo_record_exploration_forwards_seed_and_dataset_split(self):
        result, calls = self.run_command(
            "demo-record-exploration",
            arguments=("domain-20260823-train-00", "1200", "train"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start_exploration_recording", calls)
        self.assertIn("domain-20260823-train-00", calls)
        self.assertIn("1200", calls)
        self.assertIn("train", calls)
        self.assertIn(
            "ops/wait_demo_recording.sh 'domain-20260823-train-00'",
            calls,
        )

    def test_demo_record_grasp_forwards_seed_and_dataset_split(self):
        result, calls = self.run_command(
            "demo-record-grasp",
            arguments=("grasp-20260824-train-00", "2400", "train"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start_grasp_recording", calls)
        self.assertIn("grasp-20260824-train-00", calls)
        self.assertIn("2400", calls)
        self.assertIn("train", calls)

    def test_demo_record_insertion_forwards_seed_and_dataset_split(self):
        result, calls = self.run_command(
            "demo-record-insertion",
            arguments=("insert-20260824-held-12402", "12402", "held_out"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start_insertion_recording", calls)
        self.assertIn("insert-20260824-held-12402", calls)
        self.assertIn("12402", calls)
        self.assertIn("held_out", calls)

    def test_demo_record_contact_insertion_forwards_seed_and_dataset_split(self):
        result, calls = self.run_command(
            "demo-record-contact-insertion",
            arguments=("contact-insert-held-12402", "12402", "held_out"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start_contact_insertion_recording", calls)
        self.assertIn("contact-insert-held-12402", calls)
        self.assertIn("12402", calls)
        self.assertIn("held_out", calls)

    def test_grasp_recording_validation_runs_against_persistent_data(self):
        result, calls = self.run_command(
            "jepa-wm-grasp-validate",
            arguments=("grasp-20260824-held-11401", "held_out"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa_wm.grasp_recording_cli", calls)
        self.assertIn("grasp-20260824-held-11401", calls)
        self.assertIn("held_out", calls)

    def test_insertion_recording_validation_runs_against_persistent_data(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-validate",
            arguments=("insert-20260824-held-12402", "held_out"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa_wm.insertion_recording_cli", calls)
        self.assertIn("insert-20260824-held-12402", calls)
        self.assertIn("held_out", calls)

    def test_contact_insertion_validation_runs_against_persistent_data(self):
        result, calls = self.run_command(
            "jepa-wm-contact-insertion-validate",
            arguments=("contact-insert-held-12402", "held_out", "12402"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa_wm.contact_insertion_recording_cli", calls)
        self.assertIn("contact-insert-held-12402", calls)
        self.assertIn("held_out", calls)
        self.assertIn("--expected-seed 12402", calls)

    def test_contact_insertion_status_binds_expected_seed(self):
        result, calls = self.run_command(
            "jepa-wm-contact-insertion-status",
            arguments=("contact-insert-held-12402", "held_out", "12402"),
            extra_env={"FAKE_SSH_RESPONSE": "valid"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa_wm.contact_insertion_status_cli", calls)
        self.assertIn("contact-insert-held-12402", calls)
        self.assertIn("12402", calls)

    def test_partial_recording_quarantine_is_recoverable_and_scoped(self):
        result, calls = self.run_command(
            "demo-quarantine-partial-recording",
            arguments=("contact-insert-held-12402",),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("recordings/incomplete/contact-insert-held-12402", calls)
        self.assertIn("job_is_quarantinable", calls)
        self.assertIn("test ! -f", calls)
        self.assertIn("sudo mv", calls)

    def test_demo_dashboard_records_scores_and_renders_one_recording(self):
        result, calls = self.run_command(
            "demo-dashboard",
            arguments=("demo-reference", "wrist", "wrist"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("start_recording", calls)
        self.assertIn("ops/wait_demo_recording.sh", calls)
        self.assertIn("ops/encode_demo_recording.sh", calls)
        self.assertIn("ops/jepa_stages.sh report 'demo-reference'", calls)
        self.assertIn("ops/render_demo_dashboard.sh", calls)
        recording_ids = set(__import__("re").findall(r"demo-[0-9]{8}T[0-9]{6}Z", calls))
        self.assertEqual(len(recording_ids), 1)

    def test_candidate_film_records_encodes_and_renders_the_validated_session(self):
        result, calls = self.run_command(
            "jepa-wm-candidate-film",
            arguments=("candidate-proof-11401", "candidate-film-11401"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("record_candidate_demo", calls)
        self.assertIn("candidate-proof-11401", calls)
        self.assertIn("ops/encode_demo_recording.sh 'candidate-film-11401'", calls)
        self.assertIn("ops/render_demo_dashboard.sh 'candidate-film-11401'", calls)

    def test_grasp_film_reloads_readiness_then_records_encodes_and_renders(self):
        fingerprint = "a" * 64
        result, calls = self.run_command(
            "jepa-wm-grasp-film",
            arguments=("grasp-readiness-v2", "12401", "grasp-film-12401"),
            extra_env={"FAKE_SSH_RESPONSE": fingerprint},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("record_grasp_demo", calls)
        self.assertIn("grasp_control_readiness_cli", calls)
        self.assertIn("--readiness-id 'grasp-readiness-v2'", calls)
        self.assertIn("--fingerprint-only", calls)
        self.assertIn(fingerprint, calls)
        self.assertIn("grasp-readiness-v2", calls)
        self.assertIn("12401", calls)
        self.assertIn("ops/encode_demo_recording.sh 'grasp-film-12401'", calls)
        self.assertIn("ops/render_demo_dashboard.sh 'grasp-film-12401'", calls)

    def test_insertion_demo_film_reconstructs_and_encodes_the_source_run(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-demo-film",
            arguments=("insertion-demo-source", "insertion-demo-film"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("record_insertion_demo", calls)
        self.assertIn("insertion-demo-source", calls)
        self.assertIn("ops/encode_demo_recording.sh 'insertion-demo-film'", calls)
        self.assertNotIn("ops/render_demo_dashboard.sh", calls)

    def test_grasp_to_insertion_runs_one_guarded_phase_chain(self):
        source_revision = "a" * 40
        result, calls = self.run_command(
            "jepa-wm-grasp-to-insertion",
            arguments=(
                "contact-reference",
                "12401",
                "grasp-control",
                "insertion-control",
                "demo-spec",
                "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            ),
            extra_env={"FAKE_GIT_REVISION": source_revision},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_grasp_to_insertion_milestone.sh", calls)
        self.assertIn("contact-reference", calls)
        self.assertIn("grasp-control", calls)
        self.assertIn("insertion-control", calls)
        self.assertIn("demo-spec", calls)
        self.assertIn(source_revision, calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_grasp_to_insertion_refuses_a_dirty_source_tree_before_sync(self):
        result, calls = self.run_command(
            "jepa-wm-grasp-to-insertion",
            arguments=(
                "contact-reference",
                "12401",
                "grasp-control",
                "insertion-control",
                "demo-spec",
                "f" * 64,
            ),
            extra_env={"FAKE_GIT_STATUS": " M ops/aws.sh\n"},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source tree must be clean", result.stderr)
        self.assertNotIn("rsync ", calls)

    def test_grasp_transition_switches_worker_before_the_guarded_trial(self):
        result, calls = self.run_command(
            "jepa-wm-grasp-transition-trial",
            arguments=(
                "transition-run",
                "grasp-action-42",
                "contact-reference",
                "42601",
                "transition-control",
                "rolled-back-action",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        worker_stop = calls.index("ops/jepa_wm.sh control-worker-stop")
        worker_start = calls.index(
            "ops/jepa_wm.sh control-worker-start --artifacts 'transition-control'"
        )
        transition_trial = calls.index("ops/run_grasp_transition_trial.sh")
        self.assertLess(worker_stop, worker_start)
        self.assertLess(worker_start, transition_trial)
        self.assertIn("rolled-back-action", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_grasp_transition_milestone_is_one_guarded_remote_workflow(self):
        result, calls = self.run_command(
            "jepa-wm-grasp-transition-milestone",
            arguments=(
                "transition-milestone",
                "contact-reference",
                "42601",
                "grasp-control",
                "transition-control",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_grasp_transition_milestone.sh", calls)
        self.assertIn("grasp-control", calls)
        self.assertIn("transition-control", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_jepa_embed_forwards_recording_and_camera(self):
        result, calls = self.run_command(
            "jepa-embed",
            arguments=("demo-20260822T040027Z", "wrist"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ops/jepa_embed.sh 'demo-20260822T040027Z' 'wrist'",
            calls,
        )

    def test_jepa_stage_embed_forwards_recording_and_camera(self):
        result, calls = self.run_command(
            "jepa-stage-embed",
            arguments=("demo-reference", "wrist"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ops/jepa_stages.sh embed 'demo-reference' 'wrist'",
            calls,
        )

    def test_jepa_stage_report_forwards_reference_query_and_camera(self):
        result, calls = self.run_command(
            "jepa-stage-report",
            arguments=("demo-reference", "demo-held-out", "wrist"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ops/jepa_stages.sh report 'demo-reference' 'demo-held-out' 'wrist'",
            calls,
        )

    def test_jepa_wm_smoke_syncs_and_runs_on_the_remote_host(self):
        result, calls = self.run_command("jepa-wm-smoke")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn("ops/jepa_wm.sh smoke", calls)

    def test_jepa_wm_model_load_preflight_syncs_and_uses_runtime_wrapper(self):
        result, calls = self.run_command("jepa-wm-model-load-preflight")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn("ops/jepa_wm.sh model-load-preflight", calls)

    def test_jepa_wm_install_forwards_the_gated_checkpoint_url(self):
        result, calls = self.run_command(
            "jepa-wm-install",
            extra_env={
                "DINOV3_CHECKPOINT_URL": "https://weights.example/dinov3-vitl.pth"
            },
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "DINOV3_CHECKPOINT_URL=https://weights.example/dinov3-vitl.pth",
            calls,
        )

    def test_jepa_wm_status_queries_the_installed_runtime_without_syncing(self):
        result, calls = self.run_command("jepa-wm-status")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("rsync ", calls)
        self.assertIn("ops/jepa_wm.sh status", calls)

    def test_jepa_wm_eval_forwards_recording_rollout_window(self):
        result, calls = self.run_command(
            "jepa-wm-eval",
            arguments=("demo-trajectory", "wrist", "190", "8", "3"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn(
            "ops/jepa_wm.sh evaluate 'demo-trajectory' 'wrist' '190' '8' '3' 'base'",
            calls,
        )

    def test_jepa_wm_adapt_forwards_training_recording_and_steps(self):
        result, calls = self.run_command(
            "jepa-wm-adapt",
            arguments=("demo-trajectory", "wrist", "100"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn(
            "ops/jepa_wm.sh adapt 'demo-trajectory' 'wrist' '100'",
            calls,
        )

    def test_jepa_wm_adapt_set_forwards_multiple_training_recordings(self):
        result, calls = self.run_command(
            "jepa-wm-adapt-set",
            arguments=("domain-train-00,domain-train-01", "wrist", "500"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ops/jepa_wm.sh adapt-set 'domain-train-00,domain-train-01' 'wrist' '500'",
            calls,
        )

    def test_jepa_wm_adapt_set_can_preserve_a_named_adapter_checkpoint(self):
        result, calls = self.run_command(
            "jepa-wm-adapt-set",
            arguments=(
                "domain-train-00,domain-train-01",
                "wrist",
                "500",
                "quantis_isaac_wrist_goal_contrastive",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "'500' 'quantis_isaac_wrist_goal_contrastive'",
            calls,
        )

    def test_jepa_wm_eval_adapted_selects_the_persistent_adapter(self):
        result, calls = self.run_command(
            "jepa-wm-eval-adapted",
            arguments=("demo-held-out", "wrist", "0", "20", "1"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "ops/jepa_wm.sh evaluate 'demo-held-out' 'wrist' '0' '20' '1' 'adapted'",
            calls,
        )

    def test_jepa_wm_plan_benchmark_forwards_the_bounded_search_budget(self):
        result, calls = self.run_command(
            "jepa-wm-plan-benchmark",
            arguments=(
                "domain-held-00",
                "wrist",
                "4",
                "8",
                "3",
                "4",
                "128",
                "10",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn(
            "ops/jepa_wm.sh plan-benchmark --recording 'domain-held-00' "
            "--camera 'wrist' --start-index '4' --count '8' --stride '3' "
            "--iterations '4' --samples '128' --elites '10'",
            calls,
        )

    def test_jepa_wm_plan_benchmark_can_select_a_named_adapter(self):
        result, calls = self.run_command(
            "jepa-wm-plan-benchmark",
            arguments=(
                "domain-held-00",
                "wrist",
                "0",
                "8",
                "1",
                "4",
                "128",
                "10",
                "quantis_isaac_wrist_goal_contrastive",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--elites '10' --adapter 'quantis_isaac_wrist_goal_contrastive'",
            calls,
        )

    def test_jepa_wm_plan_benchmark_can_refine_a_named_proposal(self):
        result, calls = self.run_command(
            "jepa-wm-plan-benchmark",
            arguments=(
                "domain-held-00",
                "wrist",
                "0",
                "8",
                "1",
                "4",
                "128",
                "10",
                "quantis_isaac_wrist_action_adapter",
                "quantis_isaac_wrist_action_proposal_pose_spatial_12seed",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "--adapter 'quantis_isaac_wrist_action_adapter' "
            "--proposal 'quantis_isaac_wrist_action_proposal_pose_spatial_12seed'",
            calls,
        )

    def test_jepa_wm_insertion_plan_benchmark_uses_pinned_task_entrypoint(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-plan-benchmark",
            arguments=(
                "contact-insertion-held-00",
                "insertion-adapter",
                "insertion-proposal",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn(
            "ops/jepa_wm.sh insertion-plan-benchmark "
            "--recording 'contact-insertion-held-00' "
            "--adapter 'insertion-adapter' --proposal 'insertion-proposal' "
            "--profile 'sampled_readiness'",
            calls,
        )

    def test_jepa_wm_insertion_plan_benchmark_forwards_dense_profile(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-plan-benchmark",
            arguments=(
                "contact-insertion-held-00",
                "insertion-adapter",
                "insertion-proposal",
                "dense_execution",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--profile 'dense_execution'", calls)

    def test_jepa_wm_insertion_plan_summary_forwards_only_roster_and_proposal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            roster = Path(temporary_directory) / "fresh.json"
            roster.write_text('{"schema":"fresh"}\n')
            result, calls = self.run_command(
                "jepa-wm-insertion-plan-summarize",
                arguments=(str(roster), "insertion-proposal"),
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn(
            "ops/jepa_wm.sh insertion-plan-summarize " "--fresh-roster-base64 '",
            calls,
        )
        self.assertIn("--proposal 'insertion-proposal'", calls)
        self.assertIn("--profile 'sampled_readiness'", calls)
        self.assertNotIn("--adapter", calls)

    def test_jepa_wm_insertion_proposal_training_diagnostic_is_named(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-proposal-training-diagnostic",
            arguments=("insertion-proposal", "train-alignment"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn(
            "ops/jepa_wm.sh insertion-proposal-training-diagnostic "
            "--proposal 'insertion-proposal' --output 'train-alignment'",
            calls,
        )

    def test_jepa_wm_proposal_train_forwards_recording_set_and_checkpoint_name(self):
        result, calls = self.run_command(
            "jepa-wm-proposal-train",
            arguments=(
                "domain-train-00,domain-train-01",
                "wrist",
                "2000",
                "quantis_isaac_wrist_action_proposal",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh proposal-train", calls)
        self.assertIn("domain-train-00,domain-train-01", calls)
        self.assertIn("quantis_isaac_wrist_action_proposal", calls)

    def test_jepa_wm_proposal_eval_forwards_held_out_window(self):
        result, calls = self.run_command(
            "jepa-wm-proposal-eval",
            arguments=(
                "domain-held-00",
                "wrist",
                "4",
                "8",
                "8",
                "quantis_isaac_wrist_action_proposal",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh proposal-eval", calls)
        self.assertIn("--start-index '4' --count '8' --stride '8'", calls)
        self.assertIn("quantis_isaac_wrist_action_proposal", calls)

    def test_jepa_wm_proposal_summarize_forwards_whole_seed_reports(self):
        result, calls = self.run_command(
            "jepa-wm-proposal-summarize",
            arguments=(
                "domain-held-00,domain-held-01",
                "wrist",
                "4",
                "62",
                "1",
                "quantis_isaac_wrist_action_proposal_motion_state_12seed",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh proposal-summarize", calls)
        self.assertIn("domain-held-00,domain-held-01", calls)
        self.assertIn("--start-index '4' --count '62' --stride '1'", calls)

    def test_jepa_wm_grasp_proposal_commands_bind_the_complete_task_window(self):
        trained, training_calls = self.run_command(
            "jepa-wm-grasp-proposal-train",
            arguments=(
                "grasp-train-00,grasp-train-01",
                "500",
                "grasp-proposal",
                "16",
                "0.001",
                "0.01",
                "235",
            ),
        )
        evaluated, evaluation_calls = self.run_command(
            "jepa-wm-grasp-proposal-eval",
            arguments=("grasp-held-00", "grasp-proposal"),
        )
        summarized, summary_calls = self.run_command(
            "jepa-wm-grasp-proposal-summarize",
            arguments=("grasp-held-00,grasp-held-01", "grasp-proposal"),
        )

        self.assertEqual(trained.returncode, 0, trained.stderr)
        self.assertIn("ops/jepa_wm.sh grasp-proposal-train", training_calls)
        self.assertIn("--hidden-dimension '16'", training_calls)
        self.assertIn("--weight-decay '0.01' --seed '235'", training_calls)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertIn("ops/jepa_wm.sh grasp-proposal-eval", evaluation_calls)
        self.assertIn("grasp-held-00", evaluation_calls)
        self.assertEqual(summarized.returncode, 0, summarized.stderr)
        self.assertIn("ops/jepa_wm.sh grasp-proposal-summarize", summary_calls)
        self.assertIn("grasp-held-00,grasp-held-01", summary_calls)

    def test_jepa_wm_contact_grasp_commands_bind_the_contact_domain(self):
        trained, training_calls = self.run_command(
            "jepa-wm-contact-grasp-proposal-train",
            arguments=(
                "contact-train-00,contact-train-01",
                "3000",
                "contact-grasp-proposal",
                "256",
                "0.001",
                "0.0001",
                "2600",
            ),
        )
        evaluated, evaluation_calls = self.run_command(
            "jepa-wm-contact-grasp-proposal-eval",
            arguments=("contact-held-00", "contact-grasp-proposal"),
        )
        summarized, summary_calls = self.run_command(
            "jepa-wm-contact-grasp-proposal-summarize",
            arguments=(
                "contact-held-00,contact-held-01",
                "contact-grasp-proposal",
            ),
        )

        self.assertEqual(trained.returncode, 0, trained.stderr)
        self.assertIn("contact-grasp-proposal-train", training_calls)
        self.assertIn("--seed '2600'", training_calls)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertIn("contact-grasp-proposal-eval", evaluation_calls)
        self.assertEqual(summarized.returncode, 0, summarized.stderr)
        self.assertIn("contact-grasp-proposal-summarize", summary_calls)

    def test_jepa_wm_insertion_proposal_commands_bind_the_post_attachment_window(self):
        trained, training_calls = self.run_command(
            "jepa-wm-insertion-proposal-train",
            arguments=(
                "insert-train-00,insert-train-01",
                "3000",
                "insertion-proposal",
                "256",
                "0.001",
                "0.0001",
                "2600",
            ),
        )
        evaluated, evaluation_calls = self.run_command(
            "jepa-wm-insertion-proposal-eval",
            arguments=("insert-held-00", "insertion-proposal"),
        )
        summarized, summary_calls = self.run_command(
            "jepa-wm-insertion-proposal-summarize",
            arguments=(
                "insert-held-00,insert-held-01",
                "insertion-proposal",
                "insert-v9-2600",
                "2600",
            ),
        )

        self.assertEqual(trained.returncode, 0, trained.stderr)
        self.assertIn("ops/jepa_wm.sh insertion-proposal-train", training_calls)
        self.assertIn("--hidden-dimension '256'", training_calls)
        self.assertIn("--seed '2600'", training_calls)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertIn("ops/jepa_wm.sh insertion-proposal-eval", evaluation_calls)
        self.assertIn("insert-held-00", evaluation_calls)
        self.assertEqual(summarized.returncode, 0, summarized.stderr)
        self.assertIn("ops/jepa_wm.sh insertion-proposal-summarize", summary_calls)
        self.assertIn("insert-held-00,insert-held-01", summary_calls)
        self.assertIn("--experiment 'insert-v9-2600' --base-seed '2600'", summary_calls)

    def test_jepa_wm_insertion_world_model_commands_bind_one_named_adapter(self):
        frozen_fingerprint = "a" * 64
        trained, training_calls = self.run_command(
            "jepa-wm-insertion-adapt",
            arguments=(
                "insert-train-00,insert-train-01",
                "500",
                "insertion-adapter",
            ),
        )
        evaluated, evaluation_calls = self.run_command(
            "jepa-wm-insertion-wm-eval",
            arguments=("insert-held-00", "insertion-adapter"),
        )
        summarized, summary_calls = self.run_command(
            "jepa-wm-insertion-wm-summarize",
            arguments=(
                "insert-held-00,insert-held-01",
                "insertion-adapter",
                "insert-v9-2600",
                "2600",
            ),
        )
        aligned, aligned_calls = self.run_command(
            "jepa-wm-insertion-adapt",
            arguments=(
                "insert-train-00,insert-train-01",
                "1056",
                "insertion-aligned-adapter",
                "goal_aligned",
            ),
        )
        from jepa_wm.insertion_corpus import (
            FrozenInsertionAdapter,
            InsertionCorpusRoster,
            InsertionFreshEvaluationRoster,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            fresh_roster = Path(temporary_directory) / "fresh.json"
            InsertionFreshEvaluationRoster.create(
                "insert-v9-2600-fresh-22600",
                22600,
                InsertionCorpusRoster.create("insert-v9-2600", 2600),
                FrozenInsertionAdapter(
                    "insertion-aligned-adapter",
                    frozen_fingerprint,
                ),
            ).write(fresh_roster)
            fresh, fresh_calls = self.run_command(
                "jepa-wm-insertion-wm-fresh-summarize",
                arguments=(
                    str(fresh_roster),
                    "goal_aligned_relative_finetune",
                ),
            )

        self.assertEqual(trained.returncode, 0, trained.stderr)
        self.assertIn("ops/jepa_wm.sh insertion-wm-adapt", training_calls)
        self.assertIn("--steps '500' --adapter 'insertion-adapter'", training_calls)
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertIn("ops/jepa_wm.sh insertion-wm-eval", evaluation_calls)
        self.assertEqual(summarized.returncode, 0, summarized.stderr)
        self.assertIn("ops/jepa_wm.sh insertion-wm-summarize", summary_calls)
        self.assertIn(
            "--experiment 'insert-v9-2600' --base-seed '2600' "
            "--adapter-profile 'generic'",
            summary_calls,
        )
        self.assertEqual(aligned.returncode, 0, aligned.stderr)
        self.assertIn(
            "ops/jepa_wm.sh insertion-wm-adapt",
            aligned_calls,
        )
        self.assertIn(
            "--steps '1056' --adapter 'insertion-aligned-adapter' "
            "--profile 'goal_aligned'",
            aligned_calls,
        )
        self.assertEqual(fresh.returncode, 0, fresh.stderr)
        self.assertIn("ops/jepa_wm.sh insertion-wm-summarize", fresh_calls)
        self.assertIn(
            "--adapter-profile 'goal_aligned_relative_finetune' "
            "--fresh-roster-base64 '",
            fresh_calls,
        )
        self.assertNotIn("--adapter 'insertion-aligned-adapter'", fresh_calls)
        self.assertNotIn("--fresh-evaluation", fresh_calls)
        self.assertNotIn("--fresh-base-seed", fresh_calls)
        self.assertNotIn("--frozen-adapter-fingerprint", fresh_calls)

    def test_jepa_wm_control_replay_forwards_one_fresh_observation(self):
        result, calls = self.run_command(
            "jepa-wm-control-infer-replay",
            arguments=(
                "domain-held-00",
                "wrist",
                "4",
                "quantis_isaac_wrist_action_proposal",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh control-infer-replay", calls)
        self.assertIn("--context-index '4' --observation-id '1'", calls)

    def test_jepa_wm_control_worker_start_selects_one_artifact_manifest(self):
        result, calls = self.run_command(
            "jepa-wm-control-worker-start",
            arguments=("quantis_calibrated_control",),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh control-worker-start", calls)
        self.assertIn(
            "--artifacts 'quantis_calibrated_control'",
            calls,
        )

    def test_jepa_wm_control_worker_configures_one_artifact_manifest(self):
        result, calls = self.run_command(
            "jepa-wm-control-worker-configure",
            arguments=(
                "quantis_calibrated_control",
                "quantis_isaac_wrist_action_proposal",
                "quantis_isaac_wrist_action_adapter",
                "quantis_action_response",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh control-worker-configure", calls)
        self.assertIn("--name 'quantis_calibrated_control'", calls)
        self.assertIn("--calibration 'quantis_action_response'", calls)

    def test_jepa_wm_control_worker_rebases_only_the_proposal(self):
        result, calls = self.run_command(
            "jepa-wm-control-worker-rebase-proposal",
            arguments=("transition-v1", "transition-v2", "bridge-proposal"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("control-worker-rebase-proposal", calls)
        self.assertIn("--source 'transition-v1'", calls)
        self.assertIn("--name 'transition-v2'", calls)
        self.assertIn("--proposal 'bridge-proposal'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_jepa_wm_control_worker_binds_progress_margins(self):
        result, calls = self.run_command(
            "jepa-wm-control-worker-configure",
            arguments=(
                "quantis_ambitious_control",
                "proposal",
                "adapter",
                "calibration",
                "0.0005",
                "0.001",
                "0.005",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--translation-margin '0.0005'", calls)
        self.assertIn("--rotation-margin '0.001'", calls)
        self.assertIn("--gripper-margin '0.005'", calls)

    def test_jepa_wm_control_worker_binds_reproducible_search_budget(self):
        result, calls = self.run_command(
            "jepa-wm-control-worker-configure",
            arguments=(
                "quantis_search_control",
                "proposal",
                "adapter",
                "calibration",
                "0.0005",
                "0.001",
                "0.005",
                "235",
                "8",
                "256",
                "16",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--planner-seed '235'", calls)
        self.assertIn("--planner-iterations '8'", calls)
        self.assertIn("--planner-samples '256'", calls)
        self.assertIn("--planner-elites '16'", calls)

    def test_jepa_wm_control_worker_rejects_partial_progress_margins(self):
        result, calls = self.run_command(
            "jepa-wm-control-worker-configure",
            arguments=(
                "quantis_ambitious_control",
                "proposal",
                "adapter",
                "calibration",
                "0.0005",
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("all three progress margins", result.stderr)
        self.assertNotIn("control-worker-configure", calls)

    def test_jepa_wm_candidate_readiness_forwards_trial_experiments(self):
        result, calls = self.run_command(
            "jepa-wm-control-candidate-summarize",
            arguments=("candidate-a,candidate-b", "candidate-readiness-v1"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("control-candidate-summarize", calls)
        self.assertIn("--experiments 'candidate-a,candidate-b'", calls)
        self.assertIn("--output 'candidate-readiness-v1'", calls)

    def test_jepa_wm_control_step_keeps_inference_outside_isaac(self):
        result, calls = self.run_command(
            "jepa-wm-control-step",
            arguments=(
                "domain-held-00",
                "11400",
                "quantis_calibrated_control",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh control-worker-start", calls)
        self.assertIn("ops/run_control_step.sh", calls)
        self.assertIn("quantis_calibrated_control", calls)

    def test_insertion_safety_runs_no_actuation_workflow_and_always_backs_up(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-safety",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_safety_check.sh", calls)
        self.assertIn("contact-insertion-v9-2600-dense-control", calls)
        self.assertIn(f"'{CONTROL_RESOLUTION_CONTEXTS[0]}'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_safety_backs_up_a_failed_live_evaluation(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-safety",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "43",
            ),
            extra_env={"FAKE_SSH_FAIL_MATCH": "run_insertion_safety_check.sh"},
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("ops/run_insertion_safety_check.sh", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_physical_shadow_canary_is_single_non_actuating_workflow(self):
        result, calls = self.run_command("jepa-wm-physical-shadow-canary")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_physical_shadow_canary.sh", calls)
        self.assertIn("ops/backup_state.sh", calls)
        self.assertIn("finalize-recovery", calls)
        self.assertIn(
            "/mnt/quantis-assets/quantis-state/jepa-wm/checkpoints", calls
        )
        self.assertNotIn("contact-insertion-v10-drive-slow-2600-held-01", calls)
        self.assertNotIn("run_control_step.sh", calls)
        self.assertNotIn("control-apply", calls)

    def test_unknown_start_reset_terminalizes_a_recovery_failure(self):
        result, calls = self.run_command(
            "jepa-wm-unknown-start-reset",
            extra_env={"FAKE_SSH_FAIL_MATCH": "finalize-recovery"},
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("recovery_finalization:exit_7", calls)
        self.assertGreaterEqual(calls.count("ops/backup_state.sh"), 2)

    def test_physical_shadow_canary_cannot_pass_when_worker_stop_fails(self):
        result, calls = self.run_command(
            "jepa-wm-physical-shadow-canary",
            extra_env={"FAKE_SSH_FAIL_MATCH": "control-worker-stop"},
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("worker_stop:exit_7", calls)
        self.assertNotIn("finalize-recovery", calls)
        self.assertLess(calls.index("worker_stop:exit_7"), calls.index("backup_state.sh"))

    def test_physical_shadow_canary_v2_selects_its_frozen_config(self):
        result, calls = self.run_command("jepa-wm-physical-shadow-canary-v2")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa-physical-shadow-canary-v2", calls)
        self.assertIn("run_physical_shadow_canary.sh", calls)
        self.assertNotIn("run_control_step.sh", calls)

    def test_physical_shadow_canary_v3_selects_unknown_start_config(self):
        result, calls = self.run_command("jepa-wm-physical-shadow-canary-v3")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa-physical-shadow-canary-v3", calls)
        self.assertIn("run_physical_shadow_canary.sh", calls)
        self.assertNotIn("run_control_step.sh", calls)
        self.assertNotIn("control-apply", calls)

    def test_physical_shadow_canary_v4_selects_paused_unknown_start_config(self):
        result, calls = self.run_command("jepa-wm-physical-shadow-canary-v4")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa-physical-shadow-canary-v4", calls)
        self.assertNotIn("run_control_step.sh", calls)
        self.assertNotIn("control-apply", calls)

    def test_physical_shadow_canary_v5_selects_continuity_safe_reset(self):
        result, calls = self.run_command("jepa-wm-physical-shadow-canary-v5")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa-physical-shadow-canary-v5", calls)
        self.assertNotIn("run_control_step.sh", calls)
        self.assertNotIn("control-apply", calls)

    def test_physical_shadow_canary_v6_selects_paused_render_canary(self):
        result, calls = self.run_command("jepa-wm-physical-shadow-canary-v6")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("jepa-physical-shadow-canary-v6", calls)
        self.assertNotIn("run_control_step.sh", calls)
        self.assertNotIn("control-apply", calls)

    def test_unknown_start_reset_is_zero_actuation_and_recovery_gated(self):
        source_revision = "a" * 40
        result, calls = self.run_command(
            "jepa-wm-unknown-start-reset",
            extra_env={"FAKE_GIT_REVISION": source_revision},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_unknown_start_reset.sh", calls)
        self.assertIn(source_revision, calls)
        self.assertIn("ops/backup_state.sh", calls)
        self.assertIn("finalize-recovery", calls)
        self.assertIn("runtime-source-fingerprint", calls)
        self.assertNotIn("control-worker-start", calls)
        self.assertNotIn("control-apply", calls)

    def test_sync_does_not_deploy_local_log_artifacts(self):
        result, calls = self.run_command("sync")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--exclude *.log", calls)

    def test_physical_shadow_replay_is_offline_and_recovery_gated(self):
        result, calls = self.run_command("jepa-wm-physical-shadow-replay")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run_physical_shadow_replay.sh", calls)
        self.assertIn("control-worker-stop", calls)
        self.assertIn("backup_state.sh", calls)
        self.assertIn("physical_shadow_replay finalize", calls)
        self.assertNotIn("isaac_container.sh start", calls)
        self.assertNotIn("run_control_step.sh", calls)

    def test_insertion_trial_forwards_exact_source_and_always_backs_up(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-trial",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "insertion-safety-source",
                "43",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_reset_trial.sh", calls)
        self.assertIn("'insertion-safety-source'", calls)
        self.assertIn("'43'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_trial_backs_up_a_failed_execution(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-trial",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "insertion-safety-source",
                "43",
            ),
            extra_env={"FAKE_SSH_FAIL_MATCH": "run_insertion_reset_trial.sh"},
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("ops/run_insertion_reset_trial.sh", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_followup_forwards_the_exact_predecessor_and_backs_up(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "insertion-trial-previous",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_followup_trial.sh", calls)
        self.assertIn("'insertion-trial-previous'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_followup_backs_up_a_failed_execution(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "insertion-trial-previous",
            ),
            extra_env={"FAKE_SSH_FAIL_MATCH": "run_insertion_followup_trial.sh"},
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("ops/run_insertion_followup_trial.sh", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_parent_followup_reports_one_predecessor_bound_segment(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-parent-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "bridge-trial",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_followup_trial.sh", calls)
        self.assertIn("'bridge-trial'", calls)
        self.assertIn("'1' 'bridge-trial' 'true'", calls)
        self.assertLess(
            calls.index("ops/jepa_wm.sh control-worker-stop"),
            calls.index(
                "ops/jepa_wm.sh control-worker-start --artifacts "
                "'contact-insertion-v9-2600-dense-control'"
            ),
        )
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_segment_followup_keeps_the_current_worker(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-segment-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "phase2-control",
                "phase2-action1",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("ops/jepa_wm.sh control-worker-stop", calls)
        self.assertIn("'1' 'phase2-action1' 'false'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_segment_retry_forwards_the_rolled_back_runtime_owner(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-segment-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "phase2-control",
                "phase2-action2",
                "phase2-rolled-back-action3",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "'1' 'phase2-action2' 'false' 'phase2-rolled-back-action3'",
            calls,
        )
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_approach_followup_explicitly_requests_the_named_extension(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-approach-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "phase2-control",
                "terminal-demo-action",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "'1' 'terminal-demo-action' 'false' '' 'approach'",
            calls,
        )
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_alignment_followup_requests_only_the_alignment_extension(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-alignment-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "phase2-control",
                "terminal-approach-action",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "'1' 'terminal-approach-action' 'false' '' 'alignment'",
            calls,
        )

    def test_insertion_pre_insertion_followup_requests_only_the_bounded_extension(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-pre-insertion-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "phase3-control",
                "terminal-alignment-action",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "'1' 'terminal-alignment-action' 'false' '' 'pre-insertion'",
            calls,
        )
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_contact_followup_requests_only_the_contact_extension(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-contact-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "phase7-control",
                "terminal-pre-insertion-action",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "'1' 'terminal-pre-insertion-action' 'false' '' 'contact-insertion'",
            calls,
        )
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_followup_backs_up_an_early_validation_failure(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-followup",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "../invalid-predecessor",
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("ops/run_insertion_followup_trial.sh", calls)
        self.assertIn("ops/backup_state.sh", calls)
        self.assertNotIn("unbound variable", result.stderr)

    def test_insertion_two_step_runs_one_guarded_chain_and_always_backs_up(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-two-step",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "43",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_two_step_trial.sh", calls)
        self.assertIn("contact-insertion-v9-2600-dense-control", calls)
        self.assertIn("'43'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_two_step_backs_up_an_early_validation_failure(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-two-step",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "../invalid-worker",
                "43",
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("ops/run_insertion_two_step_trial.sh", calls)
        self.assertIn("ops/backup_state.sh", calls)
        self.assertNotIn("unbound variable", result.stderr)

    def test_insertion_demo_rollout_runs_one_guarded_chain_and_always_backs_up(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-demo-rollout",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "43",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_demo_rollout.sh", calls)
        self.assertIn("contact-insertion-v9-2600-dense-control", calls)
        self.assertIn("'43'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_demo_rollout_ssh_liveness_exceeds_capture_timeout(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-demo-rollout",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "contact-insertion-v9-2600-dense-control",
                "43",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        ssh_call = next(
            line
            for line in calls.splitlines()
            if line.startswith("ssh ") and "ops/run_insertion_demo_rollout.sh" in line
        )
        interval_match = re.search(r"ServerAliveInterval=(\d+)", ssh_call)
        count_match = re.search(r"ServerAliveCountMax=(\d+)", ssh_call)
        self.assertIsNotNone(interval_match, ssh_call)
        self.assertIsNotNone(count_match, ssh_call)
        liveness_seconds = int(interval_match.group(1)) * int(count_match.group(1))
        self.assertGreater(liveness_seconds, 900, ssh_call)

    def test_insertion_resolution_runs_diagnostic_and_always_backs_up(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-resolution",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "43",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_resolution_measurement.sh", calls)
        self.assertIn("'43'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_resolution_forwards_unloaded_diagnostic_mode(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-resolution",
            arguments=(
                "insertion-fresh-held-00",
                "52600",
                "64",
                "unloaded",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_insertion_resolution_measurement.sh", calls)
        self.assertIn("'64' 'unloaded'", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_insertion_resolution_backs_up_a_failed_measurement(self):
        result, calls = self.run_command(
            "jepa-wm-insertion-resolution",
            arguments=("insertion-fresh-held-00", "52600", "43"),
            extra_env={
                "FAKE_SSH_FAIL_MATCH": "run_insertion_resolution_measurement.sh"
            },
        )

        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertIn("ops/run_insertion_resolution_measurement.sh", calls)
        self.assertIn("ops/backup_state.sh", calls)

    def test_jepa_wm_control_rollout_forwards_a_bounded_step_count(self):
        result, calls = self.run_command(
            "jepa-wm-control-rollout",
            arguments=(
                "domain-held-00",
                "11400",
                "3",
                "quantis_calibrated_control",
                "44",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_control_rollout.sh", calls)
        self.assertIn("'3'", calls)
        self.assertIn("quantis_calibrated_control", calls)
        self.assertIn("direct '44'", calls)

    def test_jepa_wm_control_baseline_uses_an_explicit_non_model_policy(self):
        result, calls = self.run_command(
            "jepa-wm-control-baseline",
            arguments=("domain-held-00", "11400", "3", "scripted"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_control_rollout.sh", calls)
        self.assertIn("'baseline_scripted' 'scripted'", calls)
        self.assertNotIn("control-worker-start", calls)

    def test_jepa_wm_control_baseline_forwards_the_task_context(self):
        result, calls = self.run_command(
            "jepa-wm-control-baseline",
            arguments=("grasp-held-00", "12400", "8", "zero", "86"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("'baseline_zero' 'zero' '86'", calls)

    def test_jepa_wm_control_baselines_forwards_strict_trial_provenance(self):
        result, calls = self.run_command(
            "jepa-wm-control-baselines",
            arguments=(
                "baseline-proof",
                "direct-rollout",
                "zero-rollout",
                "scripted-rollout",
                "domain-held-00",
                "11400",
                "3",
                "quantis_isaac_wrist_action_proposal",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh control-baseline-report", calls)
        self.assertIn("--direct-rollout 'direct-rollout'", calls)
        self.assertIn("--zero-rollout 'zero-rollout'", calls)
        self.assertIn("--scripted-rollout 'scripted-rollout'", calls)
        self.assertIn("--requested-steps '3'", calls)
        self.assertIn("--direct-proposal 'quantis_isaac_wrist_action_proposal'", calls)

    def test_jepa_wm_control_baselines_can_bind_an_explicit_source_session(self):
        result, calls = self.run_command(
            "jepa-wm-control-baselines",
            arguments=(
                "baseline-proof",
                "source-direct",
                "zero-rollout",
                "scripted-rollout",
                "domain-held-00",
                "11400",
                "1",
                "quantis_isaac_wrist_action_proposal",
                "source-step-00",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--direct-sessions 'source-step-00'", calls)

    def test_jepa_wm_control_calibration_collection_is_repeatable(self):
        result, calls = self.run_command(
            "jepa-wm-control-calibration-collect",
            arguments=(
                "seed-11400-calibration",
                "domain-held-00",
                "11400",
                "6",
                "quantis_calibrated_control",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("control-worker-start", calls)
        self.assertIn("ops/run_control_calibration.sh", calls)
        self.assertIn("'seed-11400-calibration'", calls)
        self.assertIn("'6'", calls)

    def test_jepa_wm_control_candidate_is_explicitly_bound_to_one_shadow_source(self):
        result, calls = self.run_command(
            "jepa-wm-control-candidate",
            arguments=(
                "domain-held-00",
                "11400",
                "direct-rollout-00",
                "baseline-proof",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_candidate_trial.sh", calls)
        self.assertIn("'direct-rollout-00'", calls)
        self.assertIn("control-candidate-report", calls)
        self.assertIn("--baseline-experiment 'baseline-proof'", calls)

    def test_jepa_wm_objective_calibration_forwards_realized_candidate_sessions(self):
        result, calls = self.run_command(
            "jepa-wm-objective-calibrate",
            arguments=(
                "quantis_action_response",
                "candidate-00,candidate-01,candidate-02",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("control-objective-calibrate", calls)
        self.assertIn(
            "--sessions 'candidate-00,candidate-01,candidate-02'",
            calls,
        )
        self.assertIn("--output 'quantis_action_response'", calls)

    def test_jepa_wm_summarize_forwards_the_whole_experiment(self):
        result, calls = self.run_command(
            "jepa-wm-summarize",
            arguments=(
                "domain-proof",
                "domain-train-00,domain-train-01",
                "domain-held-00,domain-held-01",
                "wrist",
                "40",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/jepa_wm.sh summarize", calls)
        self.assertIn("domain-proof", calls)
        self.assertIn("domain-train-00,domain-train-01", calls)
        self.assertIn("domain-held-00,domain-held-01", calls)

    def test_jepa_wm_milestone_runs_seeded_train_and_held_out_workflow(self):
        result, calls = self.run_command(
            "jepa-wm-milestone",
            arguments=("2", "2", "10", "1200"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count("start_exploration_recording"), 4)
        for seed in (1200, 1201, 11200, 11201):
            self.assertIn(str(seed), calls)
        self.assertIn("ops/jepa_wm.sh adapt-set", calls)
        self.assertEqual(calls.count("ops/jepa_wm.sh evaluate"), 2)
        self.assertIn("ops/jepa_wm.sh summarize", calls)
        self.assertIn("ops/backup_state.sh", calls)
        self.assertIn("Experiment ID:", result.stdout)

    def test_jepa_wm_grasp_milestone_requires_and_validates_whole_seeds(self):
        result, calls = self.run_command(
            "jepa-wm-grasp-milestone",
            arguments=("12", "2", "10", "2400"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls.count("start_grasp_recording"), 14)
        self.assertEqual(calls.count("jepa_wm.grasp_recording_cli"), 14)
        self.assertIn("ops/jepa_wm.sh grasp-proposal-train", calls)
        self.assertEqual(calls.count("ops/jepa_wm.sh grasp-proposal-eval"), 2)
        self.assertIn("ops/jepa_wm.sh grasp-proposal-summarize", calls)
        self.assertIn("ops/backup_state.sh", calls)
        self.assertIn("Grasp experiment:", result.stdout)
        milestone = (REPO_ROOT / "ops/jepa_wm_grasp_milestone.sh").read_text()
        self.assertIn("readiness_status=$?", milestone)
        self.assertLess(
            milestone.index('"${aws_workflow}" jepa-wm-grasp-proposal-summarize'),
            milestone.index('"${aws_workflow}" backup-state'),
        )
        self.assertLess(
            milestone.index('"${aws_workflow}" backup-state'),
            milestone.index('exit "${readiness_status}"'),
        )

    def test_remote_bootstrap_installs_python_server_client(self):
        bootstrap = REMOTE_BOOTSTRAP.read_text()
        self.assertIn("netcat-openbsd", bootstrap)
        self.assertIn("ffmpeg", bootstrap)

    def test_remote_bootstrap_provisions_jepa_wm(self):
        bootstrap = REMOTE_BOOTSTRAP.read_text()

        self.assertIn('jepa_wm.sh" install', bootstrap)
        self.assertIn('jepa_wm.sh" smoke', bootstrap)

    def test_recording_encoder_accepts_a_safe_trajectory_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(ENCODE_RECORDING), "trajectory-20260823T025416Z"],
                env={**os.environ, "HOME": temp_dir},
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("recording manifest does not exist", result.stderr)
        self.assertNotIn("expected recording ID", result.stderr)

    def test_recording_encoder_rejects_parent_directory_identifier(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = subprocess.run(
                [str(ENCODE_RECORDING), ".."],
                env={**os.environ, "HOME": temp_dir},
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid recording ID", result.stderr)


if __name__ == "__main__":
    unittest.main()
