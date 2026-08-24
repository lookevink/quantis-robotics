import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
AWS_SCRIPT = REPO_ROOT / "ops" / "aws.sh"
REMOTE_BOOTSTRAP = REPO_ROOT / "ops" / "remote_bootstrap.sh"
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
                )
                fake_command.chmod(0o755)
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

    def test_backup_state_syncs_and_runs_on_the_remote_host(self):
        result, calls = self.run_command("backup-state")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn("ops/backup_state.sh", calls)

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
        self.assertIn("sim.runtime_loader", calls)
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
        self.assertIn("ServerAliveInterval=15", calls)
        self.assertNotIn("ops/encode_demo_recording.sh", calls)

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

    def test_jepa_wm_control_rollout_forwards_a_bounded_step_count(self):
        result, calls = self.run_command(
            "jepa-wm-control-rollout",
            arguments=(
                "domain-held-00",
                "11400",
                "3",
                "quantis_calibrated_control",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_control_rollout.sh", calls)
        self.assertIn("'3'", calls)
        self.assertIn("quantis_calibrated_control", calls)

    def test_jepa_wm_control_baseline_uses_an_explicit_non_model_policy(self):
        result, calls = self.run_command(
            "jepa-wm-control-baseline",
            arguments=("domain-held-00", "11400", "3", "scripted"),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ops/run_control_rollout.sh", calls)
        self.assertIn("'baseline_scripted' 'scripted'", calls)
        self.assertNotIn("control-worker-start", calls)

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
