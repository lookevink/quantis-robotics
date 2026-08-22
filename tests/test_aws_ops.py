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


class AwsLifecycleTests(unittest.TestCase):
    def run_command(
        self,
        command: str,
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
                    "printf '%s %s\\n' \"$(basename \"$0\")\" \"$*\" >> \"${FAKE_AWS_LOG}\"\n"
                    "if [[ \"$(basename \"$0\")\" == ssh && -n \"${FAKE_SSH_RESPONSE:-}\" ]]; then\n"
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
                [str(AWS_SCRIPT), command],
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

    def test_demo_run_syncs_and_calls_the_loopback_python_server(self):
        result, calls = self.run_command("demo-run")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("rsync ", calls)
        self.assertIn("127.0.0.1 8226", calls)
        self.assertIn("run_demo", calls)

    def test_demo_run_propagates_python_server_errors(self):
        result, _ = self.run_command(
            "demo-run",
            extra_env={
                "FAKE_SSH_RESPONSE": '{"status":"error","evalue":"bad motion"}'
            },
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("bad motion", result.stdout)

    def test_demo_record_captures_then_encodes_the_same_recording(self):
        result, calls = self.run_command("demo-record")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("record_demo", calls)
        self.assertIn("ops/encode_demo_recording.sh", calls)
        self.assertRegex(calls, r"demo-[0-9]{8}T[0-9]{6}Z")

    def test_remote_bootstrap_installs_python_server_client(self):
        bootstrap = REMOTE_BOOTSTRAP.read_text()
        self.assertIn("netcat-openbsd", bootstrap)
        self.assertIn("ffmpeg", bootstrap)


if __name__ == "__main__":
    unittest.main()
