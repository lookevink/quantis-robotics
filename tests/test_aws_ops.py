import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
AWS_SCRIPT = REPO_ROOT / "ops" / "aws.sh"


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


if __name__ == "__main__":
    unittest.main()
