from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim.demo_run_cli import main
from tests.test_demo_run import build_demo_run_spec


class DemoRunCliTest(unittest.TestCase):
    def test_preflight_reports_only_after_the_frozen_run_authenticates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = build_demo_run_spec(root)
            spec_path = root / "demo-spec.json"
            spec_path.write_text(json.dumps(spec.to_dict()))
            binding_path = root / "demo-run.binding.json"
            arguments = [
                "verify",
                "--spec",
                str(spec_path),
                "--fingerprint",
                spec.fingerprint,
                "--recording-root",
                str(root),
                "--source-revision",
                spec.source_revision,
                "--container-image-digest",
                spec.container_image_digest,
                "--run-id",
                "demo-run",
                "--binding-output",
                str(binding_path),
                "--grasp-actions",
                str(spec.terminal_contract.grasp_actions),
                "--insertion-actions",
                str(spec.terminal_contract.insertion_actions),
                "--reference-recording",
                spec.selection.reference_recording,
                "--exploration-seed",
                str(spec.selection.exploration_seed),
                "--artifact",
                f"stage_asset={spec.artifacts[0].identity.path}",
                "--worker",
                (
                    f"grasp={spec.workers[0].identity}="
                    f"{spec.workers[0].manifest.path}"
                ),
                "--worker",
                (
                    f"insertion={spec.workers[1].identity}="
                    f"{spec.workers[1].manifest.path}"
                ),
            ]

            with patch("builtins.print") as output:
                self.assertEqual(main(arguments), 0)

            payload = json.loads(output.call_args.args[0])
            self.assertEqual(payload["status"], "demo_run_spec_authenticated")
            self.assertEqual(payload["fingerprint"], spec.fingerprint)
            self.assertEqual(
                json.loads(binding_path.read_text())["spec_fingerprint"],
                spec.fingerprint,
            )


if __name__ == "__main__":
    unittest.main()
