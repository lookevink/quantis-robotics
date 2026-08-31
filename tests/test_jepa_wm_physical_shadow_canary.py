from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.physical_shadow_canary import (
    claim_canary,
    load_experiment_config,
    prepare_worker,
)
from jepa_wm.physical_shadow_canary_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)


class PhysicalShadowCanaryTest(unittest.TestCase):
    def test_frozen_contract_is_one_zero_actuation_known_start(self) -> None:
        config = load_experiment_config(
            Path(".scratch/jepa-physical-shadow-canary-v1/experiment-config.json")
        )

        self.assertEqual(config["known_start"]["seed"], 12601)
        self.assertEqual(config["known_start"]["context_index"], 110)
        self.assertEqual(config["execution"]["evaluations"], 1)
        self.assertFalse(config["execution"]["apply_action"])
        self.assertTrue(config["gate"]["require_zero_actuation"])
        self.assertEqual(
            FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "31dff3512bdddb1006435686e93e35c974ee3bfe1b0d3fe9895193c28ad83741",
        )
        self.assertEqual(
            config["evaluator"]["implementation_revision"],
            "1000bf81981fb636779f88d28c3789b2ee6ad66e",
        )

    def test_canary_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.json"

            payload = claim_canary(path, "session-1", "config-sha")

            self.assertEqual(payload["session_id"], "session-1")
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim_canary(path, "session-1", "config-sha")

    def test_runner_has_no_actuation_seam(self) -> None:
        runner = Path("ops/run_physical_shadow_canary.sh").read_text()

        self.assertIn('config["session_id"]', runner)
        self.assertIn('config["worker"]["name"]', runner)
        self.assertIn("control-shadow-session", runner)
        self.assertIn("evaluate_shadow_candidate", runner)
        self.assertIn('deployed_revision="${1:-}"', runner)
        self.assertNotIn("contact-insertion-v10-drive-slow-2600-held-01", runner)
        self.assertNotIn('exploration_seed="12601"', runner)
        self.assertNotIn("apply_control_response", runner)
        self.assertNotIn("run_control_step.sh", runner)

    def test_worker_manifest_is_derived_from_frozen_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "worker.json"
            proposal = Path(directory) / "proposal.pth"
            residual = Path(directory) / "residual.pth"
            proposal_report = Path(directory) / "proposal.pth.json"
            readiness = Path(directory) / "readiness.json"
            held_out = Path(directory) / "held-out.json"
            recording_root = Path(directory) / "recordings"
            manifest = recording_root / "held-01" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            proposal.write_bytes(b"proposal")
            residual.write_bytes(b"residual")
            proposal_report.write_text("{}")
            readiness.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "proposal_fingerprint": sha256(b"proposal").hexdigest(),
                        "held_out_evaluations": [{}, {}],
                    }
                )
            )
            held_out.write_text('{"passed": true}')
            manifest.write_text("{}")
            config = {
                "proposal": {
                    "path": str(proposal),
                    "fingerprint": sha256(b"proposal").hexdigest(),
                    "training_report_fingerprint": sha256(b"{}").hexdigest(),
                    "readiness": {
                        "path": str(readiness),
                        "fingerprint": sha256(readiness.read_bytes()).hexdigest(),
                    },
                },
                "action_model": {
                    "path": str(residual),
                    "fingerprint": sha256(b"residual").hexdigest(),
                    "held_out_gate": {
                        "path": str(held_out),
                        "fingerprint": sha256(held_out.read_bytes()).hexdigest(),
                    },
                },
                "known_start": {
                    "reference": "held-01",
                    "manifest_fingerprint": sha256(b"{}").hexdigest(),
                },
                "worker": {
                    "planner": {
                        "horizon": 3,
                        "iterations": 4,
                        "samples": 64,
                        "elites": 8,
                        "seed": 234,
                        "minimum_standard_deviation": 0.0001,
                    }
                },
            }

            worker = prepare_worker(config, output, recording_root)

            self.assertEqual(worker.proposal, Path(config["proposal"]["path"]))
            self.assertEqual(worker.adapter, Path(config["action_model"]["path"]))
            self.assertEqual(worker.planner.seed, 234)


if __name__ == "__main__":
    unittest.main()
