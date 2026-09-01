from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.physical_shadow_canary import (
    _serialized_action_scale,
    claim_canary,
    finalize_recovery,
    load_experiment_config,
    prepare_worker,
)
from jepa_wm.physical_shadow_canary_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)
from jepa_wm.physical_shadow_canary_v2_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as V2_FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)
from jepa_wm.physical_shadow_canary_v3_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as V3_FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
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
            "16f2b88f35f04eeace7de429b81945d3c5d50d29d1c4038424418ba7acf9da25",
        )
        self.assertEqual(
            config["evaluator"]["implementation_revision"],
            "3ee2d006a69eed3ed9a2537ef4d84e2583674066",
        )

    def test_canary_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.json"

            payload = claim_canary(path, "session-1", "config-sha")

            self.assertEqual(payload["session_id"], "session-1")
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim_canary(path, "session-1", "config-sha")

    def test_v2_uses_the_other_held_out_start_and_a_new_terminal_path(self) -> None:
        config = load_experiment_config(
            Path(".scratch/jepa-physical-shadow-canary-v2/experiment-config.json")
        )

        self.assertEqual(config["known_start"]["seed"], 12600)
        self.assertEqual(
            config["known_start"]["reference"],
            "contact-insertion-v10-drive-slow-2600-held-00",
        )
        self.assertTrue(config["output"].endswith("shadow-canary-v2.json"))
        self.assertEqual(
            V2_FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "329b771fb1cf700371230cfda6e94240e31f250015834ebc4257e04bff8aa7b3",
        )

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

    def test_v3_binds_authenticated_reset_and_uses_zero_actuation_capture(self) -> None:
        config = load_experiment_config(
            Path(".scratch/jepa-physical-shadow-canary-v3/experiment-config.json")
        )
        runner = Path("ops/run_physical_shadow_canary.sh").read_text()

        self.assertEqual(config["unknown_start"]["seed"], 62604)
        self.assertEqual(
            config["unknown_start"]["recording_id"],
            "unknown-start-reset-v5-62604",
        )
        self.assertFalse(config["execution"]["apply_action"])
        self.assertTrue(config["gate"]["require_zero_actuation"])
        self.assertEqual(
            V3_FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "c691b5efeadcd773df82050f24e388e0134b941ed5f100cb12fff498064e0a53",
        )
        self.assertIn("capture_unknown_start_shadow_observation", runner)

    def test_negative_safety_evidence_has_no_selected_scale(self) -> None:
        self.assertIsNone(_serialized_action_scale(None))

    def test_recovery_preserves_authenticated_model_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "checkpoints"
            output = checkpoint_root / "model" / "unknown-start-shadow-canary-v9.json"
            evaluation = output.with_name(
                "unknown-start-shadow-canary-v9-evaluation.json"
            )
            evaluation.parent.mkdir(parents=True)
            evaluation.write_text(
                json.dumps(
                    {
                        "schema": "quantis.jepa_wm_physical_shadow_canary_evaluation.v1",
                        "status": "evaluated_pending_recovery",
                        "passed": False,
                        "evaluation_passed": False,
                        "recovery_verified": False,
                        "apply_action": False,
                    }
                )
            )
            recovery_root = root / "recovery"
            recovery_evaluation = recovery_root / evaluation.relative_to(
                checkpoint_root
            )
            recovery_evaluation.parent.mkdir(parents=True)
            recovery_evaluation.write_bytes(evaluation.read_bytes())
            experiment = {
                "output": str(output),
                "action_model": {
                    "path": str(checkpoint_root / "model" / "model.pth")
                },
            }
            with patch(
                "jepa_wm.physical_shadow_canary.load_experiment_config",
                return_value=experiment,
            ), patch(
                "jepa_wm.physical_shadow_canary.authenticated_deployment_revision",
                return_value="a" * 40,
            ):
                terminal = finalize_recovery(
                    root / "config.json",
                    recovery_root,
                    "a" * 40,
                )

            self.assertFalse(terminal["passed"])
            self.assertEqual(terminal["status"], "failed_model_gate")
            self.assertTrue(terminal["model_gate_adjudicated"])

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
