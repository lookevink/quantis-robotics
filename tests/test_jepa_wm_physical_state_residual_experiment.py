from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_conditioning import (
        ACTION_CONDITIONING_SCHEMA,
        ActionConditioningContract,
        PhysicalStateResidualActionEncoder,
    )
    from jepa_wm.action_conditioning_experiment import TRAINING_RECORDINGS
    from jepa_wm.causal_routing import CausalMotionRoute
    from jepa_wm.contract import MODEL_ID
    from jepa_wm.physical_observation import PHYSICAL_ROUTING_FEATURE_NAMES
    from jepa_wm.physical_state_residual_experiment import (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        PHYSICAL_TREATMENT_SPEC,
        _applied_residual_ratio_report,
        _authenticate_training_contract,
        _expected_training_metadata,
        _is_terminal_command_phase,
        _load_experiment_config,
        _training_config,
        _write_failure_record,
    )
    from jepa_wm.training_artifact import training_configuration_fingerprint


@unittest.skipIf(torch is None, "PyTorch is required for physical residuals")
class PhysicalStateResidualExperimentTest(unittest.TestCase):
    def test_frozen_training_manifest_authenticates(self) -> None:
        experiment = _load_experiment_config(
            Path(".scratch/jepa-physical-state-residual-v1/" "experiment-config.json")
        )

        self.assertEqual(
            experiment["schema"],
            "quantis.jepa_wm_physical_state_residual_experiment.v1",
        )
        self.assertEqual(len(FROZEN_EXPERIMENT_CONFIG_FINGERPRINT), 64)
        self.assertEqual(
            PHYSICAL_TREATMENT_SPEC.physical_state_routing.maximum_residual_ratio,
            0.15,
        )

    def test_applied_residual_report_checks_all_candidates_and_hold_passthrough(
        self,
    ) -> None:
        base = torch.nn.Linear(7, 4, bias=False)
        routing = PHYSICAL_TREATMENT_SPEC.physical_state_routing
        assert routing is not None
        encoder = PhysicalStateResidualActionEncoder(base, routing)
        features = torch.zeros((3, len(PHYSICAL_ROUTING_FEATURE_NAMES)))
        encoder.router.fit_normalization(features)
        with torch.no_grad():
            base.weight.fill_(1.0)
            encoder.residuals[0].weight.fill_(20.0)
            encoder.residuals[1].weight.fill_(-20.0)
            encoder.router.output.weight.zero_()
            encoder.router.output.bias.zero_()
            encoder.router.output.bias[CausalMotionRoute.RETREAT] = 12.0
        actions = torch.ones((3, 3, 7))
        slices = ("retreat", "retreat", "retreat_hold")

        report = _applied_residual_ratio_report(
            encoder,
            actions,
            features,
            slices,
        )

        self.assertEqual(
            set(report["candidates"]),
            {"recorded", "zero", "x_zero", "x_opposed"},
        )
        self.assertLessEqual(report["maximum_applied_ratio"], 0.150001)
        self.assertFalse(report["semantic_holds_exact_base"])

    def test_training_contract_authentication_rejects_changed_report_config(
        self,
    ) -> None:
        experiment = _load_experiment_config(
            Path(".scratch/jepa-physical-state-residual-v1/" "experiment-config.json")
        )
        source_revision = "frozen-source-revision"
        metadata = _expected_training_metadata(experiment, source_revision)
        self.assertEqual(metadata.training_recordings, TRAINING_RECORDINGS)
        self.assertEqual(metadata.base_model, MODEL_ID)
        config = _training_config(experiment)
        config_fingerprint = training_configuration_fingerprint(config)
        contract = ActionConditioningContract(
            ACTION_CONDITIONING_SCHEMA,
            metadata,
            "a" * 64,
            config_fingerprint,
            FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            PHYSICAL_TREATMENT_SPEC,
        )
        report = {
            "contract": contract.to_dict(),
            "metadata": metadata.to_dict(),
            "config": config,
            "training_config_fingerprint": config_fingerprint,
            "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "control_artifact": {
                "path": "/tmp/control.pth",
                "fingerprint": config["control_artifact_fingerprint"],
            },
            "passed_router_probe": {
                "path": "/tmp/route-probe.json",
                "fingerprint": config["passed_router_probe_fingerprint"],
            },
        }

        _authenticate_training_contract(
            contract,
            report,
            experiment,
            source_revision=source_revision,
        )
        changed = deepcopy(report)
        changed["config"]["training"]["steps"] += 1

        with self.assertRaisesRegex(ValueError, "contract changed"):
            _authenticate_training_contract(
                contract,
                changed,
                experiment,
                source_revision=source_revision,
            )

    def test_failure_record_preserves_first_reconstructible_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure.json"
            _write_failure_record(
                output,
                "train",
                {"recording": [Path("/tmp/train-00")]},
                ValueError("router gate failed"),
                terminal=True,
            )
            first = json.loads(output.read_text())
            _write_failure_record(
                output,
                "evaluate-train",
                {},
                RuntimeError("later failure"),
                terminal=True,
            )

            self.assertEqual(first, json.loads(output.read_text()))
            self.assertEqual(first["command"], "train")
            self.assertEqual(first["error_type"], "ValueError")
            self.assertTrue(first["terminal_experiment_failure"])
            self.assertFalse(first["retry_same_command_authorized"])
            self.assertFalse(first["retraining_authorized"])

    def test_only_started_experiment_phases_are_terminal(self) -> None:
        self.assertTrue(_is_terminal_command_phase("train", "training_started"))
        self.assertTrue(
            _is_terminal_command_phase("evaluate-train", "evaluation_started")
        )
        self.assertFalse(_is_terminal_command_phase("preflight", None))
        self.assertFalse(_is_terminal_command_phase("train", "training_completed"))
        self.assertFalse(
            _is_terminal_command_phase("evaluate-train", "training_completed")
        )


if __name__ == "__main__":
    unittest.main()
