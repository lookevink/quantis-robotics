from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_conditioning_experiment import (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        _load_experiment_config,
        _rollout_regimes,
        summarize_canary,
    )


@unittest.skipIf(torch is None, "PyTorch is required for action conditioning")
class ActionConditioningExperimentTest(unittest.TestCase):
    def test_frozen_configuration_matches_its_declared_fingerprint(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / ".scratch/jepa-action-conditioning-48h/experiment-config.json"
        )

        payload = _load_experiment_config(path)

        self.assertEqual(
            payload["experiment_id"],
            "contact-insertion-v10-drive-slow-2600-action-conditioning-v1",
        )
        self.assertEqual(
            FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "a8e111cbd197592091c93cf1d00adb751ede97a194c537559ffe10e7b5e7de14",
        )

    def test_regime_boundary_is_semantic_and_complete(self) -> None:
        rollouts = tuple(
            SimpleNamespace(context=(SimpleNamespace(index=index),))
            for index in (113, 165, 166, 280)
        )

        regimes = _rollout_regimes(rollouts)

        torch.testing.assert_close(regimes, torch.tensor((0, 0, 1, 1)))

    @staticmethod
    def _report(treatment: str, passed: bool) -> dict:
        return {
            "treatment": treatment,
            "recording": "contact-insertion-v10-drive-slow-72600-held-00",
            "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "artifact": f"/tmp/{treatment}.pth",
            "artifact_fingerprint": treatment.lower() * 64,
            "aggregate": {"recorded_action_win_rate": 1.0},
            "retained": {"recorded_action_win_rate": 1.0},
            "post": {"recorded_action_win_rate": 1.0},
            "experimental_gate": {"passed": passed},
        }

    def test_canary_prefers_simpler_balanced_linear_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = []
            for treatment in ("A", "B", "C", "D"):
                path = root / f"{treatment}.json"
                path.write_text(
                    json.dumps(self._report(treatment, treatment != "A")) + "\n"
                )
                reports.append(path)
            output = root / "summary.json"

            summary = summarize_canary(reports, output)

        self.assertEqual(summary["outcome"], "balanced_linear_candidate")
        self.assertEqual(summary["selected_treatment"], "B")

    def test_oracle_only_success_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = []
            for treatment in ("A", "B", "C", "D"):
                path = root / f"{treatment}.json"
                path.write_text(
                    json.dumps(self._report(treatment, treatment == "D")) + "\n"
                )
                reports.append(path)

            summary = summarize_canary(reports, root / "summary.json")

        self.assertEqual(summary["outcome"], "regime_conflict_confirmed")
        self.assertIsNone(summary["selected_treatment"])


if __name__ == "__main__":
    unittest.main()
