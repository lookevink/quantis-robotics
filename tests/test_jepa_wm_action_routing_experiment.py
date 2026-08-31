from __future__ import annotations

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
        BASE_COMMAND_ROUTE,
        NEGATIVE_X_COMMAND_ROUTE,
        POSITIVE_X_COMMAND_ROUTE,
    )
    from jepa_wm.action_routing_experiment import (
        CANONICAL_HELD_OUT,
        FRESH_CANARY,
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        _authorize_evaluation,
        _functional_routes,
        _load_experiment_config,
        summarize_canary,
    )
    from jepa_wm.training_artifact import ArtifactIdentity


@unittest.skipIf(torch is None, "PyTorch is required for action routing")
class ActionRoutingExperimentTest(unittest.TestCase):
    def test_frozen_configuration_matches_declared_identity(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / ".scratch/jepa-action-routing-v1/experiment-config.json"
        )

        payload = _load_experiment_config(path)

        self.assertEqual(
            payload["experiment_id"],
            "contact-insertion-v10-drive-slow-2600-runtime-command-routing-v1",
        )
        self.assertEqual(
            FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "98fc2af503919d52a3853d3181bf007d56360136e5c1d27cd1a08a4db18bf66d",
        )

    def test_functional_routes_use_activity_and_mean_horizon_x(self) -> None:
        actions = torch.zeros((3, 4, 7))
        actions[:, 0, 0] = -2e-4
        actions[:, 1, 0] = 2e-4
        actions[:, 2, 1] = 2e-4
        actions[:, 3, 0] = torch.tensor((-4e-4, 2e-4, 2e-4))

        routes, active = _functional_routes(actions)

        torch.testing.assert_close(
            routes,
            torch.tensor(
                (
                    NEGATIVE_X_COMMAND_ROUTE,
                    POSITIVE_X_COMMAND_ROUTE,
                    BASE_COMMAND_ROUTE,
                    BASE_COMMAND_ROUTE,
                )
            ),
        )
        torch.testing.assert_close(active, torch.tensor((True, True, True, True)))

    @staticmethod
    def _report(treatment: str, passed: bool) -> dict:
        return {
            "treatment": treatment,
            "recording": FRESH_CANARY,
            "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "artifact": f"/tmp/{treatment}.pth",
            "artifact_fingerprint": treatment.lower() * 64,
            "aggregate": {"recorded_action_win_rate": 1.0},
            "retained": {"recorded_action_win_rate": 1.0},
            "post": {"recorded_action_win_rate": 1.0},
            "experimental_gate": {"passed": passed},
        }

    def test_canary_selects_only_a_passing_router(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reports = []
            for treatment in ("A", "R"):
                path = root / f"{treatment}.json"
                path.write_text(
                    json.dumps(self._report(treatment, treatment == "R")) + "\n"
                )
                reports.append(path)

            summary = summarize_canary(reports, root / "summary.json")

        self.assertEqual(summary["outcome"], "runtime_router_candidate")
        self.assertEqual(summary["selected_treatment"], "R")
        self.assertTrue(summary["canonical_authorized_offline"])
        self.assertFalse(summary["live_action_authorized"])

    def test_canonical_evaluation_requires_matching_canary_authority(self) -> None:
        artifact = ArtifactIdentity(Path("/tmp/router.pth"), "a" * 64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            summary_path = root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "quantis.jepa_wm_runtime_command_routing_"
                            "canary_summary.v1"
                        ),
                        "experiment_config_fingerprint": (
                            FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
                        ),
                        "selected_treatment": "R",
                        "selected_artifact": {
                            "path": str(artifact.path),
                            "fingerprint": artifact.fingerprint,
                        },
                        "canonical_authorized_offline": True,
                    }
                )
                + "\n"
            )

            _authorize_evaluation(
                CANONICAL_HELD_OUT[0],
                "R",
                artifact,
                summary_path,
            )

            with self.assertRaisesRegex(ValueError, "matching canary authority"):
                _authorize_evaluation(
                    CANONICAL_HELD_OUT[0],
                    "R",
                    ArtifactIdentity(Path("/tmp/other.pth"), "b" * 64),
                    summary_path,
                )

    def test_canonical_evaluation_fails_closed_without_canary_authority(self) -> None:
        artifact = ArtifactIdentity(Path("/tmp/router.pth"), "a" * 64)

        with self.assertRaisesRegex(ValueError, "requires a passing canary"):
            _authorize_evaluation(
                CANONICAL_HELD_OUT[0],
                "R",
                artifact,
                None,
            )
        with self.assertRaisesRegex(ValueError, "not authorized"):
            _authorize_evaluation(
                CANONICAL_HELD_OUT[0],
                "A",
                artifact,
                None,
            )

    def test_fresh_canary_cannot_consume_its_own_summary(self) -> None:
        artifact = ArtifactIdentity(Path("/tmp/router.pth"), "a" * 64)

        with self.assertRaisesRegex(ValueError, "cannot consume its own summary"):
            _authorize_evaluation(
                FRESH_CANARY,
                "R",
                artifact,
                Path("/tmp/summary.json"),
            )


if __name__ == "__main__":
    unittest.main()
