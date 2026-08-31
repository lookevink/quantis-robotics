from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from jepa_wm.physical_residual_held_out_v2 import (
    _claim_canonical_access,
    _claim_after_preclaim_authentication,
    authenticate_prior_evidence,
    load_experiment_config,
)
from jepa_wm.physical_residual_held_out_v2_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)


class PhysicalResidualHeldOutV2Test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_experiment_config(
            Path(".scratch/jepa-physical-state-held-out-v2/experiment-config.json")
        )

    def test_contract_binds_one_exact_two_seed_gate_and_no_expansion(self) -> None:
        corpus = self.config["corpus"]
        execution = self.config["execution"]

        self.assertEqual([item["seed"] for item in corpus["recordings"]], [12600, 12601])
        self.assertEqual(execution["evaluations"], 1)
        self.assertFalse(execution["train"])
        self.assertFalse(execution["run_isaac"])
        self.assertFalse(execution["issue_live_action"])
        self.assertFalse(execution["film"])
        self.assertEqual(
            FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "a4ed3975e75104785f495dc5d0affda48c02e527f64c783d7f51ff2f2187fb18",
        )
        self.assertEqual(
            self.config["evaluator"]["fingerprint"],
            "caed23e4763ee8163ef558c4d4027e14d917326d236ade44f8c9a172bc841d0f",
        )
        self.assertEqual(
            self.config["evaluator"]["implementation_revision"],
            "e0ea97bbb9fb8d4111ac7677122e5ce3484c5d91",
        )

    def test_contract_binds_consumed_v1_negative_and_runtime_remediation(self) -> None:
        prior = self.config["prior_attempt"]
        runtime = self.config["runtime_remediation"]

        self.assertEqual(
            prior["access_claim"]["fingerprint"],
            "b36c174f6ae8073bfde2d2618b830041f94673a69dac4fe98acfc6f1559cb358",
        )
        self.assertEqual(
            prior["failure"]["fingerprint"],
            "15b5e63c11232be938b4b8edcbf732b1e58be1a21ac7331566bc233564511d3e",
        )
        self.assertTrue(prior["require_no_evaluation_report"])
        self.assertEqual(
            runtime["report"]["fingerprint"],
            "55f6786b96bfefd95c4d8f6fa450324c364e7a04baac4665e382458f83cc0179",
        )
        self.assertEqual(
            runtime["claim"]["fingerprint"],
            "782afe7d20d5a090591b4dbd27c275713d5c894578800152dbe2b0615f14b38f",
        )

    def test_prior_evidence_rejects_mutation_and_requires_no_v1_score(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            claim = root / "claim.json"
            failure = root / "failure.json"
            runtime_report = root / "runtime.json"
            runtime_claim = root / "runtime-claim.json"
            for path, value in (
                (
                    claim,
                    b'{"schema":"quantis.jepa_wm_physical_state_residual_held_out_access.v1","evaluations_claimed":1}',
                ),
                (
                    failure,
                    b'{"schema":"quantis.jepa_wm_physical_state_residual_held_out_failure.v1","canonical_accessed":true,"terminal_experiment_failure":true,"retry_authorized":false,"retraining_authorized":false,"live_action_authorized":false,"filming_authorized":false}',
                ),
                (runtime_report, b"runtime"),
                (runtime_claim, b"runtime-claim"),
            ):
                path.write_bytes(value)
            experiment = {
                "prior_attempt": {
                    "access_claim": {"path": str(claim), "fingerprint": _sha256(claim)},
                    "failure": {"path": str(failure), "fingerprint": _sha256(failure)},
                    "evaluation_report": str(root / "must-not-exist.json"),
                    "require_no_evaluation_report": True,
                },
                "runtime_remediation": {
                    "report": {"path": str(runtime_report), "fingerprint": _sha256(runtime_report)},
                    "claim": {"path": str(runtime_claim), "fingerprint": _sha256(runtime_claim)},
                },
            }

            authenticate_prior_evidence(experiment)
            runtime_report.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "runtime remediation report"):
                authenticate_prior_evidence(experiment)
            runtime_report.write_bytes(b"runtime")
            (root / "must-not-exist.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "produced an evaluation"):
                authenticate_prior_evidence(experiment)

    def test_access_claim_is_atomic_and_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / "claim.json"
            payload = _claim_canonical_access(
                claim, (Path("held-00"), Path("held-01")), "config-sha"
            )

            self.assertEqual(payload["evaluations_claimed"], 1)
            self.assertEqual(payload["experiment_config_fingerprint"], "config-sha")
            with self.assertRaisesRegex(ValueError, "already claimed"):
                _claim_canonical_access(
                    claim, (Path("held-00"), Path("held-01")), "config-sha"
                )

    def test_claim_seam_fails_closed_until_preclaim_authentication_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / "claim.json"

            with self.assertRaisesRegex(ValueError, "complete pre-claim"):
                _claim_after_preclaim_authentication(
                    object(),
                    claim,
                    (Path("held-00"), Path("held-01")),
                )

            self.assertFalse(claim.exists())


def _sha256(path: Path) -> str:
    from hashlib import sha256

    return sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
