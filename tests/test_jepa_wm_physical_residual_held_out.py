from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from jepa_wm.physical_residual_held_out import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
    _claim_canonical_access,
    evaluate_population_gate,
    load_experiment_config,
)


def _passing_population() -> dict[str, object]:
    segment = {
        "mean_improvement_over_zero": 0.1,
        "recorded_action_win_rate": 1.0,
        "signed_order_fraction": 1.0,
    }
    return {
        "aggregate": {
            "mean_improvement_over_zero": 0.1,
            "recorded_action_win_rate": 0.95,
        },
        "retained": {"recorded_action_win_rate": 0.90},
        "post": {"recorded_action_win_rate": 0.98},
        "by_segment": {
            name: deepcopy(segment)
            for name in (
                "grasp_attach",
                "retreat",
                "retreat_hold",
                "align",
                "align_hold",
                "insert",
                "seated_hold",
            )
        },
        "router": {
            "accuracy": 0.99,
            "by_route": {
                "retreat": {"recall": 0.99},
                "advance": {"recall": 0.99},
            },
            "grasp_attach_accuracy": 1.0,
            "failed_closed_fraction": 0.01,
            "maximum_semantic_hold_owned_route_activations": 0,
        },
        "maximum_residual_ratio": 0.15000002,
        "semantic_holds_exact_base": True,
    }


class PhysicalResidualHeldOutTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_experiment_config(
            Path(".scratch/jepa-physical-state-held-out-v1/experiment-config.json")
        )

    def test_frozen_config_binds_exact_two_seed_roster(self) -> None:
        recordings = self.config["corpus"]["recordings"]

        self.assertEqual(
            FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "53882191d63b6408297b78b5780337902eaeb7d444e0cdc80cd94a43f2b2c18a",
        )
        self.assertEqual([item["seed"] for item in recordings], [12600, 12601])
        self.assertEqual(len({item["manifest_fingerprint"] for item in recordings}), 2)
        self.assertEqual(self.config["execution"]["evaluations"], 1)
        self.assertFalse(self.config["execution"]["train"])

    def test_population_gate_accepts_numerical_ratio_tolerance(self) -> None:
        result = evaluate_population_gate(_passing_population(), self.config["gate"])

        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])

    def test_population_gate_fails_closed_for_every_gate_family(self) -> None:
        mutations = {
            "energy": lambda value: value["aggregate"].update(
                mean_improvement_over_zero=0.0
            ),
            "segment": lambda value: value["by_segment"]["align"].update(
                signed_order_fraction=0.74
            ),
            "router": lambda value: value["router"].update(accuracy=0.94),
            "hold": lambda value: value.update(semantic_holds_exact_base=False),
            "ratio": lambda value: value.update(maximum_residual_ratio=0.15001),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                population = _passing_population()
                mutate(population)
                self.assertFalse(
                    evaluate_population_gate(population, self.config["gate"])[
                        "passed"
                    ]
                )

    def test_access_claim_is_atomic_and_one_shot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            claim = Path(directory) / "claim.json"
            recordings = (Path("held-00"), Path("held-01"))

            payload = _claim_canonical_access(claim, recordings)

            self.assertEqual(payload["evaluations_claimed"], 1)
            self.assertEqual(payload["recordings"], ["held-00", "held-01"])
            with self.assertRaisesRegex(ValueError, "already claimed"):
                _claim_canonical_access(claim, recordings)


if __name__ == "__main__":
    unittest.main()
