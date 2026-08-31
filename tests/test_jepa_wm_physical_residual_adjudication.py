from __future__ import annotations

import unittest

from jepa_wm.physical_residual_adjudication import adjudicate_report


class PhysicalResidualAdjudicationTest(unittest.TestCase):
    def test_adjudicates_only_the_immutable_numeric_boundary_failure(self) -> None:
        report = {
            "schema": "quantis.jepa_wm_physical_state_residual_train_evaluation.v1",
            "status": "evaluated",
            "outcome": "physical_state_residual_train_failed",
            "experiment_config_fingerprint": (
                "b296b7fc064627f13ed87c1baeaf84d4961f1b04db115f9afcc689bf05dda78d"
            ),
            "artifact": {
                "path": "/tmp/physical_state_residual.pth",
                "fingerprint": (
                    "7f3cb2a99e749dd56e4f4a988b3b1c332ab13c4661c3413118ffb270f237db51"
                ),
            },
            "training_report": {
                "path": "/tmp/physical_state_residual.pth.json",
                "fingerprint": (
                    "08f41c36f4e0987c4652244ee0011001de30c76a6e014bd94ad4966780d78aba"
                ),
            },
            "training_selection_fingerprint": (
                "f13bfc6d8eef6ca875d6f95f0bd5194a927e9bdf068377ea0ca4851c10b38d74"
            ),
            "selected_input_fingerprint": (
                "576404f64ac55f47490ef8358eb2121f4dd044f5ab72e396a2817f439fe3d839"
            ),
            "aggregate": {
                "mean_improvement_over_zero": 0.001363743911497295,
                "recorded_action_win_rate": 0.9821428656578064,
            },
            "retained": {"recorded_action_win_rate": 0.9433962106704712},
            "post": {"recorded_action_win_rate": 1.0},
            "by_segment": {
                name: {
                    "mean_improvement_over_zero": improvement,
                    "signed_order_fraction": signed_order,
                }
                for name, improvement, signed_order in (
                    ("grasp_attach", 0.0005714914877898991, 1.0),
                    ("retreat", 0.0016532990848645568, 0.9583333134651184),
                    ("retreat_hold", 0.000017448561266064644, 0.75),
                    ("align", 0.0021645897068083286, 1.0),
                    ("align_hold", 0.000007720042958681006, 1.0),
                    ("insert", 0.0007061410578899086, 1.0),
                    ("seated_hold", 0.0000005653903940583405, 1.0),
                )
            },
            "final_router": {"gate_passed": True},
            "residual_ratios": {
                "maximum_applied_ratio": 0.15000002086162567,
                "semantic_holds_exact_base": True,
            },
            "experimental_gate": {
                "passed": False,
                "minimum_overall_win_rate": 0.9,
                "minimum_retained_win_rate": 0.85,
                "minimum_post_win_rate": 0.95,
                "minimum_signed_order_fraction": {
                    "retreat": 0.75,
                    "align": 0.75,
                    "insert": 0.75,
                },
                "requires_positive_mean_each_segment": True,
                "maximum_applied_residual_to_base_embedding_ratio": 0.15,
                "requires_exact_base_in_semantic_holds": True,
            },
            "held_out_accessed": False,
            "canonical_accessed": False,
            "live_action_authorized": False,
        }
        gate = {
            "minimum_overall_win_rate": 0.9,
            "minimum_retained_win_rate": 0.85,
            "minimum_post_win_rate": 0.95,
            "minimum_signed_order_fraction": 0.75,
            "required_signed_segments": ["retreat", "align", "insert"],
            "maximum_residual_ratio": 0.15,
            "residual_ratio_absolute_tolerance": 0.000001,
            "require_positive_mean_each_segment": True,
            "require_exact_base_in_semantic_holds": True,
            "require_final_router_gate": True,
        }

        result = adjudicate_report(report, gate)

        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(
            result["original_outcome"], "physical_state_residual_train_failed"
        )
        self.assertEqual(result["outcome"], "physical_state_residual_train_candidate")
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["recordings_loaded"])
        self.assertFalse(result["rescored"])
        self.assertFalse(result["trained"])


if __name__ == "__main__":
    unittest.main()
