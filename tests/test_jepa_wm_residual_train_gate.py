from __future__ import annotations

import unittest

from jepa_wm.readiness import ResidualTrainGate, ResidualTrainReason


class ResidualTrainGateTest(unittest.TestCase):
    def test_accepts_immutable_report_at_floating_point_bound(self) -> None:
        decision = ResidualTrainGate().evaluate(
            aggregate={
                "mean_improvement_over_zero": 0.001363743911497295,
                "recorded_action_win_rate": 0.9821428656578064,
            },
            retained={"recorded_action_win_rate": 0.9433962106704712},
            post={"recorded_action_win_rate": 1.0},
            by_segment={
                "grasp_attach": {
                    "mean_improvement_over_zero": 0.0005714914877898991,
                    "signed_order_fraction": 1.0,
                },
                "retreat": {
                    "mean_improvement_over_zero": 0.0016532990848645568,
                    "signed_order_fraction": 0.9583333134651184,
                },
                "retreat_hold": {
                    "mean_improvement_over_zero": 0.000017448561266064644,
                    "signed_order_fraction": 0.75,
                },
                "align": {
                    "mean_improvement_over_zero": 0.0021645897068083286,
                    "signed_order_fraction": 1.0,
                },
                "align_hold": {
                    "mean_improvement_over_zero": 0.000007720042958681006,
                    "signed_order_fraction": 1.0,
                },
                "insert": {
                    "mean_improvement_over_zero": 0.0007061410578899086,
                    "signed_order_fraction": 1.0,
                },
                "seated_hold": {
                    "mean_improvement_over_zero": 0.0000005653903940583405,
                    "signed_order_fraction": 1.0,
                },
            },
            maximum_residual_ratio=0.15000002086162567,
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.reasons, ())

    def test_rejects_residual_ratio_outside_numerical_tolerance(self) -> None:
        decision = ResidualTrainGate().evaluate(
            aggregate={
                "mean_improvement_over_zero": 0.001,
                "recorded_action_win_rate": 0.99,
            },
            retained={"recorded_action_win_rate": 0.99},
            post={"recorded_action_win_rate": 1.0},
            by_segment={
                name: {
                    "mean_improvement_over_zero": 0.001,
                    "signed_order_fraction": 1.0,
                }
                for name in ("retreat", "align", "insert")
            },
            maximum_residual_ratio=0.15001,
        )

        self.assertFalse(decision.passed)
        self.assertEqual(
            decision.reasons,
            (ResidualTrainReason.EXCESS_RESIDUAL_RATIO,),
        )

    def test_fails_closed_for_non_finite_gate_metrics(self) -> None:
        baseline = {
            "aggregate": {
                "mean_improvement_over_zero": 0.001,
                "recorded_action_win_rate": 0.99,
            },
            "retained": {"recorded_action_win_rate": 0.99},
            "post": {"recorded_action_win_rate": 1.0},
            "by_segment": {
                name: {
                    "mean_improvement_over_zero": 0.001,
                    "signed_order_fraction": 1.0,
                }
                for name in ("retreat", "align", "insert")
            },
            "maximum_residual_ratio": 0.15,
        }
        cases = (
            ("retained", "recorded_action_win_rate", float("nan")),
            ("post", "recorded_action_win_rate", float("inf")),
            ("retreat", "mean_improvement_over_zero", float("nan")),
            ("align", "signed_order_fraction", float("inf")),
        )
        for section, metric, value in cases:
            with self.subTest(section=section, metric=metric):
                arguments = {
                    "aggregate": dict(baseline["aggregate"]),
                    "retained": dict(baseline["retained"]),
                    "post": dict(baseline["post"]),
                    "by_segment": {
                        name: dict(values)
                        for name, values in baseline["by_segment"].items()
                    },
                    "maximum_residual_ratio": baseline["maximum_residual_ratio"],
                }
                target = (
                    arguments["by_segment"][section]
                    if section in arguments["by_segment"]
                    else arguments[section]
                )
                target[metric] = value

                decision = ResidualTrainGate().evaluate(**arguments)

                self.assertFalse(decision.passed)
                self.assertIn(
                    ResidualTrainReason.NON_FINITE_GATE_METRIC,
                    decision.reasons,
                )


if __name__ == "__main__":
    unittest.main()
