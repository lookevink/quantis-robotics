from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.causal_routing import CausalMotionRoute
    from jepa_wm.physical_observation import PHYSICAL_ROUTING_FEATURE_NAMES
    from jepa_wm.physical_routing import PhysicalStateRoutingSpec
    from jepa_wm.physical_routing_training import (
        PhysicalRouterTrainingConfig,
        fit_final_physical_router,
    )


@unittest.skipIf(torch is None, "PyTorch is required for physical routing")
class PhysicalRoutingTrainingTest(unittest.TestCase):
    def test_final_router_fits_normalization_and_reports_train_routes(self) -> None:
        labels = torch.tensor((0, 1, 2, 3) * 4)
        features = torch.zeros((16, len(PHYSICAL_ROUTING_FEATURE_NAMES)))
        features[:, :4] = torch.nn.functional.one_hot(labels, num_classes=4).float()

        router, report = fit_final_physical_router(
            features,
            labels,
            PhysicalStateRoutingSpec((8, 8), 0.55, 0.15),
            PhysicalRouterTrainingConfig(
                steps=300,
                learning_rate=0.05,
                weight_decay=0.0,
                seed=19,
            ),
            device=torch.device("cpu"),
        )

        self.assertTrue(bool(router.normalization_fitted.item()))
        self.assertEqual(report["accuracy"], 1.0)
        self.assertEqual(
            report["labels"],
            {"hold": 4, "retreat": 4, "advance": 4, "active_other": 4},
        )
        self.assertEqual(
            router.decide(features).routes.tolist(),
            labels.tolist(),
        )
        self.assertEqual(
            report["by_route"]["retreat"]["recall"],
            1.0,
        )
        self.assertEqual(
            report["by_route"]["advance"]["recall"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
