"""Fit one final physical-state router before residual training."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from jepa_wm.causal_routing import CausalMotionRoute
from jepa_wm.physical_routing import (
    PHYSICAL_ROUTING_FEATURE_DIMENSION,
    PhysicalMotionRouter,
    PhysicalStateRoutingSpec,
)
from jepa_wm.route_metrics import (
    balanced_route_class_weights,
    route_metrics,
)


@dataclass(frozen=True)
class PhysicalRouterTrainingConfig:
    steps: int
    learning_rate: float
    weight_decay: float
    seed: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.steps, bool)
            or not isinstance(self.steps, int)
            or self.steps <= 0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or not isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
            or not isfinite(self.weight_decay)
            or self.weight_decay < 0.0
        ):
            raise ValueError("physical router training configuration is invalid")


def fit_final_physical_router(
    features: torch.Tensor,
    labels: torch.Tensor,
    routing: PhysicalStateRoutingSpec,
    config: PhysicalRouterTrainingConfig,
    *,
    device: torch.device,
) -> tuple[PhysicalMotionRouter, dict[str, object]]:
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if (
        features.ndim != 2
        or features.shape[0] == 0
        or features.shape[1] != PHYSICAL_ROUTING_FEATURE_DIMENSION
        or not torch.isfinite(features).all()
        or labels.shape != (features.shape[0],)
        or labels.dtype not in integer_dtypes
        or torch.any((labels < 0) | (labels >= len(CausalMotionRoute)))
    ):
        raise ValueError("final physical router training data is invalid")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    train_features = features.to(device)
    train_labels = labels.to(device)
    router = PhysicalMotionRouter(routing).to(device)
    router.fit_normalization(train_features)
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    class_weights = balanced_route_class_weights(train_labels).to(device)
    losses = []
    router.train()
    for _ in range(config.steps):
        loss = torch.nn.functional.cross_entropy(
            router(train_features),
            train_labels,
            weight=class_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    router.eval()
    with torch.inference_mode():
        decision = router.decide(train_features)
    report = {
        **route_metrics(labels.cpu(), decision.routes.cpu()),
        "failed_closed_fraction": float(decision.failed_closed.float().mean()),
        "mean_confidence": float(decision.confidence.mean()),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "normalization_fitted": bool(router.normalization_fitted.item()),
    }
    return router, report
