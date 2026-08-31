"""Grouped route probe over task-relative physical observations."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch

from jepa_wm.causal_route_probe import (
    _balanced_class_weights,
    _metrics,
)
from jepa_wm.causal_routing import CausalMotionRoute
from jepa_wm.physical_routing import (
    PHYSICAL_ROUTING_FEATURE_DIMENSION,
    PhysicalMotionRouter,
    PhysicalStateRoutingSpec,
)


@dataclass(frozen=True)
class PhysicalRouteProbeConfig:
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
        ):
            raise ValueError("physical route probe steps and seed are invalid")
        if (
            not all(
                isfinite(value) and value >= 0.0
                for value in (self.learning_rate, self.weight_decay)
            )
            or self.learning_rate == 0.0
        ):
            raise ValueError("physical route probe optimizer values are invalid")


@dataclass(frozen=True)
class PhysicalRouteProbeDataset:
    features: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor
    slices: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features.ndim != 2:
            raise ValueError("physical route probe dataset is invalid")
        count = self.features.shape[0]
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if (
            self.features.shape != (count, PHYSICAL_ROUTING_FEATURE_DIMENSION)
            or count == 0
            or not torch.isfinite(self.features).all()
            or self.labels.shape != (count,)
            or self.labels.dtype not in integer_dtypes
            or torch.any((self.labels < 0) | (self.labels >= len(CausalMotionRoute)))
            or self.groups.shape != (count,)
            or self.groups.dtype not in integer_dtypes
            or torch.unique(self.groups).numel() < 2
            or len(self.slices) != count
        ):
            raise ValueError("physical route probe dataset is invalid")


def run_grouped_physical_route_probe(
    dataset: PhysicalRouteProbeDataset,
    routing: PhysicalStateRoutingSpec,
    config: PhysicalRouteProbeConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    predictions = torch.full_like(
        dataset.labels,
        int(CausalMotionRoute.ACTIVE_OTHER),
    )
    confidences = torch.zeros(dataset.labels.shape, dtype=torch.float32)
    failed_closed = torch.ones(dataset.labels.shape, dtype=torch.bool)
    folds = []
    for fold_index, held_group in enumerate(
        int(group) for group in torch.unique(dataset.groups)
    ):
        torch.manual_seed(config.seed + fold_index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed + fold_index)
        train_mask = dataset.groups != held_group
        held_mask = ~train_mask
        train_features = dataset.features[train_mask].to(device)
        train_labels = dataset.labels[train_mask].to(device)
        if torch.unique(train_labels).numel() < 2:
            raise ValueError("physical route probe fold has fewer than two routes")
        router = PhysicalMotionRouter(routing).to(device)
        router.fit_normalization(train_features)
        optimizer = torch.optim.AdamW(
            router.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        class_weights = _balanced_class_weights(train_labels).to(device)
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
        router.eval()
        with torch.inference_mode():
            decision = router.decide(dataset.features[held_mask].to(device))
        held_predictions = decision.routes.cpu()
        predictions[held_mask] = held_predictions
        confidences[held_mask] = decision.confidence.cpu()
        failed_closed[held_mask] = decision.failed_closed.cpu()
        folds.append(
            {
                "held_group": held_group,
                **_metrics(dataset.labels[held_mask], held_predictions),
                "failed_closed_fraction": float(decision.failed_closed.float().mean()),
                "mean_confidence": float(decision.confidence.mean()),
            }
        )
    by_slice = {
        name: _metrics(dataset.labels[indices], predictions[indices])
        for name in sorted(set(dataset.slices))
        if (indices := torch.tensor([value == name for value in dataset.slices])).any()
    }
    return {
        "schema": "quantis.jepa_wm_physical_route_probe.v1",
        "grouped_holdout": True,
        "runtime_inputs": ["physical_observation"],
        "candidate_actions_used_as_router_inputs": False,
        "visual_latents_used_as_router_inputs": False,
        "routing": routing.to_dict(),
        "config": {
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "seed": config.seed,
        },
        "overall": _metrics(dataset.labels, predictions),
        "by_slice": by_slice,
        "folds": folds,
        "failed_closed_fraction": float(failed_closed.float().mean()),
        "mean_confidence": float(confidences.mean()),
    }
