"""Fast grouped probe for causal next-motion route predictability."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.causal_routing import (
    CAUSAL_MOTION_ROUTE_NAMES,
    CausalContextRoutingSpec,
    CausalMotionRoute,
    CausalMotionRouter,
    pool_context_latents,
)


@dataclass(frozen=True)
class CausalRouteProbeConfig:
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
            raise ValueError("causal route probe steps and seed are invalid")
        if (
            not all(
                isfinite(value) and value >= 0.0
                for value in (self.learning_rate, self.weight_decay)
            )
            or self.learning_rate == 0.0
        ):
            raise ValueError("causal route probe optimizer values are invalid")


@dataclass(frozen=True)
class CausalRouteProbeDataset:
    context_latents: torch.Tensor
    context_poses: torch.Tensor
    previous_actions: torch.Tensor
    labels: torch.Tensor
    groups: torch.Tensor
    slices: tuple[str, ...]

    def __post_init__(self) -> None:
        count = self.context_latents.shape[0]
        if (
            self.context_latents.ndim != 2
            or count == 0
            or self.context_poses.shape != (count, ACTION_DIMENSIONS)
            or self.previous_actions.shape != (count, ACTION_DIMENSIONS)
            or self.labels.shape != (count,)
            or self.groups.shape != (count,)
            or len(self.slices) != count
        ):
            raise ValueError("causal route probe tensors do not share one batch")
        if self.labels.dtype == torch.bool or torch.any(
            (self.labels < 0) | (self.labels >= len(CausalMotionRoute))
        ):
            raise ValueError("causal route probe labels are invalid")
        integer_dtypes = {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }
        if self.labels.dtype not in integer_dtypes:
            raise ValueError("causal route probe labels must be integers")
        if (
            self.groups.dtype not in integer_dtypes
            or torch.unique(self.groups).numel() < 2
        ):
            raise ValueError("causal route probe requires at least two groups")

    @classmethod
    def build(
        cls,
        context_latents: torch.Tensor,
        context_poses: torch.Tensor,
        previous_actions: torch.Tensor,
        future_actions: torch.Tensor,
        groups: torch.Tensor,
        slices: Sequence[str],
        *,
        routing: CausalContextRoutingSpec,
    ) -> CausalRouteProbeDataset:
        pooled = (
            pool_context_latents(
                context_latents,
                context_dimension=routing.context_dimension,
            )
            .detach()
            .cpu()
        )
        return cls(
            pooled,
            context_poses.detach().cpu(),
            previous_actions.detach().cpu(),
            routing.classify_action_horizons(future_actions).detach().cpu(),
            groups.detach().cpu(),
            tuple(slices),
        )


def _route_counts(values: torch.Tensor) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {name: counts[index] for index, name in enumerate(CAUSAL_MOTION_ROUTE_NAMES)}


def _metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, object]:
    if labels.numel() == 0 or predictions.shape != labels.shape:
        raise ValueError("causal route metrics require aligned non-empty labels")
    by_route = {}
    for route, name in enumerate(CAUSAL_MOTION_ROUTE_NAMES):
        selected = labels == route
        count = int(selected.sum())
        by_route[name] = {
            "examples": count,
            "recall": (
                float((predictions[selected] == route).float().mean())
                if count
                else None
            ),
        }
    return {
        "examples": int(labels.numel()),
        "accuracy": float((predictions == labels).float().mean()),
        "labels": _route_counts(labels),
        "predictions": _route_counts(predictions),
        "by_route": by_route,
    }


def _balanced_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=len(CausalMotionRoute)).float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = labels.numel() / (present.sum() * counts[present])
    return weights


def run_grouped_causal_route_probe(
    dataset: CausalRouteProbeDataset,
    routing: CausalContextRoutingSpec,
    config: CausalRouteProbeConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    if dataset.context_latents.shape[-1] != routing.context_dimension:
        raise ValueError("causal route probe context dimension changed")
    unique_groups = tuple(int(group) for group in torch.unique(dataset.groups))
    predictions = torch.full_like(
        dataset.labels,
        int(CausalMotionRoute.ACTIVE_OTHER),
    )
    confidences = torch.zeros(dataset.labels.shape, dtype=torch.float32)
    failed_closed = torch.ones(dataset.labels.shape, dtype=torch.bool)
    folds = []
    for fold_index, held_group in enumerate(unique_groups):
        torch.manual_seed(config.seed + fold_index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed + fold_index)
        train_mask = dataset.groups != held_group
        held_mask = ~train_mask
        train_labels = dataset.labels[train_mask].to(device)
        if torch.unique(train_labels).numel() < 2:
            raise ValueError("causal route probe fold has fewer than two routes")
        router = CausalMotionRouter(routing).to(device)
        optimizer = torch.optim.AdamW(
            router.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        train_context = dataset.context_latents[train_mask].to(device)
        train_poses = dataset.context_poses[train_mask].to(device)
        train_previous = dataset.previous_actions[train_mask].to(device)
        class_weights = _balanced_class_weights(train_labels).to(device)
        router.train()
        for _ in range(config.steps):
            loss = torch.nn.functional.cross_entropy(
                router(train_context, train_poses, train_previous),
                train_labels,
                weight=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        router.eval()
        with torch.inference_mode():
            decision = router.decide(
                dataset.context_latents[held_mask].to(device),
                dataset.context_poses[held_mask].to(device),
                dataset.previous_actions[held_mask].to(device),
            )
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
        "schema": "quantis.jepa_wm_causal_route_probe.v1",
        "grouped_holdout": True,
        "candidate_actions_used_as_router_inputs": False,
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
