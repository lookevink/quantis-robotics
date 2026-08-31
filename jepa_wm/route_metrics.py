"""Shared metrics and balancing for four-way physical motion routes."""

from __future__ import annotations

from collections import Counter

import torch

from jepa_wm.causal_routing import (
    CAUSAL_MOTION_ROUTE_NAMES,
    CausalMotionRoute,
)


def route_counts(values: torch.Tensor) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {name: counts[index] for index, name in enumerate(CAUSAL_MOTION_ROUTE_NAMES)}


def route_metrics(
    labels: torch.Tensor,
    predictions: torch.Tensor,
) -> dict[str, object]:
    if labels.numel() == 0 or predictions.shape != labels.shape:
        raise ValueError("route metrics require aligned non-empty labels")
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
        "labels": route_counts(labels),
        "predictions": route_counts(predictions),
        "by_route": by_route,
    }


def balanced_route_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=len(CausalMotionRoute)).float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = labels.numel() / (present.sum() * counts[present])
    return weights
