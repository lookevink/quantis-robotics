"""Nonlinear candidate-independent routing from task-relative physical state."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import torch

from jepa_wm.causal_routing import CausalMotionRoute, CausalRouteDecision
from jepa_wm.physical_observation import (
    PHYSICAL_ROUTING_FEATURE_NAMES,
    PHYSICAL_ROUTING_OBSERVATION_SCHEMA,
)


PHYSICAL_ROUTING_FEATURE_DIMENSION = len(PHYSICAL_ROUTING_FEATURE_NAMES)


@dataclass(frozen=True)
class PhysicalStateRoutingSpec:
    hidden_dimensions: tuple[int, ...]
    minimum_route_confidence: float
    maximum_residual_ratio: float

    def __post_init__(self) -> None:
        if not self.hidden_dimensions or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in self.hidden_dimensions
        ):
            raise ValueError("physical router hidden dimensions must be positive")
        if (
            not isfinite(self.minimum_route_confidence)
            or not 0.0 < self.minimum_route_confidence <= 1.0
        ):
            raise ValueError("physical router confidence must be in (0, 1]")
        if (
            not isfinite(self.maximum_residual_ratio)
            or not 0.0 < self.maximum_residual_ratio <= 1.0
        ):
            raise ValueError("physical residual ratio must be in (0, 1]")

    @classmethod
    def from_dict(cls, payload: Any) -> PhysicalStateRoutingSpec:
        if not isinstance(payload, dict) or set(payload) != {
            "observation_schema",
            "feature_names",
            "hidden_dimensions",
            "minimum_route_confidence",
            "maximum_residual_ratio",
        }:
            raise ValueError("physical routing specification is invalid")
        if payload["observation_schema"] != PHYSICAL_ROUTING_OBSERVATION_SCHEMA:
            raise ValueError("physical routing specification is invalid")
        if payload["feature_names"] != list(PHYSICAL_ROUTING_FEATURE_NAMES):
            raise ValueError("physical routing specification is invalid")
        hidden = payload["hidden_dimensions"]
        if not isinstance(hidden, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in hidden
        ):
            raise ValueError("physical routing specification is invalid")
        try:
            return cls(
                tuple(hidden),
                float(payload["minimum_route_confidence"]),
                float(payload["maximum_residual_ratio"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("physical routing specification is invalid") from error

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_schema": PHYSICAL_ROUTING_OBSERVATION_SCHEMA,
            "feature_names": list(PHYSICAL_ROUTING_FEATURE_NAMES),
            "hidden_dimensions": list(self.hidden_dimensions),
            "minimum_route_confidence": self.minimum_route_confidence,
            "maximum_residual_ratio": self.maximum_residual_ratio,
        }


class PhysicalMotionRouter(torch.nn.Module):
    """Route normalized physical observations through a small nonlinear map."""

    def __init__(self, spec: PhysicalStateRoutingSpec) -> None:
        super().__init__()
        self.spec = spec
        self.register_buffer(
            "feature_mean",
            torch.zeros(PHYSICAL_ROUTING_FEATURE_DIMENSION),
        )
        self.register_buffer(
            "feature_scale",
            torch.ones(PHYSICAL_ROUTING_FEATURE_DIMENSION),
        )
        self.register_buffer("normalization_fitted", torch.tensor(False))
        dimensions = (PHYSICAL_ROUTING_FEATURE_DIMENSION, *spec.hidden_dimensions)
        self.hidden = torch.nn.ModuleList(
            torch.nn.Linear(input_dimension, output_dimension)
            for input_dimension, output_dimension in zip(
                dimensions,
                dimensions[1:],
            )
        )
        self.output = torch.nn.Linear(dimensions[-1], len(CausalMotionRoute))
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def _validated_features(self, features: torch.Tensor) -> torch.Tensor:
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != PHYSICAL_ROUTING_FEATURE_DIMENSION
            or not torch.isfinite(features).all()
        ):
            raise ValueError("physical router features must be finite [batch, 26]")
        return features.to(
            device=self.feature_mean.device,
            dtype=self.feature_mean.dtype,
        )

    def fit_normalization(self, features: torch.Tensor) -> None:
        values = self._validated_features(features)
        with torch.no_grad():
            mean = values.mean(dim=0)
            scale = values.std(dim=0, unbiased=False)
            scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
            self.feature_mean.copy_(mean)
            self.feature_scale.copy_(scale)
            self.normalization_fitted.fill_(True)

    def normalized_features(self, features: torch.Tensor) -> torch.Tensor:
        values = self._validated_features(features)
        if not bool(self.normalization_fitted.item()):
            raise ValueError("physical router normalization has not been fitted")
        return (values - self.feature_mean) / self.feature_scale

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        values = self.normalized_features(features)
        for layer in self.hidden:
            values = torch.nn.functional.silu(layer(values))
        return self.output(values)

    def decide(self, features: torch.Tensor) -> CausalRouteDecision:
        logits = self(features)
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted = probabilities.max(dim=-1)
        accepted = confidence >= self.spec.minimum_route_confidence
        active_other = torch.full_like(predicted, int(CausalMotionRoute.ACTIVE_OTHER))
        routes = torch.where(accepted, predicted, active_other)
        failed_closed = (~accepted) | (routes == int(CausalMotionRoute.ACTIVE_OTHER))
        residual_weights = torch.stack(
            (
                accepted & (routes == int(CausalMotionRoute.RETREAT)),
                accepted & (routes == int(CausalMotionRoute.ADVANCE)),
            ),
            dim=-1,
        ).to(dtype=logits.dtype)
        return CausalRouteDecision(
            routes,
            confidence,
            residual_weights,
            failed_closed,
        )
