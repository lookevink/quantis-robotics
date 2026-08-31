"""Candidate-independent causal routing for context-conditioned action maps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import isfinite
from typing import Any, Sequence

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.insertion_layout import (
    CONTACT_INSERTION_PASSTHROUGH_SEGMENTS,
    ContactInsertionSegment,
)


class CausalMotionRoute(IntEnum):
    """Observable next-motion classes; only two classes own residual experts."""

    HOLD = 0
    RETREAT = 1
    ADVANCE = 2
    ACTIVE_OTHER = 3


CAUSAL_MOTION_ROUTE_NAMES = tuple(route.name.lower() for route in CausalMotionRoute)


@dataclass(frozen=True)
class RecordedMotionLabelSpec:
    """Classify demonstrated intent while preserving declared semantic holds."""

    signed_x_deadband: float
    translation_activity_deadband: float
    rotation_activity_deadband: float
    gripper_activity_deadband: float

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value >= 0.0
            for value in (
                self.signed_x_deadband,
                self.translation_activity_deadband,
                self.rotation_activity_deadband,
                self.gripper_activity_deadband,
            )
        ):
            raise ValueError(
                "motion route label deadbands must be finite and non-negative"
            )

    @classmethod
    def from_dict(cls, payload: Any) -> RecordedMotionLabelSpec:
        expected = {
            "signed_x_deadband",
            "translation_activity_deadband",
            "rotation_activity_deadband",
            "gripper_activity_deadband",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("motion route labeling specification is invalid")
        try:
            return cls(
                signed_x_deadband=float(payload["signed_x_deadband"]),
                translation_activity_deadband=float(
                    payload["translation_activity_deadband"]
                ),
                rotation_activity_deadband=float(
                    payload["rotation_activity_deadband"]
                ),
                gripper_activity_deadband=float(payload["gripper_activity_deadband"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "motion route labeling specification is invalid"
            ) from error

    def classify_action_horizons(self, actions: torch.Tensor) -> torch.Tensor:
        if (
            actions.ndim != 3
            or actions.shape[0] == 0
            or actions.shape[-1] != ACTION_DIMENSIONS
        ):
            raise ValueError("future actions must have shape [horizon, batch, 7]")
        by_rollout = actions.transpose(0, 1)
        translation_active = (
            torch.linalg.vector_norm(by_rollout[..., :3], dim=-1)
            > self.translation_activity_deadband
        )
        rotation_active = (
            torch.linalg.vector_norm(by_rollout[..., 3:6], dim=-1)
            > self.rotation_activity_deadband
        )
        gripper_active = by_rollout[..., 6].abs() > self.gripper_activity_deadband
        active = (translation_active | rotation_active | gripper_active).any(dim=1)
        mean_x = by_rollout[..., 0].mean(dim=1)
        routes = torch.full(
            (by_rollout.shape[0],),
            int(CausalMotionRoute.HOLD),
            dtype=torch.long,
            device=actions.device,
        )
        routes = torch.where(active, int(CausalMotionRoute.ACTIVE_OTHER), routes)
        routes = torch.where(
            active & (mean_x < -self.signed_x_deadband),
            int(CausalMotionRoute.RETREAT),
            routes,
        )
        return torch.where(
            active & (mean_x > self.signed_x_deadband),
            int(CausalMotionRoute.ADVANCE),
            routes,
        )

    def classify_recorded_horizons(
        self,
        actions: torch.Tensor,
        segments: Sequence[ContactInsertionSegment],
    ) -> torch.Tensor:
        routes = self.classify_action_horizons(actions)
        if len(segments) != routes.shape[0] or any(
            not isinstance(segment, ContactInsertionSegment) for segment in segments
        ):
            raise ValueError("recorded route segments do not match the action batch")
        passthrough = torch.tensor(
            [segment in CONTACT_INSERTION_PASSTHROUGH_SEGMENTS for segment in segments],
            dtype=torch.bool,
            device=routes.device,
        )
        return torch.where(passthrough, int(CausalMotionRoute.HOLD), routes)


@dataclass(frozen=True)
class CausalContextRoutingSpec:
    context_dimension: int
    router_hidden_dimension: int
    signed_x_deadband: float
    translation_activity_deadband: float
    rotation_activity_deadband: float
    gripper_activity_deadband: float
    minimum_route_confidence: float
    maximum_residual_ratio: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_dimension, bool)
            or not isinstance(self.context_dimension, int)
            or self.context_dimension <= 0
            or isinstance(self.router_hidden_dimension, bool)
            or not isinstance(self.router_hidden_dimension, int)
            or self.router_hidden_dimension <= 0
        ):
            raise ValueError("causal routing dimensions must be positive integers")
        deadbands = (
            self.signed_x_deadband,
            self.translation_activity_deadband,
            self.rotation_activity_deadband,
            self.gripper_activity_deadband,
        )
        if not all(isfinite(value) and value >= 0.0 for value in deadbands):
            raise ValueError("causal routing deadbands must be finite and non-negative")
        if (
            not isfinite(self.minimum_route_confidence)
            or not 0.0 < self.minimum_route_confidence <= 1.0
        ):
            raise ValueError("causal routing confidence must be in (0, 1]")
        if (
            not isfinite(self.maximum_residual_ratio)
            or not 0.0 < self.maximum_residual_ratio <= 1.0
        ):
            raise ValueError("causal residual ratio must be in (0, 1]")

    @classmethod
    def from_dict(cls, payload: Any) -> CausalContextRoutingSpec:
        expected = {
            "context_dimension",
            "router_hidden_dimension",
            "signed_x_deadband",
            "translation_activity_deadband",
            "rotation_activity_deadband",
            "gripper_activity_deadband",
            "minimum_route_confidence",
            "maximum_residual_ratio",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("causal context routing specification is invalid")
        if any(
            isinstance(payload[name], bool) or not isinstance(payload[name], int)
            for name in ("context_dimension", "router_hidden_dimension")
        ):
            raise ValueError("causal context routing specification is invalid")
        try:
            return cls(
                context_dimension=int(payload["context_dimension"]),
                router_hidden_dimension=int(payload["router_hidden_dimension"]),
                signed_x_deadband=float(payload["signed_x_deadband"]),
                translation_activity_deadband=float(
                    payload["translation_activity_deadband"]
                ),
                rotation_activity_deadband=float(payload["rotation_activity_deadband"]),
                gripper_activity_deadband=float(payload["gripper_activity_deadband"]),
                minimum_route_confidence=float(payload["minimum_route_confidence"]),
                maximum_residual_ratio=float(payload["maximum_residual_ratio"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "causal context routing specification is invalid"
            ) from error

    def to_dict(self) -> dict[str, int | float]:
        return {
            "context_dimension": self.context_dimension,
            "router_hidden_dimension": self.router_hidden_dimension,
            "signed_x_deadband": self.signed_x_deadband,
            "translation_activity_deadband": self.translation_activity_deadband,
            "rotation_activity_deadband": self.rotation_activity_deadband,
            "gripper_activity_deadband": self.gripper_activity_deadband,
            "minimum_route_confidence": self.minimum_route_confidence,
            "maximum_residual_ratio": self.maximum_residual_ratio,
        }

    def classify_action_horizons(self, actions: torch.Tensor) -> torch.Tensor:
        """Label complete 7D future horizons without collapsing active non-X motion."""
        return self.labeling_spec.classify_action_horizons(actions)

    def classify_recorded_horizons(
        self,
        actions: torch.Tensor,
        segments: Sequence[ContactInsertionSegment],
    ) -> torch.Tensor:
        """Label authenticated demonstrations without turning hold drift into intent."""

        return self.labeling_spec.classify_recorded_horizons(actions, segments)

    @property
    def labeling_spec(self) -> RecordedMotionLabelSpec:
        return RecordedMotionLabelSpec(
            signed_x_deadband=self.signed_x_deadband,
            translation_activity_deadband=self.translation_activity_deadband,
            rotation_activity_deadband=self.rotation_activity_deadband,
            gripper_activity_deadband=self.gripper_activity_deadband,
        )


@dataclass(frozen=True)
class CausalRouteDecision:
    routes: torch.Tensor
    confidence: torch.Tensor
    residual_weights: torch.Tensor
    failed_closed: torch.Tensor

    def detached(self) -> CausalRouteDecision:
        return CausalRouteDecision(
            self.routes.detach(),
            self.confidence.detach(),
            self.residual_weights.detach(),
            self.failed_closed.detach(),
        )


def pool_context_latents(
    context_latents: torch.Tensor,
    *,
    context_dimension: int,
) -> torch.Tensor:
    if (
        context_latents.ndim < 2
        or context_latents.shape[0] == 0
        or context_latents.shape[-1] != context_dimension
    ):
        raise ValueError("context latents have the wrong batch or feature dimension")
    reduction_dimensions = tuple(range(1, context_latents.ndim - 1))
    if not reduction_dimensions:
        return context_latents
    return context_latents.mean(dim=reduction_dimensions)


class CausalMotionRouter(torch.nn.Module):
    """Predict one candidate-independent next-motion route from causal state."""

    def __init__(self, spec: CausalContextRoutingSpec) -> None:
        super().__init__()
        self.spec = spec
        input_dimension = spec.context_dimension + ACTION_DIMENSIONS * 2
        self.hidden = torch.nn.Linear(input_dimension, spec.router_hidden_dimension)
        self.output = torch.nn.Linear(
            spec.router_hidden_dimension,
            len(CausalMotionRoute),
        )
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def features(
        self,
        context_latents: torch.Tensor,
        context_poses: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:
        pooled = pool_context_latents(
            context_latents,
            context_dimension=self.spec.context_dimension,
        )
        expected = (pooled.shape[0], ACTION_DIMENSIONS)
        if context_poses.shape != expected or previous_actions.shape != expected:
            raise ValueError(
                "causal routing pose and action batches must be [batch, 7]"
            )
        return torch.cat(
            (
                pooled,
                context_poses.to(device=pooled.device, dtype=pooled.dtype),
                previous_actions.to(device=pooled.device, dtype=pooled.dtype),
            ),
            dim=-1,
        )

    def forward(
        self,
        context_latents: torch.Tensor,
        context_poses: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> torch.Tensor:
        features = self.features(context_latents, context_poses, previous_actions)
        return self.output(torch.nn.functional.silu(self.hidden(features)))

    def decide(
        self,
        context_latents: torch.Tensor,
        context_poses: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> CausalRouteDecision:
        logits = self(context_latents, context_poses, previous_actions)
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
            routes=routes,
            confidence=confidence,
            residual_weights=residual_weights,
            failed_closed=failed_closed,
        )
