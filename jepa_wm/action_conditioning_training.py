"""Balanced sampling and signed negatives for action-conditioning experiments."""

from __future__ import annotations

from collections import Counter
from math import isfinite

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.action_conditioning import (
    NEGATIVE_X_COMMAND_ROUTE,
    POSITIVE_X_COMMAND_ROUTE,
    POST_REGIME,
    RETAINED_REGIME,
)


def signed_x_negatives(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 3 or actions.shape[-1] != ACTION_DIMENSIONS:
        raise ValueError("actions must have shape [horizon, batch, 7]")
    zero = actions.clone()
    opposed = actions.clone()
    zero[..., 0] = 0.0
    opposed[..., 0] = -actions[..., 0]
    return zero, opposed


def signed_x_margin_loss(
    recorded: torch.Tensor,
    negative: torch.Tensor,
    actions: torch.Tensor,
    *,
    weight: float,
    margin: float,
    minimum_activity: float,
) -> torch.Tensor:
    if (
        recorded.ndim != 1
        or negative.shape != recorded.shape
        or actions.ndim != 3
        or actions.shape[1] != recorded.shape[0]
        or actions.shape[-1] != ACTION_DIMENSIONS
    ):
        raise ValueError("signed-X loss tensors do not share one rollout batch")
    if not all(isfinite(value) and value >= 0.0 for value in (weight, margin)):
        raise ValueError("signed-X loss weight and margin must be non-negative")
    if not isfinite(minimum_activity) or minimum_activity <= 0.0:
        raise ValueError("signed-X activity threshold must be positive")
    active = actions[..., 0].abs().amax(dim=0) >= minimum_activity
    per_rollout = torch.relu(margin + recorded - negative)
    if not torch.any(active):
        return per_rollout.sum() * 0.0
    return weight * per_rollout[active].mean()


class AlternatingStratumSampler:
    """Deterministically alternate retained and post-stratum examples."""

    def __init__(self, regimes: torch.Tensor, *, seed: int) -> None:
        if regimes.ndim != 1 or regimes.numel() == 0:
            raise ValueError("training regimes must be a non-empty vector")
        if torch.any((regimes != RETAINED_REGIME) & (regimes != POST_REGIME)):
            raise ValueError("training regimes must contain only zero or one")
        self._indices = {
            regime: torch.nonzero(regimes == regime, as_tuple=False).flatten()
            for regime in (RETAINED_REGIME, POST_REGIME)
        }
        if any(indices.numel() == 0 for indices in self._indices.values()):
            raise ValueError("balanced training requires both regimes")
        self.seed = seed
        self.samples_drawn = 0
        self._samples_by_regime: Counter[int] = Counter()
        self._generators = {
            regime: torch.Generator(device="cpu").manual_seed(seed + regime)
            for regime in (RETAINED_REGIME, POST_REGIME)
        }
        self._orders = {
            regime: self._shuffle(regime) for regime in (RETAINED_REGIME, POST_REGIME)
        }
        self._cursors = {RETAINED_REGIME: 0, POST_REGIME: 0}

    def _shuffle(self, regime: int) -> torch.Tensor:
        indices = self._indices[regime]
        order = torch.randperm(indices.numel(), generator=self._generators[regime])
        return indices[order]

    def next_index(self) -> int:
        regime = self.samples_drawn % 2
        cursor = self._cursors[regime]
        if cursor == self._orders[regime].numel():
            self._orders[regime] = self._shuffle(regime)
            cursor = 0
        index = int(self._orders[regime][cursor])
        self._cursors[regime] = cursor + 1
        self.samples_drawn += 1
        self._samples_by_regime[regime] += 1
        return index

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": "alternating_seeded_shuffled_strata",
            "seed": self.seed,
            "samples_drawn": self.samples_drawn,
            "samples_by_regime": {
                "retained": self._samples_by_regime[RETAINED_REGIME],
                "post": self._samples_by_regime[POST_REGIME],
            },
            "rollouts_by_regime": {
                "retained": self._indices[RETAINED_REGIME].numel(),
                "post": self._indices[POST_REGIME].numel(),
            },
        }


class AlternatingCommandRouteSampler:
    """Deterministically balance negative- and positive-X motion routes."""

    _routes = (NEGATIVE_X_COMMAND_ROUTE, POSITIVE_X_COMMAND_ROUTE)

    def __init__(self, routes: torch.Tensor, *, seed: int) -> None:
        if routes.ndim != 1 or routes.numel() == 0:
            raise ValueError("training routes must be a non-empty vector")
        self._indices = {
            route: torch.nonzero(routes == route, as_tuple=False).flatten()
            for route in self._routes
        }
        if any(indices.numel() == 0 for indices in self._indices.values()):
            raise ValueError("balanced training requires both command routes")
        self.seed = seed
        self.samples_drawn = 0
        self._samples_by_route: Counter[int] = Counter()
        self._generators = {
            route: torch.Generator(device="cpu").manual_seed(seed + route)
            for route in self._routes
        }
        self._orders = {route: self._shuffle(route) for route in self._routes}
        self._cursors = {route: 0 for route in self._routes}

    def _shuffle(self, route: int) -> torch.Tensor:
        indices = self._indices[route]
        order = torch.randperm(indices.numel(), generator=self._generators[route])
        return indices[order]

    def next_index(self) -> int:
        route = self._routes[self.samples_drawn % len(self._routes)]
        cursor = self._cursors[route]
        if cursor == self._orders[route].numel():
            self._orders[route] = self._shuffle(route)
            cursor = 0
        index = int(self._orders[route][cursor])
        self._cursors[route] = cursor + 1
        self.samples_drawn += 1
        self._samples_by_route[route] += 1
        return index

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": "alternating_seeded_shuffled_command_routes",
            "seed": self.seed,
            "samples_drawn": self.samples_drawn,
            "samples_by_route": {
                "negative_x": self._samples_by_route[NEGATIVE_X_COMMAND_ROUTE],
                "positive_x": self._samples_by_route[POSITIVE_X_COMMAND_ROUTE],
            },
            "rollouts_by_route": {
                "negative_x": self._indices[NEGATIVE_X_COMMAND_ROUTE].numel(),
                "positive_x": self._indices[POSITIVE_X_COMMAND_ROUTE].numel(),
            },
        }
