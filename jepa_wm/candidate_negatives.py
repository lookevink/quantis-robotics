"""Train-only local candidate generation and energy-based negative mining."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.planner import PlannerActionBounds
from jepa_wm.rollout_scoring import score_actions


@dataclass(frozen=True)
class CandidateMiningConfig:
    candidates_per_rollout: int = 4
    noise_scale: float = 0.25
    bounds: PlannerActionBounds = PlannerActionBounds()

    def __post_init__(self) -> None:
        if self.candidates_per_rollout < 2:
            raise ValueError("candidate mining requires at least two candidates")
        if not isfinite(self.noise_scale) or self.noise_scale <= 0.0:
            raise ValueError("candidate mining noise scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_per_rollout": self.candidates_per_rollout,
            "noise_scale": self.noise_scale,
            "bounds": self.bounds.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateMiningConfig:
        return cls(
            candidates_per_rollout=int(payload["candidates_per_rollout"]),
            noise_scale=float(payload["noise_scale"]),
            bounds=PlannerActionBounds.from_dict(payload["bounds"]),
        )


def sample_local_candidates(
    recorded_actions: torch.Tensor,
    *,
    config: CandidateMiningConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample bounded perturbations with shape [horizon, batch, candidates, 7]."""

    if recorded_actions.ndim != 3 or recorded_actions.shape[-1] != ACTION_DIMENSIONS:
        raise ValueError("recorded actions must have shape [horizon, batch, 7]")
    scales = torch.tensor(
        config.bounds.initial_standard_deviation * config.noise_scale,
        dtype=recorded_actions.dtype,
        device=recorded_actions.device,
    )
    noise = torch.randn(
        (*recorded_actions.shape[:2], config.candidates_per_rollout, ACTION_DIMENSIONS),
        dtype=recorded_actions.dtype,
        device=recorded_actions.device,
        generator=generator,
    )
    candidates = recorded_actions.unsqueeze(2) + noise * scales
    return config.bounds.clip_tensor(candidates)


def mine_lowest_energy_candidates(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Select the current model's most deceptive candidate for each rollout."""

    if candidates.ndim != 4 or candidates.shape[-1] != ACTION_DIMENSIONS:
        raise ValueError(
            "candidate actions must have shape [horizon, batch, candidates, 7]"
        )
    horizon, batch, candidate_count, _ = candidates.shape
    if context.shape[0] != batch or target.shape[0] != batch:
        raise ValueError("candidate batch does not match context and target batches")
    flattened = candidates.reshape(horizon, batch * candidate_count, ACTION_DIMENSIONS)
    with torch.no_grad():
        energies = score_actions(
            model,
            context.repeat_interleave(candidate_count, dim=0),
            target.repeat_interleave(candidate_count, dim=0),
            flattened,
        ).reshape(batch, candidate_count)
        selected_indices = energies.argmin(dim=1)
    by_rollout = candidates.permute(1, 2, 0, 3)
    selected = by_rollout[
        torch.arange(batch, device=candidates.device), selected_indices
    ]
    return selected.permute(1, 0, 2).contiguous()
