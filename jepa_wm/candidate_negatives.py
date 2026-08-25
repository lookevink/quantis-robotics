"""Train-only local candidate generation and energy-based negative mining."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.action_activity import DroidActionActivityThresholds
from jepa_wm.planner import PlannerActionBounds
from jepa_wm.rollout_scoring import score_actions


@dataclass(frozen=True)
class CandidateMiningConfig:
    candidates_per_rollout: int = 4
    scoring_batch_size: int = 2
    noise_scale: float = 0.25
    bounds: PlannerActionBounds = PlannerActionBounds()
    minimum_goal_cosine: float | None = None
    first_action_activity: DroidActionActivityThresholds = (
        DroidActionActivityThresholds()
    )

    def __post_init__(self) -> None:
        if self.candidates_per_rollout < 2:
            raise ValueError("candidate mining requires at least two candidates")
        if self.scoring_batch_size <= 0:
            raise ValueError("candidate scoring batch size must be positive")
        if not isfinite(self.noise_scale) or self.noise_scale <= 0.0:
            raise ValueError("candidate mining noise scale must be positive")
        if self.minimum_goal_cosine is not None and not (
            isfinite(self.minimum_goal_cosine)
            and 0.0 <= self.minimum_goal_cosine <= 1.0
        ):
            raise ValueError("candidate mining goal cosine must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates_per_rollout": self.candidates_per_rollout,
            "scoring_batch_size": self.scoring_batch_size,
            "noise_scale": self.noise_scale,
            "bounds": self.bounds.to_dict(),
            "minimum_goal_cosine": self.minimum_goal_cosine,
            "first_action_activity": self.first_action_activity.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateMiningConfig:
        return cls(
            candidates_per_rollout=int(payload["candidates_per_rollout"]),
            scoring_batch_size=int(payload.get("scoring_batch_size", 2)),
            noise_scale=float(payload["noise_scale"]),
            bounds=PlannerActionBounds.from_dict(payload["bounds"]),
            minimum_goal_cosine=(
                float(payload["minimum_goal_cosine"])
                if payload.get("minimum_goal_cosine") is not None
                else None
            ),
            first_action_activity=DroidActionActivityThresholds.from_dict(
                payload.get(
                    "first_action_activity",
                    DroidActionActivityThresholds().to_dict(),
                )
            ),
        )


def _goal_align_first_actions(
    candidates: torch.Tensor,
    recorded_actions: torch.Tensor,
    goal_actions: torch.Tensor | None,
    minimum_cosine: float,
    first_action_activity: DroidActionActivityThresholds,
) -> torch.Tensor:
    batch = recorded_actions.shape[1]
    if (
        goal_actions is None
        or goal_actions.shape != (batch, ACTION_DIMENSIONS)
        or not torch.isfinite(goal_actions).all()
    ):
        raise ValueError("goal-aligned candidate mining requires finite [batch, 7] goals")
    goals = goal_actions[:, None, :]
    recorded_first = recorded_actions[0]
    recorded_stationary = ~first_action_activity.active_tensor(recorded_first)
    recorded_cosines = torch.nn.functional.cosine_similarity(
        recorded_first,
        goal_actions,
        dim=-1,
    )
    if torch.any(
        ~recorded_stationary & (recorded_cosines < minimum_cosine - 1e-6)
    ):
        raise ValueError("recorded first action does not satisfy the goal alignment")
    first_actions = candidates[0]
    candidate_cosines = torch.nn.functional.cosine_similarity(
        first_actions,
        goals,
        dim=-1,
    )
    keep_candidate = (
        ~recorded_stationary[:, None]
        & (candidate_cosines >= minimum_cosine - 1e-6)
    )
    aligned_first = torch.where(
        keep_candidate.unsqueeze(-1),
        first_actions,
        recorded_first[:, None, :],
    )
    aligned = candidates.clone()
    aligned[0] = aligned_first
    return aligned


def sample_local_candidates(
    recorded_actions: torch.Tensor,
    *,
    config: CandidateMiningConfig,
    generator: torch.Generator,
    goal_actions: torch.Tensor | None = None,
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
    candidates = config.bounds.clip_tensor(candidates)
    if config.minimum_goal_cosine is not None:
        if not torch.equal(
            config.bounds.clip_tensor(recorded_actions[0]),
            recorded_actions[0],
        ):
            raise ValueError("goal-aligned recorded fallback exceeds planner bounds")
        candidates = _goal_align_first_actions(
            candidates,
            recorded_actions,
            goal_actions,
            config.minimum_goal_cosine,
            config.first_action_activity,
        )
    return candidates


def mine_lowest_energy_candidates(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    *,
    scoring_batch_size: int,
) -> torch.Tensor:
    """Select the current model's most deceptive candidate for each rollout."""

    if candidates.ndim != 4 or candidates.shape[-1] != ACTION_DIMENSIONS:
        raise ValueError(
            "candidate actions must have shape [horizon, batch, candidates, 7]"
        )
    horizon, batch, candidate_count, _ = candidates.shape
    if context.shape[0] != batch or target.shape[0] != batch:
        raise ValueError("candidate batch does not match context and target batches")
    if scoring_batch_size <= 0:
        raise ValueError("candidate scoring batch size must be positive")
    flattened = candidates.reshape(horizon, batch * candidate_count, ACTION_DIMENSIONS)
    repeated_context = context.repeat_interleave(candidate_count, dim=0)
    repeated_target = target.repeat_interleave(candidate_count, dim=0)
    with torch.no_grad():
        energy_chunks = tuple(
            score_actions(
                model,
                repeated_context[start : start + scoring_batch_size],
                repeated_target[start : start + scoring_batch_size],
                flattened[:, start : start + scoring_batch_size],
            )
            for start in range(0, batch * candidate_count, scoring_batch_size)
        )
        energies = torch.cat(energy_chunks).reshape(batch, candidate_count)
        selected_indices = energies.argmin(dim=1)
    by_rollout = candidates.permute(1, 2, 0, 3)
    selected = by_rollout[
        torch.arange(batch, device=candidates.device), selected_indices
    ]
    return selected.permute(1, 0, 2).contiguous()
