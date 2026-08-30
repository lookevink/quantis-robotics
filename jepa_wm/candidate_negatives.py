"""Train-only local candidate generation and energy-based negative mining."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.action_activity import DroidActionActivityThresholds
from jepa_wm.candidate_policy import CandidateNoisePolicy, CandidateNoiseReference
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
    noise_policy: CandidateNoisePolicy = CandidateNoisePolicy()

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
        payload = {
            "candidates_per_rollout": self.candidates_per_rollout,
            "scoring_batch_size": self.scoring_batch_size,
            "noise_scale": self.noise_scale,
            "bounds": self.bounds.to_dict(),
            "minimum_goal_cosine": self.minimum_goal_cosine,
            "first_action_activity": self.first_action_activity.to_dict(),
        }
        if self.noise_policy.reference is not CandidateNoiseReference.PLANNER_BOUNDS:
            payload["noise_policy"] = self.noise_policy.to_dict()
        return payload

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
            noise_policy=(
                CandidateNoisePolicy.from_dict(payload["noise_policy"])
                if "noise_policy" in payload
                else CandidateNoisePolicy()
            ),
        )


def _candidate_noise_standard_deviation(
    recorded_actions: torch.Tensor,
    config: CandidateMiningConfig,
) -> torch.Tensor:
    planner_scale = torch.tensor(
        config.bounds.initial_standard_deviation * config.noise_scale,
        dtype=recorded_actions.dtype,
        device=recorded_actions.device,
    )
    if config.noise_policy.reference is CandidateNoiseReference.PLANNER_BOUNDS:
        return planner_scale

    translation = torch.linalg.vector_norm(
        recorded_actions[..., :3], dim=-1, keepdim=True
    ).clamp(
        min=config.noise_policy.translation_floor,
        max=config.bounds.maximum_translation_norm,
    )
    rotation = torch.linalg.vector_norm(
        recorded_actions[..., 3:6], dim=-1, keepdim=True
    ).clamp(
        min=config.noise_policy.rotation_floor,
        max=config.bounds.maximum_rotation_norm,
    )
    gripper = recorded_actions[..., 6:].abs().clamp(
        min=config.noise_policy.gripper_floor,
        max=config.bounds.maximum_gripper_delta,
    )
    return torch.cat(
        (
            translation.expand(*translation.shape[:-1], 3),
            rotation.expand(*rotation.shape[:-1], 3),
            gripper,
        ),
        dim=-1,
    ) * config.noise_scale


def _goal_align_first_actions(
    candidates: torch.Tensor,
    recorded_first: torch.Tensor,
    bounded_fallback_first: torch.Tensor,
    goal_actions: torch.Tensor | None,
    minimum_cosine: float,
    first_action_activity: DroidActionActivityThresholds,
) -> torch.Tensor:
    batch = recorded_first.shape[0]
    if (
        goal_actions is None
        or goal_actions.shape != (batch, ACTION_DIMENSIONS)
        or not torch.isfinite(goal_actions).all()
    ):
        raise ValueError("goal-aligned candidate mining requires finite [batch, 7] goals")
    goals = goal_actions[:, None, :]
    recorded_stationary = ~first_action_activity.active_tensor(recorded_first)
    bounded_fallback_cosines = torch.nn.functional.cosine_similarity(
        bounded_fallback_first,
        goal_actions,
        dim=-1,
    )
    if torch.any(
        ~recorded_stationary
        & (bounded_fallback_cosines < minimum_cosine - 1e-6)
    ):
        raise ValueError(
            "bounded recorded first action does not satisfy the goal alignment"
        )
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
        bounded_fallback_first[:, None, :],
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
    scales = _candidate_noise_standard_deviation(recorded_actions, config)
    noise = torch.randn(
        (*recorded_actions.shape[:2], config.candidates_per_rollout, ACTION_DIMENSIONS),
        dtype=recorded_actions.dtype,
        device=recorded_actions.device,
        generator=generator,
    )
    candidates = recorded_actions.unsqueeze(2) + noise * (
        scales.unsqueeze(2) if scales.ndim == recorded_actions.ndim else scales
    )
    candidates = config.bounds.clip_tensor(candidates)
    if config.minimum_goal_cosine is not None:
        recorded_first = recorded_actions[0]
        bounded_fallback_first = config.bounds.clip_tensor(recorded_first)
        candidates = _goal_align_first_actions(
            candidates,
            recorded_first,
            bounded_fallback_first,
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
    regimes: torch.Tensor | None = None,
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
    if regimes is not None and regimes.shape != (batch,):
        raise ValueError("candidate regimes must match the rollout batch")
    flattened = candidates.reshape(horizon, batch * candidate_count, ACTION_DIMENSIONS)
    repeated_context = context.repeat_interleave(candidate_count, dim=0)
    repeated_target = target.repeat_interleave(candidate_count, dim=0)
    repeated_regimes = (
        regimes.repeat_interleave(candidate_count) if regimes is not None else None
    )
    with torch.no_grad():
        energy_chunks = []
        for start in range(0, batch * candidate_count, scoring_batch_size):
            arguments = (
                model,
                repeated_context[start : start + scoring_batch_size],
                repeated_target[start : start + scoring_batch_size],
                flattened[:, start : start + scoring_batch_size],
            )
            energy_chunks.append(
                score_actions(
                    *arguments,
                    **(
                        {"regimes": repeated_regimes[start : start + scoring_batch_size]}
                        if repeated_regimes is not None
                        else {}
                    ),
                )
            )
        energies = torch.cat(tuple(energy_chunks)).reshape(batch, candidate_count)
        selected_indices = energies.argmin(dim=1)
    by_rollout = candidates.permute(1, 2, 0, 3)
    selected = by_rollout[
        torch.arange(batch, device=candidates.device), selected_indices
    ]
    return selected.permute(1, 0, 2).contiguous()
