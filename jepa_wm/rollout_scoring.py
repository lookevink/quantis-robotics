"""Tensor contract and energy comparison shared by adaptation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from jepa_wm.objective import terminal_l2_energy
from jepa_wm.trajectory import RecordedRollout


@dataclass(frozen=True)
class RolloutEnergies:
    recorded: torch.Tensor
    zero: torch.Tensor


def rollout_action_tensor(
    rollouts: Sequence[RecordedRollout],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    return torch.tensor(
        [[action.values for action in rollout.actions] for rollout in rollouts],
        device=device,
        dtype=torch.float32,
    ).transpose(0, 1)


def score_recorded_against_zero(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
) -> RolloutEnergies:
    recorded_prediction = model.unroll(context, actions)
    zero_prediction = model.unroll(context, torch.zeros_like(actions))
    return RolloutEnergies(
        recorded=terminal_l2_energy(recorded_prediction, target),
        zero=terminal_l2_energy(zero_prediction, target),
    )
