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


def score_actions(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    return terminal_l2_energy(model.unroll(context, actions), target)


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
    return RolloutEnergies(
        recorded=score_actions(model, context, target, actions),
        zero=score_actions(model, context, target, torch.zeros_like(actions)),
    )
