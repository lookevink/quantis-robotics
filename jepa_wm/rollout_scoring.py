"""Tensor contract and energy comparison shared by adaptation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from jepa_wm.action_conditioning import action_regime_context
from jepa_wm.objective import terminal_l2_energy
from jepa_wm.trajectory import RecordedRollout


@dataclass(frozen=True)
class RolloutEnergies:
    recorded: torch.Tensor
    zero: torch.Tensor


@dataclass(frozen=True)
class ContrastiveRolloutEnergies:
    recorded: torch.Tensor
    zero: torch.Tensor
    mismatched_negative: torch.Tensor


def score_actions(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
    regimes: torch.Tensor | None = None,
) -> torch.Tensor:
    if regimes is None:
        return terminal_l2_energy(model.unroll(context, actions), target)
    with action_regime_context(model, regimes):
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
    regimes: torch.Tensor | None = None,
) -> RolloutEnergies:
    return RolloutEnergies(
        recorded=score_actions(model, context, target, actions, regimes),
        zero=score_actions(
            model,
            context,
            target,
            torch.zeros_like(actions),
            regimes,
        ),
    )


def score_recorded_against_mismatched(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
    mismatched_actions: torch.Tensor,
    regimes: torch.Tensor | None = None,
) -> ContrastiveRolloutEnergies:
    if mismatched_actions.shape != actions.shape:
        raise ValueError("mismatched actions must match recorded actions")
    baseline = score_recorded_against_zero(
        model,
        context,
        target,
        actions,
        regimes,
    )
    return ContrastiveRolloutEnergies(
        recorded=baseline.recorded,
        zero=baseline.zero,
        mismatched_negative=score_actions(
            model,
            context,
            target,
            mismatched_actions,
            regimes,
        ),
    )
