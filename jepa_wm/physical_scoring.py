"""Candidate-invariant scoring with physical-state action conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.action_conditioning import physical_state_encoder
from jepa_wm.causal_routing import CausalRouteDecision
from jepa_wm.objective import terminal_l2_energy


@dataclass(frozen=True)
class PhysicalCandidateScores:
    energies: dict[str, torch.Tensor]
    decision: CausalRouteDecision


class PhysicalCandidateScorer:
    """Score every candidate under one route derived before candidate scoring."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def score(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        candidates: Mapping[str, torch.Tensor],
        *,
        physical_observations: torch.Tensor,
    ) -> PhysicalCandidateScores:
        if not candidates:
            raise ValueError("physical scoring requires at least one candidate set")
        batch = context.shape[0]
        if target.shape[0] != batch or physical_observations.shape[0] != batch:
            raise ValueError("physical scoring observation batches differ")
        for name, actions in candidates.items():
            if (
                not name
                or actions.ndim != 3
                or actions.shape[0] == 0
                or actions.shape[1] != batch
                or actions.shape[-1] != ACTION_DIMENSIONS
            ):
                raise ValueError(
                    "physical candidate actions must be named "
                    "[horizon, batch, 7] tensors"
                )
        encoder = physical_state_encoder(self.model)
        with encoder.use_physical_observations(physical_observations) as decision:
            energies = {
                name: terminal_l2_energy(
                    self.model.unroll(context, actions),
                    target,
                )
                for name, actions in candidates.items()
            }
        return PhysicalCandidateScores(energies, decision.detached())
