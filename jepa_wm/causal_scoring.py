"""One scoring seam for candidate-invariant causal action conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.action_conditioning import causal_context_encoder
from jepa_wm.causal_routing import CausalRouteDecision
from jepa_wm.objective import terminal_l2_energy


@dataclass(frozen=True)
class CausalCandidateScores:
    energies: dict[str, torch.Tensor]
    decision: CausalRouteDecision


class CausalCandidateScorer:
    """Own the serial scoring scope and reuse one decision across candidates."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def score(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        candidates: Mapping[str, torch.Tensor],
        *,
        context_poses: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> CausalCandidateScores:
        if not candidates:
            raise ValueError("causal scoring requires at least one candidate set")
        batch = context.shape[0]
        if target.shape[0] != batch:
            raise ValueError("causal scoring context and target batches differ")
        for name, actions in candidates.items():
            if (
                not name
                or actions.ndim != 3
                or actions.shape[0] == 0
                or actions.shape[1] != batch
                or actions.shape[-1] != ACTION_DIMENSIONS
            ):
                raise ValueError(
                    "causal candidate actions must be named [horizon, batch, 7] tensors"
                )
        encoder = causal_context_encoder(self.model)
        with encoder.use_causal_context(
            context,
            context_poses,
            previous_actions,
        ) as decision:
            energies = {
                name: terminal_l2_energy(
                    self.model.unroll(context, actions),
                    target,
                )
                for name, actions in candidates.items()
            }
        return CausalCandidateScores(energies, decision.detached())
