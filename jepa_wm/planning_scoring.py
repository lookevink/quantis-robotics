"""GPU scoring adapter between bounded candidates and JEPA-WM goal energy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jepa_wm.action_conditioning import physical_state_encoder
from jepa_wm.objective import terminal_l2_energy
from jepa_wm.physical_observation import PhysicalRoutingObservation


class LatentGoalScorer:
    """Score action candidates against one encoded target observation."""

    def __init__(
        self,
        model: Any,
        context: torch.Tensor,
        target: torch.Tensor,
        *,
        device: torch.device,
        batch_size: int = 64,
        physical_routing: PhysicalRoutingObservation | None = None,
    ) -> None:
        if context.shape[0] != 1 or target.shape[0] != 1:
            raise ValueError("planner scorer requires one context and one target")
        if batch_size <= 0:
            raise ValueError("planner scoring batch size must be positive")
        self._model = model
        self._context = context
        self._target = target
        self._device = device
        self._batch_size = batch_size
        try:
            self._physical_encoder = physical_state_encoder(model)
        except ValueError:
            self._physical_encoder = None
        if self._physical_encoder is not None and physical_routing is None:
            raise ValueError(
                "physical action conditioning requires one observed router input"
            )
        if self._physical_encoder is None and physical_routing is not None:
            raise ValueError(
                "physical router input requires physical action conditioning"
            )
        self._physical_routing = physical_routing

    def __call__(self, candidates: np.ndarray) -> np.ndarray:
        energies = []
        with torch.inference_mode():
            for start in range(0, len(candidates), self._batch_size):
                candidate_batch = candidates[start : start + self._batch_size]
                actions = torch.as_tensor(
                    candidate_batch,
                    device=self._device,
                    dtype=torch.float32,
                ).transpose(0, 1)
                batch = len(candidate_batch)
                context = self._context.expand(batch, *self._context.shape[1:])
                target = self._target.expand(batch, *self._target.shape[1:])
                if self._physical_encoder is None:
                    prediction = self._model.unroll(context, actions)
                else:
                    assert self._physical_routing is not None
                    physical = torch.tensor(
                        self._physical_routing.values,
                        device=self._device,
                        dtype=torch.float32,
                    ).unsqueeze(0).expand(batch, -1)
                    with self._physical_encoder.use_physical_observations(physical):
                        prediction = self._model.unroll(context, actions)
                energies.append(terminal_l2_energy(prediction, target).cpu())
        return torch.cat(energies).numpy()
