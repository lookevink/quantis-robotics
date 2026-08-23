"""GPU scoring adapter between bounded candidates and JEPA-WM goal energy."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jepa_wm.objective import terminal_l2_energy


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
                prediction = self._model.unroll(context, actions)
                energies.append(terminal_l2_energy(prediction, target).cpu())
        return torch.cat(energies).numpy()
