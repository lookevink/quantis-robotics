"""Latent objectives shared by JEPA-WM evaluation and adaptation."""

from __future__ import annotations

import torch


def terminal_l2_energy(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    terminal = prediction[-1]
    target_frame = target[:, -1]
    reduction_dimensions = tuple(range(1, terminal.ndim))
    return (target_frame - terminal).pow(2).mean(dim=reduction_dimensions)
