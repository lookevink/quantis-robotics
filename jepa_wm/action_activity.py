"""Axis-aware activity thresholds for one DROID action."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Sequence


@dataclass(frozen=True)
class DroidActionActivityThresholds:
    translation_norm: float = 0.001
    rotation_norm: float = 0.005
    gripper_delta: float = 0.02

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value >= 0.0
            for value in (
                self.translation_norm,
                self.rotation_norm,
                self.gripper_delta,
            )
        ):
            raise ValueError("DROID action activity thresholds must be non-negative")

    @staticmethod
    def _norm(values: Sequence[float]) -> float:
        return sqrt(sum(value * value for value in values))

    def is_active(self, values: Sequence[float]) -> bool:
        if len(values) != 7:
            raise ValueError("DROID action activity requires seven values")
        return (
            self._norm(values[:3]) > self.translation_norm
            or self._norm(values[3:6]) > self.rotation_norm
            or abs(values[6]) > self.gripper_delta
        )

    def active_tensor(self, actions):
        if actions.ndim < 1 or actions.shape[-1] != 7:
            raise ValueError("DROID action activity tensor must end with seven values")
        import torch

        return (
            torch.linalg.vector_norm(actions[..., :3], dim=-1)
            > self.translation_norm
        ) | (
            torch.linalg.vector_norm(actions[..., 3:6], dim=-1)
            > self.rotation_norm
        ) | (actions[..., 6].abs() > self.gripper_delta)

    def to_dict(self) -> dict[str, float]:
        return {
            "translation_norm": self.translation_norm,
            "rotation_norm": self.rotation_norm,
            "gripper_delta": self.gripper_delta,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DroidActionActivityThresholds:
        if not isinstance(payload, dict):
            raise ValueError("DROID action activity thresholds must be an object")
        try:
            return cls(
                translation_norm=float(payload["translation_norm"]),
                rotation_norm=float(payload["rotation_norm"]),
                gripper_delta=float(payload["gripper_delta"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("DROID action activity thresholds are incomplete") from error
