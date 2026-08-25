"""Dependency-light candidate-mining policy vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping


class CandidateNoiseReference(str, Enum):
    """Scale perturbations from global bounds or the demonstrated action."""

    PLANNER_BOUNDS = "planner_bounds"
    RECORDED_ACTION = "recorded_action"


@dataclass(frozen=True)
class CandidateNoisePolicy:
    """Reference and explicit per-axis floors for candidate perturbations."""

    reference: CandidateNoiseReference = CandidateNoiseReference.PLANNER_BOUNDS
    translation_floor: float = 0.0
    rotation_floor: float = 0.0
    gripper_floor: float = 0.0

    def __post_init__(self) -> None:
        floors = (
            self.translation_floor,
            self.rotation_floor,
            self.gripper_floor,
        )
        if not isinstance(self.reference, CandidateNoiseReference):
            raise ValueError("candidate noise reference is invalid")
        if not all(isfinite(value) and value >= 0.0 for value in floors):
            raise ValueError("candidate noise floors must be finite and non-negative")
        if self.reference is CandidateNoiseReference.PLANNER_BOUNDS and any(floors):
            raise ValueError("planner-bound candidate noise does not use action floors")

    @classmethod
    def recorded_action(
        cls,
        *,
        translation_floor: float,
        rotation_floor: float,
        gripper_floor: float,
    ) -> CandidateNoisePolicy:
        return cls(
            reference=CandidateNoiseReference.RECORDED_ACTION,
            translation_floor=translation_floor,
            rotation_floor=rotation_floor,
            gripper_floor=gripper_floor,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference.value,
            "floors": {
                "translation": self.translation_floor,
                "rotation": self.rotation_floor,
                "gripper": self.gripper_floor,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateNoisePolicy:
        floors = payload.get("floors", {})
        return cls(
            reference=CandidateNoiseReference(payload["reference"]),
            translation_floor=float(floors.get("translation", 0.0)),
            rotation_floor=float(floors.get("rotation", 0.0)),
            gripper_floor=float(floors.get("gripper", 0.0)),
        )
