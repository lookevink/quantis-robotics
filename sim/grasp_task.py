"""Simulator-independent reach-and-grasp task contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

import numpy as np


class GraspAcquisitionFailure(str, Enum):
    OUTSIDE_GRASP_REGION = "outside_grasp_region"
    GRIPPER_OPEN = "gripper_open"


@dataclass(frozen=True)
class GraspAcquisitionLimits:
    maximum_hand_error_meters: float = 0.025
    maximum_gripper_width_meters: float = 0.03

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                self.maximum_hand_error_meters,
                self.maximum_gripper_width_meters,
            )
        ):
            raise ValueError("grasp acquisition limits must be positive and finite")


@dataclass(frozen=True)
class GraspAcquisitionDecision:
    hand_error_meters: float
    gripper_width_meters: float
    failures: tuple[GraspAcquisitionFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "hand_error_meters": self.hand_error_meters,
            "gripper_width_meters": self.gripper_width_meters,
            "failures": [failure.value for failure in self.failures],
        }


def evaluate_grasp_acquisition(
    hand_position: Sequence[float],
    grasp_hand_position: Sequence[float],
    gripper_width_meters: float,
    limits: GraspAcquisitionLimits = GraspAcquisitionLimits(),
) -> GraspAcquisitionDecision:
    """Require both spatial alignment and a closed gripper before attachment."""

    hand = np.asarray(hand_position, dtype=np.float64)
    target = np.asarray(grasp_hand_position, dtype=np.float64)
    if (
        hand.shape != (3,)
        or target.shape != (3,)
        or not np.all(np.isfinite(hand))
        or not np.all(np.isfinite(target))
        or not isfinite(gripper_width_meters)
        or gripper_width_meters < 0.0
    ):
        raise ValueError("grasp acquisition state is invalid")
    hand_error = float(np.linalg.norm(hand - target))
    failures = []
    if hand_error > limits.maximum_hand_error_meters:
        failures.append(GraspAcquisitionFailure.OUTSIDE_GRASP_REGION)
    if gripper_width_meters > limits.maximum_gripper_width_meters:
        failures.append(GraspAcquisitionFailure.GRIPPER_OPEN)
    return GraspAcquisitionDecision(
        hand_error,
        float(gripper_width_meters),
        tuple(failures),
    )
