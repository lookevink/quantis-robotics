"""Simulator-independent reach-and-grasp task contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Sequence

import numpy as np


class GraspAcquisitionFailure(str, Enum):
    OUTSIDE_GRASP_REGION = "outside_grasp_region"
    GRIPPER_OPEN = "gripper_open"


class AttachmentMechanism(str, Enum):
    """Simulator mechanism retaining the connector after acquisition."""

    KINEMATIC_FOLLOW = "kinematic_follow"
    FIXED_JOINT = "fixed_joint"


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

    @classmethod
    def from_dict(cls, payload: Any) -> GraspAcquisitionDecision:
        if not isinstance(payload, dict):
            raise ValueError("grasp acquisition decision must be an object")
        try:
            decision = cls(
                float(payload["hand_error_meters"]),
                float(payload["gripper_width_meters"]),
                tuple(
                    GraspAcquisitionFailure(failure)
                    for failure in payload["failures"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("grasp acquisition decision is incomplete") from error
        if (
            not isfinite(decision.hand_error_meters)
            or decision.hand_error_meters < 0.0
            or not isfinite(decision.gripper_width_meters)
            or decision.gripper_width_meters < 0.0
            or payload.get("passed") is not decision.passed
        ):
            raise ValueError("grasp acquisition decision is inconsistent")
        return decision


@dataclass(frozen=True)
class GraspAcquisitionEvidence:
    hand_position: tuple[float, ...]
    grasp_hand_position: tuple[float, ...]
    gripper_width_meters: float
    decision: GraspAcquisitionDecision

    def __post_init__(self) -> None:
        if self.decision != evaluate_grasp_acquisition(
            self.hand_position,
            self.grasp_hand_position,
            self.gripper_width_meters,
        ):
            raise ValueError("grasp acquisition evidence is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "hand_position": list(self.hand_position),
            "grasp_hand_position": list(self.grasp_hand_position),
            "gripper_width_meters": self.gripper_width_meters,
            "decision": self.decision.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> GraspAcquisitionEvidence:
        if not isinstance(payload, dict):
            raise ValueError("grasp acquisition evidence must be an object")
        try:
            return cls(
                tuple(float(value) for value in payload["hand_position"]),
                tuple(float(value) for value in payload["grasp_hand_position"]),
                float(payload["gripper_width_meters"]),
                GraspAcquisitionDecision.from_dict(payload["decision"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("grasp acquisition evidence is incomplete") from error


def observe_grasp_acquisition(
    hand_position: Sequence[float],
    grasp_hand_position: Sequence[float],
    gripper_width_meters: float,
) -> GraspAcquisitionEvidence:
    hand = tuple(float(value) for value in hand_position)
    target = tuple(float(value) for value in grasp_hand_position)
    return GraspAcquisitionEvidence(
        hand,
        target,
        float(gripper_width_meters),
        evaluate_grasp_acquisition(hand, target, gripper_width_meters),
    )


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
