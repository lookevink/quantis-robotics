"""Evidence gate for a meaningful reach-and-grasp control rollout."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


class ReachAndGraspFailure(str, Enum):
    NO_ATTACHMENT_TRANSITION = "no_attachment_transition"
    ATTACHMENT_LOST = "attachment_lost"
    INSUFFICIENT_RETENTION = "insufficient_retention"
    INSUFFICIENT_LIFT = "insufficient_lift"
    TRACKING_FAILED = "tracking_failed"
    COLLISION_DETECTED = "collision_detected"
    CONTACT_FORCE_EXCEEDED = "contact_force_exceeded"


@dataclass(frozen=True)
class GraspTaskLimits:
    minimum_attached_observations: int = 2
    minimum_retained_displacement_meters: float = 0.02
    maximum_contact_force_newtons: float = 2.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_attached_observations, bool)
            or self.minimum_attached_observations < 2
            or not isfinite(self.minimum_retained_displacement_meters)
            or self.minimum_retained_displacement_meters <= 0.0
            or not isfinite(self.maximum_contact_force_newtons)
            or self.maximum_contact_force_newtons <= 0.0
        ):
            raise ValueError("grasp task limits are invalid")


@dataclass(frozen=True)
class GraspTaskStep:
    plug_position: tuple[float, float, float]
    plug_attached: bool
    tracking_passed: bool
    collision_detected: bool
    contact_force_newtons: float

    def __post_init__(self) -> None:
        if (
            len(self.plug_position) != 3
            or not all(isfinite(value) for value in self.plug_position)
            or not isinstance(self.plug_attached, bool)
            or not isinstance(self.tracking_passed, bool)
            or not isinstance(self.collision_detected, bool)
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
        ):
            raise ValueError("grasp task step is invalid")


@dataclass(frozen=True)
class ReachAndGraspDecision:
    acquisition_index: int | None
    attached_observations: int
    maximum_retained_displacement_meters: float
    failures: tuple[ReachAndGraspFailure, ...]

    def __post_init__(self) -> None:
        if (
            (self.acquisition_index is not None and self.acquisition_index < 1)
            or self.attached_observations < 0
            or not isfinite(self.maximum_retained_displacement_meters)
            or self.maximum_retained_displacement_meters < 0.0
        ):
            raise ValueError("reach-and-grasp decision is invalid")

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "acquisition_index": self.acquisition_index,
            "attached_observations": self.attached_observations,
            "maximum_retained_displacement_meters": (
                self.maximum_retained_displacement_meters
            ),
            "failures": [failure.value for failure in self.failures],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReachAndGraspDecision:
        try:
            failures = payload["failures"]
            if not isinstance(failures, (list, tuple)):
                raise ValueError("reach-and-grasp failures are invalid")
            instance = cls(
                (
                    int(payload["acquisition_index"])
                    if payload.get("acquisition_index") is not None
                    else None
                ),
                int(payload["attached_observations"]),
                float(payload["maximum_retained_displacement_meters"]),
                tuple(ReachAndGraspFailure(value) for value in failures),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("reach-and-grasp decision is incomplete") from error
        if payload.get("passed") is not instance.passed:
            raise ValueError("reach-and-grasp pass claim is invalid")
        return instance


def evaluate_reach_and_grasp(
    steps: Sequence[GraspTaskStep],
    limits: GraspTaskLimits = GraspTaskLimits(),
) -> ReachAndGraspDecision:
    """Require acquisition, continuous retention, lift, tracking, and safety."""

    acquisition_index = next(
        (
            index
            for index in range(1, len(steps))
            if not steps[index - 1].plug_attached and steps[index].plug_attached
        ),
        None,
    )
    failures = []
    attached_steps: Sequence[GraspTaskStep] = ()
    displacement = 0.0
    if acquisition_index is None:
        failures.append(ReachAndGraspFailure.NO_ATTACHMENT_TRANSITION)
    else:
        attached_steps = steps[acquisition_index:]
        if any(not step.plug_attached for step in attached_steps):
            failures.append(ReachAndGraspFailure.ATTACHMENT_LOST)
        retained = tuple(step for step in attached_steps if step.plug_attached)
        if len(retained) < limits.minimum_attached_observations:
            failures.append(ReachAndGraspFailure.INSUFFICIENT_RETENTION)
        origin = np.asarray(steps[acquisition_index].plug_position)
        displacement = max(
            (
                float(np.linalg.norm(np.asarray(step.plug_position) - origin))
                for step in retained
            ),
            default=0.0,
        )
        if displacement < limits.minimum_retained_displacement_meters:
            failures.append(ReachAndGraspFailure.INSUFFICIENT_LIFT)
    if any(not step.tracking_passed for step in steps):
        failures.append(ReachAndGraspFailure.TRACKING_FAILED)
    if any(step.collision_detected for step in steps):
        failures.append(ReachAndGraspFailure.COLLISION_DETECTED)
    if any(
        step.contact_force_newtons > limits.maximum_contact_force_newtons
        for step in steps
    ):
        failures.append(ReachAndGraspFailure.CONTACT_FORCE_EXCEEDED)
    return ReachAndGraspDecision(
        acquisition_index,
        len(tuple(step for step in attached_steps if step.plug_attached)),
        displacement,
        tuple(failures),
    )
