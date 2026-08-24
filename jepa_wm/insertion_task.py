"""Evidence gate for a rearward-grasp cable insertion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

import numpy as np


Vector3 = tuple[float, float, float]


class InsertionFailure(str, Enum):
    NO_ATTACHMENT_TRANSITION = "no_attachment_transition"
    INSUFFICIENT_GRASP_CLEARANCE = "insufficient_grasp_clearance"
    ATTACHMENT_LOST_BEFORE_SEATING = "attachment_lost_before_seating"
    NOT_SEATED = "not_seated"
    TRACKING_FAILED = "tracking_failed"
    COLLISION_DETECTED = "collision_detected"
    CONTACT_FORCE_EXCEEDED = "contact_force_exceeded"


@dataclass(frozen=True)
class InsertionTarget:
    socket_position: Vector3
    insertion_axis: Vector3

    def __post_init__(self) -> None:
        for name, values in (
            ("socket position", self.socket_position),
            ("insertion axis", self.insertion_axis),
        ):
            if len(values) != 3 or not all(isfinite(value) for value in values):
                raise ValueError(f"{name} must contain three finite values")
        axis_norm = float(np.linalg.norm(self.insertion_axis))
        if abs(axis_norm - 1.0) > 1e-6:
            raise ValueError("insertion axis must be a unit vector")


@dataclass(frozen=True)
class InsertionTaskLimits:
    minimum_grasp_clearance_meters: float = 0.03
    maximum_depth_error_meters: float = 0.003
    maximum_lateral_error_meters: float = 0.003
    maximum_orientation_error_rad: float = 0.05
    maximum_contact_force_newtons: float = 2.0

    def __post_init__(self) -> None:
        values = (
            self.minimum_grasp_clearance_meters,
            self.maximum_depth_error_meters,
            self.maximum_lateral_error_meters,
            self.maximum_orientation_error_rad,
            self.maximum_contact_force_newtons,
        )
        if not all(isfinite(value) and value > 0.0 for value in values):
            raise ValueError("insertion task limits must be positive and finite")


@dataclass(frozen=True)
class InsertionGeometryStep:
    plug_tip_position: Vector3
    gripper_frame_position: Vector3
    plug_attached: bool
    orientation_error_rad: float

    def __post_init__(self) -> None:
        if any(
            len(values) != 3 or not all(isfinite(value) for value in values)
            for values in (self.plug_tip_position, self.gripper_frame_position)
        ):
            raise ValueError("insertion task positions must contain three finite values")
        if (
            not isinstance(self.plug_attached, bool)
            or not isfinite(self.orientation_error_rad)
            or self.orientation_error_rad < 0.0
        ):
            raise ValueError("insertion geometry step is invalid")


@dataclass(frozen=True)
class InsertionTaskStep(InsertionGeometryStep):
    tracking_passed: bool
    collision_detected: bool
    contact_force_newtons: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            not isinstance(self.tracking_passed, bool)
            or not isinstance(self.collision_detected, bool)
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
        ):
            raise ValueError("insertion task step is invalid")


@dataclass(frozen=True)
class InsertionDecision:
    acquisition_index: int | None
    seated_index: int | None
    seated_indices: tuple[int, ...]
    grasp_clearance_meters: float
    seating_depth_error_meters: float
    seating_lateral_error_meters: float
    seating_orientation_error_rad: float
    failures: tuple[InsertionFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_insertion_geometry(
    steps: Sequence[InsertionGeometryStep],
    target: InsertionTarget,
    limits: InsertionTaskLimits = InsertionTaskLimits(),
    *,
    eligible_seating_indices: frozenset[int] | None = None,
) -> InsertionDecision:
    """Require a usable grasp retained through an aligned seating event."""

    axis = np.asarray(target.insertion_axis, dtype=np.float64)
    socket = np.asarray(target.socket_position, dtype=np.float64)
    acquisition_index = next(
        (
            index
            for index in range(1, len(steps))
            if not steps[index - 1].plug_attached and steps[index].plug_attached
        ),
        None,
    )
    failures = []
    grasp_clearance = 0.0
    seated_index = None
    seated_indices = []
    depth_error = float("inf")
    lateral_error = float("inf")
    orientation_error = float("inf")
    if acquisition_index is None:
        failures.append(InsertionFailure.NO_ATTACHMENT_TRANSITION)
    else:
        acquisition = steps[acquisition_index]
        gripper_from_tip = np.asarray(
            acquisition.gripper_frame_position
        ) - np.asarray(acquisition.plug_tip_position)
        grasp_clearance = float(np.dot(gripper_from_tip, -axis))
        if grasp_clearance < limits.minimum_grasp_clearance_meters:
            failures.append(InsertionFailure.INSUFFICIENT_GRASP_CLEARANCE)

        for index, step in enumerate(steps[acquisition_index:], acquisition_index):
            if not step.plug_attached:
                break
            if (
                eligible_seating_indices is not None
                and index not in eligible_seating_indices
            ):
                continue
            offset = np.asarray(step.plug_tip_position) - socket
            axial = abs(float(np.dot(offset, axis)))
            lateral_vector = offset - np.dot(offset, axis) * axis
            lateral = float(np.linalg.norm(lateral_vector))
            if (
                axial <= limits.maximum_depth_error_meters
                and lateral <= limits.maximum_lateral_error_meters
                and step.orientation_error_rad <= limits.maximum_orientation_error_rad
            ):
                seated_indices.append(index)
                if seated_index is None:
                    seated_index = index
                    depth_error = axial
                    lateral_error = lateral
                    orientation_error = step.orientation_error_rad
        if seated_index is None:
            failures.append(InsertionFailure.NOT_SEATED)
            if any(not step.plug_attached for step in steps[acquisition_index + 1 :]):
                failures.append(InsertionFailure.ATTACHMENT_LOST_BEFORE_SEATING)

    return InsertionDecision(
        acquisition_index,
        seated_index,
        tuple(seated_indices),
        grasp_clearance,
        depth_error,
        lateral_error,
        orientation_error,
        tuple(failures),
    )


def evaluate_insertion(
    steps: Sequence[InsertionTaskStep],
    target: InsertionTarget,
    limits: InsertionTaskLimits = InsertionTaskLimits(),
) -> InsertionDecision:
    """Require a usable grasp retained through a safe, aligned seating event."""

    geometry = evaluate_insertion_geometry(steps, target, limits)
    failures = list(geometry.failures)
    if any(not step.tracking_passed for step in steps):
        failures.append(InsertionFailure.TRACKING_FAILED)
    if any(step.collision_detected for step in steps):
        failures.append(InsertionFailure.COLLISION_DETECTED)
    if any(
        step.contact_force_newtons > limits.maximum_contact_force_newtons
        for step in steps
    ):
        failures.append(InsertionFailure.CONTACT_FORCE_EXCEEDED)
    return InsertionDecision(
        geometry.acquisition_index,
        geometry.seated_index,
        geometry.seated_indices,
        geometry.grasp_clearance_meters,
        geometry.seating_depth_error_meters,
        geometry.seating_lateral_error_meters,
        geometry.seating_orientation_error_rad,
        tuple(failures),
    )
