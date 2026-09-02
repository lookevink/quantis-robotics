"""Evidence gate for a rearward-grasp cable insertion."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import acos, isfinite
from typing import Any, Sequence

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

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "socket_position": list(self.socket_position),
            "insertion_axis": list(self.insertion_axis),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTarget:
        if not isinstance(payload, dict):
            raise ValueError("insertion target must be an object")
        try:
            return cls(
                tuple(float(value) for value in payload["socket_position"]),
                tuple(float(value) for value in payload["insertion_axis"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion target is incomplete") from error

    def bind_live_target(
        self,
        live: InsertionTarget,
        expected_translation: Sequence[float],
        *,
        maximum_position_error_meters: float = 1e-4,
    ) -> InsertionTarget:
        """Authenticate a live socket as the reference under one translation."""

        translation = np.asarray(expected_translation, dtype=np.float64)
        expected_socket = np.asarray(self.socket_position) + translation
        if (
            translation.shape != (3,)
            or not np.all(np.isfinite(translation))
            or live.insertion_axis != self.insertion_axis
            or not isfinite(maximum_position_error_meters)
            or maximum_position_error_meters <= 0.0
            or np.linalg.norm(np.asarray(live.socket_position) - expected_socket)
            > maximum_position_error_meters
        ):
            raise ValueError("live insertion target is inconsistent with reference")
        return live


def quaternion_orientation_error(
    left_wxyz: Sequence[float],
    right_wxyz: Sequence[float],
) -> float:
    """Return the sign-invariant angular distance between two quaternions."""

    left = np.asarray(left_wxyz, dtype=np.float64)
    right = np.asarray(right_wxyz, dtype=np.float64)
    if (
        left.shape != (4,)
        or right.shape != (4,)
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
        or np.linalg.norm(left) <= 0.0
        or np.linalg.norm(right) <= 0.0
    ):
        raise ValueError("insertion orientation evidence is invalid")
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    cosine = min(1.0, abs(float(np.dot(left, right))))
    return 2.0 * acos(cosine)


@dataclass(frozen=True)
class InsertionTaskLimits:
    minimum_grasp_clearance_meters: float = 0.03
    maximum_depth_error_meters: float = 0.003
    maximum_lateral_error_meters: float = 0.003
    maximum_orientation_error_rad: float = 0.05
    maximum_contact_force_newtons: float = 2.0
    maximum_arm_tracking_error_rad: float = 0.01
    maximum_gripper_tracking_error_m: float = 0.003

    def __post_init__(self) -> None:
        values = (
            self.minimum_grasp_clearance_meters,
            self.maximum_depth_error_meters,
            self.maximum_lateral_error_meters,
            self.maximum_orientation_error_rad,
            self.maximum_contact_force_newtons,
            self.maximum_arm_tracking_error_rad,
            self.maximum_gripper_tracking_error_m,
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

    def to_dict(self) -> dict[str, object]:
        return {
            "plug_tip_position": list(self.plug_tip_position),
            "gripper_frame_position": list(self.gripper_frame_position),
            "plug_attached": self.plug_attached,
            "orientation_error_rad": self.orientation_error_rad,
            "tracking_passed": self.tracking_passed,
            "collision_detected": self.collision_detected,
            "contact_force_newtons": self.contact_force_newtons,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTaskStep:
        if not isinstance(payload, dict):
            raise ValueError("insertion task step must be an object")
        try:
            return cls(
                tuple(float(value) for value in payload["plug_tip_position"]),
                tuple(float(value) for value in payload["gripper_frame_position"]),
                payload["plug_attached"],
                float(payload["orientation_error_rad"]),
                payload["tracking_passed"],
                payload["collision_detected"],
                float(payload["contact_force_newtons"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion task step is incomplete") from error


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

    def evidence_dict(self) -> dict[str, object]:
        return {
            "acquisition_index": self.acquisition_index,
            "seated_index": self.seated_index,
            "seated_observations": len(self.seated_indices),
            "grasp_clearance_meters": self.grasp_clearance_meters,
            "seating_depth_error_meters": self.seating_depth_error_meters,
            "seating_lateral_error_meters": self.seating_lateral_error_meters,
            "seating_orientation_error_rad": self.seating_orientation_error_rad,
        }


def evaluate_insertion_geometry(
    steps: Sequence[InsertionGeometryStep],
    target: InsertionTarget,
    limits: InsertionTaskLimits = InsertionTaskLimits(),
    *,
    eligible_seating_indices: frozenset[int] | None = None,
    require_terminal_attachment: bool = False,
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

        retained_steps = steps[acquisition_index:]
        for index, step in enumerate(retained_steps, acquisition_index):
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
        if require_terminal_attachment and any(
            not step.plug_attached for step in steps[acquisition_index + 1 :]
        ):
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
    *,
    eligible_seating_indices: frozenset[int] | None = None,
    require_terminal_attachment: bool = False,
) -> InsertionDecision:
    """Require a usable grasp retained through a safe, aligned seating event."""

    geometry = evaluate_insertion_geometry(
        steps,
        target,
        limits,
        eligible_seating_indices=eligible_seating_indices,
        require_terminal_attachment=require_terminal_attachment,
    )
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
