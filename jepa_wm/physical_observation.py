"""Dependency-light task-relative state for physical motion routing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Any, Mapping, Sequence

from jepa_wm.action import DroidAction


PHYSICAL_ROUTING_OBSERVATION_SCHEMA = "quantis.jepa_wm_physical_routing_observation.v2"
PHYSICAL_ROUTING_FEATURE_NAMES = (
    "plug_to_socket_axial_m",
    "plug_to_socket_lateral_x_m",
    "plug_to_socket_lateral_y_m",
    "plug_to_socket_lateral_z_m",
    "end_effector_to_socket_x_m",
    "end_effector_to_socket_y_m",
    "end_effector_to_socket_z_m",
    "gripper_frame_to_socket_x_m",
    "gripper_frame_to_socket_y_m",
    "gripper_frame_to_socket_z_m",
    "plug_to_socket_orientation_w",
    "plug_to_socket_orientation_x",
    "plug_to_socket_orientation_y",
    "plug_to_socket_orientation_z",
    "gripper_width_m",
    "arm_tracking_error_rad",
    "gripper_tracking_error_m",
    "contact_force_newtons",
    "plug_attached",
    "previous_action_x",
    "previous_action_y",
    "previous_action_z",
    "previous_action_rx",
    "previous_action_ry",
    "previous_action_rz",
    "previous_action_gripper",
)


def _vector(payload: Any, *, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        raise ValueError(f"{name} must contain {length} finite values")
    try:
        values = tuple(float(value) for value in payload)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} finite values") from error
    if len(values) != length or not all(isfinite(value) for value in values):
        raise ValueError(f"{name} must contain {length} finite values")
    return values


def _difference(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(
        left_value - right_value for left_value, right_value in zip(left, right)
    )


def _unit_quaternion(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    quaternion = _vector(values, length=4, name=name)
    norm = sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return tuple(value / norm for value in quaternion)


def _quaternion_product(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return (
        left_w * right_w - left_x * right_x - left_y * right_y - left_z * right_z,
        left_w * right_x + left_x * right_w + left_y * right_z - left_z * right_y,
        left_w * right_y - left_x * right_z + left_y * right_w + left_z * right_x,
        left_w * right_z + left_x * right_y - left_y * right_x + left_z * right_w,
    )


def _relative_quaternion(
    plug: Sequence[float],
    socket: Sequence[float],
) -> tuple[float, float, float, float]:
    plug_unit = _unit_quaternion(plug, name="plug orientation")
    socket_unit = _unit_quaternion(socket, name="socket orientation")
    socket_inverse = (
        socket_unit[0],
        -socket_unit[1],
        -socket_unit[2],
        -socket_unit[3],
    )
    relative = _quaternion_product(socket_inverse, plug_unit)
    for value in relative:
        if abs(value) <= 1e-12:
            continue
        return tuple(-component for component in relative) if value < 0.0 else relative
    raise ValueError("relative plug orientation is invalid")


@dataclass(frozen=True)
class PhysicalRoutingObservation:
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != len(PHYSICAL_ROUTING_FEATURE_NAMES) or not all(
            isfinite(value) for value in self.values
        ):
            raise ValueError("physical routing observation is invalid")

    @classmethod
    def from_recorded_step(
        cls,
        step: Mapping[str, Any],
        insertion_target: Mapping[str, Any],
        previous_action: DroidAction,
    ) -> PhysicalRoutingObservation:
        socket = _vector(
            insertion_target.get("socket_position"),
            length=3,
            name="socket position",
        )
        axis = _vector(
            insertion_target.get("insertion_axis"),
            length=3,
            name="insertion axis",
        )
        socket_orientation = _unit_quaternion(
            insertion_target.get("socket_orientation_wxyz"),
            name="socket orientation",
        )
        plug = _vector(step.get("plug_position"), length=3, name="plug position")
        plug_delta = _difference(plug, socket)
        axial = sum(value * direction for value, direction in zip(plug_delta, axis))
        lateral = tuple(
            value - axial * direction for value, direction in zip(plug_delta, axis)
        )
        end_effector = _vector(
            step.get("end_effector_world_position"),
            length=3,
            name="end-effector position",
        )
        gripper = _vector(
            step.get("gripper_frame_world_position"),
            length=3,
            name="gripper-frame position",
        )
        plug_orientation = _unit_quaternion(
            step.get("plug_orientation_wxyz"),
            name="plug orientation",
        )
        attached = step.get("plug_attached")
        if not isinstance(attached, bool):
            raise ValueError("plug attachment must be boolean")
        scalars = _vector(
            (
                step.get("gripper_width_m"),
                step.get("arm_tracking_error_rad"),
                step.get("gripper_tracking_error_m"),
                step.get("contact_force_newtons"),
            ),
            length=4,
            name="physical routing scalars",
        )
        return cls(
            (
                axial,
                *lateral,
                *_difference(end_effector, socket),
                *_difference(gripper, socket),
                *_relative_quaternion(plug_orientation, socket_orientation),
                *scalars,
                float(attached),
                *previous_action.values,
            )
        )
