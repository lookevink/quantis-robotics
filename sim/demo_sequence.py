"""Simulator-independent waypoint contract for the plug-in demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence

from jepa_wm.insertion_contract import REARWARD_GRASP_OFFSET_METERS


Vector3 = tuple[float, float, float]


class Phase(str, Enum):
    READY = "ready"
    PRE_GRASP = "pre_grasp"
    GRASP = "grasp"
    PRE_INSERTION = "pre_insertion"
    INSERT = "insert"
    RELEASE = "release"


class PlugAction(str, Enum):
    KEEP = "keep"
    ATTACH = "attach"
    DETACH = "detach"


@dataclass(frozen=True)
class DemoGeometry:
    plug_position: Vector3
    socket_position: Vector3
    ready_position: Vector3

    def __post_init__(self) -> None:
        for name, position in (
            ("plug_position", self.plug_position),
            ("socket_position", self.socket_position),
            ("ready_position", self.ready_position),
        ):
            if len(position) != 3 or not all(isfinite(value) for value in position):
                raise ValueError(f"{name} must contain three finite coordinates")


@dataclass(frozen=True)
class Waypoint:
    phase: Phase
    target_position: Vector3
    gripper_width_m: float
    plug_action: PlugAction = PlugAction.KEEP


def _add(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return tuple(left[index] + right[index] for index in range(3))  # type: ignore[return-value]


def build_demo_sequence(
    geometry: DemoGeometry,
    *,
    approach_clearance_m: float = 0.10,
    insertion_clearance_m: float = 0.03,
    grasp_offset_m: float = REARWARD_GRASP_OFFSET_METERS,
    open_width_m: float = 0.07,
    grasp_width_m: float = 0.018,
) -> tuple[Waypoint, ...]:
    """Build the ordered Cartesian targets for a negative-X insertion."""

    if geometry.socket_position[0] >= geometry.plug_position[0]:
        raise ValueError("socket must be on the negative X side of the plug")
    if (
        approach_clearance_m <= 0
        or insertion_clearance_m <= 0
        or grasp_offset_m <= 0
    ):
        raise ValueError("clearances and grasp offset must be positive")
    if not 0 <= grasp_width_m < open_width_m:
        raise ValueError("grasp width must be non-negative and smaller than open width")

    grasp_offset = (grasp_offset_m, 0.0, 0.0)
    grasp = _add(geometry.plug_position, grasp_offset)
    insert = _add(geometry.socket_position, grasp_offset)
    pre_grasp = _add(grasp, (approach_clearance_m, 0.0, 0.0))
    pre_insert = _add(insert, (insertion_clearance_m, 0.0, 0.0))

    return (
        Waypoint(Phase.READY, geometry.ready_position, open_width_m),
        Waypoint(Phase.PRE_GRASP, pre_grasp, open_width_m),
        Waypoint(Phase.GRASP, grasp, grasp_width_m, PlugAction.ATTACH),
        Waypoint(Phase.PRE_INSERTION, pre_insert, grasp_width_m),
        Waypoint(Phase.INSERT, insert, grasp_width_m),
        Waypoint(Phase.RELEASE, insert, open_width_m, PlugAction.DETACH),
    )
