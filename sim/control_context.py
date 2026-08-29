"""Validated recorded state used to initialize one live control context."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path

from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
    INSERTION_TASK_ID,
)
from jepa_wm.task_windows import CONTACT_GRASP_PROPOSAL_WINDOW
from sim.exploration import ExplorationPlan, exploration_prefix


GRASP_TASK_CONTEXT_START = 69
GRASP_TASK_CONTEXT_END = 98
EXPLORATION_CONTEXT_BOUNDARIES = tuple(range(4, 69, 4))


class ControlContextPurpose(str, Enum):
    """The narrow task phase that an initial live capture may replay."""

    STANDARD = "standard"
    CONTACT_GRASP = "contact_grasp"


def recording_task(recording: Path) -> str | None:
    manifest = json.loads((recording / "manifest.json").read_text())
    metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
    task = metadata.get("task") if isinstance(metadata, dict) else None
    return task if isinstance(task, str) else None


@dataclass(frozen=True)
class RecordedControlStep:
    index: int
    arm_positions: tuple[float, ...]
    gripper_width_m: float
    plug_attached: bool
    plug_position: tuple[float, ...]
    plug_orientation_wxyz: tuple[float, ...]

    @classmethod
    def from_dict(cls, payload: object) -> RecordedControlStep:
        if not isinstance(payload, dict):
            raise ValueError("recorded control step must be an object")
        try:
            step = cls(
                int(payload["index"]),
                tuple(float(value) for value in payload["arm_positions"]),
                float(payload["gripper_width_m"]),
                payload["plug_attached"],
                tuple(float(value) for value in payload["plug_position"]),
                tuple(float(value) for value in payload["plug_orientation_wxyz"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("recorded control step is incomplete") from error
        if (
            step.index < 0
            or len(step.arm_positions) != 7
            or not all(isfinite(value) for value in step.arm_positions)
            or not isfinite(step.gripper_width_m)
            or not 0.0 <= step.gripper_width_m <= 0.08
            or not isinstance(step.plug_attached, bool)
            or len(step.plug_position) != 3
            or not all(isfinite(value) for value in step.plug_position)
            or len(step.plug_orientation_wxyz) != 4
            or not all(isfinite(value) for value in step.plug_orientation_wxyz)
            or sum(value * value for value in step.plug_orientation_wxyz) <= 0.0
        ):
            raise ValueError("recorded control step is invalid")
        return step

    def to_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "arm_positions": list(self.arm_positions),
            "gripper_width_m": self.gripper_width_m,
            "plug_attached": self.plug_attached,
            "plug_position": list(self.plug_position),
            "plug_orientation_wxyz": list(self.plug_orientation_wxyz),
        }


def load_control_context(
    recording: Path,
    context_index: int,
    plan: ExplorationPlan,
    purpose: ControlContextPurpose = ControlContextPurpose.STANDARD,
) -> tuple[RecordedControlStep, ...]:
    """Load an exact prefix and admit task interiors only for grasp recordings."""

    task = recording_task(recording)
    if task == INSERTION_TASK_ID:
        if purpose is ControlContextPurpose.CONTACT_GRASP:
            if context_index != CONTACT_GRASP_PROPOSAL_WINDOW.start_index:
                raise ValueError(
                    "contact grasp capture must start at its canonical context"
                )
        elif context_index not in (
            CONTACT_INSERTION_RECORDING.insertion_command_window.context_indices
        ):
            raise ValueError("control context is outside the insertion command window")
    elif purpose is not ControlContextPurpose.STANDARD:
        raise ValueError("contact grasp capture requires an insertion recording")
    elif task == GRASP_TASK_ID:
        if not (
            context_index in EXPLORATION_CONTEXT_BOUNDARIES
            or GRASP_TASK_CONTEXT_START <= context_index <= GRASP_TASK_CONTEXT_END
        ):
            raise ValueError(
                "grasp control context is outside a complete task boundary"
            )
    else:
        exploration_prefix(plan, context_index)
    steps = tuple(
        RecordedControlStep.from_dict(json.loads(line))
        for line in (recording / "steps.jsonl").read_text().splitlines()
        if line
    )
    if context_index >= len(steps) - 3 or tuple(step.index for step in steps) != tuple(
        range(len(steps))
    ):
        raise ValueError(
            "recorded control context cannot provide a three-action target"
        )
    if steps[0].plug_attached:
        raise ValueError("recorded control context starts with an attached plug")
    return steps[: context_index + 1]
