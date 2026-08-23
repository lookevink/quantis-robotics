"""Validated action/frame transitions from a recorded Isaac trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from jepa_wm.action import (
    DEFAULT_ACTION_SELECTION_BOUNDS,
    ActionRecordingContract,
    ActionSelectionBounds,
    DroidAction,
    DroidPose,
    action_between,
)


@dataclass(frozen=True)
class RecordedTransition:
    current_index: int
    next_index: int
    current_frame: Path
    next_frame: Path
    action: DroidAction


@dataclass(frozen=True)
class TransitionWindow:
    """Validated selection and naming contract for trajectory transitions."""

    start_index: int
    count: int
    stride: int

    def __post_init__(self) -> None:
        if self.start_index < 0:
            raise ValueError("start index must be non-negative")
        if self.count <= 0 or self.stride <= 0:
            raise ValueError("count and stride must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "start_index": self.start_index,
            "count": self.count,
            "stride": self.stride,
        }

    def report_name(self, camera: str) -> str:
        return (
            f"{camera}_transition_eval_" f"{self.start_index:06d}_{self.count:03d}.json"
        )


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _frame_path(recording: Path, step: dict[str, Any], camera: str) -> Path:
    try:
        relative_path = step["frames"][camera]
    except (KeyError, TypeError) as error:
        raise ValueError(f"recording step has no {camera!r} frame") from error
    if not isinstance(relative_path, str):
        raise ValueError("recording frame path must be a string")
    root = recording.resolve()
    frame = (root / relative_path).resolve()
    try:
        frame.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"recording frame escapes its root: {relative_path}"
        ) from error
    if not frame.is_file():
        raise ValueError(f"recording frame does not exist: {frame}")
    return frame


def load_transitions(
    recording: Path,
    *,
    camera: str,
    window: TransitionWindow,
    bounds: ActionSelectionBounds = DEFAULT_ACTION_SELECTION_BOUNDS,
) -> tuple[RecordedTransition, ...]:
    recording = recording.resolve()
    manifest = _read_object(recording / "manifest.json")
    action_contract = ActionRecordingContract.from_mapping(manifest.get("action"))
    cameras = manifest.get("cameras")
    if not isinstance(cameras, list) or camera not in cameras:
        raise ValueError(f"recording has no {camera!r} camera")
    steps = [
        json.loads(line)
        for line in (recording / "steps.jsonl").read_text().splitlines()
        if line
    ]
    if len(steps) != manifest.get("frames"):
        raise ValueError("manifest and telemetry frame counts differ")
    for index, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("index") != index:
            raise ValueError(f"recording step indices are not contiguous at {index}")

    transitions = []
    for current_index in range(
        window.start_index,
        len(steps) - window.stride,
        window.stride,
    ):
        next_index = current_index + window.stride
        current = steps[current_index]
        following = steps[next_index]
        try:
            previous_pose = DroidPose(tuple(current[action_contract.pose_field]))
            current_pose = DroidPose(tuple(following[action_contract.pose_field]))
            action = action_between(previous_pose, current_pose)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"recording step {next_index} has no valid DROID action"
            ) from error
        if not bounds.accepts(action):
            continue
        transitions.append(
            RecordedTransition(
                current_index=current_index,
                next_index=next_index,
                current_frame=_frame_path(recording, current, camera),
                next_frame=_frame_path(recording, following, camera),
                action=action,
            )
        )
        if len(transitions) == window.count:
            break
    if len(transitions) != window.count:
        raise ValueError(
            f"recording has only {len(transitions)} qualifying transitions, "
            f"expected {window.count}"
        )
    return tuple(transitions)
