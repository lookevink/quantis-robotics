"""Validated action/frame rollouts from a recorded Isaac trajectory."""

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
from jepa_wm.control_protocol import (
    ControlObservation,
    ControlTarget,
    TaskContextIndex,
)
from jepa_wm.rollout_protocol import (
    DROID_ROLLOUT_PROTOCOL,
    RolloutProtocol,
    RolloutWindow,
)


@dataclass(frozen=True)
class RecordedFrame:
    index: int
    path: Path

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("recorded frame index must be non-negative")


@dataclass(frozen=True)
class RecordedRollout:
    context: tuple[RecordedFrame, ...]
    context_pose: DroidPose
    previous_action: DroidAction
    target: RecordedFrame
    target_pose: DroidPose
    actions: tuple[DroidAction, ...]

    @property
    def context_paths(self) -> tuple[Path, ...]:
        return tuple(frame.path for frame in self.context)

    @property
    def target_clip(self) -> tuple[Path, ...]:
        return (self.target.path,)

    @property
    def goal_action(self) -> DroidAction:
        """Base-frame delta from the observed pose to the rollout goal pose."""

        return action_between(self.context_pose, self.target_pose)

    @property
    def task_context_index(self) -> TaskContextIndex:
        """Recorded task position shared by identically scripted domain trials."""

        return TaskContextIndex(self.context[0].index)


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


def load_rollouts(
    recording: Path,
    *,
    camera: str,
    protocol: RolloutProtocol = DROID_ROLLOUT_PROTOCOL,
    bounds: ActionSelectionBounds = DEFAULT_ACTION_SELECTION_BOUNDS,
) -> tuple[RecordedRollout, ...]:
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

    poses = []
    for index, step in enumerate(steps):
        try:
            poses.append(DroidPose(tuple(step[action_contract.pose_field])))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"recording step {index} has no valid DROID pose"
            ) from error

    rollouts = []
    candidate_stop = len(steps) - protocol.context_frames - protocol.action_horizon + 1
    for context_start in range(candidate_stop):
        context_stop = context_start + protocol.context_frames
        target_index = context_stop - 1 + protocol.action_horizon
        actions = tuple(
            action_between(poses[index], poses[index + 1])
            for index in range(context_stop - 1, target_index)
        )
        if not bounds.accepts_rollout(actions):
            continue
        context_indices = tuple(range(context_start, context_stop))
        rollouts.append(
            RecordedRollout(
                context=tuple(
                    RecordedFrame(
                        index,
                        _frame_path(recording, steps[index], camera),
                    )
                    for index in context_indices
                ),
                context_pose=poses[context_stop - 1],
                previous_action=(
                    action_between(poses[context_stop - 2], poses[context_stop - 1])
                    if context_stop >= 2
                    else DroidAction((0.0,) * 7)
                ),
                target=RecordedFrame(
                    target_index,
                    _frame_path(recording, steps[target_index], camera),
                ),
                target_pose=poses[target_index],
                actions=actions,
            )
        )
    return tuple(rollouts)


def load_rollout_at(
    recording: Path,
    *,
    camera: str,
    context_index: int,
    protocol: RolloutProtocol = DROID_ROLLOUT_PROTOCOL,
    bounds: ActionSelectionBounds = DEFAULT_ACTION_SELECTION_BOUNDS,
) -> RecordedRollout:
    matches = tuple(
        rollout
        for rollout in load_rollouts(
            recording,
            camera=camera,
            protocol=protocol,
            bounds=bounds,
        )
        if rollout.context[0].index == context_index
    )
    if len(matches) != 1:
        raise ValueError(
            f"recording has {len(matches)} rollouts at context index {context_index}"
        )
    return matches[0]


def validate_observation_target(
    observation: ControlObservation,
    recording: Path,
    *,
    frame_root: Path,
    camera: str = "wrist",
) -> None:
    rollout = load_rollout_at(
        recording,
        camera=camera,
        context_index=observation.warmup_frames,
        bounds=ActionSelectionBounds(minimum_action_norm=0.0),
    )
    expected = ControlTarget(
        rollout.target.path.relative_to(frame_root.resolve()),
        rollout.target_pose,
    )
    if observation.target != expected:
        raise ValueError("control target does not match its reference telemetry")
