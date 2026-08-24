"""Versioned wire messages for simulator-only JEPA-WM control."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from jepa_wm.action import DroidAction, DroidPose, action_between


CONTROL_SCHEMA = "quantis.jepa_wm_control.v1"
DROID_ACTION_HORIZON = 3


def _strict_positive_int(payload: dict[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _strict_nonnegative_int(payload: dict[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class ControlTarget:
    frame: Path
    pose: DroidPose | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"frame": str(self.frame)}
        if self.pose is not None:
            payload["pose"] = list(self.pose.values)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ControlTarget:
        if not isinstance(payload, dict):
            raise ValueError("control target is incomplete")
        try:
            return cls(
                frame=Path(payload["frame"]),
                pose=(
                    DroidPose(tuple(payload["pose"]))
                    if payload.get("pose") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control target is incomplete") from error


@dataclass(frozen=True)
class ControlObservation:
    observation_id: int
    captured_at_unix_seconds: float
    context_frame: Path
    target: ControlTarget
    expected_proposal: Path
    pose: DroidPose
    previous_action: DroidAction
    warmup_frames: int

    @property
    def target_frame(self) -> Path:
        return self.target.frame

    @property
    def target_pose(self) -> DroidPose | None:
        return self.target.pose

    @property
    def goal_action(self) -> DroidAction:
        if self.target_pose is None:
            raise ValueError("control observation has no target pose")
        return action_between(self.pose, self.target_pose)

    def __post_init__(self) -> None:
        if isinstance(self.observation_id, bool) or self.observation_id <= 0:
            raise ValueError("control observation ID must be positive")
        if not isfinite(self.captured_at_unix_seconds):
            raise ValueError("control capture time must be finite")
        if isinstance(self.warmup_frames, bool) or self.warmup_frames < 0:
            raise ValueError("control warm-up frame count must be non-negative")
        if not self.expected_proposal.is_absolute():
            raise ValueError("expected control proposal path must be absolute")

    @classmethod
    def from_dict(cls, payload: Any) -> ControlObservation:
        if not isinstance(payload, dict) or payload.get("schema") != CONTROL_SCHEMA:
            raise ValueError("control observation schema is unsupported")
        try:
            return cls(
                observation_id=_strict_positive_int(payload, "observation_id"),
                captured_at_unix_seconds=float(payload["captured_at_unix_seconds"]),
                context_frame=Path(payload["context_frame"]),
                target=(
                    ControlTarget.from_dict(payload["target"])
                    if payload.get("target") is not None
                    else ControlTarget(
                        frame=Path(payload["target_frame"]),
                        pose=(
                            DroidPose(tuple(payload["target_pose"]))
                            if payload.get("target_pose") is not None
                            else None
                        ),
                    )
                ),
                expected_proposal=Path(payload["expected_proposal"]),
                pose=DroidPose(tuple(payload["pose"])),
                previous_action=DroidAction(tuple(payload["previous_action"])),
                warmup_frames=_strict_nonnegative_int(payload, "warmup_frames"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control observation is incomplete") from error

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CONTROL_SCHEMA,
            "observation_id": self.observation_id,
            "captured_at_unix_seconds": self.captured_at_unix_seconds,
            "context_frame": str(self.context_frame),
            "target": self.target.to_dict(),
            "expected_proposal": str(self.expected_proposal),
            "pose": list(self.pose.values),
            "previous_action": list(self.previous_action.values),
            "warmup_frames": self.warmup_frames,
        }
        return payload


@dataclass(frozen=True)
class ProposedControl:
    observation_id: int
    created_at_unix_seconds: float
    actions: tuple[DroidAction, ...]
    proposal: Path

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_id, bool)
            or self.observation_id <= 0
            or not isfinite(self.created_at_unix_seconds)
        ):
            raise ValueError("proposed control identity is invalid")
        if len(self.actions) != DROID_ACTION_HORIZON:
            raise ValueError("proposed control must contain the DROID action horizon")
        if not self.proposal.is_absolute():
            raise ValueError("control proposal path must be absolute")

    @property
    def first_action(self) -> DroidAction:
        return self.actions[0]

    @classmethod
    def from_dict(cls, payload: Any) -> ProposedControl:
        if not isinstance(payload, dict) or payload.get("schema") != CONTROL_SCHEMA:
            raise ValueError("proposed control schema is unsupported")
        try:
            return cls(
                observation_id=_strict_positive_int(payload, "observation_id"),
                created_at_unix_seconds=float(payload["created_at_unix_seconds"]),
                actions=tuple(DroidAction(tuple(action)) for action in payload["actions"]),
                proposal=Path(payload["proposal"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("proposed control is incomplete") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_SCHEMA,
            "observation_id": self.observation_id,
            "created_at_unix_seconds": self.created_at_unix_seconds,
            "actions": [list(action.values) for action in self.actions],
            "proposal": str(self.proposal),
        }
