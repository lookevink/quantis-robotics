"""Immutable scheduling and identity contracts for live control capture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from math import acos, isfinite, sqrt
from pathlib import Path
from typing import Any

from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.insertion_contract import insertion_control_target_policy
from jepa_wm.training_artifact import (
    artifact_fingerprint,
    validate_artifact_fingerprint,
)
from sim.control_context import ControlContextPurpose, RecordedControlStep
from sim.recording import validate_recording_id


CONTROL_CAPTURE_SCHEDULE_SCHEMA = "quantis.control_capture_schedule.v1"
CONTROL_KNOWN_START_SCHEMA = "quantis.control_known_start.v1"
CONTROL_CAPTURE_CLIENT_TIMEOUT_SECONDS = 900
KNOWN_START_POSITION_TOLERANCE_METERS = 1e-5
KNOWN_START_ORIENTATION_TOLERANCE_RADIANS = 1e-3


@dataclass(frozen=True)
class ControlCaptureTimingBudget:
    """Synthetic wall-clock budget for each externally visible capture phase."""

    phases: tuple[tuple[str, int], ...]
    maximum_total_seconds: int = CONTROL_CAPTURE_CLIENT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if (
            not self.phases
            or len({phase for phase, _ in self.phases}) != len(self.phases)
            or any(not phase or seconds <= 0 for phase, seconds in self.phases)
            or sum(seconds for _, seconds in self.phases) > self.maximum_total_seconds
        ):
            raise ValueError("control capture timing budget is invalid")

    def phase_seconds(self, phase: str) -> int:
        try:
            return dict(self.phases)[phase]
        except KeyError as error:
            raise ValueError(
                f"control capture phase is not budgeted: {phase}"
            ) from error

    def validate_elapsed(self, phase: str, elapsed_seconds: float) -> None:
        if (
            not isfinite(elapsed_seconds)
            or elapsed_seconds < 0.0
            or elapsed_seconds > self.phase_seconds(phase)
        ):
            raise ValueError(f"control capture phase exceeded its budget: {phase}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_total_seconds": self.maximum_total_seconds,
            "phases": {phase: seconds for phase, seconds in self.phases},
        }


KNOWN_START_CAPTURE_TIMING_BUDGET = ControlCaptureTimingBudget(
    (
        ("reset", 120),
        ("known_start", 60),
        ("terminal_camera_and_stabilization", 600),
        ("terminal_snapshot", 60),
    )
)


def canonical_control_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def control_reference_fingerprint(recording: Path) -> str:
    """Bind the manifest and complete canonical telemetry used by control."""

    return canonical_control_fingerprint(
        {
            name: artifact_fingerprint(recording / name)
            for name in ("manifest.json", "steps.jsonl")
        }
    )


def validate_known_start_pose(
    label: str,
    actual_position: tuple[float, ...],
    actual_orientation_wxyz: tuple[float, ...],
    expected_position: tuple[float, ...],
    expected_orientation_wxyz: tuple[float, ...],
) -> None:
    """Fail closed when a live scene pose differs from its recorded binding."""

    if (
        any(
            len(vector) != length or not all(isfinite(value) for value in vector)
            for vector, length in (
                (actual_position, 3),
                (actual_orientation_wxyz, 4),
                (expected_position, 3),
                (expected_orientation_wxyz, 4),
            )
        )
        or sqrt(
            sum(
                (actual - expected) ** 2
                for actual, expected in zip(actual_position, expected_position)
            )
        )
        > KNOWN_START_POSITION_TOLERANCE_METERS
    ):
        raise RuntimeError(f"known-start {label} position does not match reference")
    actual_norm = sqrt(sum(value * value for value in actual_orientation_wxyz))
    expected_norm = sqrt(sum(value * value for value in expected_orientation_wxyz))
    if actual_norm <= 0.0 or expected_norm <= 0.0:
        raise RuntimeError(f"known-start {label} orientation does not match reference")
    dot = abs(
        sum(
            actual * expected
            for actual, expected in zip(
                actual_orientation_wxyz,
                expected_orientation_wxyz,
            )
        )
        / (actual_norm * expected_norm)
    )
    orientation_error = 2.0 * acos(min(dot, 1.0))
    if orientation_error > KNOWN_START_ORIENTATION_TOLERANCE_RADIANS:
        raise RuntimeError(f"known-start {label} orientation does not match reference")


def validate_known_start_collision_configuration(
    target_metadata: Any,
    compliant_parts: tuple[str, ...],
    collision_configuration: tuple[tuple[str, bool], ...],
) -> None:
    """Authenticate the complete authored connector collision configuration."""

    if not isinstance(target_metadata, dict):
        raise RuntimeError("known-start connector collision configuration is invalid")
    expected_compliant_parts = tuple(
        sorted(target_metadata.get("compliant_collision_parts", ()))
    )
    if (
        target_metadata.get("connector_collisions_enabled") is not True
        or compliant_parts != expected_compliant_parts
        or not collision_configuration
        or any(
            not isinstance(path, str)
            or not isinstance(enabled, bool)
            or enabled != (path.rsplit("/", 1)[-1] not in expected_compliant_parts)
            for path, enabled in collision_configuration
        )
    ):
        raise RuntimeError("known-start connector collision configuration is invalid")


def requires_stable_insertion_capture(
    policy: ControlExecutionPolicy,
    *,
    insertion_control: bool,
    step_index: int,
    context_index: int,
    context_purpose: ControlContextPurpose = ControlContextPurpose.STANDARD,
) -> bool:
    """Require a strict stable frame before insertion safety or execution."""

    return (
        insertion_control
        and step_index == context_index
        and (
            context_purpose is ControlContextPurpose.CONTACT_GRASP
            or policy is ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT
            or insertion_control_target_policy(policy) is not None
        )
    )


def should_record_control_warmup_frame(
    *,
    context_purpose: ControlContextPurpose,
    step_index: int,
    context_index: int,
) -> bool:
    """Persist only the RGB frames that the control observation can consume."""

    if step_index < 0 or context_index < 0 or step_index > context_index:
        raise ValueError("control warm-up frame index is invalid")
    return (
        context_purpose is not ControlContextPurpose.CONTACT_GRASP
        or step_index == context_index
    )


@dataclass(frozen=True)
class ControlWarmupFramePlan:
    task_index: int
    record_rgb: bool
    stabilize: bool
    observe_safety: bool


@dataclass(frozen=True)
class ControlCaptureSchedule:
    """A fingerprinted reset/replay/render schedule for one initial capture."""

    frames: tuple[ControlWarmupFramePlan, ...]
    initialization_task_index: int
    defer_camera_activation: bool

    def __post_init__(self) -> None:
        if (
            not self.frames
            or tuple(frame.task_index for frame in self.frames)
            != tuple(range(len(self.frames)))
            or self.initialization_task_index not in range(len(self.frames))
            or not isinstance(self.defer_camera_activation, bool)
        ):
            raise ValueError("control capture schedule is invalid")
        if self.defer_camera_activation != (self.initialization_task_index > 0):
            raise ValueError("control capture renderer schedule is invalid")

    @property
    def replay_frames(self) -> tuple[ControlWarmupFramePlan, ...]:
        return self.frames[self.initialization_task_index + 1 :]

    @property
    def recorded_task_indices(self) -> tuple[int, ...]:
        return tuple(frame.task_index for frame in self.frames if frame.record_rgb)

    @property
    def progress_units(self) -> int:
        """Bound progress independently from the full task-context index."""

        return len(self.replay_frames) + 5

    @property
    def timing_budget(self) -> ControlCaptureTimingBudget | None:
        return (
            KNOWN_START_CAPTURE_TIMING_BUDGET if self.defer_camera_activation else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_CAPTURE_SCHEDULE_SCHEMA,
            "initialization_task_index": self.initialization_task_index,
            "defer_camera_activation": self.defer_camera_activation,
            "timing_budget": (
                self.timing_budget.to_dict() if self.timing_budget is not None else None
            ),
            "frames": [
                {
                    "task_index": frame.task_index,
                    "record_rgb": frame.record_rgb,
                    "stabilize": frame.stabilize,
                    "observe_safety": frame.observe_safety,
                }
                for frame in self.frames
            ],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlCaptureSchedule:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != CONTROL_CAPTURE_SCHEDULE_SCHEMA
            or not isinstance(payload.get("frames"), list)
        ):
            raise ValueError("control capture schedule payload is invalid")
        try:
            frames = tuple(
                ControlWarmupFramePlan(
                    int(frame["task_index"]),
                    frame["record_rgb"],
                    frame["stabilize"],
                    frame["observe_safety"],
                )
                for frame in payload["frames"]
            )
            if any(
                not isinstance(value, bool)
                for frame in frames
                for value in (
                    frame.record_rgb,
                    frame.stabilize,
                    frame.observe_safety,
                )
            ):
                raise ValueError
            schedule = cls(
                frames,
                int(payload["initialization_task_index"]),
                payload["defer_camera_activation"],
            )
            expected_timing_budget = (
                schedule.timing_budget.to_dict()
                if schedule.timing_budget is not None
                else None
            )
            if payload.get("timing_budget") != expected_timing_budget:
                raise ValueError
            return schedule
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control capture schedule payload is invalid") from error

    @property
    def fingerprint(self) -> str:
        return canonical_control_fingerprint(self.to_dict())


@dataclass(frozen=True)
class ControlKnownStartAuthority:
    """Authenticated inputs that define one direct initialization boundary."""

    reference_fingerprint: str
    stage_asset_fingerprint: str
    exploration_plan_fingerprint: str
    context_fingerprint: str
    target_fingerprint: str
    collision_configuration_fingerprint: str

    def __post_init__(self) -> None:
        for fingerprint in (
            self.reference_fingerprint,
            self.stage_asset_fingerprint,
            self.exploration_plan_fingerprint,
            self.context_fingerprint,
            self.target_fingerprint,
            self.collision_configuration_fingerprint,
        ):
            validate_artifact_fingerprint(fingerprint)

    def to_dict(self) -> dict[str, str]:
        return {
            "reference_fingerprint": self.reference_fingerprint,
            "stage_asset_fingerprint": self.stage_asset_fingerprint,
            "exploration_plan_fingerprint": self.exploration_plan_fingerprint,
            "context_fingerprint": self.context_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "collision_configuration_fingerprint": (
                self.collision_configuration_fingerprint
            ),
        }


@dataclass(frozen=True)
class ControlKnownStart:
    """The exact authenticated reset state used instead of physical prefix replay."""

    reference_recording: str
    seed: int
    task_index: int
    arm_positions: tuple[float, ...]
    gripper_width_m: float
    plug_attached: bool
    plug_position: tuple[float, ...]
    plug_orientation_wxyz: tuple[float, ...]
    socket_position: tuple[float, ...]
    socket_orientation_wxyz: tuple[float, ...]
    schedule_fingerprint: str
    authority: ControlKnownStartAuthority

    def __post_init__(self) -> None:
        validate_recording_id(self.reference_recording)
        validate_artifact_fingerprint(self.schedule_fingerprint)
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or isinstance(self.task_index, bool)
            or not isinstance(self.task_index, int)
            or self.task_index < 0
            or len(self.arm_positions) != 7
            or not all(isfinite(value) for value in self.arm_positions)
            or not isfinite(self.gripper_width_m)
            or not 0.0 <= self.gripper_width_m <= 0.08
            or self.plug_attached
            or len(self.plug_position) != 3
            or not all(isfinite(value) for value in self.plug_position)
            or len(self.plug_orientation_wxyz) != 4
            or not all(isfinite(value) for value in self.plug_orientation_wxyz)
            or sum(value * value for value in self.plug_orientation_wxyz) <= 0.0
            or len(self.socket_position) != 3
            or not all(isfinite(value) for value in self.socket_position)
            or len(self.socket_orientation_wxyz) != 4
            or not all(isfinite(value) for value in self.socket_orientation_wxyz)
            or sum(value * value for value in self.socket_orientation_wxyz) <= 0.0
        ):
            raise ValueError("control known start is invalid")

    @classmethod
    def from_context(
        cls,
        reference_recording: str,
        seed: int,
        step: RecordedControlStep,
        socket_position: tuple[float, ...],
        socket_orientation_wxyz: tuple[float, ...],
        schedule: ControlCaptureSchedule,
        authority: ControlKnownStartAuthority,
    ) -> ControlKnownStart:
        if step.index != schedule.initialization_task_index:
            raise ValueError("control known start does not match its schedule")
        return cls(
            reference_recording,
            seed,
            step.index,
            step.arm_positions,
            step.gripper_width_m,
            step.plug_attached,
            step.plug_position,
            step.plug_orientation_wxyz,
            socket_position,
            socket_orientation_wxyz,
            schedule.fingerprint,
            authority,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_KNOWN_START_SCHEMA,
            "reference_recording": self.reference_recording,
            "seed": self.seed,
            "task_index": self.task_index,
            "arm_positions": list(self.arm_positions),
            "gripper_width_m": self.gripper_width_m,
            "plug_attached": self.plug_attached,
            "plug_position": list(self.plug_position),
            "plug_orientation_wxyz": list(self.plug_orientation_wxyz),
            "socket_position": list(self.socket_position),
            "socket_orientation_wxyz": list(self.socket_orientation_wxyz),
            "schedule_fingerprint": self.schedule_fingerprint,
            "authority": self.authority.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_control_fingerprint(self.to_dict())


def control_warmup_plan(
    policy: ControlExecutionPolicy,
    *,
    insertion_control: bool,
    context_index: int,
    context_purpose: ControlContextPurpose,
) -> tuple[ControlWarmupFramePlan, ...]:
    """Plan task contexts independently from sparse RGB persistence."""

    return tuple(
        ControlWarmupFramePlan(
            task_index=step_index,
            record_rgb=should_record_control_warmup_frame(
                context_purpose=context_purpose,
                step_index=step_index,
                context_index=context_index,
            ),
            stabilize=requires_stable_insertion_capture(
                policy,
                insertion_control=insertion_control,
                step_index=step_index,
                context_index=context_index,
                context_purpose=context_purpose,
            ),
            observe_safety=insertion_control,
        )
        for step_index in range(context_index + 1)
    )


def control_capture_schedule(
    policy: ControlExecutionPolicy,
    *,
    insertion_control: bool,
    context_index: int,
    context_purpose: ControlContextPurpose,
) -> ControlCaptureSchedule:
    frames = control_warmup_plan(
        policy,
        insertion_control=insertion_control,
        context_index=context_index,
        context_purpose=context_purpose,
    )
    known_start = context_purpose is ControlContextPurpose.CONTACT_GRASP
    return ControlCaptureSchedule(
        frames,
        context_index if known_start else 0,
        known_start,
    )
