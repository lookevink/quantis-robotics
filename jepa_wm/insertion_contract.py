"""Shared identity, geometry, and capture contract for reach-and-insert."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist, isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping

from jepa_wm.action import (
    ActionSelectionBounds,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.control_safety import ACTION_SCALES, ORIENTATION_HOLD_ACTION_SCALES
from jepa_wm.identifiers import validate_safe_identifier
from jepa_wm.trajectory import (
    RecordedRollout,
    RolloutProtocol,
    RolloutWindow,
    load_rollout_at,
)

INSERTION_TASK_ID = "reach_and_insert"
INSERTION_AXIS = (-1.0, 0.0, 0.0)
REARWARD_GRASP_OFFSET_METERS = 0.04
KINEMATIC_INSERTION_MODE = "kinematic_scripted_baseline"
CONTACT_AWARE_INSERTION_MODE = "contact_aware_scripted_baseline"
CONNECTOR_CONTACT_SENSOR_ID = "connector_tip"
COMPLIANT_COLLISION_PARTS = ("latch",)


@dataclass(frozen=True)
class InsertionControlTargetPolicy:
    minimum_translation_meters: float = 5e-4
    orientation_hold_tolerance_radians: float | None = 1.25e-3
    minimum_action_horizon: int = 3
    maximum_action_horizon: int = 8
    camera: str = "wrist"
    action_bounds: ActionSelectionBounds = ActionSelectionBounds(
        minimum_action_norm=0.0
    )

    def __post_init__(self) -> None:
        validate_safe_identifier(self.camera)
        if (
            not isfinite(self.minimum_translation_meters)
            or self.minimum_translation_meters <= 0.0
            or (
                self.orientation_hold_tolerance_radians is not None
                and (
                    not isfinite(self.orientation_hold_tolerance_radians)
                    or self.orientation_hold_tolerance_radians <= 0.0
                )
            )
            or isinstance(self.minimum_action_horizon, bool)
            or not isinstance(self.minimum_action_horizon, int)
            or self.minimum_action_horizon <= 0
            or isinstance(self.maximum_action_horizon, bool)
            or not isinstance(self.maximum_action_horizon, int)
            or self.maximum_action_horizon < self.minimum_action_horizon
        ):
            raise ValueError("insertion control target policy is invalid")

    def projection_scales(
        self,
        current: DroidPose,
        target: DroidPose | None,
    ) -> tuple[DroidActionScale, ...]:
        """Hold orientation when its target error is below measured resolution."""

        if target is None or self.orientation_hold_tolerance_radians is None:
            return ACTION_SCALES
        relative = action_between(current, target)
        rotation_error = sqrt(sum(value * value for value in relative.values[3:6]))
        return (
            ORIENTATION_HOLD_ACTION_SCALES
            if rotation_error <= self.orientation_hold_tolerance_radians
            else ACTION_SCALES
        )

    def select(
        self,
        recording: Path,
        *,
        context_index: int,
    ) -> RecordedRollout:
        """Select the nearest bounded-horizon target above control resolution."""

        for horizon in range(
            self.minimum_action_horizon,
            self.maximum_action_horizon + 1,
        ):
            rollout = load_rollout_at(
                recording,
                camera=self.camera,
                context_index=context_index,
                protocol=RolloutProtocol(action_horizon=horizon),
                bounds=self.action_bounds,
            )
            if (
                dist(
                    rollout.context_pose.values[:3],
                    rollout.target_pose.values[:3],
                )
                >= self.minimum_translation_meters
            ):
                return rollout
        raise ValueError(
            "insertion control target has no resolvable bounded-horizon goal"
        )

    def validate_observation(
        self,
        observation: ControlObservation,
        recording: Path,
        *,
        frame_root: Path,
    ) -> None:
        rollout = self.select(
            recording,
            context_index=observation.warmup_frames,
        )
        expected = ControlTarget(
            rollout.target.path.relative_to(frame_root.resolve()),
            rollout.target_pose,
        )
        if observation.target != expected:
            raise ValueError("insertion control observation target is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "minimum_translation_meters": self.minimum_translation_meters,
            "minimum_action_horizon": self.minimum_action_horizon,
            "maximum_action_horizon": self.maximum_action_horizon,
            "camera": self.camera,
            "action_bounds": self.action_bounds.to_dict(),
        }
        if self.orientation_hold_tolerance_radians is not None:
            payload["orientation_hold_tolerance_radians"] = (
                self.orientation_hold_tolerance_radians
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionControlTargetPolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("insertion control target policy must be an object")
        minimum_horizon = payload.get("minimum_action_horizon")
        maximum_horizon = payload.get("maximum_action_horizon")
        camera = payload.get("camera")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (minimum_horizon, maximum_horizon)
        ) or not isinstance(camera, str):
            raise ValueError("insertion target policy fields are invalid")
        try:
            return cls(
                minimum_translation_meters=float(
                    payload["minimum_translation_meters"]
                ),
                orientation_hold_tolerance_radians=(
                    float(payload["orientation_hold_tolerance_radians"])
                    if "orientation_hold_tolerance_radians" in payload
                    else None
                ),
                minimum_action_horizon=minimum_horizon,
                maximum_action_horizon=maximum_horizon,
                camera=camera,
                action_bounds=ActionSelectionBounds.from_dict(
                    payload["action_bounds"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion control target policy is incomplete") from error


INSERTION_CONTROL_TARGET_POLICY = InsertionControlTargetPolicy()


def insertion_control_target_policy(
    execution_policy: ControlExecutionPolicy,
) -> InsertionControlTargetPolicy | None:
    return (
        INSERTION_CONTROL_TARGET_POLICY
        if execution_policy
        in (
            ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            ControlExecutionPolicy.INSERTION_RESET_TRIAL,
        )
        else None
    )


class ContactInsertionSegment(str, Enum):
    INITIAL = "initial"
    PRE_GRASP = "pre_grasp"
    GRASP_OPEN = "grasp_open"
    GRASP_CLOSE = "grasp_close"
    GRASP_ATTACH = "grasp_attach"
    RETREAT = "retreat"
    RETREAT_HOLD = "retreat_hold"
    ALIGN = "align"
    ALIGN_HOLD = "align_hold"
    INSERT = "insert"
    SEATED_HOLD = "seated_hold"


@dataclass(frozen=True)
class ContactInsertionSpan:
    segment: ContactInsertionSegment
    phase: str
    stage: str
    frames: int
    attached: bool


@dataclass(frozen=True)
class ContactInsertionRecordingContract:
    attachment: str = "dynamic_fixed_joint"
    socket_scale: float = 1.05
    spans: tuple[ContactInsertionSpan, ...] = (
        ContactInsertionSpan(ContactInsertionSegment.INITIAL, "initial", "approaching_cable", 1, False),
        ContactInsertionSpan(ContactInsertionSegment.PRE_GRASP, "pre_grasp", "approaching_cable", 8, False),
        ContactInsertionSpan(ContactInsertionSegment.GRASP_OPEN, "grasp", "approaching_cable", 8, False),
        ContactInsertionSpan(ContactInsertionSegment.GRASP_CLOSE, "grasp_close", "approaching_cable", 4, False),
        ContactInsertionSpan(ContactInsertionSegment.GRASP_ATTACH, "grasp_attached", "cable_grasped", 1, True),
        ContactInsertionSpan(ContactInsertionSegment.RETREAT, "pre_insertion", "cable_grasped", 8, True),
        ContactInsertionSpan(ContactInsertionSegment.RETREAT_HOLD, "pre_insertion_settle", "cable_grasped", 4, True),
        ContactInsertionSpan(ContactInsertionSegment.ALIGN, "pre_insertion", "cable_grasped", 8, True),
        ContactInsertionSpan(ContactInsertionSegment.ALIGN_HOLD, "pre_insertion_settle", "aligned_with_socket", 2, True),
        ContactInsertionSpan(ContactInsertionSegment.INSERT, "insert", "aligned_with_socket", 64, True),
        ContactInsertionSpan(ContactInsertionSegment.SEATED_HOLD, "insert_settle", "plug_seated", 4, True),
    )

    def span(self, segment: ContactInsertionSegment) -> ContactInsertionSpan:
        return next(span for span in self.spans if span.segment is segment)

    def start_index(self, segment: ContactInsertionSegment) -> int:
        start = 0
        for span in self.spans:
            if span.segment is segment:
                return start
            start += span.frames
        raise ValueError(f"unknown insertion segment: {segment.value}")

    @property
    def insertion_steps(self) -> int:
        return self.span(ContactInsertionSegment.INSERT).frames

    @property
    def insertion_command_window(self) -> RolloutWindow:
        return RolloutWindow(
            self.start_index(ContactInsertionSegment.INSERT) - 1,
            self.insertion_steps,
            1,
        )

    @property
    def frame_count(self) -> int:
        return sum(span.frames for span in self.spans)

    @property
    def phase_roster(self) -> tuple[str, ...]:
        return tuple(
            span.phase for span in self.spans for _ in range(span.frames)
        )

    @property
    def stage_roster(self) -> tuple[str, ...]:
        return tuple(
            span.stage for span in self.spans for _ in range(span.frames)
        )

    @property
    def attachment_roster(self) -> tuple[bool, ...]:
        return tuple(
            span.attached for span in self.spans for _ in range(span.frames)
        )

    def instrumentation_metadata(
        self,
        compliant_collision_parts: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "connector_collisions_enabled": True,
            "contact_sensor": CONNECTOR_CONTACT_SENSOR_ID,
            "compliant_collision_parts": list(compliant_collision_parts),
            "attachment": self.attachment,
            "socket_scale": self.socket_scale,
            "insertion_steps": self.insertion_steps,
            "expected_frames": self.frame_count,
        }

    def validate_instrumentation(self, payload: Mapping[str, Any]) -> None:
        expected = self.instrumentation_metadata(COMPLIANT_COLLISION_PARTS)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("contact-aware insertion instrumentation is incomplete")


CONTACT_INSERTION_RECORDING = ContactInsertionRecordingContract()
