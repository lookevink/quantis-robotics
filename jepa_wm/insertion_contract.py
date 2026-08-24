"""Shared identity, geometry, and capture contract for reach-and-insert."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

INSERTION_TASK_ID = "reach_and_insert"
INSERTION_AXIS = (-1.0, 0.0, 0.0)
REARWARD_GRASP_OFFSET_METERS = 0.04
KINEMATIC_INSERTION_MODE = "kinematic_scripted_baseline"
CONTACT_AWARE_INSERTION_MODE = "contact_aware_scripted_baseline"
CONNECTOR_CONTACT_SENSOR_ID = "connector_tip"
COMPLIANT_COLLISION_PARTS = ("latch",)


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

    @property
    def insertion_steps(self) -> int:
        return self.span(ContactInsertionSegment.INSERT).frames

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
