"""Dependency-light phase layout for the drive-only insertion recording."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


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


DRIVE_ONLY_ARM_MOTION_FRAMES = 48
DRIVE_ONLY_GRIPPER_CLOSE_FRAMES = 16


@dataclass(frozen=True)
class ContactInsertionSpan:
    segment: ContactInsertionSegment
    phase: str
    stage: str
    frames: int
    attached: bool


@dataclass(frozen=True)
class ContactInsertionLayout:
    spans: tuple[ContactInsertionSpan, ...] = (
        ContactInsertionSpan(
            ContactInsertionSegment.INITIAL,
            "initial",
            "approaching_cable",
            1,
            False,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.PRE_GRASP,
            "pre_grasp",
            "approaching_cable",
            DRIVE_ONLY_ARM_MOTION_FRAMES,
            False,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.GRASP_OPEN,
            "grasp",
            "approaching_cable",
            DRIVE_ONLY_ARM_MOTION_FRAMES,
            False,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.GRASP_CLOSE,
            "grasp_close",
            "approaching_cable",
            DRIVE_ONLY_GRIPPER_CLOSE_FRAMES,
            False,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.GRASP_ATTACH,
            "grasp_attached",
            "cable_grasped",
            1,
            True,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.RETREAT,
            "pre_insertion",
            "cable_grasped",
            DRIVE_ONLY_ARM_MOTION_FRAMES,
            True,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.RETREAT_HOLD,
            "pre_insertion_settle",
            "cable_grasped",
            4,
            True,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.ALIGN,
            "pre_insertion",
            "cable_grasped",
            DRIVE_ONLY_ARM_MOTION_FRAMES,
            True,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.ALIGN_HOLD,
            "pre_insertion_settle",
            "aligned_with_socket",
            2,
            True,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.INSERT,
            "insert",
            "aligned_with_socket",
            64,
            True,
        ),
        ContactInsertionSpan(
            ContactInsertionSegment.SEATED_HOLD,
            "insert_settle",
            "plug_seated",
            4,
            True,
        ),
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

    def segment_for_index(self, index: int) -> ContactInsertionSegment:
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("insertion layout index must be a non-negative integer")
        start = 0
        for span in self.spans:
            stop = start + span.frames
            if index < stop:
                return span.segment
            start = stop
        raise ValueError("index is outside the insertion layout")

    @property
    def insertion_steps(self) -> int:
        return self.span(ContactInsertionSegment.INSERT).frames

    @property
    def insertion_command_context_indices(self) -> tuple[int, ...]:
        start = self.start_index(ContactInsertionSegment.INSERT) - 1
        return tuple(range(start, start + self.insertion_steps))

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


CONTACT_INSERTION_LAYOUT = ContactInsertionLayout()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the canonical drive-only insertion layout."
    )
    parser.add_argument(
        "field",
        choices=("initial-command-context", "command-contexts"),
    )
    arguments = parser.parse_args(argv)
    contexts = CONTACT_INSERTION_LAYOUT.insertion_command_context_indices
    if arguments.field == "initial-command-context":
        print(contexts[0])
    else:
        print("\n".join(str(context) for context in contexts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
