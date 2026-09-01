"""Canonical rollout windows for task-conditioned proposal checkpoints."""

from __future__ import annotations

import argparse
from typing import Sequence

from jepa_wm.insertion_layout import (
    CONTACT_INSERTION_LAYOUT,
    ContactInsertionSegment,
)
from jepa_wm.rollout_protocol import DROID_ROLLOUT_PROTOCOL, RolloutWindow


GRASP_PROPOSAL_WINDOW = RolloutWindow(69, 30, 1)
_CONTACT_GRASP_START = (
    CONTACT_INSERTION_LAYOUT.start_index(ContactInsertionSegment.GRASP_ATTACH)
    - DROID_ROLLOUT_PROTOCOL.action_horizon
)
LEGACY_CONTACT_GRASP_PROPOSAL_WINDOW = RolloutWindow(_CONTACT_GRASP_START, 8, 1)
LEGACY_CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW = RolloutWindow(
    0,
    CONTACT_INSERTION_LAYOUT.start_index(ContactInsertionSegment.GRASP_ATTACH)
    + 5,
    1,
)
# Frame 128 is at least 23.9 mm beyond attachment in every authenticated
# TRAIN and HELD_OUT recording.  Keep the target window beyond the unchanged
# 20 mm retained-displacement gate instead of ending at frame 120 (5.8 mm),
# which made faithful task completion geometrically impossible.
_CONTACT_GRASP_RETAINED_TARGET_INDEX = 128
_CONTACT_GRASP_RETAINED_CONTEXT_INDEX = (
    _CONTACT_GRASP_RETAINED_TARGET_INDEX - DROID_ROLLOUT_PROTOCOL.action_horizon
)
CONTACT_GRASP_PROPOSAL_WINDOW = RolloutWindow(
    _CONTACT_GRASP_START,
    _CONTACT_GRASP_RETAINED_CONTEXT_INDEX - _CONTACT_GRASP_START + 1,
    1,
)
CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW = RolloutWindow(
    0,
    _CONTACT_GRASP_RETAINED_CONTEXT_INDEX + 1,
    1,
)
_INSERTION_START = CONTACT_INSERTION_LAYOUT.start_index(
    ContactInsertionSegment.GRASP_ATTACH
)
INSERTION_PROPOSAL_WINDOW = RolloutWindow(
    _INSERTION_START,
    CONTACT_INSERTION_LAYOUT.frame_count
    - DROID_ROLLOUT_PROTOCOL.context_frames
    - DROID_ROLLOUT_PROTOCOL.action_horizon
    + 1
    - _INSERTION_START,
    1,
)


def proposal_window(task: str) -> RolloutWindow:
    try:
        return {
            "grasp": GRASP_PROPOSAL_WINDOW,
            "contact-grasp": CONTACT_GRASP_PROPOSAL_WINDOW,
            "contact-grasp-acquisition": (
                CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW
            ),
            "insertion": INSERTION_PROPOSAL_WINDOW,
        }[task]
    except KeyError as error:
        raise ValueError(f"unknown proposal task: {task}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=(
            "grasp",
            "contact-grasp",
            "contact-grasp-acquisition",
            "insertion",
        ),
    )
    parser.add_argument(
        "field",
        nargs="?",
        choices=("window", "start-index"),
        default="window",
    )
    arguments = parser.parse_args(argv)
    window = proposal_window(arguments.task)
    if arguments.field == "start-index":
        print(window.start_index)
    else:
        print(window.start_index, window.count, window.stride)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
