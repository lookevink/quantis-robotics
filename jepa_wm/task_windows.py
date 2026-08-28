"""Canonical rollout windows for task-conditioned proposal checkpoints."""

from __future__ import annotations

import argparse
from typing import Sequence

from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
)
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL, RolloutWindow


GRASP_PROPOSAL_WINDOW = RolloutWindow(69, 30, 1)
_CONTACT_GRASP_START = (
    CONTACT_INSERTION_RECORDING.start_index(ContactInsertionSegment.GRASP_ATTACH)
    - DROID_ROLLOUT_PROTOCOL.action_horizon
)
CONTACT_GRASP_PROPOSAL_WINDOW = RolloutWindow(_CONTACT_GRASP_START, 8, 1)
_INSERTION_START = CONTACT_INSERTION_RECORDING.start_index(
    ContactInsertionSegment.GRASP_ATTACH
)
INSERTION_PROPOSAL_WINDOW = RolloutWindow(
    _INSERTION_START,
    CONTACT_INSERTION_RECORDING.frame_count
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
            "insertion": INSERTION_PROPOSAL_WINDOW,
        }[task]
    except KeyError as error:
        raise ValueError(f"unknown proposal task: {task}") from error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("grasp", "contact-grasp", "insertion"))
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
