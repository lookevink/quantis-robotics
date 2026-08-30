"""Canonical epoch sizing for the authenticated insertion corpus."""

from __future__ import annotations

import argparse
from typing import Sequence

from jepa_wm.insertion_corpus import TRAINING_RECORDINGS
from jepa_wm.task_windows import INSERTION_PROPOSAL_WINDOW


INSERTION_ROLLOUTS_PER_RECORDING = INSERTION_PROPOSAL_WINDOW.count
INSERTION_TRAINING_BATCH_SIZE = 1
_INSERTION_EPOCH_SAMPLES = TRAINING_RECORDINGS * INSERTION_ROLLOUTS_PER_RECORDING
INSERTION_EPOCH_STEPS = (
    _INSERTION_EPOCH_SAMPLES + INSERTION_TRAINING_BATCH_SIZE - 1
) // INSERTION_TRAINING_BATCH_SIZE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print the canonical insertion training epoch size."
    )
    parser.add_argument(
        "field",
        nargs="?",
        choices=("epoch-steps", "batch-size"),
        default="epoch-steps",
    )
    arguments = parser.parse_args(argv)
    print(
        INSERTION_EPOCH_STEPS
        if arguments.field == "epoch-steps"
        else INSERTION_TRAINING_BATCH_SIZE
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
