"""Canonical insertion adapter profiles shared by training and planning."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from jepa_wm.candidate_negatives import CandidateMiningConfig


@dataclass(frozen=True)
class InsertionAdapterProfileDescriptor:
    artifact_stem: str
    minimum_goal_cosine: float | None

    def candidate_mining_config(self) -> CandidateMiningConfig:
        from jepa_wm.candidate_negatives import CandidateMiningConfig

        return CandidateMiningConfig(minimum_goal_cosine=self.minimum_goal_cosine)


class InsertionAdapterProfile(str, Enum):
    GENERIC = "generic"
    GOAL_ALIGNED = "goal_aligned"

    @property
    def descriptor(self) -> InsertionAdapterProfileDescriptor:
        return INSERTION_ADAPTER_PROFILES[self]


INSERTION_ADAPTER_PROFILES = {
    InsertionAdapterProfile.GENERIC: InsertionAdapterProfileDescriptor(
        artifact_stem="insertion_adapter_s",
        minimum_goal_cosine=None,
    ),
    InsertionAdapterProfile.GOAL_ALIGNED: InsertionAdapterProfileDescriptor(
        artifact_stem="insertion_adapter_goal_aligned_s",
        minimum_goal_cosine=0.95,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "profile",
        choices=tuple(profile.value for profile in InsertionAdapterProfile),
    )
    parser.add_argument("field", choices=("artifact-stem", "minimum-goal-cosine"))
    args = parser.parse_args(argv)
    descriptor = InsertionAdapterProfile(args.profile).descriptor
    value = (
        descriptor.artifact_stem
        if args.field == "artifact-stem"
        else descriptor.minimum_goal_cosine
    )
    print("" if value is None else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
