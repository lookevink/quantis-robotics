"""Dependency-light insertion planner profile identity for orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


@dataclass(frozen=True)
class InsertionPlannerProfileDescriptor:
    report_suffix: str
    readiness_scope: str


class InsertionPlannerProfileName(str, Enum):
    SAMPLED_READINESS = "sampled_readiness"
    DENSE_EXECUTION = "dense_execution"

    @property
    def descriptor(self) -> InsertionPlannerProfileDescriptor:
        return INSERTION_PLANNER_PROFILE_DESCRIPTORS[self]


INSERTION_PLANNER_PROFILE_DESCRIPTORS = {
    InsertionPlannerProfileName.SAMPLED_READINESS: InsertionPlannerProfileDescriptor(
        report_suffix="insertion_planner_readiness",
        readiness_scope="offline frozen insertion planner; no live insertion",
    ),
    InsertionPlannerProfileName.DENSE_EXECUTION: InsertionPlannerProfileDescriptor(
        report_suffix="insertion_dense_planner_readiness",
        readiness_scope="offline frozen dense insertion planner; no live insertion",
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("field", choices=("name", "report-suffix"))
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in InsertionPlannerProfileName),
        default=InsertionPlannerProfileName.SAMPLED_READINESS.value,
    )
    arguments = parser.parse_args(argv)
    profile = InsertionPlannerProfileName(arguments.profile)
    print(
        profile.value
        if arguments.field == "name"
        else profile.descriptor.report_suffix
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
