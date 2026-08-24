"""Shared all-observed-whole-seeds readiness decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MINIMUM_WHOLE_SEEDS = 2


@dataclass(frozen=True)
class WholeSeedReadiness:
    whole_seed_count: int
    pass_count: int

    def __post_init__(self) -> None:
        if (
            self.whole_seed_count < 0
            or not 0 <= self.pass_count <= self.whole_seed_count
        ):
            raise ValueError("whole-seed readiness counts are invalid")

    @classmethod
    def from_passes(cls, passes: Iterable[bool]) -> WholeSeedReadiness:
        decisions = tuple(passes)
        if any(type(decision) is not bool for decision in decisions):
            raise ValueError("whole-seed readiness decisions must be booleans")
        return cls(len(decisions), sum(decisions))

    @property
    def passed(self) -> bool:
        return (
            self.whole_seed_count >= MINIMUM_WHOLE_SEEDS
            and self.pass_count == self.whole_seed_count
        )

    @property
    def minimum_whole_seeds(self) -> int:
        return MINIMUM_WHOLE_SEEDS

    @property
    def production_authority_granted(self) -> bool:
        return False
