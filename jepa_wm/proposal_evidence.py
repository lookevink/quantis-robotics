"""Typed per-rollout provenance emitted by proposal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from jepa_wm.action import DroidAction

if TYPE_CHECKING:
    from jepa_wm.trajectory import RecordedRollout


@dataclass(frozen=True)
class ProposalGoalEvidence:
    context_index: int
    target_index: int
    goal_delta: DroidAction

    def __post_init__(self) -> None:
        if self.context_index < 0 or self.target_index <= self.context_index:
            raise ValueError("proposal goal evidence indices are invalid")

    @classmethod
    def from_rollout(cls, rollout: RecordedRollout) -> ProposalGoalEvidence:
        return cls(
            rollout.context[0].index,
            rollout.target.index,
            rollout.goal_action,
        )

    @classmethod
    def from_dict(cls, payload: Any) -> ProposalGoalEvidence:
        if not isinstance(payload, dict):
            raise ValueError("proposal goal evidence is incomplete")
        try:
            context_index = payload["context_index"]
            target_index = payload["target_index"]
            if type(context_index) is not int or type(target_index) is not int:
                raise ValueError("proposal goal evidence indices must be integers")
            return cls(
                context_index,
                target_index,
                DroidAction(tuple(payload["goal_delta"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("proposal goal evidence is incomplete") from error

    def validates(self, rollout: RecordedRollout) -> bool:
        return self == self.from_rollout(rollout)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_index": self.context_index,
            "target_index": self.target_index,
            "goal_delta": list(self.goal_delta.values),
        }
