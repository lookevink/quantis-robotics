"""Fast reachability contract for the integrated grasp-to-insertion run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    INSERTION_CONTROL_TARGET_POLICY,
)
from jepa_wm.insertion_layout import ContactInsertionSegment
from jepa_wm.insertion_rollout import GRASP_TO_INSERTION_ROLLOUT


@dataclass(frozen=True)
class IntegratedInsertionSchedule:
    """Align a bounded live rollout with its authenticated terminal hold."""

    action_count: int = GRASP_TO_INSERTION_ROLLOUT.maximum_steps
    terminal_observations: int = 4

    @property
    def final_reference_index(self) -> int:
        return CONTACT_INSERTION_RECORDING.frame_count - 1

    @property
    def final_context_index(self) -> int:
        return (
            self.final_reference_index
            - INSERTION_CONTROL_TARGET_POLICY.minimum_action_horizon
        )

    @property
    def initial_context_index(self) -> int:
        return self.final_context_index - self.action_count + 1

    @property
    def context_indices(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.initial_context_index,
                self.final_context_index + 1,
            )
        )

    @property
    def terminal_context_indices(self) -> tuple[int, ...]:
        return self.context_indices[-self.terminal_observations :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_count": self.action_count,
            "terminal_observations": self.terminal_observations,
            "initial_context_index": self.initial_context_index,
            "final_context_index": self.final_context_index,
            "final_reference_index": self.final_reference_index,
        }

    def __post_init__(self) -> None:
        seated_start = CONTACT_INSERTION_RECORDING.start_index(
            ContactInsertionSegment.SEATED_HOLD
        )
        grasp_attach = CONTACT_INSERTION_RECORDING.start_index(
            ContactInsertionSegment.GRASP_ATTACH
        )
        if (
            self.action_count < self.terminal_observations
            or self.terminal_observations < 1
            or len(self.context_indices) != self.action_count
            or self.initial_context_index < 0
            or self.initial_context_index != grasp_attach
            or self.context_indices
            != tuple(range(grasp_attach, self.final_context_index + 1))
            or self.terminal_context_indices[0]
            + INSERTION_CONTROL_TARGET_POLICY.minimum_action_horizon
            < seated_start
            or any(
                not (
                    INSERTION_CONTROL_TARGET_POLICY.minimum_action_horizon
                    <= self.final_reference_index - context
                    <= INSERTION_CONTROL_TARGET_POLICY.maximum_action_horizon
                )
                for context in self.terminal_context_indices
            )
        ):
            raise ValueError("integrated insertion schedule cannot reach terminal hold")


INTEGRATED_INSERTION_SCHEDULE = IntegratedInsertionSchedule()
