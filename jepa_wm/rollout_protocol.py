"""Dependency-light temporal contract for recorded JEPA-WM rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jepa_wm.trajectory import RecordedRollout


@dataclass(frozen=True)
class RolloutProtocol:
    """Temporal shape used by the released DROID planner."""

    context_frames: int = 1
    action_horizon: int = 3

    def __post_init__(self) -> None:
        if self.context_frames <= 0 or self.action_horizon <= 0:
            raise ValueError("context frames and action horizon must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "context_frames": self.context_frames,
            "action_horizon": self.action_horizon,
        }


DROID_ROLLOUT_PROTOCOL = RolloutProtocol()


@dataclass(frozen=True)
class RolloutWindow:
    """Validated selection and naming contract for recorded rollouts."""

    start_index: int
    count: int
    stride: int

    def __post_init__(self) -> None:
        if self.start_index < 0:
            raise ValueError("start index must be non-negative")
        if self.count <= 0 or self.stride <= 0:
            raise ValueError("count and stride must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "start_index": self.start_index,
            "count": self.count,
            "stride": self.stride,
        }

    @property
    def context_indices(self) -> tuple[int, ...]:
        return tuple(
            self.start_index + offset * self.stride
            for offset in range(self.count)
        )

    def report_name(self, camera: str) -> str:
        return (
            f"{camera}_rollout_eval_"
            f"{self.start_index:06d}_{self.count:03d}.json"
        )

    def select(
        self, rollouts: tuple[RecordedRollout, ...]
    ) -> tuple[RecordedRollout, ...]:
        by_context = {rollout.context[0].index: rollout for rollout in rollouts}
        if len(by_context) != len(rollouts):
            raise ValueError("recording contains duplicate rollout contexts")
        try:
            return tuple(by_context[index] for index in self.context_indices)
        except KeyError as error:
            raise ValueError(
                f"recording is missing required rollout context {error.args[0]}"
            ) from error
