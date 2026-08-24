"""Exact rollout selection shared by proposal and world-model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.training_artifact import (
    TrainingRecordingSelection,
    rollout_training_selection_fingerprint,
)
from jepa_wm.trajectory import RecordedRollout, RolloutWindow, load_rollouts


@dataclass(frozen=True)
class RolloutTrainingSelection:
    rollouts: tuple[RecordedRollout, ...]
    recordings: tuple[TrainingRecordingSelection, ...]
    window: RolloutWindow | None
    bounds: ActionSelectionBounds

    def __post_init__(self) -> None:
        if not self.rollouts or not self.recordings:
            raise ValueError("rollout training selection must not be empty")

    @classmethod
    def load(
        cls,
        recordings: Sequence[Path],
        *,
        camera: str,
        bounds: ActionSelectionBounds,
        window: RolloutWindow | None,
    ) -> RolloutTrainingSelection:
        selected_rollouts = []
        selections = []
        for recording in recordings:
            available = load_rollouts(recording, camera=camera, bounds=bounds)
            selected = window.select(available) if window is not None else available
            selected_rollouts.extend(selected)
            selections.append(
                TrainingRecordingSelection(
                    recording.name,
                    tuple(rollout.context[0].index for rollout in selected),
                )
            )
        return cls(tuple(selected_rollouts), tuple(selections), window, bounds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window.to_dict() if self.window is not None else None,
            "selection_bounds": self.bounds.to_dict(),
            "recording_selections": [
                selection.to_dict() for selection in self.recordings
            ],
            "rollouts": len(self.rollouts),
        }

    @property
    def fingerprint(self) -> str:
        return rollout_training_selection_fingerprint(self.to_dict())
