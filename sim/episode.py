"""Small, simulator-independent episode manifest writer."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "quantis.episode.v1"


@dataclass(frozen=True)
class Step:
    index: int
    timestamp_seconds: float
    frame: str
    action: list[float]
    state: dict[str, Any]


class EpisodeWriter:
    """Write synchronized image references, actions, and state as JSONL."""

    def __init__(
        self,
        episode_dir: Path,
        *,
        task: str,
        robot: str,
        action_labels: Sequence[str],
        fps: float,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not action_labels:
            raise ValueError("action_labels cannot be empty")

        self.episode_dir = episode_dir
        self.task = task
        self.robot = robot
        self.action_labels = list(action_labels)
        self.fps = float(fps)
        self._steps: list[Step] = []
        self.episode_dir.mkdir(parents=True, exist_ok=False)

    def add_step(
        self,
        *,
        frame: Path,
        action: Sequence[float],
        state: dict[str, Any],
    ) -> None:
        if len(action) != len(self.action_labels):
            raise ValueError(
                f"expected {len(self.action_labels)} action values, got {len(action)}"
            )
        try:
            relative_frame = frame.relative_to(self.episode_dir)
        except ValueError as error:
            raise ValueError("frame must be inside the episode directory") from error

        index = len(self._steps)
        self._steps.append(
            Step(
                index=index,
                timestamp_seconds=index / self.fps,
                frame=relative_frame.as_posix(),
                action=[float(value) for value in action],
                state=state,
            )
        )

    def finish(self, *, success: bool, extra: dict[str, Any] | None = None) -> None:
        if not self._steps:
            raise ValueError("cannot finish an empty episode")

        steps_path = self.episode_dir / "steps.jsonl"
        with steps_path.open("w", encoding="utf-8") as output:
            for step in self._steps:
                output.write(json.dumps(asdict(step), separators=(",", ":")))
                output.write("\n")

        manifest = {
            "schema": SCHEMA_VERSION,
            "task": self.task,
            "robot": self.robot,
            "fps": self.fps,
            "frames": len(self._steps),
            "action_labels": self.action_labels,
            "success": bool(success),
            "extra": extra or {},
        }
        (self.episode_dir / "episode.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
