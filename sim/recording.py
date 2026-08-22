"""Simulator-independent artifact writer for deterministic demo recordings."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from sim.demo_sequence import Phase


RECORDING_SCHEMA = "quantis.demo_recording.v1"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RecordingMoment(str, Enum):
    INITIAL = "initial"
    MOTION = "motion"
    SETTLE = "settle"
    CLOSE = "close"
    ATTACHED = "attached"
    COMPLETE = "complete"


@dataclass(frozen=True)
class RecordingLabel:
    moment: RecordingMoment
    phase: Phase | None = None

    def __post_init__(self) -> None:
        has_phase = self.phase is not None
        if (self.moment == RecordingMoment.INITIAL) == has_phase:
            raise ValueError("initial has no task phase; every other moment requires one")

    @property
    def value(self) -> str:
        if self.moment == RecordingMoment.INITIAL:
            return self.moment.value
        if self.phase is None:
            raise AssertionError("validated recording label has no phase")
        if self.moment == RecordingMoment.MOTION:
            return self.phase.value
        return f"{self.phase.value}_{self.moment.value}"


@dataclass(frozen=True)
class RecordingSnapshot:
    phase: RecordingLabel
    arm_positions: Sequence[float]
    gripper_width_m: float
    plug_position: Sequence[float]
    plug_attached: bool


@dataclass(frozen=True)
class RecordingStep:
    index: int
    timestamp_seconds: float
    phase: str
    frames: dict[str, str]
    arm_positions: list[float]
    gripper_width_m: float
    plug_position: list[float]
    plug_attached: bool


class RecordingWriter:
    """Write synchronized camera frames, robot state, and a video manifest."""

    def __init__(
        self,
        root: Path,
        *,
        recording_id: str,
        fps: int,
        cameras: Sequence[str],
    ) -> None:
        if not _SAFE_NAME.fullmatch(recording_id):
            raise ValueError(
                "recording_id must contain only letters, numbers, dot, dash, or underscore"
            )
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("cameras must be non-empty and unique")
        if any(not _SAFE_NAME.fullmatch(camera) for camera in cameras):
            raise ValueError("camera names must be safe path components")

        self.recording_id = recording_id
        self.fps = int(fps)
        self.cameras = tuple(cameras)
        self.output_dir = root / recording_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for camera in self.cameras:
            (self.output_dir / camera).mkdir()
        self._steps: list[RecordingStep] = []

    def frame_paths(self) -> dict[str, Path]:
        index = len(self._steps)
        return {
            camera: self.output_dir / camera / f"frame_{index:06d}.png"
            for camera in self.cameras
        }

    def add_step(self, snapshot: RecordingSnapshot) -> None:
        index = len(self._steps)
        frames = self.frame_paths()
        missing = [str(path) for path in frames.values() if not path.is_file()]
        if missing:
            raise ValueError(f"camera frames must exist before adding a step: {missing}")
        self._steps.append(
            RecordingStep(
                index=index,
                timestamp_seconds=index / self.fps,
                phase=snapshot.phase.value,
                frames={
                    camera: path.relative_to(self.output_dir).as_posix()
                    for camera, path in frames.items()
                },
                arm_positions=[float(value) for value in snapshot.arm_positions],
                gripper_width_m=float(snapshot.gripper_width_m),
                plug_position=[float(value) for value in snapshot.plug_position],
                plug_attached=bool(snapshot.plug_attached),
            )
        )

    def finish(self) -> Path:
        if not self._steps:
            raise ValueError("cannot finish an empty recording")

        with (self.output_dir / "steps.jsonl").open("w", encoding="utf-8") as output:
            for step in self._steps:
                output.write(json.dumps(asdict(step), separators=(",", ":")))
                output.write("\n")

        manifest = {
            "schema": RECORDING_SCHEMA,
            "recording_id": self.recording_id,
            "fps": self.fps,
            "frames": len(self._steps),
            "cameras": list(self.cameras),
            "videos": {camera: f"{camera}.mp4" for camera in self.cameras},
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return self.output_dir
