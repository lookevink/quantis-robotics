"""Simulator-independent artifact writer for deterministic demo recordings."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from jepa.contract import ObservationStage
from jepa_wm.action import ACTION_RECORDING_CONTRACT, DroidPose, action_between
from sim.demo_sequence import Phase


RECORDING_SCHEMA = "quantis.demo_recording.v3"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_recording_id(recording_id: str) -> None:
    if not _SAFE_NAME.fullmatch(recording_id):
        raise ValueError(
            "recording_id must contain only letters, numbers, dot, dash, or underscore"
        )


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
            raise ValueError(
                "initial has no task phase; every other moment requires one"
            )

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
    stage: ObservationStage
    arm_positions: Sequence[float]
    gripper_width_m: float
    plug_position: Sequence[float]
    plug_attached: bool
    end_effector_pose: DroidPose


@dataclass(frozen=True)
class RecordingStep:
    index: int
    timestamp_seconds: float
    phase: str
    stage: str
    frames: dict[str, str]
    arm_positions: list[float]
    gripper_width_m: float
    plug_position: list[float]
    plug_attached: bool
    end_effector_pose: list[float]
    action_from_previous: list[float] | None


class RecordingWriter:
    """Write synchronized camera frames, robot state, and a video manifest."""

    def __init__(
        self,
        root: Path,
        *,
        recording_id: str,
        fps: int,
        camera_resolutions: Mapping[str, tuple[int, int]],
    ) -> None:
        validate_recording_id(recording_id)
        if fps <= 0:
            raise ValueError("fps must be positive")
        cameras = tuple(camera_resolutions)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("cameras must be non-empty and unique")
        if any(not _SAFE_NAME.fullmatch(camera) for camera in cameras):
            raise ValueError("camera names must be safe path components")

        self.recording_id = recording_id
        self.fps = int(fps)
        self.camera_resolutions = dict(camera_resolutions)
        self.cameras = tuple(self.camera_resolutions)
        if any(
            width <= 0 or height <= 0
            for width, height in self.camera_resolutions.values()
        ):
            raise ValueError("camera resolutions must be positive")
        self.output_dir = root / recording_id
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for camera in self.cameras:
            (self.output_dir / camera).mkdir()
        self._steps: list[RecordingStep] = []
        self._stage_frame_counts = {stage: 0 for stage in ObservationStage}
        self._previous_end_effector_pose: DroidPose | None = None

    @property
    def frame_count(self) -> int:
        return len(self._steps)

    def stage_frame_count(self, stage: ObservationStage) -> int:
        return self._stage_frame_counts[stage]

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
            raise ValueError(
                f"camera frames must exist before adding a step: {missing}"
            )
        end_effector_pose = snapshot.end_effector_pose
        action = (
            action_between(self._previous_end_effector_pose, end_effector_pose)
            if self._previous_end_effector_pose is not None
            else None
        )
        self._steps.append(
            RecordingStep(
                index=index,
                timestamp_seconds=index / self.fps,
                phase=snapshot.phase.value,
                stage=snapshot.stage.value,
                frames={
                    camera: path.relative_to(self.output_dir).as_posix()
                    for camera, path in frames.items()
                },
                arm_positions=[float(value) for value in snapshot.arm_positions],
                gripper_width_m=float(snapshot.gripper_width_m),
                plug_position=[float(value) for value in snapshot.plug_position],
                plug_attached=bool(snapshot.plug_attached),
                end_effector_pose=list(end_effector_pose.values),
                action_from_previous=(
                    list(action.values) if action is not None else None
                ),
            )
        )
        self._previous_end_effector_pose = end_effector_pose
        self._stage_frame_counts[snapshot.stage] += 1

    def finish(self) -> Path:
        if not self._steps:
            raise ValueError("cannot finish an empty recording")

        with (self.output_dir / "steps.jsonl").open("w", encoding="utf-8") as output:
            for step in self._steps:
                output.write(json.dumps(asdict(step), separators=(",", ":")))
                output.write("\n")

        stage_frames = {
            stage.value: count
            for stage, count in self._stage_frame_counts.items()
            if count
        }

        manifest = {
            "schema": RECORDING_SCHEMA,
            "recording_id": self.recording_id,
            "fps": self.fps,
            "frames": len(self._steps),
            "stage_frames": stage_frames,
            "cameras": list(self.cameras),
            "resolutions": {
                camera: list(self.camera_resolutions[camera]) for camera in self.cameras
            },
            "videos": {camera: f"{camera}.mp4" for camera in self.cameras},
            "action": ACTION_RECORDING_CONTRACT.to_dict(),
        }
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return self.output_dir
