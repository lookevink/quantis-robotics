"""Simulator-independent artifact writer for deterministic demo recordings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

from jepa.contract import ObservationStage
from jepa_wm.action import ACTION_RECORDING_CONTRACT, DroidPose, action_between
from jepa_wm.identifiers import validate_safe_identifier
from sim.demo_sequence import Phase


RECORDING_SCHEMA_V5 = "quantis.demo_recording.v5"
RECORDING_SCHEMA_V6 = "quantis.demo_recording.v6"
RECORDING_SCHEMA_V7 = "quantis.demo_recording.v7"
RECORDING_SCHEMA_V8 = "quantis.demo_recording.v8"
RECORDING_SCHEMA = "quantis.demo_recording.v9"


def validate_recording_id(recording_id: str) -> None:
    try:
        validate_safe_identifier(recording_id)
    except ValueError as error:
        raise ValueError(
            "recording_id must contain only letters, numbers, dot, dash, or underscore"
        ) from error
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
class RecordingSafetyTelemetry:
    collision_detected: bool = False
    contact_force_newtons: float = 0.0
    arm_tracking_error_rad: float = 0.0
    gripper_tracking_error_m: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.collision_detected, bool)
            or isinstance(self.contact_force_newtons, bool)
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
            or isinstance(self.arm_tracking_error_rad, bool)
            or not isfinite(self.arm_tracking_error_rad)
            or self.arm_tracking_error_rad < 0.0
            or isinstance(self.gripper_tracking_error_m, bool)
            or not isfinite(self.gripper_tracking_error_m)
            or self.gripper_tracking_error_m < 0.0
        ):
            raise ValueError("recording safety telemetry is invalid")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RecordingSafetyTelemetry:
        try:
            raw_values = (
                payload["contact_force_newtons"],
                payload["arm_tracking_error_rad"],
                payload["gripper_tracking_error_m"],
            )
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in raw_values
            ):
                raise ValueError("numeric telemetry must be a JSON number")
            return cls(
                collision_detected=payload["collision_detected"],
                contact_force_newtons=float(raw_values[0]),
                arm_tracking_error_rad=float(raw_values[1]),
                gripper_tracking_error_m=float(raw_values[2]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("recording safety telemetry is invalid") from error


@dataclass(frozen=True)
class RecordingSnapshot:
    phase: RecordingLabel
    stage: ObservationStage
    arm_positions: Sequence[float]
    gripper_width_m: float
    plug_position: Sequence[float]
    plug_orientation_wxyz: Sequence[float]
    plug_attached: bool
    end_effector_pose: DroidPose
    end_effector_world_position: Sequence[float]
    gripper_frame_world_position: Sequence[float]
    simulation_time_seconds: float | None = None
    safety: RecordingSafetyTelemetry = RecordingSafetyTelemetry()


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
    plug_orientation_wxyz: list[float]
    plug_attached: bool
    end_effector_pose: list[float]
    end_effector_world_position: list[float]
    gripper_frame_world_position: list[float]
    action_from_previous: list[float] | None
    simulation_time_seconds: float | None
    collision_detected: bool
    contact_force_newtons: float
    arm_tracking_error_rad: float
    gripper_tracking_error_m: float


class RecordingWriter:
    """Write synchronized camera frames, robot state, and a video manifest."""

    def __init__(
        self,
        root: Path,
        *,
        recording_id: str,
        fps: int,
        camera_resolutions: Mapping[str, tuple[int, int]],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        validate_recording_id(recording_id)
        if fps <= 0:
            raise ValueError("fps must be positive")
        cameras = tuple(camera_resolutions)
        if not cameras or len(set(cameras)) != len(cameras):
            raise ValueError("cameras must be non-empty and unique")
        try:
            for camera in cameras:
                validate_safe_identifier(camera)
        except ValueError as error:
            raise ValueError("camera names must be safe path components") from error

        self.recording_id = recording_id
        self.fps = int(fps)
        self.camera_resolutions = dict(camera_resolutions)
        self.metadata = dict(metadata or {})
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
                plug_orientation_wxyz=[
                    float(value) for value in snapshot.plug_orientation_wxyz
                ],
                plug_attached=bool(snapshot.plug_attached),
                end_effector_pose=list(end_effector_pose.values),
                end_effector_world_position=[
                    float(value) for value in snapshot.end_effector_world_position
                ],
                gripper_frame_world_position=[
                    float(value) for value in snapshot.gripper_frame_world_position
                ],
                action_from_previous=(
                    list(action.values) if action is not None else None
                ),
                simulation_time_seconds=snapshot.simulation_time_seconds,
                collision_detected=snapshot.safety.collision_detected,
                contact_force_newtons=snapshot.safety.contact_force_newtons,
                arm_tracking_error_rad=snapshot.safety.arm_tracking_error_rad,
                gripper_tracking_error_m=snapshot.safety.gripper_tracking_error_m,
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
        if self.metadata:
            manifest["metadata"] = self.metadata
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        return self.output_dir
