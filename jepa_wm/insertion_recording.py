"""Validated kinematic reach-and-insert demonstration evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from jepa_wm.domain_recording import DomainRecording
from jepa_wm.insertion_contract import (
    INSERTION_TASK_ID,
    KINEMATIC_INSERTION_MODE,
    REARWARD_GRASP_OFFSET_METERS,
)
from jepa_wm.action import DROID_FPS
from jepa_wm.insertion_task import (
    InsertionGeometryStep,
    InsertionTarget,
    InsertionTaskLimits,
    evaluate_insertion_geometry,
)
from jepa.contract import ObservationStage
from sim.exploration import DatasetSplit
from sim.recording import RECORDING_SCHEMA


def _vector(payload: Any, length: int, name: str) -> np.ndarray:
    values = np.asarray(payload, dtype=np.float64)
    if values.shape != (length,) or not np.all(np.isfinite(values)):
        raise ValueError(f"insertion demonstration {name} is invalid")
    return values


def _orientation_error(left: np.ndarray, right: np.ndarray) -> float:
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    cosine = min(1.0, abs(float(np.dot(left, right))))
    return 2.0 * acos(cosine)


@dataclass(frozen=True)
class InsertionDemonstrationEvidence:
    recording: str
    acquisition_index: int
    seated_index: int
    seated_observations: int
    grasp_clearance_meters: float
    seating_depth_error_meters: float
    seating_lateral_error_meters: float
    seating_orientation_error_rad: float

    @property
    def kinematic_only(self) -> bool:
        return True

    @classmethod
    def from_recording(
        cls,
        path: Path,
        *,
        expected_split: str,
        limits: InsertionTaskLimits = InsertionTaskLimits(),
    ) -> InsertionDemonstrationEvidence:
        recording = DomainRecording.from_path(
            path,
            expected_split=DatasetSplit(expected_split),
        )
        manifest = recording.manifest
        metadata = manifest.get("metadata")
        target_payload = (
            metadata.get("insertion_target") if isinstance(metadata, dict) else None
        )
        if (
            manifest.get("schema") != RECORDING_SCHEMA
            or not isinstance(metadata, dict)
            or metadata.get("task") != INSERTION_TASK_ID
            or manifest.get("fps") != DROID_FPS
            or not isinstance(target_payload, dict)
            or target_payload.get("evidence_mode") != KINEMATIC_INSERTION_MODE
        ):
            raise ValueError("recording is not a kinematic insertion demonstration")
        target = InsertionTarget(
            tuple(_vector(target_payload.get("socket_position"), 3, "socket")),
            tuple(_vector(target_payload.get("insertion_axis"), 3, "axis")),
        )
        socket_orientation = _vector(
            target_payload.get("socket_orientation_wxyz"),
            4,
            "socket orientation",
        )
        if float(np.linalg.norm(socket_orientation)) <= 0.0:
            raise ValueError("insertion demonstration socket orientation is invalid")
        claimed_grasp_offset = target_payload.get("grasp_offset_meters")
        if (
            isinstance(claimed_grasp_offset, bool)
            or not isinstance(claimed_grasp_offset, (int, float))
            or not isfinite(float(claimed_grasp_offset))
            or abs(
                float(claimed_grasp_offset) - REARWARD_GRASP_OFFSET_METERS
            )
            > 1e-9
        ):
            raise ValueError("insertion demonstration grasp offset is invalid")

        steps = recording.load_steps()

        geometry_steps = []
        for step in steps:
            tip = _vector(step.get("plug_position"), 3, "plug position")
            gripper_frame = _vector(
                step.get("gripper_frame_world_position"),
                3,
                "gripper frame",
            )
            orientation = _vector(
                step.get("plug_orientation_wxyz"),
                4,
                "plug orientation",
            )
            if float(np.linalg.norm(orientation)) <= 0.0:
                raise ValueError("insertion demonstration plug orientation is invalid")
            geometry_steps.append(
                InsertionGeometryStep(
                    plug_tip_position=tuple(tip),
                    gripper_frame_position=tuple(gripper_frame),
                    plug_attached=step.get("plug_attached") is True,
                    orientation_error_rad=_orientation_error(
                        orientation,
                        socket_orientation,
                    ),
                )
            )
        decision = evaluate_insertion_geometry(
            tuple(geometry_steps),
            target,
            limits,
            eligible_seating_indices=frozenset(
                index
                for index, step in enumerate(steps)
                if step.get("stage") == ObservationStage.PLUG_SEATED.value
            ),
        )
        if not decision.passed:
            reasons = ", ".join(failure.value for failure in decision.failures)
            raise ValueError(f"insertion demonstration is invalid: {reasons}")
        if (
            abs(decision.grasp_clearance_meters - REARWARD_GRASP_OFFSET_METERS)
            > 0.003
        ):
            raise ValueError("insertion demonstration grasp offset is inconsistent")
        if len(decision.seated_indices) < 4:
            raise ValueError("insertion demonstration does not retain a seated plug")
        return cls(
            recording.name,
            decision.acquisition_index,
            decision.seated_index,
            len(decision.seated_indices),
            decision.grasp_clearance_meters,
            decision.seating_depth_error_meters,
            decision.seating_lateral_error_meters,
            decision.seating_orientation_error_rad,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "recording": self.recording,
            "kinematic_only": self.kinematic_only,
            "acquisition_index": self.acquisition_index,
            "seated_index": self.seated_index,
            "seated_observations": self.seated_observations,
            "grasp_clearance_meters": self.grasp_clearance_meters,
            "seating_depth_error_meters": self.seating_depth_error_meters,
            "seating_lateral_error_meters": self.seating_lateral_error_meters,
            "seating_orientation_error_rad": self.seating_orientation_error_rad,
        }
