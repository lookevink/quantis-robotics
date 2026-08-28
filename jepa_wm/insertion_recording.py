"""Validated kinematic reach-and-insert demonstration evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite
from pathlib import Path
from typing import Any

import numpy as np

from jepa_wm.domain_recording import DomainRecording
from jepa_wm.insertion_contract import (
    CONTACT_AWARE_INSERTION_MODE,
    CONTACT_INSERTION_RECORDING,
    INSERTION_TASK_ID,
    KINEMATIC_INSERTION_MODE,
    REARWARD_GRASP_OFFSET_METERS,
)
from jepa_wm.action import DROID_FPS
from jepa_wm.insertion_task import (
    InsertionGeometryStep,
    InsertionDecision,
    InsertionTarget,
    InsertionTaskLimits,
    InsertionTaskStep,
    evaluate_insertion,
    evaluate_insertion_geometry,
)
from jepa_wm.task_windows import CONTACT_GRASP_PROPOSAL_WINDOW
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL
from jepa.contract import ObservationStage
from sim.exploration import DatasetSplit
from sim.recording import (
    RECORDING_SCHEMA,
    RECORDING_SCHEMA_V5,
    RECORDING_SCHEMA_V6,
    RECORDING_SCHEMA_V7,
    RECORDING_SCHEMA_V8,
    RecordingSafetyTelemetry,
)


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
class _ParsedInsertionRecording:
    recording: DomainRecording
    target_payload: dict[str, Any]
    target: InsertionTarget
    steps: tuple[dict[str, Any], ...]
    geometry_steps: tuple[InsertionGeometryStep, ...]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_split: str,
        expected_mode: str,
        expected_seed: int | None = None,
    ) -> _ParsedInsertionRecording:
        recording = DomainRecording.from_path(
            path,
            expected_split=DatasetSplit(expected_split),
        )
        if expected_seed is not None and recording.seed != expected_seed:
            raise ValueError(
                f"recording seed {recording.seed} does not match {expected_seed}"
            )
        manifest = dict(recording.manifest)
        metadata = manifest.get("metadata")
        target_payload = (
            metadata.get("insertion_target") if isinstance(metadata, dict) else None
        )
        accepted_schemas = (
            {
                RECORDING_SCHEMA_V5,
                RECORDING_SCHEMA_V6,
                RECORDING_SCHEMA_V7,
                RECORDING_SCHEMA_V8,
                RECORDING_SCHEMA,
            }
            if expected_mode == KINEMATIC_INSERTION_MODE
            else {RECORDING_SCHEMA}
        )
        if (
            manifest.get("schema") not in accepted_schemas
            or not isinstance(metadata, dict)
            or metadata.get("task") != INSERTION_TASK_ID
            or manifest.get("fps") != DROID_FPS
            or not isinstance(target_payload, dict)
            or target_payload.get("evidence_mode") != expected_mode
        ):
            raise ValueError("recording is not the expected insertion demonstration")
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
        return cls(
            recording,
            target_payload,
            target,
            steps,
            tuple(geometry_steps),
        )

    @property
    def seating_indices(self) -> frozenset[int]:
        return frozenset(
            index
            for index, step in enumerate(self.steps)
            if step.get("stage") == ObservationStage.PLUG_SEATED.value
        )


def _validate_contact_recording_contract(
    parsed: _ParsedInsertionRecording,
) -> None:
    CONTACT_INSERTION_RECORDING.validate_instrumentation(parsed.target_payload)
    if len(parsed.steps) != CONTACT_INSERTION_RECORDING.frame_count:
        raise ValueError("contact-aware insertion frame contract is invalid")
    if (
        tuple(step.get("phase") for step in parsed.steps)
        != CONTACT_INSERTION_RECORDING.phase_roster
    ):
        raise ValueError("contact-aware insertion phase contract is invalid")
    if (
        tuple(step.get("stage") for step in parsed.steps)
        != CONTACT_INSERTION_RECORDING.stage_roster
    ):
        raise ValueError("contact-aware insertion stage contract is invalid")
    if (
        tuple(step.get("plug_attached") for step in parsed.steps)
        != CONTACT_INSERTION_RECORDING.attachment_roster
    ):
        raise ValueError("contact-aware insertion attachment contract is invalid")


@dataclass(frozen=True)
class ContactGraspEvidence:
    recording: str
    maximum_contact_force_newtons: float
    maximum_arm_tracking_error_rad: float
    maximum_gripper_tracking_error_m: float

    @classmethod
    def from_recording(
        cls,
        path: Path,
        *,
        expected_split: str,
        limits: InsertionTaskLimits = InsertionTaskLimits(),
        expected_seed: int | None = None,
    ) -> ContactGraspEvidence:
        parsed = _ParsedInsertionRecording.load(
            path,
            expected_split=expected_split,
            expected_mode=CONTACT_AWARE_INSERTION_MODE,
            expected_seed=expected_seed,
        )
        _validate_contact_recording_contract(parsed)
        final_target_index = (
            CONTACT_GRASP_PROPOSAL_WINDOW.context_indices[-1]
            + DROID_ROLLOUT_PROTOCOL.action_horizon
        )
        safety = tuple(
            RecordingSafetyTelemetry.from_dict(step)
            for step in parsed.steps[: final_target_index + 1]
        )
        if any(
            item.collision_detected
            or item.contact_force_newtons > limits.maximum_contact_force_newtons
            or item.arm_tracking_error_rad
            > limits.maximum_arm_tracking_error_rad
            or item.gripper_tracking_error_m
            > limits.maximum_gripper_tracking_error_m
            for item in safety
        ):
            raise ValueError("contact-aware grasp safety evidence is invalid")
        return cls(
            parsed.recording.name,
            max(item.contact_force_newtons for item in safety),
            max(item.arm_tracking_error_rad for item in safety),
            max(item.gripper_tracking_error_m for item in safety),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "recording": self.recording,
            "contact_aware": True,
            "maximum_contact_force_newtons": self.maximum_contact_force_newtons,
            "maximum_arm_tracking_error_rad": self.maximum_arm_tracking_error_rad,
            "maximum_gripper_tracking_error_m": self.maximum_gripper_tracking_error_m,
        }


@dataclass(frozen=True)
class InsertionDemonstrationEvidence:
    recording: str
    decision: InsertionDecision

    @property
    def acquisition_index(self) -> int | None:
        return self.decision.acquisition_index

    @property
    def seated_index(self) -> int | None:
        return self.decision.seated_index

    @property
    def seated_observations(self) -> int:
        return len(self.decision.seated_indices)

    @property
    def grasp_clearance_meters(self) -> float:
        return self.decision.grasp_clearance_meters

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
        parsed = _ParsedInsertionRecording.load(
            path,
            expected_split=expected_split,
            expected_mode=KINEMATIC_INSERTION_MODE,
        )
        decision = evaluate_insertion_geometry(
            parsed.geometry_steps,
            parsed.target,
            limits,
            eligible_seating_indices=parsed.seating_indices,
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
        return cls(parsed.recording.name, decision)

    def to_dict(self) -> dict[str, object]:
        return {
            "recording": self.recording,
            "kinematic_only": self.kinematic_only,
            **self.decision.evidence_dict(),
        }


@dataclass(frozen=True)
class ContactInsertionEvidence:
    recording: str
    decision: InsertionDecision
    maximum_contact_force_newtons: float
    maximum_arm_tracking_error_rad: float
    maximum_gripper_tracking_error_m: float

    @property
    def seated_observations(self) -> int:
        return len(self.decision.seated_indices)

    @classmethod
    def from_recording(
        cls,
        path: Path,
        *,
        expected_split: str,
        limits: InsertionTaskLimits = InsertionTaskLimits(),
        expected_seed: int | None = None,
    ) -> ContactInsertionEvidence:
        parsed = _ParsedInsertionRecording.load(
            path,
            expected_split=expected_split,
            expected_mode=CONTACT_AWARE_INSERTION_MODE,
            expected_seed=expected_seed,
        )
        _validate_contact_recording_contract(parsed)
        task_steps = []
        arm_errors = []
        gripper_errors = []
        contact_forces = []
        for geometry, raw in zip(parsed.geometry_steps, parsed.steps):
            try:
                safety = RecordingSafetyTelemetry.from_dict(raw)
            except ValueError as error:
                raise ValueError("contact-aware insertion telemetry is invalid") from error
            arm_errors.append(safety.arm_tracking_error_rad)
            gripper_errors.append(safety.gripper_tracking_error_m)
            contact_forces.append(safety.contact_force_newtons)
            task_steps.append(
                InsertionTaskStep(
                    plug_tip_position=geometry.plug_tip_position,
                    gripper_frame_position=geometry.gripper_frame_position,
                    plug_attached=geometry.plug_attached,
                    orientation_error_rad=geometry.orientation_error_rad,
                    tracking_passed=(
                        safety.arm_tracking_error_rad
                        <= limits.maximum_arm_tracking_error_rad
                        and safety.gripper_tracking_error_m
                        <= limits.maximum_gripper_tracking_error_m
                    ),
                    collision_detected=safety.collision_detected,
                    contact_force_newtons=safety.contact_force_newtons,
                )
            )
        decision = evaluate_insertion(
            tuple(task_steps),
            parsed.target,
            limits,
            eligible_seating_indices=parsed.seating_indices,
        )
        if not decision.passed:
            reasons = ", ".join(failure.value for failure in decision.failures)
            raise ValueError(f"contact-aware insertion is invalid: {reasons}")
        if (
            abs(decision.grasp_clearance_meters - REARWARD_GRASP_OFFSET_METERS)
            > 0.0005
        ):
            raise ValueError("contact-aware insertion grasp offset is inconsistent")
        if len(decision.seated_indices) < 4:
            raise ValueError("contact-aware insertion does not retain a seated plug")
        return cls(
            parsed.recording.name,
            decision,
            max(contact_forces),
            max(arm_errors),
            max(gripper_errors),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "recording": self.recording,
            "contact_aware": True,
            "kinematic_only": False,
            **self.decision.evidence_dict(),
            "maximum_contact_force_newtons": self.maximum_contact_force_newtons,
            "maximum_arm_tracking_error_rad": self.maximum_arm_tracking_error_rad,
            "maximum_gripper_tracking_error_m": self.maximum_gripper_tracking_error_m,
        }
