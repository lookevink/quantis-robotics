"""Pure contract checks for handing an authenticated reset to control."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite, sqrt
from typing import Any

from jepa_wm.identifiers import validate_safe_identifier
from sim.unknown_start_reset import (
    UNKNOWN_START_RESET_CONTRACT,
    UnknownStartResetEvidence,
)


UNKNOWN_START_HANDOFF_SCHEMA = "quantis.unknown_start_control_handoff.v1"
MAXIMUM_HANDOFF_JOINT_DRIFT_RAD = 1e-4
MAXIMUM_HANDOFF_GRIPPER_DRIFT_M = 1e-4
MAXIMUM_HANDOFF_POSITION_DRIFT_M = 1e-5
MAXIMUM_HANDOFF_ORIENTATION_DRIFT_RAD = 1e-5


def _valid_fingerprint(value: str) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class UnknownStartControlHandoff:
    session_id: str
    reset_recording_id: str
    reset_seed: int
    reset_result_fingerprint: str
    reset_evidence_fingerprint: str
    reset_contract_fingerprint: str
    reference_recording: str
    reference_seed: int
    context_fingerprint: str
    routing_target_fingerprint: str
    routing_step_fingerprint: str
    request_fingerprint: str
    state_fingerprint: str
    applied_actions: int = 0

    def __post_init__(self) -> None:
        for identifier in (
            self.session_id,
            self.reset_recording_id,
            self.reference_recording,
        ):
            if not isinstance(identifier, str):
                raise ValueError("unknown-start control handoff is invalid")
            validate_safe_identifier(identifier)
        if (
            isinstance(self.reset_seed, bool)
            or not isinstance(self.reset_seed, int)
            or self.reset_seed < 0
            or isinstance(self.reference_seed, bool)
            or not isinstance(self.reference_seed, int)
            or self.reference_seed < 0
            or isinstance(self.applied_actions, bool)
            or not isinstance(self.applied_actions, int)
            or self.applied_actions != 0
            or any(
                not _valid_fingerprint(value)
                for value in (
                    self.reset_result_fingerprint,
                    self.reset_evidence_fingerprint,
                    self.reset_contract_fingerprint,
                    self.context_fingerprint,
                    self.routing_target_fingerprint,
                    self.routing_step_fingerprint,
                    self.request_fingerprint,
                    self.state_fingerprint,
                )
            )
        ):
            raise ValueError("unknown-start control handoff is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": UNKNOWN_START_HANDOFF_SCHEMA,
            "session_id": self.session_id,
            "reset_recording_id": self.reset_recording_id,
            "reset_seed": self.reset_seed,
            "reset_result_fingerprint": self.reset_result_fingerprint,
            "reset_evidence_fingerprint": self.reset_evidence_fingerprint,
            "reset_contract_fingerprint": self.reset_contract_fingerprint,
            "reference_recording": self.reference_recording,
            "reference_seed": self.reference_seed,
            "context_fingerprint": self.context_fingerprint,
            "routing_target_fingerprint": self.routing_target_fingerprint,
            "routing_step_fingerprint": self.routing_step_fingerprint,
            "request_fingerprint": self.request_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "applied_actions": self.applied_actions,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> UnknownStartControlHandoff:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "session_id",
            "reset_recording_id",
            "reset_seed",
            "reset_result_fingerprint",
            "reset_evidence_fingerprint",
            "reset_contract_fingerprint",
            "reference_recording",
            "reference_seed",
            "context_fingerprint",
            "routing_target_fingerprint",
            "routing_step_fingerprint",
            "request_fingerprint",
            "state_fingerprint",
            "applied_actions",
        } or payload.get("schema") != UNKNOWN_START_HANDOFF_SCHEMA:
            raise ValueError("unknown-start control handoff payload is invalid")
        try:
            return cls(
                session_id=payload["session_id"],
                reset_recording_id=payload["reset_recording_id"],
                reset_seed=payload["reset_seed"],
                reset_result_fingerprint=payload["reset_result_fingerprint"],
                reset_evidence_fingerprint=payload["reset_evidence_fingerprint"],
                reset_contract_fingerprint=payload["reset_contract_fingerprint"],
                reference_recording=payload["reference_recording"],
                reference_seed=payload["reference_seed"],
                context_fingerprint=payload["context_fingerprint"],
                routing_target_fingerprint=payload["routing_target_fingerprint"],
                routing_step_fingerprint=payload["routing_step_fingerprint"],
                request_fingerprint=payload["request_fingerprint"],
                state_fingerprint=payload["state_fingerprint"],
                applied_actions=payload["applied_actions"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("unknown-start control handoff payload is invalid") from error


def _maximum_error(left: Any, right: Any) -> float:
    if len(left) != len(right):
        return float("inf")
    return max(abs(float(a) - float(b)) for a, b in zip(left, right))


def _quaternion_error(left: Any, right: Any) -> float:
    if len(left) != 4 or len(right) != 4:
        return float("inf")
    left_norm = sqrt(sum(float(value) ** 2 for value in left))
    right_norm = sqrt(sum(float(value) ** 2 for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return float("inf")
    dot = abs(
        sum(float(a) * float(b) for a, b in zip(left, right))
        / (left_norm * right_norm)
    )
    return 2.0 * acos(min(1.0, max(-1.0, dot)))


def validate_unknown_start_handoff(
    evidence: UnknownStartResetEvidence,
    *,
    arm_positions: tuple[float, ...],
    gripper_width_m: float,
    connector_position_m: tuple[float, ...],
    socket_position_m: tuple[float, ...],
    gripper_frame_position_m: tuple[float, ...],
    connector_orientation_wxyz: tuple[float, ...],
    expected_connector_orientation_wxyz: tuple[float, ...],
    socket_orientation_wxyz: tuple[float, ...],
    expected_socket_orientation_wxyz: tuple[float, ...],
    camera_offset_m: tuple[float, ...],
    socket_scale: float,
    light_exposure_deltas: tuple[float, ...],
    plug_attached: bool,
    collision_detected: bool,
    contact_force_newtons: float,
) -> None:
    """Reject any live-state drift before a model request is created."""

    evidence.validate(UNKNOWN_START_RESET_CONTRACT)
    if (
        len(arm_positions) != 7
        or len(connector_position_m) != 3
        or len(socket_position_m) != 3
        or len(gripper_frame_position_m) != 3
        or len(camera_offset_m) != 3
        or not light_exposure_deltas
        or not all(
            isfinite(float(value))
            for values in (
                arm_positions,
                connector_position_m,
                socket_position_m,
                gripper_frame_position_m,
                connector_orientation_wxyz,
                expected_connector_orientation_wxyz,
                socket_orientation_wxyz,
                expected_socket_orientation_wxyz,
                camera_offset_m,
                light_exposure_deltas,
            )
            for value in values
        )
        or not isfinite(gripper_width_m)
        or not isfinite(contact_force_newtons)
        or not isfinite(socket_scale)
        or _maximum_error(arm_positions, evidence.observed_arm_positions_radians)
        > MAXIMUM_HANDOFF_JOINT_DRIFT_RAD
        or abs(gripper_width_m - evidence.observed_gripper_width_m)
        > MAXIMUM_HANDOFF_GRIPPER_DRIFT_M
        or _maximum_error(
            connector_position_m,
            evidence.workspace.connector_position_m,
        )
        > MAXIMUM_HANDOFF_POSITION_DRIFT_M
        or _maximum_error(socket_position_m, evidence.workspace.socket_position_m)
        > MAXIMUM_HANDOFF_POSITION_DRIFT_M
        or _maximum_error(
            gripper_frame_position_m,
            evidence.workspace.gripper_control_frame_position_m,
        )
        > MAXIMUM_HANDOFF_POSITION_DRIFT_M
        or _quaternion_error(
            connector_orientation_wxyz,
            expected_connector_orientation_wxyz,
        )
        > MAXIMUM_HANDOFF_ORIENTATION_DRIFT_RAD
        or _quaternion_error(
            socket_orientation_wxyz,
            expected_socket_orientation_wxyz,
        )
        > MAXIMUM_HANDOFF_ORIENTATION_DRIFT_RAD
        or _maximum_error(camera_offset_m, evidence.realization.camera_offset_m)
        > UNKNOWN_START_RESET_CONTRACT.realization_tolerances.camera_offset_m
        or abs(socket_scale - evidence.workspace.socket_scale)
        > UNKNOWN_START_RESET_CONTRACT.workspace.realization_scale_tolerance
        or any(
            abs(delta - evidence.realization.light_exposure_delta)
            > UNKNOWN_START_RESET_CONTRACT.realization_tolerances.light_exposure_delta
            for delta in light_exposure_deltas
        )
        or not isinstance(plug_attached, bool)
        or not isinstance(collision_detected, bool)
        or plug_attached
        or collision_detected
        or contact_force_newtons != 0.0
    ):
        raise ValueError("unknown-start live state drifted after authentication")
