"""Typed persistence contract for a single-use Isaac control session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from typing import Any

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import ControlGateDecision
from jepa_wm.control_tracking import ActionTrackingDecision
from jepa_wm.persistence import write_json_atomic
from sim.recording import validate_recording_id


QUANTIS_DATA_ROOT = Path("/isaac-sim/.local/share/ov/data/quantis")
CONTROL_ROOT = QUANTIS_DATA_ROOT / "control_sessions"
RECORDING_ROOT = QUANTIS_DATA_ROOT / "recordings"


class ControlCaptureStatus(str, Enum):
    OBSERVATION_READY = "observation_ready"


class ControlResultStatus(str, Enum):
    BLOCKED = "blocked"
    APPLIED = "applied"
    ROLLED_BACK_TRACKING = "rolled_back_after_tracking_failure"
    ROLLED_BACK_CONTACT = "rolled_back_after_contact"
    ROLLED_BACK_EXECUTION = "rolled_back_after_execution_failure"


@dataclass(frozen=True)
class ControlSessionState:
    session_id: str
    reference_recording: str
    seed: int
    recording: str
    current_joint_positions: tuple[float, ...]
    collision_detected: bool
    contact_force_newtons: float
    previous_session_id: str | None = None

    @classmethod
    def from_dict(cls, payload: Any) -> ControlSessionState:
        if not isinstance(payload, dict):
            raise ValueError("control session state must be an object")
        try:
            state = cls(
                session_id=str(payload["session_id"]),
                reference_recording=str(payload["reference_recording"]),
                seed=int(payload["seed"]),
                recording=str(payload["recording"]),
                current_joint_positions=tuple(
                    float(value) for value in payload["current_joint_positions"]
                ),
                collision_detected=payload["collision_detected"],
                contact_force_newtons=float(payload["contact_force_newtons"]),
                previous_session_id=(
                    str(payload["previous_session_id"])
                    if payload.get("previous_session_id") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control session state is incomplete") from error
        validate_recording_id(state.session_id)
        validate_recording_id(state.reference_recording)
        validate_recording_id(state.recording)
        if state.previous_session_id is not None:
            validate_recording_id(state.previous_session_id)
        if (
            state.seed < 0
            or len(state.current_joint_positions) != 7
            or not all(isfinite(value) for value in state.current_joint_positions)
            or not isinstance(state.collision_detected, bool)
            or not isfinite(state.contact_force_newtons)
            or state.contact_force_newtons < 0.0
        ):
            raise ValueError("control session state is invalid")
        return state

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "reference_recording": self.reference_recording,
            "seed": self.seed,
            "recording": self.recording,
            "current_joint_positions": list(self.current_joint_positions),
            "collision_detected": self.collision_detected,
            "contact_force_newtons": self.contact_force_newtons,
            "previous_session_id": self.previous_session_id,
        }


@dataclass(frozen=True)
class SafetyProjectionAttempt:
    scale: DroidActionScale
    gate: ControlGateDecision
    maximum_joint_delta_rad: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.maximum_joint_delta_rad)
            or self.maximum_joint_delta_rad < 0.0
        ):
            raise ValueError("safety projection evidence is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.to_dict(),
            "gate": self.gate.to_dict(),
            "maximum_joint_delta_rad": self.maximum_joint_delta_rad,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> SafetyProjectionAttempt:
        if not isinstance(payload, dict):
            raise ValueError("safety projection attempt must be an object")
        try:
            return cls(
                DroidActionScale.from_payload(payload["scale"]),
                ControlGateDecision.from_dict(payload["gate"]),
                float(payload["maximum_joint_delta_rad"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("safety projection attempt is incomplete") from error


@dataclass(frozen=True)
class PostActionEvidence:
    raw_proposed_action: DroidAction
    commanded_action: DroidAction
    actual_action: DroidAction
    tracking: ActionTrackingDecision
    pose: DroidPose
    joint_positions: tuple[float, ...]
    maximum_joint_tracking_error_rad: float
    contact_force_newtons: float
    collision_detected: bool
    frame: dict[str, Any]

    def __post_init__(self) -> None:
        if len(self.joint_positions) != 7 or not all(
            isfinite(value) for value in self.joint_positions
        ):
            raise ValueError("post-action joint evidence must be seven-dimensional")
        scalars = (
            self.maximum_joint_tracking_error_rad,
            self.contact_force_newtons,
        )
        if not all(isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError("post-action safety evidence is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_proposed_action": list(self.raw_proposed_action.values),
            "commanded_action": list(self.commanded_action.values),
            "actual_action": list(self.actual_action.values),
            "action_tracking": self.tracking.to_dict(),
            "post_action_pose": list(self.pose.values),
            "post_action_joint_positions": list(self.joint_positions),
            "maximum_joint_tracking_error_rad": self.maximum_joint_tracking_error_rad,
            "post_action_contact_force_newtons": self.contact_force_newtons,
            "post_action_collision_detected": self.collision_detected,
            "post_action_frame": self.frame,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> PostActionEvidence:
        if not isinstance(payload, dict):
            raise ValueError("post-action evidence must be an object")
        try:
            frame = payload["post_action_frame"]
            if not isinstance(frame, dict):
                raise ValueError("post-action frame must be an object")
            evidence = cls(
                raw_proposed_action=DroidAction(tuple(payload["raw_proposed_action"])),
                commanded_action=DroidAction(tuple(payload["commanded_action"])),
                actual_action=DroidAction(tuple(payload["actual_action"])),
                tracking=ActionTrackingDecision.from_dict(payload["action_tracking"]),
                pose=DroidPose(tuple(payload["post_action_pose"])),
                joint_positions=tuple(
                    float(value) for value in payload["post_action_joint_positions"]
                ),
                maximum_joint_tracking_error_rad=float(
                    payload["maximum_joint_tracking_error_rad"]
                ),
                contact_force_newtons=float(
                    payload["post_action_contact_force_newtons"]
                ),
                collision_detected=payload["post_action_collision_detected"],
                frame=frame,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("post-action evidence is incomplete") from error
        if not isinstance(evidence.collision_detected, bool):
            raise ValueError("post-action collision evidence is invalid")
        return evidence


@dataclass(frozen=True)
class ControlResult:
    status: ControlResultStatus
    session_id: str
    gate: ControlGateDecision
    projection_attempts: tuple[SafetyProjectionAttempt, ...]
    selected_action_scale: DroidActionScale | None
    inference_age_seconds: float
    ik_position_error_m: float | None
    ik_orientation_error_rad: float | None
    pre_action_contact_force_newtons: float
    post_action: PostActionEvidence | None = None
    execution_error: str | None = None

    def __post_init__(self) -> None:
        blocked = self.status == ControlResultStatus.BLOCKED
        execution_failed = self.status == ControlResultStatus.ROLLED_BACK_EXECUTION
        has_post_action = self.post_action is not None
        selected_attempts = tuple(
            attempt
            for attempt in self.projection_attempts
            if attempt.scale == self.selected_action_scale and attempt.gate.passed
        )
        if not self.projection_attempts or (
            blocked
            and (self.gate.passed or self.selected_action_scale is not None or has_post_action)
        ) or (
            not blocked
            and (not self.gate.passed or self.selected_action_scale is None)
        ) or (not blocked and execution_failed == has_post_action) or (
            execution_failed != (self.execution_error is not None)
        ) or any(
            attempt.gate.observation_id != self.gate.observation_id
            for attempt in self.projection_attempts
        ) or (
            self.selected_action_scale is not None and not selected_attempts
        ) or (
            self.execution_error is not None and not self.execution_error
        ):
            raise ValueError("control result status and evidence are inconsistent")
        scalars = tuple(
            value
            for value in (
                self.inference_age_seconds,
                self.ik_position_error_m,
                self.ik_orientation_error_rad,
                self.pre_action_contact_force_newtons,
            )
            if value is not None
        )
        if not all(isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError("control result metrics are invalid")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status.value,
            "session": self.session_id,
            "gate": self.gate.to_dict(),
            "safety_projection_attempts": [
                attempt.to_dict() for attempt in self.projection_attempts
            ],
            "selected_action_scale": (
                self.selected_action_scale.to_dict()
                if self.selected_action_scale is not None
                else None
            ),
            "inference_age_seconds": self.inference_age_seconds,
            "ik_position_error_m": self.ik_position_error_m,
            "ik_orientation_error_rad": self.ik_orientation_error_rad,
            "pre_action_contact_force_newtons": self.pre_action_contact_force_newtons,
        }
        if self.post_action is not None:
            payload.update(self.post_action.to_dict())
        if self.execution_error is not None:
            payload["execution_error"] = self.execution_error
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResult:
        if not isinstance(payload, dict):
            raise ValueError("control result must be an object")
        try:
            selected_scale = payload.get("selected_action_scale")
            post_action = (
                PostActionEvidence.from_dict(payload)
                if "post_action_pose" in payload
                else None
            )
            execution_error = payload.get("execution_error")
            if execution_error is not None:
                execution_error = str(execution_error)
            return cls(
                status=ControlResultStatus(payload["status"]),
                session_id=str(payload["session"]),
                gate=ControlGateDecision.from_dict(payload["gate"]),
                projection_attempts=tuple(
                    SafetyProjectionAttempt.from_dict(attempt)
                    for attempt in payload["safety_projection_attempts"]
                ),
                selected_action_scale=(
                    DroidActionScale.from_payload(selected_scale)
                    if selected_scale is not None
                    else None
                ),
                inference_age_seconds=float(payload["inference_age_seconds"]),
                ik_position_error_m=(
                    float(payload["ik_position_error_m"])
                    if payload.get("ik_position_error_m") is not None
                    else None
                ),
                ik_orientation_error_rad=(
                    float(payload["ik_orientation_error_rad"])
                    if payload.get("ik_orientation_error_rad") is not None
                    else None
                ),
                pre_action_contact_force_newtons=float(
                    payload["pre_action_contact_force_newtons"]
                ),
                post_action=post_action,
                execution_error=execution_error,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control result is incomplete") from error


@dataclass(frozen=True)
class ControlCaptureResult:
    session_id: str
    observation: ControlObservation
    request_path: Path
    contact_force_newtons: float
    collision_detected: bool
    status: ControlCaptureStatus = ControlCaptureStatus.OBSERVATION_READY

    def __post_init__(self) -> None:
        validate_recording_id(self.session_id)
        if (
            not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
            or not isinstance(self.collision_detected, bool)
        ):
            raise ValueError("control capture safety evidence is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "session": self.session_id,
            "observation_id": self.observation.observation_id,
            "request": str(self.request_path),
            "context_frame": str(self.observation.context_frame),
            "target_frame": str(self.observation.target_frame),
            "expected_proposal": str(self.observation.expected_proposal),
            "contact_force_newtons": self.contact_force_newtons,
            "collision_detected": self.collision_detected,
        }


@dataclass(frozen=True)
class ControlSession:
    path: Path

    @classmethod
    def at(cls, root: Path, session_id: str) -> ControlSession:
        validate_recording_id(session_id)
        return cls(root / session_id)

    @property
    def session_id(self) -> str:
        return self.path.name

    @property
    def request_path(self) -> Path:
        return self.path / "request.json"

    @property
    def response_path(self) -> Path:
        return self.path / "response.json"

    @property
    def state_path(self) -> Path:
        return self.path / "state.json"

    @property
    def result_path(self) -> Path:
        return self.path / "result.json"

    @property
    def execution_path(self) -> Path:
        return self.path / "execution_started.json"

    def create(self) -> None:
        if self.path.exists():
            raise ValueError(f"control session already exists: {self.session_id}")
        self.path.mkdir(parents=True)

    def write_capture(
        self, observation: ControlObservation, state: ControlSessionState
    ) -> None:
        if not self.path.exists():
            self.create()
        elif self.request_path.exists() or self.state_path.exists():
            raise ValueError(f"control session capture already exists: {self.session_id}")
        write_json_atomic(self.request_path, observation.to_dict())
        write_json_atomic(self.state_path, state.to_dict())

    def load_capture(self) -> tuple[ControlObservation, ControlSessionState]:
        try:
            observation = ControlObservation.from_dict(
                json.loads(self.request_path.read_text())
            )
            state = ControlSessionState.from_dict(json.loads(self.state_path.read_text()))
        except FileNotFoundError as error:
            raise ValueError(f"control session is incomplete: {self.session_id}") from error
        if state.session_id != self.session_id:
            raise ValueError("control state belongs to a different session")
        return observation, state

    def load_result(self) -> ControlResult:
        try:
            result = ControlResult.from_dict(json.loads(self.result_path.read_text()))
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid result: {self.session_id}"
            ) from error
        if result.session_id != self.session_id:
            raise ValueError("control result belongs to a different session")
        return result

    def load(self) -> tuple[ControlObservation, ProposedControl, ControlSessionState]:
        if self.result_path.exists():
            raise ValueError(f"control session was already consumed: {self.session_id}")
        if self.execution_path.exists():
            raise ValueError(f"control session execution already started: {self.session_id}")
        try:
            observation, state = self.load_capture()
            proposal = self.load_response()
        except FileNotFoundError as error:
            raise ValueError(f"control session is incomplete: {self.session_id}") from error
        return observation, proposal, state

    def load_response(self) -> ProposedControl:
        try:
            return ProposedControl.from_dict(json.loads(self.response_path.read_text()))
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid response: {self.session_id}"
            ) from error

    def claim_execution(self) -> None:
        try:
            with self.execution_path.open("x", encoding="utf-8") as output:
                json.dump({"session": self.session_id}, output)
                output.write("\n")
        except FileExistsError as error:
            raise ValueError(
                f"control session execution already started: {self.session_id}"
            ) from error

    def write_result(self, result: ControlResult) -> None:
        if result.session_id != self.session_id:
            raise ValueError("control result belongs to a different session")
        write_json_atomic(self.result_path, result.to_dict())
