"""Typed persistence contract for a single-use Isaac control session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any

from jepa_wm.action import DROID_FPS, DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import (
    ControlInterlockEvidence,
    ControlGateDecision,
    ControlGateReason,
    INSERTION_TARGET_PROGRESS,
    SafetyProjectionAttempt,
    SimulatorControlGate,
    SimulatorSafetyState,
)
from jepa_wm.control_tracking import ActionTrackingDecision
from jepa_wm.direct_safety import (
    ControlSafetySnapshot,
    DirectInsertionSafetyEvidence,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.experimental_candidate import (
    CandidateExecutionEvidence,
    CandidateSourceEvidence,
    ExperimentalCandidateBinding,
)
from jepa_wm.insertion_trial import (
    InsertionTrialBinding,
    InsertionTrialExecutionEvidence,
    InsertionTrialSourceEvidence,
)
from jepa_wm.insertion_contract import (
    INSERTION_TASK_ID,
    InsertionControlTargetPolicy,
    insertion_control_target_policy,
)
from jepa_wm.trajectory import validate_observation_target
from jepa_wm.persistence import write_json_atomic
from jepa_wm.shadow_planning import ShadowPlanningRequest, ShadowSearchEvidence
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from jepa_wm.trial_equivalence import ControlTrialContext, TrialResetState
from sim.control_context import recording_task
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
    ROLLED_BACK_ATTACHMENT = "rolled_back_after_attachment_loss"
    ROLLED_BACK_EXECUTION = "rolled_back_after_execution_failure"
    ROLLBACK_FAILED = "rollback_failed"


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
    execution_policy: ControlExecutionPolicy = ControlExecutionPolicy.DIRECT
    plug_position: tuple[float, ...] | None = None
    plug_attached: bool = False
    current_gripper_width_m: float | None = None
    insertion_target_policy: InsertionControlTargetPolicy | None = None

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
                execution_policy=ControlExecutionPolicy(
                    payload.get("execution_policy", ControlExecutionPolicy.DIRECT.value)
                ),
                plug_position=(
                    tuple(float(value) for value in payload["plug_position"])
                    if payload.get("plug_position") is not None
                    else None
                ),
                plug_attached=payload.get("plug_attached", False),
                current_gripper_width_m=(
                    float(payload["current_gripper_width_m"])
                    if payload.get("current_gripper_width_m") is not None
                    else None
                ),
                insertion_target_policy=(
                    InsertionControlTargetPolicy.from_dict(
                        payload["insertion_target_policy"]
                    )
                    if payload.get("insertion_target_policy") is not None
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
            or (
                state.plug_position is not None
                and (
                    len(state.plug_position) != 3
                    or not all(isfinite(value) for value in state.plug_position)
                )
            )
            or not isinstance(state.plug_attached, bool)
            or (
                state.current_gripper_width_m is not None
                and (
                    not isfinite(state.current_gripper_width_m)
                    or not 0.0 <= state.current_gripper_width_m <= 0.08
                )
            )
        ):
            raise ValueError("control session state is invalid")
        return state

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "session_id": self.session_id,
            "reference_recording": self.reference_recording,
            "seed": self.seed,
            "recording": self.recording,
            "current_joint_positions": list(self.current_joint_positions),
            "collision_detected": self.collision_detected,
            "contact_force_newtons": self.contact_force_newtons,
            "previous_session_id": self.previous_session_id,
            "execution_policy": self.execution_policy.value,
            "plug_position": (
                list(self.plug_position) if self.plug_position is not None else None
            ),
            "plug_attached": self.plug_attached,
            "current_gripper_width_m": self.current_gripper_width_m,
        }
        if self.insertion_target_policy is not None:
            payload["insertion_target_policy"] = (
                self.insertion_target_policy.to_dict()
            )
        return payload

    def require_safety_snapshot(self) -> ControlSafetySnapshot:
        if self.current_gripper_width_m is None or self.plug_position is None:
            raise ValueError("control session has incomplete safety state")
        return ControlSafetySnapshot(
            joint_positions=self.current_joint_positions,
            gripper_width_m=self.current_gripper_width_m,
            plug_position=self.plug_position,
            contact_force_newtons=self.contact_force_newtons,
            collision_detected=self.collision_detected,
            plug_attached=self.plug_attached,
        )

    def validate_observation_target(
        self,
        observation: ControlObservation,
        recording: Path,
        *,
        frame_root: Path,
    ) -> None:
        configured_policy = insertion_control_target_policy(
            self.execution_policy
        )
        if self.insertion_target_policy is not None:
            if configured_policy is None:
                raise ValueError(
                    "insertion target policy is invalid for the execution policy"
                )
            self.insertion_target_policy.validate_observation(
                observation,
                recording,
                frame_root=frame_root,
            )
        else:
            validate_observation_target(
                observation,
                recording,
                frame_root=frame_root,
            )


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
    plug_position: tuple[float, ...] | None = None
    plug_attached: bool = False

    def __post_init__(self) -> None:
        if len(self.joint_positions) != 7 or not all(
            isfinite(value) for value in self.joint_positions
        ):
            raise ValueError("post-action joint evidence must be seven-dimensional")
        if self.plug_position is not None and (
            len(self.plug_position) != 3
            or not all(isfinite(value) for value in self.plug_position)
        ):
            raise ValueError("post-action plug position must be three-dimensional")
        if not isinstance(self.plug_attached, bool):
            raise ValueError("post-action plug attachment evidence must be boolean")
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
            "post_action_plug_position": (
                list(self.plug_position) if self.plug_position is not None else None
            ),
            "post_action_plug_attached": self.plug_attached,
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
                plug_position=(
                    tuple(float(value) for value in payload["post_action_plug_position"])
                    if payload.get("post_action_plug_position") is not None
                    else None
                ),
                plug_attached=payload.get("post_action_plug_attached", False),
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
    observation_age_seconds: float
    ik_position_error_m: float | None
    ik_orientation_error_rad: float | None
    pre_action_contact_force_newtons: float
    post_action: PostActionEvidence | None = None
    execution_error: str | None = None
    execution_interlock: ControlInterlockEvidence | None = None

    def __post_init__(self) -> None:
        blocked = self.status == ControlResultStatus.BLOCKED
        execution_failed = self.status == ControlResultStatus.ROLLED_BACK_EXECUTION
        rollback_failed = self.status == ControlResultStatus.ROLLBACK_FAILED
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
        ) or (
            execution_failed and has_post_action
        ) or (
            not blocked
            and not execution_failed
            and not rollback_failed
            and not has_post_action
        ) or (
            (execution_failed or rollback_failed)
            != (self.execution_error is not None)
        ) or any(
            attempt.gate.observation_id != self.gate.observation_id
            for attempt in self.projection_attempts
        ) or (
            self.selected_action_scale is not None and not selected_attempts
        ) or (
            self.execution_error is not None and not self.execution_error
        ) or (
            self.execution_interlock is not None and blocked
        ):
            raise ValueError("control result status and evidence are inconsistent")
        scalars = tuple(
            value
            for value in (
                self.observation_age_seconds,
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
            "observation_age_seconds": self.observation_age_seconds,
            "ik_position_error_m": self.ik_position_error_m,
            "ik_orientation_error_rad": self.ik_orientation_error_rad,
            "pre_action_contact_force_newtons": self.pre_action_contact_force_newtons,
        }
        if self.post_action is not None:
            payload.update(self.post_action.to_dict())
        if self.execution_error is not None:
            payload["execution_error"] = self.execution_error
        if self.execution_interlock is not None:
            payload["execution_interlock"] = self.execution_interlock.to_dict()
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
            execution_interlock = payload.get("execution_interlock")
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
                observation_age_seconds=float(
                    payload.get(
                        "observation_age_seconds",
                        payload.get("inference_age_seconds"),
                    )
                ),
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
                execution_interlock=(
                    ControlInterlockEvidence.from_dict(execution_interlock)
                    if execution_interlock is not None
                    else None
                ),
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
    def shadow_path(self) -> Path:
        return self.path / "shadow.json"

    @property
    def shadow_request_path(self) -> Path:
        return self.path / "shadow_request.json"

    @property
    def shadow_safety_path(self) -> Path:
        return self.path / "shadow_safety.json"

    @property
    def direct_safety_path(self) -> Path:
        return self.path / "direct_insertion_safety.json"

    @property
    def execution_path(self) -> Path:
        return self.path / "execution_started.json"

    @property
    def candidate_binding_path(self) -> Path:
        return self.path / "experimental_candidate.json"

    @property
    def insertion_trial_binding_path(self) -> Path:
        return self.path / "insertion_trial.json"

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
        if insertion_control_target_policy(state.execution_policy) is not None:
            state.validate_observation_target(
                observation,
                RECORDING_ROOT / state.reference_recording,
                frame_root=QUANTIS_DATA_ROOT,
            )
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
        is_candidate = (
            state.execution_policy is ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
        )
        is_insertion_trial = (
            state.execution_policy is ControlExecutionPolicy.INSERTION_RESET_TRIAL
        )
        if is_candidate:
            self.load_candidate_binding(proposal)
        elif self.candidate_binding_path.exists():
            raise ValueError("non-experimental session contains a candidate binding")
        if is_insertion_trial:
            self.load_insertion_trial_binding(proposal)
        elif self.insertion_trial_binding_path.exists():
            raise ValueError("non-insertion session contains an insertion trial binding")
        return observation, proposal, state

    def load_response(self) -> ProposedControl:
        try:
            return ProposedControl.from_dict(json.loads(self.response_path.read_text()))
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid response: {self.session_id}"
            ) from error

    def write_response(self, response: ProposedControl) -> None:
        if self.response_path.exists():
            raise ValueError(f"control session already has a response: {self.session_id}")
        observation, _ = self.load_capture()
        if (
            response.observation_id != observation.observation_id
            or response.proposal != observation.expected_proposal
            or response.created_at_unix_seconds
            < observation.captured_at_unix_seconds
        ):
            raise ValueError("control response is not bound to its session")
        write_json_atomic(self.response_path, response.to_dict())

    def _validate_candidate_binding(
        self,
        binding: ExperimentalCandidateBinding,
        response: ProposedControl | None,
        source_evidence: CandidateSourceEvidence | None = None,
    ) -> None:
        observation, state = self.load_capture()
        source = ControlSession.at(self.path.parent, binding.source_session_id)
        if source_evidence is None:
            source_evidence = source.load_candidate_source_evidence()
        if (
            binding.execution_session_id != self.session_id
            or binding.source_session_id != source.session_id
        ):
            raise ValueError("experimental candidate is not bound to its source trial")
        binding.validate_execution(
            source_evidence,
            CandidateExecutionEvidence(
                self.trial_context(observation, state),
                response,
            ),
        )

    @staticmethod
    def trial_context(
        observation: ControlObservation,
        state: ControlSessionState,
    ) -> ControlTrialContext:
        return ControlTrialContext(
            observation,
            TrialResetState(
                observation.pose,
                state.current_joint_positions,
                state.collision_detected,
                state.contact_force_newtons,
                state.plug_position,
                state.plug_attached,
            ),
            state.execution_policy,
            state.reference_recording,
            state.seed,
            state.previous_session_id,
        )

    def load_candidate_source_evidence(self) -> CandidateSourceEvidence:
        observation, state = self.load_capture()
        return CandidateSourceEvidence(
            self.trial_context(observation, state),
            self.load_shadow(),
            self.load_shadow_safety(),
        )

    def write_candidate_binding(
        self,
        binding: ExperimentalCandidateBinding,
        source_evidence: CandidateSourceEvidence | None = None,
    ) -> None:
        if self.candidate_binding_path.exists():
            raise ValueError(
                f"control session already has candidate evidence: {self.session_id}"
            )
        self._validate_candidate_binding(binding, None, source_evidence)
        write_json_atomic(self.candidate_binding_path, binding.to_dict())

    def load_candidate_binding(
        self,
        response: ProposedControl | None = None,
    ) -> ExperimentalCandidateBinding:
        try:
            binding = ExperimentalCandidateBinding.from_dict(
                json.loads(self.candidate_binding_path.read_text())
            )
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid candidate evidence: {self.session_id}"
            ) from error
        self._validate_candidate_binding(binding, response or self.load_response())
        return binding

    def load_insertion_trial_source_evidence(self) -> InsertionTrialSourceEvidence:
        observation, state = self.load_capture()
        if (
            recording_task(
                self.path.parent.parent / "recordings" / state.reference_recording
            )
            != INSERTION_TASK_ID
        ):
            raise ValueError("insertion trial source does not reference insertion evidence")
        return InsertionTrialSourceEvidence(
            self.trial_context(observation, state),
            self.load_response(),
            self.load_direct_safety(),
        )

    def _validate_insertion_trial_binding(
        self,
        binding: InsertionTrialBinding,
        response: ProposedControl | None,
        source_evidence: InsertionTrialSourceEvidence | None = None,
    ) -> None:
        observation, state = self.load_capture()
        source = ControlSession.at(self.path.parent, binding.source_session_id)
        if source_evidence is None:
            source_evidence = source.load_insertion_trial_source_evidence()
        if (
            binding.execution_session_id != self.session_id
            or binding.source_session_id != source.session_id
        ):
            raise ValueError("insertion trial is not bound to its safety source")
        binding.validate_execution(
            source_evidence,
            InsertionTrialExecutionEvidence(
                self.trial_context(observation, state),
                response,
            ),
        )

    def write_insertion_trial_binding(
        self,
        binding: InsertionTrialBinding,
        source_evidence: InsertionTrialSourceEvidence | None = None,
    ) -> None:
        if self.insertion_trial_binding_path.exists():
            raise ValueError(
                f"control session already has insertion trial evidence: {self.session_id}"
            )
        self._validate_insertion_trial_binding(binding, None, source_evidence)
        write_json_atomic(self.insertion_trial_binding_path, binding.to_dict())

    def load_insertion_trial_binding(
        self,
        response: ProposedControl | None = None,
    ) -> InsertionTrialBinding:
        try:
            binding = InsertionTrialBinding.from_dict(
                json.loads(self.insertion_trial_binding_path.read_text())
            )
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid insertion trial evidence: {self.session_id}"
            ) from error
        self._validate_insertion_trial_binding(
            binding, response or self.load_response()
        )
        return binding

    def load_shadow(self) -> ShadowSearchEvidence:
        try:
            shadow_request = ShadowPlanningRequest.from_dict(
                json.loads(self.shadow_request_path.read_text())
            )
            evidence = ShadowSearchEvidence.from_dict(
                json.loads(self.shadow_path.read_text())
            )
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid shadow evidence: {self.session_id}"
            ) from error
        observation, _ = self.load_capture()
        response = self.load_response()
        task_progress = evidence.task_progress
        expected_calibration = shadow_request.expected_calibration
        calibrated_binding_invalid = (
            expected_calibration is None and task_progress is not None
        ) or (
            expected_calibration is not None
            and (
                task_progress is None
                or observation.target_pose is None
                or task_progress.start != observation.pose
                or task_progress.target != observation.target_pose
                or task_progress.calibration.fingerprint
                != expected_calibration.fingerprint
            )
        )
        if (
            shadow_request.observation != observation
            or shadow_request.direct_control != response
            or evidence.observation_id != observation.observation_id
            or evidence.proposal != response.proposal
            or evidence.direct.actions != response.actions
            or evidence.adapter != shadow_request.expected_adapter
            or evidence.config.planner != shadow_request.expected_planner
            or calibrated_binding_invalid
        ):
            raise ValueError("shadow evidence is not bound to its control session")
        return evidence

    def load_shadow_safety(self) -> ShadowSafetyEvidence:
        try:
            evidence = ShadowSafetyEvidence.from_dict(
                json.loads(self.shadow_safety_path.read_text())
            )
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid shadow safety evidence: {self.session_id}"
            ) from error
        self._validate_shadow_safety_binding(evidence)
        return evidence

    def _validate_shadow_safety_binding(
        self, evidence: ShadowSafetyEvidence
    ) -> None:
        shadow = self.load_shadow()
        observation, state = self.load_capture()
        response = self.load_response()
        if (
            evidence.observation_id != shadow.observation_id
            or evidence.planned_actions != shadow.planned.actions
            or not isclose(
                evidence.counterfactual_as_of_unix_seconds,
                response.created_at_unix_seconds,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("shadow safety evidence is not bound to its search")
        self._validate_projection_attempts(
            observation,
            state,
            response,
            evidence.attempts,
            evidence.counterfactual_as_of_unix_seconds,
            shadow.planned.actions,
        )

    @staticmethod
    def _validate_projection_attempts(
        observation: ControlObservation,
        state: ControlSessionState,
        response: ProposedControl,
        attempts: tuple[SafetyProjectionAttempt, ...],
        as_of_unix_seconds: float,
        actions: tuple[DroidAction, ...],
        live_state: ControlSafetySnapshot | None = None,
    ) -> None:
        current_joints = (
            live_state.joint_positions
            if live_state is not None
            else state.current_joint_positions
        )
        contact_force = (
            live_state.contact_force_newtons
            if live_state is not None
            else state.contact_force_newtons
        )
        collision_detected = (
            live_state.collision_detected
            if live_state is not None
            else state.collision_detected
        )
        for attempt in attempts:
            scaled_action = attempt.scale.apply(actions[0])
            candidate = response.with_actions(
                (scaled_action, *actions[1:])
            )
            safety_state = SimulatorSafetyState(
                observed_joint_positions=state.current_joint_positions,
                current_joint_positions=current_joints,
                proposed_joint_positions=attempt.proposed_joint_positions,
                control_period_seconds=1.0 / DROID_FPS,
                contact_force_newtons=contact_force,
                collision_detected=collision_detected,
            )
            expected_gate = SimulatorControlGate().evaluate(
                observation,
                candidate,
                safety_state,
                now_unix_seconds=as_of_unix_seconds,
            )
            if state.execution_policy in (
                ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
                ControlExecutionPolicy.INSERTION_RESET_TRIAL,
            ):
                expected_gate = INSERTION_TARGET_PROGRESS.apply(
                    expected_gate,
                    observation.pose,
                    observation.target_pose,
                )
            expected_delta = max(
                abs(proposed - current)
                for proposed, current in zip(
                    attempt.proposed_joint_positions, current_joints
                )
            )
            if not isclose(
                attempt.maximum_joint_delta_rad,
                expected_delta,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError("shadow safety joint delta is inconsistent")
            if attempt.gate.reasons == (ControlGateReason.IK_SOLUTION_FAILED,):
                try:
                    expected_pose = observation.pose.applied(scaled_action)
                except ValueError as error:
                    raise ValueError("shadow IK failure pose is invalid") from error
                if not expected_gate.passed or attempt.gate.next_pose != expected_pose:
                    raise ValueError("shadow IK failure evidence is inconsistent")
            elif attempt.gate != expected_gate:
                raise ValueError("control safety gate evidence is inconsistent")

    def write_shadow_safety(self, evidence: ShadowSafetyEvidence) -> None:
        if self.shadow_safety_path.exists():
            raise ValueError(f"shadow safety was already evaluated: {self.session_id}")
        self._validate_shadow_safety_binding(evidence)
        write_json_atomic(self.shadow_safety_path, evidence.to_dict())

    def _validate_direct_safety_binding(
        self, evidence: DirectInsertionSafetyEvidence
    ) -> None:
        observation, state = self.load_capture()
        response = self.load_response()
        if (
            state.execution_policy
            is not ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION
            or response.proposal_fingerprint is None
            or evidence.observation_id != observation.observation_id
            or evidence.proposed_actions != response.actions
            or evidence.proposal.path != response.proposal
            or evidence.proposal.fingerprint != response.proposal_fingerprint
            or evidence.evaluated_at_unix_seconds < response.created_at_unix_seconds
        ):
            raise ValueError("direct insertion safety is not bound to its session")
        try:
            evidence.live_state.validate_continuity(state.require_safety_snapshot())
        except ValueError as error:
            raise ValueError(
                "direct insertion safety is not bound to its session"
            ) from error
        self._validate_projection_attempts(
            observation,
            state,
            response,
            evidence.attempts,
            evidence.evaluated_at_unix_seconds,
            response.actions,
            evidence.live_state,
        )

    def load_direct_safety(self) -> DirectInsertionSafetyEvidence:
        if self.execution_path.exists() or self.result_path.exists():
            raise ValueError("direct insertion safety session was consumed")
        try:
            evidence = DirectInsertionSafetyEvidence.from_dict(
                json.loads(self.direct_safety_path.read_text())
            )
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"control session has no valid direct insertion safety: {self.session_id}"
            ) from error
        self._validate_direct_safety_binding(evidence)
        return evidence

    def write_direct_safety(self, evidence: DirectInsertionSafetyEvidence) -> None:
        if self.execution_path.exists() or self.result_path.exists():
            raise ValueError("cannot evaluate an executed control session without actuation")
        if self.direct_safety_path.exists():
            raise ValueError(
                f"direct insertion safety was already evaluated: {self.session_id}"
            )
        self._validate_direct_safety_binding(evidence)
        write_json_atomic(self.direct_safety_path, evidence.to_dict())

    def claim_execution(self) -> None:
        _, state = self.load_capture()
        if (
            state.execution_policy
            in (
                ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
                ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT,
            )
            or self.direct_safety_path.exists()
        ):
            raise ValueError(
                "restricted insertion diagnostic sessions cannot be executed"
            )
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
