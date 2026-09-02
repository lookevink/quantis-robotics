"""Typed persistence contract for a single-use Isaac control session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any

from jepa_wm.action import (
    DROID_FPS,
    MAX_GRIPPER_WIDTH_M,
    DroidAction,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlInterlockEvidence,
    ControlGateDecision,
    ControlGateReason,
    INSERTION_TARGET_PROGRESS,
    SafetyProjectionAttempt,
    SimulatorControlGate,
    SimulatorSafetyLimits,
    SimulatorSafetyState,
)
from jepa_wm.control_tracking import (
    ActionTrackingDecision,
    ActionTrackingLimits,
    CommandRealizationLimits,
    CommandRealizationDecision,
    evaluate_action_tracking,
    evaluate_command_realization,
    tracking_limits_for_policy,
)
from jepa_wm.joint_settlement import JointSettlementAttempt
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.direct_safety import DirectInsertionSafetyEvidence
from jepa_wm.insertion_refresh import (
    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
    MAXIMUM_SYNCHRONIZED_GRIPPER_ERROR_METERS,
    ControlSafetySnapshot,
    InsertionEvaluationRefresh,
)
from jepa_wm.insertion_rollout import (
    TWO_STEP_INSERTION_ROLLOUT,
    InsertionRolloutPosition,
    is_insertion_rollout_policy,
)
from jepa_wm.integrated_insertion import INTEGRATED_INSERTION_SCHEDULE
from jepa_wm.insertion_task import (
    InsertionTarget,
    InsertionTaskStep,
    quaternion_orientation_error,
)
from jepa_wm.control_policy import (
    ControlExecutionPolicy,
    is_insertion_trial_execution_policy,
)
from jepa_wm.contact_grasp_target import ContactGraspTargetPolicy
from jepa_wm.experimental_candidate import (
    CandidateExecutionEvidence,
    CandidateSourceEvidence,
    ExperimentalCandidateBinding,
)
from jepa_wm.insertion_trial import (
    InsertionTrialDriveEvidence,
    InsertionTrialBinding,
    InsertionTrialExecutionEvidence,
    InsertionTrialPostActionEvidence,
    InsertionTrialRollbackEvidence,
    InsertionTrialRollbackFailure,
    InsertionTrialRollbackOutcome,
    InsertionTrialRollbackReason,
    InsertionTrialSourceEvidence,
    insertion_trial_rollback_outcome_from_dict,
)
from jepa_wm.insertion_transition import (
    InsertionProposalContinuation,
    insertion_proposal_continuation_from_dict,
    resolve_insertion_followup_proposal,
)
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
    INSERTION_TASK_ID,
    InsertionControlTargetPolicy,
    InsertionTargetOrigin,
    insertion_control_target_policy,
)
from jepa_wm.target_progress import RealizedTargetProgressDecision
from jepa_wm.trajectory import validate_observation_target
from jepa_wm.persistence import write_json_atomic
from jepa_wm.shadow_planning import ShadowPlanningRequest, ShadowSearchEvidence
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from jepa_wm.trial_equivalence import ControlTrialContext, TrialResetState
from sim.control_context import recording_task
from sim.grasp_task import AttachmentMechanism, GraspAcquisitionEvidence
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
    ROLLED_BACK_PROGRESS = "rolled_back_after_insufficient_target_progress"
    ROLLED_BACK_EXECUTION = "rolled_back_after_execution_failure"
    ROLLBACK_FAILED = "rollback_failed"

    @classmethod
    def from_insertion_rollback_reason(
        cls,
        reason: InsertionTrialRollbackReason | None,
    ) -> ControlResultStatus:
        return {
            None: cls.APPLIED,
            InsertionTrialRollbackReason.TRACKING: cls.ROLLED_BACK_TRACKING,
            InsertionTrialRollbackReason.CONTACT: cls.ROLLED_BACK_CONTACT,
            InsertionTrialRollbackReason.ATTACHMENT: cls.ROLLED_BACK_ATTACHMENT,
            InsertionTrialRollbackReason.PROGRESS: cls.ROLLED_BACK_PROGRESS,
        }[reason]


CONTROL_SESSION_STATE_SCHEMA = "quantis.control_session_state.v2"


class ControlTargetContract(str, Enum):
    GENERIC = "generic"
    INSERTION = "insertion"
    CONTACT_GRASP = "contact_grasp"


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
    active_drive_target: JointDriveTarget | None = None
    insertion_rollout_position: InsertionRolloutPosition | None = None
    contact_grasp_target_policy: ContactGraspTargetPolicy | None = None
    schema: str | None = CONTROL_SESSION_STATE_SCHEMA

    @property
    def target_contract(self) -> ControlTargetContract:
        if self.contact_grasp_target_policy is not None:
            return ControlTargetContract.CONTACT_GRASP
        if self.insertion_target_policy is not None:
            return ControlTargetContract.INSERTION
        return ControlTargetContract.GENERIC

    @classmethod
    def from_dict(cls, payload: Any) -> ControlSessionState:
        if not isinstance(payload, dict):
            raise ValueError("control session state must be an object")
        try:
            schema = payload.get("schema")
            if schema not in (None, CONTROL_SESSION_STATE_SCHEMA):
                raise ValueError("control session state schema is invalid")
            claimed_target_contract = payload.get("target_contract")
            if schema is None:
                if claimed_target_contract is not None:
                    raise ValueError("legacy control target contract is invalid")
            else:
                claimed_target_contract = ControlTargetContract(
                    claimed_target_contract
                )
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
                active_drive_target=(
                    JointDriveTarget.from_dict(payload["active_drive_target"])
                    if payload.get("active_drive_target") is not None
                    else None
                ),
                insertion_rollout_position=(
                    InsertionRolloutPosition.from_dict(
                        payload["insertion_rollout_position"]
                    )
                    if payload.get("insertion_rollout_position") is not None
                    else None
                ),
                contact_grasp_target_policy=(
                    ContactGraspTargetPolicy.from_dict(
                        payload["contact_grasp_target_policy"]
                    )
                    if payload.get("contact_grasp_target_policy") is not None
                    else None
                ),
                schema=schema,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control session state is incomplete") from error
        validate_recording_id(state.session_id)
        validate_recording_id(state.reference_recording)
        validate_recording_id(state.recording)
        if state.previous_session_id is not None:
            validate_recording_id(state.previous_session_id)
        if (
            state.schema is not None
            and claimed_target_contract is not state.target_contract
        ):
            raise ValueError("control target contract is invalid")
        if (
            state.schema not in (None, CONTROL_SESSION_STATE_SCHEMA)
            or state.seed < 0
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
            or (
                state.active_drive_target is not None
                and not isinstance(state.active_drive_target, JointDriveTarget)
            )
            or (
                state.insertion_rollout_position is not None
                and not isinstance(
                    state.insertion_rollout_position,
                    InsertionRolloutPosition,
                )
            )
            or (
                state.insertion_rollout_position is not None
                and (
                    state.insertion_target_policy is None
                    or not is_insertion_rollout_policy(state.execution_policy)
                )
            )
            or (
                state.contact_grasp_target_policy is not None
                and (
                    not isinstance(
                        state.contact_grasp_target_policy,
                        ContactGraspTargetPolicy,
                    )
                    or state.execution_policy is not ControlExecutionPolicy.DIRECT
                    or state.insertion_target_policy is not None
                    or state.active_drive_target is None
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
        if self.schema is not None:
            payload["schema"] = self.schema
            payload["target_contract"] = self.target_contract.value
        if self.insertion_target_policy is not None:
            payload["insertion_target_policy"] = (
                self.insertion_target_policy.to_dict()
            )
        if self.active_drive_target is not None:
            payload["active_drive_target"] = self.active_drive_target.to_dict()
        if self.insertion_rollout_position is not None:
            payload["insertion_rollout_position"] = (
                self.insertion_rollout_position.to_dict()
            )
        if self.contact_grasp_target_policy is not None:
            payload["contact_grasp_target_policy"] = (
                self.contact_grasp_target_policy.to_dict()
            )
        return payload

    def require_current_contact_grasp_policy(self) -> ContactGraspTargetPolicy:
        if (
            self.schema != CONTROL_SESSION_STATE_SCHEMA
            or not isinstance(
                self.contact_grasp_target_policy,
                ContactGraspTargetPolicy,
            )
        ):
            raise ValueError(
                "contact-grasp target policy is not current execution authority"
            )
        return self.contact_grasp_target_policy

    def resolved_insertion_rollout_position(self) -> InsertionRolloutPosition:
        """Return current position or the bounded legacy two-step interpretation."""

        if self.insertion_rollout_position is not None:
            return self.insertion_rollout_position
        if (
            self.insertion_target_policy is None
            or not is_insertion_rollout_policy(self.execution_policy)
        ):
            raise ValueError("control session has no insertion rollout position")
        return InsertionRolloutPosition(
            1 if self.previous_session_id is None else 2,
            TWO_STEP_INSERTION_ROLLOUT.maximum_steps,
        )

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
        if self.contact_grasp_target_policy is not None:
            if self.insertion_target_policy is not None:
                raise ValueError("control target policies are mutually exclusive")
            self.contact_grasp_target_policy.validate_observation_target(
                observation,
                recording,
                frame_root=frame_root,
                require_initial=self.previous_session_id is None,
            )
        elif self.insertion_target_policy is not None:
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

    def insertion_projection_scales(
        self,
        observation: ControlObservation,
        proposed_action: DroidAction | None = None,
    ) -> tuple[DroidActionScale, ...]:
        if self.insertion_target_policy is None:
            return ACTION_SCALES
        return self.insertion_target_policy.projection_scales(
            observation.pose,
            observation.target_pose,
            proposed_action,
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
    insertion_trial: InsertionTrialPostActionEvidence | None = None
    grasp_acquisition: GraspAcquisitionEvidence | None = None
    attachment_mechanism: AttachmentMechanism | None = None
    insertion_task_step: InsertionTaskStep | None = None
    insertion_target: InsertionTarget | None = None
    command_realization: CommandRealizationDecision | None = None
    plug_orientation_wxyz: tuple[float, ...] | None = None
    socket_orientation_wxyz: tuple[float, ...] | None = None
    gripper_frame_world_position: tuple[float, ...] | None = None

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
        if self.attachment_mechanism is not None and not isinstance(
            self.attachment_mechanism,
            AttachmentMechanism,
        ):
            raise ValueError("post-action attachment mechanism is invalid")
        if (self.insertion_task_step is None) != (self.insertion_target is None):
            raise ValueError("post-action insertion geometry is incomplete")
        if (self.plug_orientation_wxyz is None) != (
            self.socket_orientation_wxyz is None
        ) or (self.insertion_task_step is None) != (
            self.plug_orientation_wxyz is None
        ):
            raise ValueError("post-action insertion orientation is incomplete")
        if (self.insertion_task_step is None) != (
            self.gripper_frame_world_position is None
        ):
            raise ValueError("post-action gripper-frame evidence is incomplete")
        if self.gripper_frame_world_position is not None and (
            len(self.gripper_frame_world_position) != 3
            or not all(isfinite(value) for value in self.gripper_frame_world_position)
        ):
            raise ValueError("post-action gripper-frame evidence is invalid")
        if self.plug_orientation_wxyz is not None and any(
            len(values) != 4 or not all(isfinite(value) for value in values)
            for values in (
                self.plug_orientation_wxyz,
                self.socket_orientation_wxyz,
            )
        ):
            raise ValueError("post-action insertion orientation is invalid")
        scalars = (
            self.maximum_joint_tracking_error_rad,
            self.contact_force_newtons,
        )
        if not all(isfinite(value) and value >= 0.0 for value in scalars):
            raise ValueError("post-action safety evidence is invalid")

    def require_safety_snapshot(self) -> ControlSafetySnapshot:
        if self.plug_position is None:
            raise ValueError("post-action plug position is missing")
        return ControlSafetySnapshot(
            self.joint_positions,
            (1.0 - self.pose.values[6]) * MAX_GRIPPER_WIDTH_M,
            self.plug_position,
            self.contact_force_newtons,
            self.collision_detected,
            self.plug_attached,
        )

    def validate_followup_capture(
        self,
        observation: ControlObservation,
        state: ControlSessionState,
        *,
        maximum_gripper_error_meters: float = (
            MAXIMUM_SYNCHRONIZED_GRIPPER_ERROR_METERS
        ),
    ) -> None:
        if state.active_drive_target is None:
            raise ValueError("follow-up capture has no active drive target")
        state.require_safety_snapshot().validate_followup_continuity(
            self.require_safety_snapshot(),
            state.active_drive_target,
            maximum_gripper_error_meters=maximum_gripper_error_meters,
        )
        drift = action_between(self.pose, observation.pose)
        limits = ActionTrackingLimits()
        if (
            sum(value * value for value in drift.values[:3])
            > limits.maximum_translation_error_meters**2
            or sum(value * value for value in drift.values[3:6])
            > limits.maximum_rotation_error_radians**2
            or abs(drift.values[6]) > limits.maximum_gripper_error
        ):
            raise ValueError("follow-up capture changed after its applied result")

    def validate_derived_evidence(
        self,
        commanded: DroidAction,
        actual: DroidAction,
        execution_policy: ControlExecutionPolicy,
    ) -> None:
        """Recompute every derived motion/geometry claim from persisted inputs."""

        expected_tracking = evaluate_action_tracking(
            commanded,
            actual,
            tracking_limits_for_policy(execution_policy),
        )
        expected_realization = evaluate_command_realization(
            commanded,
            actual,
            CommandRealizationLimits(),
        )
        if self.actual_action != actual or self.tracking != expected_tracking:
            raise ValueError("post-action tracking evidence is inconsistent")
        if (
            self.command_realization is not None
            and self.command_realization != expected_realization
        ):
            raise ValueError("post-action completion evidence is inconsistent")
        if self.insertion_task_step is not None:
            if (
                self.command_realization is None
                or self.plug_position is None
                or self.plug_orientation_wxyz is None
                or self.socket_orientation_wxyz is None
            ):
                raise ValueError("post-action insertion evidence is incomplete")
            expected_step = InsertionTaskStep(
                tuple(self.plug_position),
                self.gripper_frame_world_position,
                self.plug_attached,
                quaternion_orientation_error(
                    self.plug_orientation_wxyz,
                    self.socket_orientation_wxyz,
                ),
                expected_tracking.passed and expected_realization.passed,
                self.collision_detected,
                self.contact_force_newtons,
            )
            if self.insertion_task_step != expected_step:
                raise ValueError("post-action insertion evidence is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.grasp_acquisition is not None:
            payload["grasp_acquisition"] = self.grasp_acquisition.to_dict()
        if self.attachment_mechanism is not None:
            payload["attachment_mechanism"] = self.attachment_mechanism.value
        if self.insertion_task_step is not None:
            payload["insertion_task_step"] = self.insertion_task_step.to_dict()
        if self.insertion_target is not None:
            payload["insertion_target"] = self.insertion_target.to_dict()
        if self.command_realization is not None:
            payload["command_realization"] = self.command_realization.to_dict()
        if self.plug_orientation_wxyz is not None:
            payload["post_action_plug_orientation_wxyz"] = list(
                self.plug_orientation_wxyz
            )
            payload["post_action_socket_orientation_wxyz"] = list(
                self.socket_orientation_wxyz
            )
            payload["post_action_gripper_frame_world_position"] = list(
                self.gripper_frame_world_position
            )
        if self.insertion_trial is not None:
            payload["insertion_trial_post_action"] = self.insertion_trial.to_dict()
        return payload

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
                grasp_acquisition=(
                    GraspAcquisitionEvidence.from_dict(payload["grasp_acquisition"])
                    if "grasp_acquisition" in payload
                    else None
                ),
                attachment_mechanism=(
                    AttachmentMechanism(payload["attachment_mechanism"])
                    if payload.get("attachment_mechanism") is not None
                    else None
                ),
                insertion_task_step=(
                    InsertionTaskStep.from_dict(payload["insertion_task_step"])
                    if "insertion_task_step" in payload
                    else None
                ),
                insertion_target=(
                    InsertionTarget.from_dict(payload["insertion_target"])
                    if "insertion_target" in payload
                    else None
                ),
                command_realization=(
                    CommandRealizationDecision.from_dict(
                        payload["command_realization"]
                    )
                    if "command_realization" in payload
                    else None
                ),
                plug_orientation_wxyz=(
                    tuple(
                        float(value)
                        for value in payload["post_action_plug_orientation_wxyz"]
                    )
                    if "post_action_plug_orientation_wxyz" in payload
                    else None
                ),
                socket_orientation_wxyz=(
                    tuple(
                        float(value)
                        for value in payload["post_action_socket_orientation_wxyz"]
                    )
                    if "post_action_socket_orientation_wxyz" in payload
                    else None
                ),
                gripper_frame_world_position=(
                    tuple(
                        float(value)
                        for value in payload[
                            "post_action_gripper_frame_world_position"
                        ]
                    )
                    if "post_action_gripper_frame_world_position" in payload
                    else None
                ),
                insertion_trial=(
                    InsertionTrialPostActionEvidence.from_dict(
                        payload["insertion_trial_post_action"]
                    )
                    if "insertion_trial_post_action" in payload
                    else None
                ),
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
    insertion_trial_refresh: InsertionEvaluationRefresh | None = None
    insertion_trial_drive: InsertionTrialDriveEvidence | None = None
    insertion_trial_rollback: InsertionTrialRollbackOutcome | None = None
    insertion_trial_settlement_failure: JointSettlementAttempt | None = None

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
        ) or (
            self.insertion_trial_drive is not None
            and (blocked or self.insertion_trial_refresh is None)
        ) or (
            isinstance(self.insertion_trial_rollback, InsertionTrialRollbackEvidence)
            and self.status
            not in (
                ControlResultStatus.ROLLED_BACK_TRACKING,
                ControlResultStatus.ROLLED_BACK_CONTACT,
                ControlResultStatus.ROLLED_BACK_ATTACHMENT,
                ControlResultStatus.ROLLED_BACK_PROGRESS,
                ControlResultStatus.ROLLED_BACK_EXECUTION,
            )
        ) or (
            isinstance(self.insertion_trial_rollback, InsertionTrialRollbackFailure)
            and not rollback_failed
        ) or (
            self.insertion_trial_settlement_failure is not None
            and (
                has_post_action
                or self.status not in (
                    ControlResultStatus.ROLLED_BACK_EXECUTION,
                    ControlResultStatus.ROLLBACK_FAILED,
                )
            )
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
        if self.insertion_trial_refresh is not None:
            payload["insertion_trial_refresh"] = (
                self.insertion_trial_refresh.to_dict()
            )
        if self.insertion_trial_drive is not None:
            payload["insertion_trial_drive"] = self.insertion_trial_drive.to_dict()
        if self.insertion_trial_rollback is not None:
            payload["insertion_trial_rollback"] = (
                self.insertion_trial_rollback.to_dict()
            )
        if self.insertion_trial_settlement_failure is not None:
            payload["insertion_trial_settlement_failure"] = (
                self.insertion_trial_settlement_failure.to_dict()
            )
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
            insertion_trial_refresh = payload.get("insertion_trial_refresh")
            insertion_trial_drive = payload.get("insertion_trial_drive")
            insertion_trial_rollback = payload.get("insertion_trial_rollback")
            insertion_trial_settlement_failure = payload.get(
                "insertion_trial_settlement_failure"
            )
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
                insertion_trial_refresh=(
                    InsertionEvaluationRefresh.from_dict(
                        insertion_trial_refresh
                    )
                    if insertion_trial_refresh is not None
                    else None
                ),
                insertion_trial_drive=(
                    InsertionTrialDriveEvidence.from_dict(insertion_trial_drive)
                    if insertion_trial_drive is not None
                    else None
                ),
                insertion_trial_rollback=(
                    insertion_trial_rollback_outcome_from_dict(
                        insertion_trial_rollback
                    )
                    if insertion_trial_rollback is not None
                    else None
                ),
                insertion_trial_settlement_failure=(
                    JointSettlementAttempt.from_dict(
                        insertion_trial_settlement_failure
                    )
                    if insertion_trial_settlement_failure is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control result is incomplete") from error

    def applied_drive_target(
        self,
        *,
        held_gripper_width_m: float | None = None,
    ) -> JointDriveTarget:
        """Reconstruct the exact drive target of one applied generic action.

        Contact-grasp motion holds the previously authenticated gripper target
        after attachment while continuing to update the arm target.  Its source
        state supplies that held width so reconstruction matches execution.
        """

        if (
            self.status is not ControlResultStatus.APPLIED
            or not self.projection_attempts
            or not self.gate.passed
            or self.gate.next_pose is None
        ):
            raise ValueError("control result has no applied drive target")
        target = JointDriveTarget.for_command(
            self.projection_attempts[-1].proposed_joint_positions,
            (1.0 - self.gate.next_pose.values[6]) * MAX_GRIPPER_WIDTH_M,
        )
        if self.insertion_trial_drive is not None:
            target = self.insertion_trial_drive.forward_target
        if held_gripper_width_m is None:
            return target
        return JointDriveTarget(
            target.joint_positions,
            held_gripper_width_m,
        )


@dataclass(frozen=True)
class InsertionFollowupLineage:
    """One applied insertion result authorized to produce one follow-up source."""

    observation: ControlObservation
    state: ControlSessionState
    result: ControlResult
    next_maximum_steps: int | None = None

    def __post_init__(self) -> None:
        post_action = self.result.post_action
        if (
            self.result.session_id != self.state.session_id
            or self.result.status is not ControlResultStatus.APPLIED
            or post_action is None
            or post_action.insertion_trial is None
            or not post_action.insertion_trial.realized_target_progress.passed
            or self.result.insertion_trial_drive is None
            or self.state.insertion_target_policy is None
            or not is_insertion_trial_execution_policy(self.state.execution_policy)
        ):
            raise ValueError("follow-up lineage requires one safe applied insertion")
        self.rollout_position.followup(self.next_maximum_steps)

    @property
    def rollout_position(self) -> InsertionRolloutPosition:
        return self.state.resolved_insertion_rollout_position()

    @property
    def post_action(self) -> PostActionEvidence:
        if self.result.post_action is None:
            raise ValueError("follow-up lineage has no post-action evidence")
        return self.result.post_action

    @property
    def active_drive_target(self) -> JointDriveTarget:
        if self.result.insertion_trial_drive is None:
            raise ValueError("follow-up lineage has no drive evidence")
        return self.result.insertion_trial_drive.forward_target

    @property
    def realized_target_progress(self) -> RealizedTargetProgressDecision:
        insertion_trial = self.post_action.insertion_trial
        if insertion_trial is None:
            raise ValueError("follow-up lineage has no progress evidence")
        return insertion_trial.realized_target_progress

    @property
    def followup_position(self) -> InsertionRolloutPosition:
        return self.rollout_position.followup(self.next_maximum_steps)

    def validate_source(
        self,
        observation: ControlObservation,
        state: ControlSessionState,
        *,
        expected_proposal: Path,
    ) -> None:
        if (
            state.previous_session_id != self.result.session_id
            or observation.warmup_frames != self.observation.warmup_frames + 1
            or observation.expected_proposal != expected_proposal
            or state.reference_recording != self.state.reference_recording
            or state.seed != self.state.seed
            or not self.state.insertion_target_policy.authorizes_followup(
                state.insertion_target_policy
            )
            or state.resolved_insertion_rollout_position()
            != self.followup_position
            or state.active_drive_target != self.active_drive_target
            or observation.previous_action
            != action_between(self.observation.pose, observation.pose)
        ):
            raise ValueError("insertion follow-up source lineage is invalid")
        self.post_action.validate_followup_capture(observation, state)


@dataclass(frozen=True)
class GraspToInsertionLineage:
    """One applied contact-aware grasp authorized to start insertion."""

    observation: ControlObservation
    state: ControlSessionState
    result: ControlResult

    def __post_init__(self) -> None:
        post_action = self.result.post_action
        if (
            self.result.session_id != self.state.session_id
            or self.result.status is not ControlResultStatus.APPLIED
            or self.state.execution_policy is not ControlExecutionPolicy.DIRECT
            or self.state.insertion_target_policy is not None
            or post_action is None
            or not post_action.tracking.passed
            or post_action.collision_detected
            or post_action.contact_force_newtons
            > SimulatorSafetyLimits().maximum_contact_force_newtons
            or not post_action.plug_attached
            or not self.result.projection_attempts
            or not self.result.gate.passed
        ):
            raise ValueError(
                "grasp-to-insertion lineage requires one safe applied grasp"
            )

    @property
    def post_action(self) -> PostActionEvidence:
        if self.result.post_action is None:
            raise ValueError("grasp-to-insertion lineage has no post-action evidence")
        return self.result.post_action

    @property
    def active_drive_target(self) -> JointDriveTarget:
        return self.result.applied_drive_target(
            held_gripper_width_m=(
                self.state.active_drive_target.gripper_width_m
                if self.state.plug_attached
                and self.state.active_drive_target is not None
                else None
            )
        )

    def validate_source(
        self,
        observation: ControlObservation,
        state: ControlSessionState,
    ) -> None:
        rollout_position = state.resolved_insertion_rollout_position()
        expected_context = (
            INTEGRATED_INSERTION_SCHEDULE.initial_context_index
            if rollout_position.maximum_steps
            == INTEGRATED_INSERTION_SCHEDULE.action_count
            else CONTACT_INSERTION_RECORDING.start_index(
                ContactInsertionSegment.GRASP_ATTACH
            )
        )
        if (
            state.previous_session_id != self.result.session_id
            or state.execution_policy
            is not ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION
            or observation.warmup_frames != expected_context
            or state.reference_recording != self.state.reference_recording
            or state.seed != self.state.seed
            or state.insertion_target_policy is None
            or state.insertion_target_policy.target_origin
            is not InsertionTargetOrigin.LIVE_OBSERVATION
            or state.resolved_insertion_rollout_position().step_index != 1
            or state.active_drive_target != self.active_drive_target
            or observation.previous_action
            != action_between(self.observation.pose, observation.pose)
        ):
            raise ValueError("grasp-to-insertion source lineage is invalid")
        self.post_action.validate_followup_capture(
            observation,
            state,
            maximum_gripper_error_meters=(
                MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
            ),
        )


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
    data_root: Path

    @classmethod
    def at(cls, root: Path, session_id: str) -> ControlSession:
        validate_recording_id(session_id)
        return cls(root / session_id, root.parent)

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
    def resolution_capture_failure_path(self) -> Path:
        return self.path / "control_resolution_capture_failure.json"

    @property
    def execution_path(self) -> Path:
        return self.path / "execution_started.json"

    @property
    def candidate_binding_path(self) -> Path:
        return self.path / "experimental_candidate.json"

    @property
    def insertion_trial_binding_path(self) -> Path:
        return self.path / "insertion_trial.json"

    def insertion_proposal_handoff_path(self, followup_session_id: str) -> Path:
        validate_recording_id(followup_session_id)
        return self.path / f"proposal_handoff_{followup_session_id}.json"

    def contact_grasp_acquisition_handoff_path(
        self,
        followup_session_id: str,
    ) -> Path:
        validate_recording_id(followup_session_id)
        return self.path / f"acquisition_handoff_{followup_session_id}.json"

    def load_insertion_proposal_handoff(
        self,
        followup_session_id: str,
    ) -> InsertionProposalContinuation:
        try:
            payload = json.loads(
                self.insertion_proposal_handoff_path(
                    followup_session_id
                ).read_text()
            )
            return insertion_proposal_continuation_from_dict(payload)
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"insertion proposal handoff is invalid: {self.session_id}"
            ) from error

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
        if (
            insertion_control_target_policy(state.execution_policy) is not None
            or state.contact_grasp_target_policy is not None
        ):
            state.validate_observation_target(
                observation,
                self.data_root / "recordings" / state.reference_recording,
                frame_root=self.data_root,
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
        if (
            state.execution_policy is ControlExecutionPolicy.DIRECT
            and state.insertion_target_policy is None
            and state.active_drive_target is not None
            and recording_task(
                self.data_root / "recordings" / state.reference_recording
            )
            == INSERTION_TASK_ID
        ):
            state.require_current_contact_grasp_policy()
        is_candidate = (
            state.execution_policy is ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
        )
        is_insertion_trial = is_insertion_trial_execution_policy(
            state.execution_policy
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
        reference_task = recording_task(
            self.path.parent.parent / "recordings" / state.reference_recording
        )
        if reference_task != INSERTION_TASK_ID:
            raise ValueError("insertion trial source does not reference insertion evidence")
        proposal_handoff = None
        if state.previous_session_id is not None:
            from jepa_wm.control_rollout import ControlStepSummary

            previous = ControlSession.at(self.path.parent, state.previous_session_id)
            previous_step = ControlStepSummary.from_session(previous)
            if (
                previous_step.state.execution_policy
                is ControlExecutionPolicy.DIRECT
            ):
                lineage = GraspToInsertionLineage(
                    previous_step.observation,
                    previous_step.state,
                    previous_step.result,
                )
                lineage.validate_source(observation, state)
            else:
                lineage = InsertionFollowupLineage(
                    previous_step.observation,
                    previous_step.state,
                    previous_step.result,
                    (
                        state.resolved_insertion_rollout_position().maximum_steps
                        if state.resolved_insertion_rollout_position().maximum_steps
                        != previous_step.state.resolved_insertion_rollout_position()
                        .maximum_steps
                        else None
                    ),
                )
                proposal_changed = (
                    previous_step.observation.expected_proposal
                    != observation.expected_proposal
                )
                proposal_handoff = (
                        previous.load_insertion_proposal_handoff(self.session_id)
                    if proposal_changed
                    else None
                )
                expected_proposal = resolve_insertion_followup_proposal(
                    previous_step.observation.expected_proposal,
                    observation.expected_proposal,
                    previous_proposal_fingerprint=(
                        previous_step.response.proposal_fingerprint
                    ),
                    handoff=proposal_handoff,
                )
                lineage.validate_source(
                    observation,
                    state,
                    expected_proposal=expected_proposal,
                )
        direct_safety = self.load_direct_safety()
        if (
            proposal_handoff is not None
            and direct_safety.proposal != proposal_handoff.requested
        ):
            raise ValueError("insertion proposal handoff parent evidence is invalid")
        return InsertionTrialSourceEvidence(
            self.trial_context(observation, state),
            self.load_response(),
            direct_safety,
            state.active_drive_target,
        )

    def _validate_insertion_trial_binding(
        self,
        binding: InsertionTrialBinding,
        response: ProposedControl | None,
        source_evidence: InsertionTrialSourceEvidence | None = None,
    ) -> None:
        observation, state = self.load_capture()
        source = ControlSession.at(self.path.parent, binding.source_session_id)
        _, source_state = source.load_capture()
        if source_evidence is None:
            source_evidence = source.load_insertion_trial_source_evidence()
        if (
            binding.execution_session_id != self.session_id
            or binding.source_session_id != source.session_id
            or (
                (
                    state.insertion_rollout_position is not None
                    or source_state.insertion_rollout_position is not None
                )
                and state.resolved_insertion_rollout_position()
                != source_state.resolved_insertion_rollout_position()
            )
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
            or (
                evidence.active_drive_target is not None
                and evidence.active_drive_target != state.active_drive_target
            )
        ):
            raise ValueError("direct insertion safety is not bound to its session")
        expected_scales = state.insertion_projection_scales(
            observation,
            response.first_action,
        )
        evaluation_observation = observation
        evaluation_response = response
        try:
            if evidence.live_pose is not None:
                refresh = evidence.evaluation
                evaluation_observation, evaluation_response = refresh.authorize(
                    observation,
                    response,
                    state.require_safety_snapshot(),
                )
                expected_scales = state.insertion_projection_scales(
                    evaluation_observation,
                    evaluation_response.first_action,
                )
            else:
                evidence.live_state.validate_continuity(
                    state.require_safety_snapshot()
                )
        except ValueError as error:
            raise ValueError(
                "direct insertion safety is not bound to its session"
            ) from error
        attempted_scales = tuple(attempt.scale for attempt in evidence.attempts)
        if attempted_scales != expected_scales[: len(attempted_scales)]:
            raise ValueError("direct insertion safety used the wrong projection policy")
        self._validate_projection_attempts(
            evaluation_observation,
            state,
            evaluation_response,
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
        if is_insertion_trial_execution_policy(state.execution_policy):
            self.load_insertion_trial_binding().require_current_execution()
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
