"""Reset-bound authority for one realized insertion proposal action."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isclose, isfinite
from pathlib import Path
from time import time
from typing import Any, Mapping, Union

from jepa_wm.action import DROID_FPS, DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import (
    ControlInterlockEvidence,
    ProjectedTargetProgressPolicy,
    SimulatorSafetyLimits,
    insertion_projection_policy_for_scale,
)
from jepa_wm.target_progress import (
    RealizedTargetProgressDecision,
    RealizedTargetProgressPolicy,
)
from jepa_wm.direct_safety import (
    ControlSafetySnapshot,
    DirectInsertionSafetyEvidence,
)
from jepa_wm.joint_settlement import (
    GripperTrackedJointSettlementAttempt,
    GripperTrackedJointSettlementEvidence,
    JointSettlementEvidence,
    TrackedJointSettlementPolicy,
)
from jepa_wm.joint_drive import JointDriveBiasCompensation, JointDriveTarget
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.trial_equivalence import ControlTrialContext, validate_reset_equivalence
from sim.recording import validate_recording_id


INSERTION_TRIAL_SCHEMA = "quantis.jepa_wm_insertion_trial.v1"
INSERTION_TRIAL_REFRESH_SCHEMA = "quantis.jepa_wm_insertion_trial_refresh.v1"
INSERTION_ROLLBACK_GRIPPER_ERROR_METERS = 1e-3
INSERTION_MAXIMUM_DRIVE_BIAS_RADIANS = 0.002
INSERTION_TRIAL_SETTLEMENT_MAXIMUM_UPDATES = 48


class InsertionTrialAuthority(str, Enum):
    RESET_TRIAL_ONLY = "reset_trial_only"


class InsertionTrialRollbackReason(str, Enum):
    TRACKING = "tracking"
    CONTACT = "contact"
    ATTACHMENT = "attachment"
    PROGRESS = "progress"


@dataclass(frozen=True)
class InsertionTrialOutcomeObservation:
    final_joint_tracking_error_radians: float
    action_tracking_passed: bool
    maximum_contact_force_newtons: float
    collision_detected: bool
    plug_attached: bool
    contact_force_limit_newtons: float
    expected_attachment: bool

    def __post_init__(self) -> None:
        if (
            not all(
                isfinite(value) and value >= 0.0
                for value in (
                    self.final_joint_tracking_error_radians,
                    self.maximum_contact_force_newtons,
                    self.contact_force_limit_newtons,
                )
            )
            or self.contact_force_limit_newtons <= 0.0
            or not all(
                isinstance(value, bool)
                for value in (
                    self.action_tracking_passed,
                    self.collision_detected,
                    self.plug_attached,
                    self.expected_attachment,
                )
            )
        ):
            raise ValueError("insertion trial outcome observation is invalid")


@dataclass(frozen=True)
class InsertionTrialSourceEvidence:
    context: ControlTrialContext
    response: ProposedControl
    safety: DirectInsertionSafetyEvidence
    active_drive_target: JointDriveTarget | None = None


@dataclass(frozen=True)
class InsertionTrialExecutionEvidence:
    context: ControlTrialContext
    response: ProposedControl | None


@dataclass(frozen=True)
class InsertionTrialExecutionRefresh:
    """Reauthorize an exact bound action after live reset continuity passes."""

    refreshed_at_unix_seconds: float
    live_state: ControlSafetySnapshot
    live_pose: DroidPose | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.refreshed_at_unix_seconds):
            raise ValueError("insertion trial execution refresh time is invalid")

    def authorize(
        self,
        captured: ControlObservation,
        response: ProposedControl,
        captured_state: ControlSafetySnapshot,
    ) -> tuple[ControlObservation, ProposedControl]:
        self.live_state.validate_continuity(captured_state)
        if self.refreshed_at_unix_seconds < max(
            captured.captured_at_unix_seconds,
            response.created_at_unix_seconds,
        ):
            raise ValueError("insertion trial execution refresh precedes its source")
        return (
            replace(
                captured,
                captured_at_unix_seconds=self.refreshed_at_unix_seconds,
                pose=(self.live_pose if self.live_pose is not None else captured.pose),
            ),
            replace(
                response,
                created_at_unix_seconds=self.refreshed_at_unix_seconds,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": INSERTION_TRIAL_REFRESH_SCHEMA,
            "refreshed_at_unix_seconds": self.refreshed_at_unix_seconds,
            "live_state": self.live_state.to_dict(),
        }
        if self.live_pose is not None:
            payload["live_pose"] = list(self.live_pose.values)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTrialExecutionRefresh:
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != INSERTION_TRIAL_REFRESH_SCHEMA
        ):
            raise ValueError("insertion trial execution refresh schema is invalid")
        try:
            return cls(
                float(payload["refreshed_at_unix_seconds"]),
                ControlSafetySnapshot.from_dict(payload["live_state"]),
                (
                    DroidPose(tuple(payload["live_pose"]))
                    if "live_pose" in payload
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "insertion trial execution refresh is incomplete"
            ) from error


@dataclass(frozen=True)
class InsertionTrialPolicy:
    joint_settlement: TrackedJointSettlementPolicy = TrackedJointSettlementPolicy(
        maximum_updates=INSERTION_TRIAL_SETTLEMENT_MAXIMUM_UPDATES
    )
    realized_progress: RealizedTargetProgressPolicy = RealizedTargetProgressPolicy()
    rollback_gripper_error_meters: float = (
        INSERTION_ROLLBACK_GRIPPER_ERROR_METERS
    )
    drive_bias_compensation: JointDriveBiasCompensation | None = (
        JointDriveBiasCompensation(INSERTION_MAXIMUM_DRIVE_BIAS_RADIANS)
    )
    control_period_seconds: float = 1.0 / DROID_FPS

    def __post_init__(self) -> None:
        if (
            not isfinite(self.rollback_gripper_error_meters)
            or not isclose(
                self.rollback_gripper_error_meters,
                INSERTION_ROLLBACK_GRIPPER_ERROR_METERS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("insertion rollback gripper error is invalid")
        if self.drive_bias_compensation is not None and not isinstance(
            self.drive_bias_compensation,
            JointDriveBiasCompensation,
        ):
            raise ValueError("insertion drive bias compensation is invalid")
        if self.drive_bias_compensation is not None and not isclose(
            self.drive_bias_compensation.maximum_bias_radians,
            INSERTION_MAXIMUM_DRIVE_BIAS_RADIANS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("insertion drive bias compensation is invalid")
        if not isclose(
            self.control_period_seconds,
            1.0 / DROID_FPS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("insertion control period is invalid")

    @property
    def projected_progress(self) -> ProjectedTargetProgressPolicy:
        return ProjectedTargetProgressPolicy(
            self.realized_progress.minimum_translation_error_reduction_fraction
        )

    def forward_drive_target(
        self,
        desired_joint_positions: tuple[float, ...],
        desired_gripper_width_meters: float,
        active_drive_target: JointDriveTarget,
        stable_joint_positions: tuple[float, ...],
        safety_limits: SimulatorSafetyLimits,
    ) -> JointDriveTarget:
        if self.drive_bias_compensation is None:
            target = JointDriveTarget.for_command(
                desired_joint_positions,
                desired_gripper_width_meters,
            )
        else:
            target = JointDriveTarget.for_command(
                self.drive_bias_compensation.compensated_joint_target(
                    desired_joint_positions,
                    active_drive_target,
                    stable_joint_positions,
                    safety_limits,
                ),
                desired_gripper_width_meters,
            )
        maximum_motion = (
            safety_limits.maximum_joint_velocity_radians_per_second
            * self.control_period_seconds
        )
        if max(
            abs(target_value - start_value)
            for target_value, start_value in zip(
                target.joint_positions,
                stable_joint_positions,
            )
        ) > maximum_motion:
            raise ValueError("compensated insertion target exceeds velocity gate")
        return target

    def rollback_reason(
        self,
        evidence: InsertionTrialPostActionEvidence,
        observation: InsertionTrialOutcomeObservation,
    ) -> InsertionTrialRollbackReason | None:
        """Apply the one canonical fail-closed insertion outcome priority."""

        if (
            not evidence.final_joint_tracking_passed(
                observation.final_joint_tracking_error_radians
            )
            or not observation.action_tracking_passed
        ):
            return InsertionTrialRollbackReason.TRACKING
        if (
            observation.collision_detected
            or observation.maximum_contact_force_newtons
            > observation.contact_force_limit_newtons
        ):
            return InsertionTrialRollbackReason.CONTACT
        if observation.plug_attached is not observation.expected_attachment:
            return InsertionTrialRollbackReason.ATTACHMENT
        if not evidence.realized_target_progress.passed:
            return InsertionTrialRollbackReason.PROGRESS
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_settlement": self.joint_settlement.to_dict(),
            "realized_progress": self.realized_progress.to_dict(),
            "rollback_gripper_error_meters": self.rollback_gripper_error_meters,
            **(
                {
                    "drive_bias_compensation": (
                        self.drive_bias_compensation.to_dict()
                    )
                }
                if self.drive_bias_compensation is not None
                else {}
            ),
            "control_period_seconds": self.control_period_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTrialPolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("insertion trial policy must be an object")
        gripper_error = payload.get(
            "rollback_gripper_error_meters",
            INSERTION_ROLLBACK_GRIPPER_ERROR_METERS,
        )
        if isinstance(gripper_error, bool) or not isinstance(
            gripper_error, (int, float)
        ):
            raise ValueError("insertion rollback gripper error must be numeric")
        try:
            return cls(
                TrackedJointSettlementPolicy.from_dict(
                    payload["joint_settlement"]
                ),
                RealizedTargetProgressPolicy.from_dict(
                    payload["realized_progress"]
                ),
                float(gripper_error),
                (
                    JointDriveBiasCompensation.from_dict(
                        payload["drive_bias_compensation"]
                    )
                    if "drive_bias_compensation" in payload
                    else None
                ),
                float(payload.get("control_period_seconds", 1.0 / DROID_FPS)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion trial policy is incomplete") from error


@dataclass(frozen=True)
class InsertionTrialDriveEvidence:
    active_target: JointDriveTarget
    forward_target: JointDriveTarget

    def __post_init__(self) -> None:
        if not isinstance(self.active_target, JointDriveTarget) or not isinstance(
            self.forward_target,
            JointDriveTarget,
        ):
            raise ValueError("insertion trial drive evidence is invalid")

    def validate(
        self,
        policy: InsertionTrialPolicy,
        *,
        desired_joint_positions: tuple[float, ...],
        desired_gripper_width_meters: float,
        stable_joint_positions: tuple[float, ...],
        safety_limits: SimulatorSafetyLimits,
    ) -> None:
        expected = policy.forward_drive_target(
            desired_joint_positions,
            desired_gripper_width_meters,
            self.active_target,
            stable_joint_positions,
            safety_limits,
        )
        if self.forward_target != expected:
            raise ValueError("insertion trial forward drive target is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_target": self.active_target.to_dict(),
            "forward_target": self.forward_target.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTrialDriveEvidence:
        if not isinstance(payload, Mapping):
            raise ValueError("insertion trial drive evidence must be an object")
        try:
            return cls(
                JointDriveTarget.from_dict(payload["active_target"]),
                JointDriveTarget.from_dict(payload["forward_target"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion trial drive evidence is incomplete") from error


@dataclass(frozen=True)
class InsertionTrialPostActionEvidence:
    joint_settlement: JointSettlementEvidence
    realized_target_progress: RealizedTargetProgressDecision

    def validate(
        self,
        policy: InsertionTrialPolicy,
        *,
        expected_requested_motion_radians: float,
        initial_pose: DroidPose,
        target_pose: DroidPose,
        realized_pose: DroidPose,
        final_joint_tracking_error_radians: float,
    ) -> None:
        self.joint_settlement.validate(
            policy.joint_settlement,
            expected_requested_motion_radians=expected_requested_motion_radians,
        )
        if (
            self.realized_target_progress
            != policy.realized_progress.evaluate(
                initial_pose,
                target_pose,
                realized_pose,
            )
            or not isfinite(final_joint_tracking_error_radians)
        ):
            raise ValueError("insertion trial post-action evidence is inconsistent")

    def final_joint_tracking_passed(
        self,
        final_joint_tracking_error_radians: float,
    ) -> bool:
        if (
            not isfinite(final_joint_tracking_error_radians)
            or final_joint_tracking_error_radians < 0.0
        ):
            raise ValueError("final joint tracking error is invalid")
        return (
            final_joint_tracking_error_radians
            <= self.joint_settlement.required_tracking_error_radians
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_settlement": self.joint_settlement.to_dict(),
            "realized_target_progress": self.realized_target_progress.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTrialPostActionEvidence:
        if not isinstance(payload, Mapping):
            raise ValueError("insertion trial post-action evidence must be an object")
        try:
            return cls(
                JointSettlementEvidence.from_dict(payload["joint_settlement"]),
                RealizedTargetProgressDecision.from_dict(
                    payload["realized_target_progress"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "insertion trial post-action evidence is incomplete"
            ) from error


@dataclass(frozen=True)
class InsertionTrialRollbackEvidence:
    start_joint_positions: tuple[float, ...]
    target_joint_positions: tuple[float, ...]
    end_joint_positions: tuple[float, ...]
    settlement: GripperTrackedJointSettlementEvidence
    plug_attached: bool
    drive_target: JointDriveTarget | None = None

    def __post_init__(self) -> None:
        positions = (
            self.start_joint_positions,
            self.target_joint_positions,
            self.end_joint_positions,
        )
        if (
            any(len(values) != 7 for values in positions)
            or not all(isfinite(value) for values in positions for value in values)
            or not isinstance(self.plug_attached, bool)
            or not isinstance(
                self.settlement,
                GripperTrackedJointSettlementEvidence,
            )
            or (
                self.drive_target is not None
                and not isinstance(self.drive_target, JointDriveTarget)
            )
        ):
            raise ValueError("insertion trial rollback evidence is invalid")

    def validate(
        self,
        policy: TrackedJointSettlementPolicy,
        *,
        expected_target_joint_positions: tuple[float, ...],
        expected_drive_target: JointDriveTarget | None = None,
        expected_attachment: bool,
        expected_target_gripper_width_meters: float,
        expected_gripper_error_meters: float,
    ) -> None:
        requested_motion = max(
            abs(start - target)
            for start, target in zip(
                self.start_joint_positions,
                self.target_joint_positions,
            )
        )
        final_error = max(
            abs(end - target)
            for end, target in zip(
                self.end_joint_positions,
                self.target_joint_positions,
            )
        )
        self.settlement.validate(
            policy,
            expected_requested_motion_radians=requested_motion,
            expected_target_gripper_width_meters=(
                expected_target_gripper_width_meters
            ),
            expected_gripper_error_meters=expected_gripper_error_meters,
        )
        if (
            self.target_joint_positions != expected_target_joint_positions
            or self.drive_target != expected_drive_target
            or self.plug_attached is not expected_attachment
            or final_error
            > self.settlement.joint.required_tracking_error_radians
        ):
            raise ValueError("insertion trial rollback evidence is inconsistent")

    @property
    def joint_settlement(self) -> JointSettlementEvidence:
        return self.settlement.joint

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": "settled",
            "start_joint_positions": list(self.start_joint_positions),
            "target_joint_positions": list(self.target_joint_positions),
            "end_joint_positions": list(self.end_joint_positions),
            **self.settlement.to_dict(),
            "plug_attached": self.plug_attached,
        }
        if self.drive_target is not None:
            payload["drive_target"] = self.drive_target.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTrialRollbackEvidence:
        if not isinstance(payload, Mapping):
            raise ValueError("insertion trial rollback evidence must be an object")
        try:
            return cls(
                tuple(float(value) for value in payload["start_joint_positions"]),
                tuple(float(value) for value in payload["target_joint_positions"]),
                tuple(float(value) for value in payload["end_joint_positions"]),
                GripperTrackedJointSettlementEvidence.from_dict(payload),
                payload["plug_attached"],
                (
                    JointDriveTarget.from_dict(payload["drive_target"])
                    if "drive_target" in payload
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion trial rollback evidence is incomplete") from error


class InsertionTrialRollbackFailureReason(str, Enum):
    DRIVE_COMMAND_REJECTED = "drive_command_rejected"
    SETTLEMENT_TIMEOUT = "settlement_timeout"
    SAFETY_INTERLOCK = "safety_interlock"
    ATTACHMENT_CHANGED = "attachment_changed"

    @classmethod
    def from_evidence(
        cls,
        *,
        settlement_attempt: GripperTrackedJointSettlementAttempt | None,
        plug_attached: bool,
        expected_attachment: bool,
        interlock: ControlInterlockEvidence,
        maximum_contact_force_newtons: float,
        drive_command_accepted: bool,
    ) -> InsertionTrialRollbackFailureReason:
        """Select the sole fail-closed rollback-failure priority."""

        if plug_attached is not expected_attachment:
            return cls.ATTACHMENT_CHANGED
        if (
            interlock.collision_detected
            or interlock.maximum_contact_force_newtons
            > maximum_contact_force_newtons
        ):
            return cls.SAFETY_INTERLOCK
        if settlement_attempt is not None:
            return cls.SETTLEMENT_TIMEOUT
        if not drive_command_accepted:
            return cls.DRIVE_COMMAND_REJECTED
        raise ValueError("rollback failure has no supported typed reason")


@dataclass(frozen=True)
class InsertionTrialRollbackFailure:
    """Raw terminal rollback state when reset verification could not complete."""

    start_joint_positions: tuple[float, ...]
    target_joint_positions: tuple[float, ...]
    end_joint_positions: tuple[float, ...]
    plug_attached: bool
    reason: InsertionTrialRollbackFailureReason
    interlock: ControlInterlockEvidence
    drive_command_accepted: bool
    error: str
    settlement_attempt: GripperTrackedJointSettlementAttempt | None = None
    drive_target: JointDriveTarget | None = None

    def __post_init__(self) -> None:
        positions = (
            self.start_joint_positions,
            self.target_joint_positions,
            self.end_joint_positions,
        )
        if (
            any(len(values) != 7 for values in positions)
            or not all(isfinite(value) for values in positions for value in values)
            or not isinstance(self.plug_attached, bool)
            or not isinstance(self.reason, InsertionTrialRollbackFailureReason)
            or not isinstance(self.interlock, ControlInterlockEvidence)
            or not isinstance(self.drive_command_accepted, bool)
            or not self.error
            or (
                self.drive_target is not None
                and not isinstance(self.drive_target, JointDriveTarget)
            )
        ):
            raise ValueError("insertion trial rollback failure is invalid")

    def validate(
        self,
        policy: TrackedJointSettlementPolicy,
        *,
        expected_target_joint_positions: tuple[float, ...],
        expected_drive_target: JointDriveTarget | None = None,
        expected_attachment: bool,
        maximum_contact_force_newtons: float,
        expected_target_gripper_width_meters: float,
        expected_gripper_error_meters: float,
    ) -> None:
        if (
            not isfinite(maximum_contact_force_newtons)
            or maximum_contact_force_newtons <= 0.0
        ):
            raise ValueError("rollback failure force limit is invalid")
        if (
            not isfinite(expected_gripper_error_meters)
            or expected_gripper_error_meters <= 0.0
        ):
            raise ValueError("rollback failure gripper limit is invalid")
        requested_motion = max(
            abs(start - target)
            for start, target in zip(
                self.start_joint_positions,
                self.target_joint_positions,
            )
        )
        if self.target_joint_positions != expected_target_joint_positions:
            raise ValueError("rollback failure target is inconsistent")
        if self.drive_target != expected_drive_target:
            raise ValueError("rollback failure drive target is inconsistent")
        if self.settlement_attempt is not None:
            if not self.drive_command_accepted:
                raise ValueError("rollback timeout evidence is incomplete")
            self.settlement_attempt.validate(
                policy,
                expected_requested_motion_radians=requested_motion,
                expected_target_joint_positions=self.target_joint_positions,
                expected_target_gripper_width_meters=(
                    expected_target_gripper_width_meters
                ),
                expected_gripper_error_meters=expected_gripper_error_meters,
            )
            if self.settlement_attempt.final_joint_positions != self.end_joint_positions:
                raise ValueError("rollback failure endpoint is inconsistent")
        expected_reason = InsertionTrialRollbackFailureReason.from_evidence(
            settlement_attempt=self.settlement_attempt,
            plug_attached=self.plug_attached,
            expected_attachment=expected_attachment,
            interlock=self.interlock,
            maximum_contact_force_newtons=maximum_contact_force_newtons,
            drive_command_accepted=self.drive_command_accepted,
        )
        if self.reason is not expected_reason:
            raise ValueError("rollback failure reason is not supported by its evidence")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": "failed",
            "start_joint_positions": list(self.start_joint_positions),
            "target_joint_positions": list(self.target_joint_positions),
            "end_joint_positions": list(self.end_joint_positions),
            "plug_attached": self.plug_attached,
            "reason": self.reason.value,
            "interlock": self.interlock.to_dict(),
            "drive_command_accepted": self.drive_command_accepted,
            "error": self.error,
        }
        if self.settlement_attempt is not None:
            payload["settlement_attempt"] = self.settlement_attempt.to_dict()
        if self.drive_target is not None:
            payload["drive_target"] = self.drive_target.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTrialRollbackFailure:
        if not isinstance(payload, Mapping) or payload.get("status") != "failed":
            raise ValueError("insertion trial rollback failure is invalid")
        try:
            return cls(
                tuple(float(value) for value in payload["start_joint_positions"]),
                tuple(float(value) for value in payload["target_joint_positions"]),
                tuple(float(value) for value in payload["end_joint_positions"]),
                payload["plug_attached"],
                InsertionTrialRollbackFailureReason(payload["reason"]),
                ControlInterlockEvidence.from_dict(payload["interlock"]),
                payload["drive_command_accepted"],
                str(payload["error"]),
                (
                    GripperTrackedJointSettlementAttempt.from_dict(
                        payload["settlement_attempt"]
                    )
                    if "settlement_attempt" in payload
                    else None
                ),
                (
                    JointDriveTarget.from_dict(payload["drive_target"])
                    if "drive_target" in payload
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion trial rollback failure is incomplete") from error


InsertionTrialRollbackOutcome = Union[
    InsertionTrialRollbackEvidence,
    InsertionTrialRollbackFailure,
]


def insertion_trial_rollback_outcome_from_dict(
    payload: Any,
) -> InsertionTrialRollbackOutcome:
    if not isinstance(payload, Mapping):
        raise ValueError("insertion trial rollback outcome must be an object")
    if payload.get("status") == "failed":
        if set(payload) - {
            "status",
            "start_joint_positions",
            "target_joint_positions",
            "end_joint_positions",
            "plug_attached",
            "reason",
            "interlock",
            "drive_command_accepted",
            "error",
            "settlement_attempt",
            "drive_target",
        }:
            raise ValueError("rollback failure has contradictory fields")
        return InsertionTrialRollbackFailure.from_dict(payload)
    if payload.get("status") not in (None, "settled"):
        raise ValueError("insertion trial rollback outcome status is invalid")
    if set(payload) - {
        "status",
        "start_joint_positions",
        "target_joint_positions",
        "end_joint_positions",
        "joint_settlement",
        "gripper_settlement",
        "plug_attached",
        "drive_target",
    }:
        raise ValueError("settled rollback has contradictory fields")
    return InsertionTrialRollbackEvidence.from_dict(payload)


@dataclass(frozen=True)
class InsertionTrialBinding:
    execution_session_id: str
    source_session_id: str
    source_observation_id: int
    execution_observation_id: int
    proposal: ArtifactIdentity
    actions: tuple[DroidAction, ...]
    source_selected_action_scale: DroidActionScale
    source_active_drive_target: JointDriveTarget | None = None
    trial_policy: InsertionTrialPolicy | None = InsertionTrialPolicy()
    authority: InsertionTrialAuthority = InsertionTrialAuthority.RESET_TRIAL_ONLY

    def __post_init__(self) -> None:
        validate_recording_id(self.execution_session_id)
        validate_recording_id(self.source_session_id)
        try:
            insertion_projection_policy_for_scale(self.source_selected_action_scale)
        except ValueError as error:
            raise ValueError("insertion trial binding is invalid") from error
        if (
            self.execution_session_id == self.source_session_id
            or self.source_observation_id <= 0
            or self.execution_observation_id <= 0
            or len(self.actions) != 3
            or self.authority is not InsertionTrialAuthority.RESET_TRIAL_ONLY
            or (
                self.source_active_drive_target is not None
                and not isinstance(
                    self.source_active_drive_target,
                    JointDriveTarget,
                )
            )
            or (
                self.trial_policy is not None
                and self.trial_policy.drive_bias_compensation is not None
                and self.source_active_drive_target is None
            )
        ):
            raise ValueError("insertion trial binding is invalid")

    @property
    def production_authority_granted(self) -> bool:
        return False

    def require_current_execution(self) -> InsertionTrialPolicy:
        if (
            self.trial_policy is None
            or self.trial_policy.drive_bias_compensation is None
            or self.source_active_drive_target is None
        ):
            raise ValueError("legacy insertion trial cannot be executed")
        return self.trial_policy

    @property
    def allowed_projection_scales(self) -> tuple[DroidActionScale, ...]:
        policy = insertion_projection_policy_for_scale(
            self.source_selected_action_scale
        )
        source_index = policy.index(self.source_selected_action_scale)
        return policy[source_index:]

    def validate_attempted_projection_scales(
        self,
        attempted: tuple[DroidActionScale, ...],
    ) -> None:
        if (
            not attempted
            or attempted != self.allowed_projection_scales[: len(attempted)]
        ):
            raise ValueError("insertion trial exceeded its source projection")

    def validate_execution(
        self,
        source: InsertionTrialSourceEvidence,
        execution: InsertionTrialExecutionEvidence,
    ) -> None:
        source_context = source.context
        execution_context = execution.context
        source_response = source.response
        safety = source.safety
        response = execution.response
        source_identity = (
            ArtifactIdentity(source_response.proposal, source_response.proposal_fingerprint)
            if source_response.proposal_fingerprint is not None
            else None
        )
        if (
            self.source_observation_id
            != source_context.observation.observation_id
            or self.source_observation_id != source_response.observation_id
            or self.source_observation_id != safety.observation_id
            or self.execution_observation_id
            != execution_context.observation.observation_id
            or source_identity != self.proposal
            or safety.proposal != self.proposal
            or source_response.actions != self.actions
            or safety.proposed_actions != self.actions
            or safety.selected_action_scale != self.source_selected_action_scale
            or source.active_drive_target != self.source_active_drive_target
            or not safety.passed
            or not safety.live_state.plug_attached
            or source_context.policy
            is not ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION
            or execution_context.policy
            is not ControlExecutionPolicy.INSERTION_RESET_TRIAL
            or execution_context.reference_recording
            != source_context.reference_recording
            or execution_context.seed != source_context.seed
            or execution_context.previous_session_id
            != source_context.previous_session_id
            or execution_context.observation.target
            != source_context.observation.target
            or execution_context.observation.warmup_frames
            != source_context.observation.warmup_frames
            or execution_context.observation.previous_action
            != source_context.observation.previous_action
        ):
            raise ValueError("insertion trial is not bound to its safety source")
        validate_reset_equivalence(source_context.reset, execution_context.reset)
        if response is not None and (
            response.observation_id != self.execution_observation_id
            or response.actions != self.actions
            or response.proposal != self.proposal.path
            or response.proposal_fingerprint != self.proposal.fingerprint
        ):
            raise ValueError("insertion trial response does not match its binding")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": INSERTION_TRIAL_SCHEMA,
            "execution_session_id": self.execution_session_id,
            "source_session_id": self.source_session_id,
            "source_observation_id": self.source_observation_id,
            "execution_observation_id": self.execution_observation_id,
            "proposal": self.proposal.to_dict(),
            "actions": [list(action.values) for action in self.actions],
            "source_selected_action_scale": self.source_selected_action_scale.to_dict(),
            "authority": self.authority.value,
            "production_authority_granted": self.production_authority_granted,
        }
        if self.source_active_drive_target is not None:
            payload["source_active_drive_target"] = (
                self.source_active_drive_target.to_dict()
            )
        if self.trial_policy is not None:
            payload["trial_policy"] = self.trial_policy.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InsertionTrialBinding:
        if payload.get("schema") != INSERTION_TRIAL_SCHEMA:
            raise ValueError("insertion trial schema is invalid")
        if payload.get("production_authority_granted") is not False:
            raise ValueError("insertion trial cannot have production authority")
        try:
            return cls(
                execution_session_id=str(payload["execution_session_id"]),
                source_session_id=str(payload["source_session_id"]),
                source_observation_id=int(payload["source_observation_id"]),
                execution_observation_id=int(payload["execution_observation_id"]),
                proposal=ArtifactIdentity.from_dict(payload["proposal"]),
                actions=tuple(
                    DroidAction(tuple(values)) for values in payload["actions"]
                ),
                source_selected_action_scale=DroidActionScale.from_payload(
                    payload["source_selected_action_scale"]
                ),
                source_active_drive_target=(
                    JointDriveTarget.from_dict(payload["source_active_drive_target"])
                    if "source_active_drive_target" in payload
                    else None
                ),
                trial_policy=(
                    InsertionTrialPolicy.from_dict(payload["trial_policy"])
                    if "trial_policy" in payload
                    else None
                ),
                authority=InsertionTrialAuthority(payload["authority"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion trial binding is incomplete") from error


def build_insertion_trial_response(
    *,
    execution_session_id: str,
    source_session_id: str,
    execution: ControlTrialContext,
    source: InsertionTrialSourceEvidence,
    created_at_unix_seconds: float | None = None,
) -> tuple[InsertionTrialBinding, ProposedControl]:
    """Rebind a passing no-actuation proposal to an equivalent fresh reset."""

    source_response = source.response
    safety = source.safety
    if source_response.proposal_fingerprint is None or safety.selected_action_scale is None:
        raise ValueError("insertion trial source has no selected exact proposal")
    binding = InsertionTrialBinding(
        execution_session_id=execution_session_id,
        source_session_id=source_session_id,
        source_observation_id=source_response.observation_id,
        execution_observation_id=execution.observation.observation_id,
        proposal=ArtifactIdentity(
            source_response.proposal, source_response.proposal_fingerprint
        ),
        actions=source_response.actions,
        source_selected_action_scale=safety.selected_action_scale,
        source_active_drive_target=source.active_drive_target,
    )
    response = ProposedControl(
        observation_id=execution.observation.observation_id,
        created_at_unix_seconds=(
            time() if created_at_unix_seconds is None else created_at_unix_seconds
        ),
        actions=binding.actions,
        proposal=binding.proposal.path,
        proposal_fingerprint=binding.proposal.fingerprint,
    )
    binding.validate_execution(
        source,
        InsertionTrialExecutionEvidence(execution, response),
    )
    return binding, response
