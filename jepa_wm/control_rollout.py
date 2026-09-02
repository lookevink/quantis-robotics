"""Typed validation and metrics for chained simulator control rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import (
    MAX_GRIPPER_WIDTH_M,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.contact_grasp_target import (
    ContactGraspTargetPolicy,
    ContactGraspTargetStep,
)
from jepa_wm.control_protocol import (
    CONTROL_SCHEMA,
    ControlObservation,
    ProposedControl,
)
from jepa_wm.control_policy import (
    ControlExecutionPolicy,
    is_insertion_trial_execution_policy,
)
from jepa_wm.grasp_task import (
    MAXIMUM_CONTACT_GRASP_ACTIONS,
    GraspTaskStep,
    ReachAndGraspDecision,
    evaluate_reach_and_grasp,
)
from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.insertion_rollout import is_insertion_rollout_policy
from jepa_wm.insertion_task import InsertionTarget
from jepa_wm.insertion_refresh import (
    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
)
from jepa_wm.control_safety import (
    ControlGateReason,
    SimulatorSafetyLimits,
    contact_grasp_action_scales,
)
from jepa_wm.insertion_trial import (
    InsertionTrialOutcomeObservation,
    InsertionTrialRollbackEvidence,
    InsertionTrialRollbackFailure,
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.shadow_planning import ShadowSearchEvidence
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
)
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from sim.recording import validate_recording_id


ROLLOUT_SCHEMA = "quantis.jepa_wm_control_rollout.v1"
STANDARD_MAX_CONTROL_ROLLOUT_STEPS = 8
MAX_CONTROL_ROLLOUT_STEPS = MAXIMUM_CONTACT_GRASP_ACTIONS
PROJECTION_SCALE_RECONSTRUCTION_ABSOLUTE_TOLERANCE = 1e-15


def _projection_scale_policy_matches(
    attempted: Sequence[DroidActionScale],
    expected: Sequence[DroidActionScale],
) -> bool:
    """Compare regenerated scales across runtimes without accepting drift."""

    return len(attempted) == len(expected) and all(
        isclose(
            actual.translation,
            regenerated.translation,
            rel_tol=0.0,
            abs_tol=PROJECTION_SCALE_RECONSTRUCTION_ABSOLUTE_TOLERANCE,
        )
        and isclose(
            actual.rotation,
            regenerated.rotation,
            rel_tol=0.0,
            abs_tol=PROJECTION_SCALE_RECONSTRUCTION_ABSOLUTE_TOLERANCE,
        )
        and isclose(
            actual.gripper,
            regenerated.gripper,
            rel_tol=0.0,
            abs_tol=PROJECTION_SCALE_RECONSTRUCTION_ABSOLUTE_TOLERANCE,
        )
        for actual, regenerated in zip(attempted, expected)
    )


class IncompleteStepStatus(str, Enum):
    ORCHESTRATION_FAILED = "orchestration_failed"


class OrchestrationOperation(str, Enum):
    INITIAL_CONTROL_STEP = "initial_control_step"
    INITIAL_STATUS = "initial_status"
    FOLLOWUP_CAPTURE = "followup_capture"
    FOLLOWUP_INFERENCE = "followup_inference"
    FOLLOWUP_SAFETY = "followup_safety"
    FOLLOWUP_SOURCE_PREFLIGHT = "followup_source_preflight"
    FOLLOWUP_BINDING = "followup_binding"
    FOLLOWUP_APPLY = "followup_apply"
    FOLLOWUP_STATUS = "followup_status"
    RESET_TRIAL_SOURCE_PREFLIGHT = "reset_trial_source_preflight"
    RESET_TRIAL_CAPTURE = "reset_trial_capture"
    RESET_TRIAL_BINDING = "reset_trial_binding"
    RESET_TRIAL_APPLY = "reset_trial_apply"

    @property
    def requires_step_index(self) -> bool:
        return self in (
            OrchestrationOperation.FOLLOWUP_CAPTURE,
            OrchestrationOperation.FOLLOWUP_INFERENCE,
            OrchestrationOperation.FOLLOWUP_SAFETY,
            OrchestrationOperation.FOLLOWUP_SOURCE_PREFLIGHT,
            OrchestrationOperation.FOLLOWUP_BINDING,
            OrchestrationOperation.FOLLOWUP_APPLY,
            OrchestrationOperation.FOLLOWUP_STATUS,
        )

    @classmethod
    def parse_phase(
        cls, encoded_phase: str
    ) -> tuple[OrchestrationOperation, int | None]:
        for operation in cls:
            if not operation.requires_step_index and encoded_phase == operation.value:
                return operation, None
            prefix = f"{operation.value}_"
            if operation.requires_step_index and encoded_phase.startswith(prefix):
                encoded_index = encoded_phase[len(prefix) :]
                if encoded_index.isdigit():
                    return operation, int(encoded_index)
        raise ValueError("control rollout orchestration phase is malformed")


@dataclass(frozen=True)
class OrchestrationFailure:
    operation: OrchestrationOperation
    exit_code: int
    step_index: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.exit_code, bool)
            or not 1 <= self.exit_code <= 255
            or self.operation.requires_step_index != (self.step_index is not None)
            or (self.step_index is not None and self.step_index <= 0)
        ):
            raise ValueError("control rollout orchestration failure is invalid")

    @classmethod
    def parse(cls, value: str) -> OrchestrationFailure:
        try:
            phase, encoded_exit_code = value.rsplit(":exit_", 1)
            exit_code = int(encoded_exit_code)
            operation, step_index = OrchestrationOperation.parse_phase(phase)
            return cls(operation, exit_code, step_index)
        except (TypeError, ValueError) as error:
            raise ValueError("control rollout orchestration failure is malformed") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "step_index": self.step_index,
            "exit_code": self.exit_code,
        }


@dataclass(frozen=True)
class PoseError:
    translation_meters: float
    rotation_radians: float
    gripper_closedness: float

    @classmethod
    def between(cls, pose: DroidPose, target: DroidPose) -> PoseError:
        delta = action_between(pose, target)
        return cls(
            float(np.linalg.norm(delta.values[:3])),
            float(Rotation.from_euler("xyz", delta.values[3:6]).magnitude()),
            abs(delta.values[6]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "translation_meters": self.translation_meters,
            "rotation_radians": self.rotation_radians,
            "gripper_closedness": self.gripper_closedness,
        }


@dataclass(frozen=True)
class ControlStepTiming:
    observation_age_seconds: float
    inference_latency_seconds: float
    command_age_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.observation_age_seconds,
            self.inference_latency_seconds,
            self.command_age_seconds,
        )
        if not all(isfinite(value) and value >= 0.0 for value in values) or not np.isclose(
            self.observation_age_seconds,
            self.inference_latency_seconds + self.command_age_seconds,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("control timing evidence is inconsistent")

    @classmethod
    def from_step(
        cls,
        observation: ControlObservation,
        response: ProposedControl,
        result: ControlResult,
    ) -> ControlStepTiming:
        inference_latency = (
            response.created_at_unix_seconds
            - observation.captured_at_unix_seconds
        )
        return cls(
            result.observation_age_seconds,
            inference_latency,
            result.observation_age_seconds - inference_latency,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "observation_age_seconds": self.observation_age_seconds,
            "inference_latency_seconds": self.inference_latency_seconds,
            "command_age_seconds": self.command_age_seconds,
        }


@dataclass(frozen=True)
class ControlStepSummary:
    state: ControlSessionState
    observation: ControlObservation
    response: ProposedControl
    result: ControlResult
    shadow: ShadowSearchEvidence | None = None
    shadow_safety: ShadowSafetyEvidence | None = None

    def contact_grasp_drive_target(self) -> JointDriveTarget:
        """Reconstruct the arm target plus any attached-state gripper hold."""

        return self.result.applied_drive_target(
            held_gripper_width_m=(
                self.state.active_drive_target.gripper_width_m
                if self.state.plug_attached
                and self.state.active_drive_target is not None
                else None
            )
        )

    @classmethod
    def from_session(cls, session: ControlSession) -> ControlStepSummary:
        observation, state = session.load_capture()
        response = session.load_response()
        captured_observation = observation
        captured_response = response
        result = session.load_result()
        limits = SimulatorSafetyLimits()
        potential_contact_grasp = (
            state.execution_policy is ControlExecutionPolicy.DIRECT
            and state.insertion_target_policy is None
            and state.active_drive_target is not None
        )
        reference_task = None
        if potential_contact_grasp:
            manifest = json.loads(
                (
                    session.data_root
                    / "recordings"
                    / state.reference_recording
                    / "manifest.json"
                ).read_text()
            )
            metadata = (
                manifest.get("metadata") if isinstance(manifest, dict) else None
            )
            reference_task = (
                metadata.get("task") if isinstance(metadata, dict) else None
            )
        contact_grasp_execution = (
            potential_contact_grasp and reference_task == INSERTION_TASK_ID
        )
        reset_trial_candidate = (
            state.execution_policy
            is ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
        )
        if is_insertion_trial_execution_policy(state.execution_policy):
            binding = session.load_insertion_trial_binding(response)
            attempted_scales = tuple(
                attempt.scale for attempt in result.projection_attempts
            )
            try:
                binding.validate_attempted_projection_scales(attempted_scales)
            except ValueError as error:
                raise ValueError(
                    f"insertion trial exceeded its source projection: {session.session_id}"
                ) from error
            if (
                result.status is not ControlResultStatus.BLOCKED
                and result.execution_interlock is None
            ):
                raise ValueError(
                    f"insertion trial execution interlock is missing: {session.session_id}"
                )
            if binding.trial_policy is not None:
                if (
                    result.insertion_trial_refresh is None
                    or result.insertion_trial_refresh.live_pose is None
                ):
                    raise ValueError(
                        f"insertion trial live-pose refresh is missing: {session.session_id}"
                    )
                observation, response = result.insertion_trial_refresh.authorize(
                    observation,
                    response,
                    state.require_safety_snapshot(),
                )
            elif result.insertion_trial_refresh is not None:
                observation, response = result.insertion_trial_refresh.authorize(
                    observation,
                    response,
                    state.require_safety_snapshot(),
                )
            drive_rejected = result.gate.reasons == (
                ControlGateReason.DRIVE_TARGET_INVALID,
            )
            if (
                ControlGateReason.DRIVE_TARGET_INVALID in result.gate.reasons
                and not drive_rejected
            ):
                raise ValueError(
                    f"insertion drive rejection is inconsistent: {session.session_id}"
                )
            if drive_rejected:
                rejected_attempt = next(
                    (
                        attempt
                        for attempt in reversed(result.projection_attempts)
                        if attempt.gate.passed
                    ),
                    None,
                )
                if (
                    result.status is not ControlResultStatus.BLOCKED
                    or binding.trial_policy is None
                    or binding.trial_policy.drive_bias_compensation is None
                    or state.active_drive_target is None
                    or result.insertion_trial_refresh is None
                    or rejected_attempt is None
                ):
                    raise ValueError(
                        f"insertion drive rejection is incomplete: {session.session_id}"
                    )
                try:
                    binding.trial_policy.forward_drive_target(
                        rejected_attempt.proposed_joint_positions,
                        (
                            (1.0 - rejected_attempt.gate.next_pose.values[-1])
                            * MAX_GRIPPER_WIDTH_M
                        ),
                        state.active_drive_target,
                        tuple(
                            result.insertion_trial_refresh.live_state.joint_positions
                        ),
                        limits,
                    )
                except ValueError:
                    pass
                else:
                    raise ValueError(
                        f"insertion drive rejection is inconsistent: {session.session_id}"
                    )
            selected_attempt = next(
                (
                    attempt
                    for attempt in result.projection_attempts
                    if attempt.scale == result.selected_action_scale
                    and attempt.gate.passed
                ),
                None,
            )
            if (
                binding.trial_policy is not None
                and binding.trial_policy.drive_bias_compensation is not None
                and result.status is not ControlResultStatus.BLOCKED
            ):
                if (
                    state.active_drive_target is None
                    or result.insertion_trial_drive is None
                    or selected_attempt is None
                ):
                    raise ValueError(
                        f"insertion trial drive evidence is missing: {session.session_id}"
                    )
                if (
                    result.insertion_trial_drive.active_target
                    != state.active_drive_target
                ):
                    raise ValueError(
                        f"insertion trial active drive target is inconsistent: {session.session_id}"
                    )
                result.insertion_trial_drive.validate(
                    binding.trial_policy,
                    desired_joint_positions=selected_attempt.proposed_joint_positions,
                    desired_gripper_width_meters=(
                        (1.0 - selected_attempt.gate.next_pose.values[-1])
                        * MAX_GRIPPER_WIDTH_M
                    ),
                    stable_joint_positions=tuple(
                        result.insertion_trial_refresh.live_state.joint_positions
                    ),
                    safety_limits=limits,
                )
            elif result.insertion_trial_drive is not None:
                raise ValueError(
                    f"legacy insertion trial has unexpected drive evidence: {session.session_id}"
                )
            if result.post_action is not None:
                trial_evidence = result.post_action.insertion_trial
                if binding.trial_policy is None:
                    if trial_evidence is not None:
                        raise ValueError(
                            f"legacy insertion trial has unexpected current evidence: {session.session_id}"
                        )
                elif observation.target_pose is None or trial_evidence is None:
                    raise ValueError(
                        f"insertion trial post-action evidence is missing: {session.session_id}"
                    )
                else:
                    selected_attempt = next(
                        attempt
                        for attempt in result.projection_attempts
                        if attempt.scale == result.selected_action_scale
                        and attempt.gate.passed
                    )
                    final_joint_error = float(
                        np.max(
                            np.abs(
                                np.asarray(result.post_action.joint_positions)
                                - np.asarray(selected_attempt.proposed_joint_positions)
                            )
                        )
                    )
                    try:
                        trial_evidence.validate(
                            binding.trial_policy,
                            expected_requested_motion_radians=(
                                selected_attempt.maximum_joint_delta_rad
                            ),
                            initial_pose=observation.pose,
                            target_pose=observation.target_pose,
                            realized_pose=result.post_action.pose,
                            final_joint_tracking_error_radians=final_joint_error,
                        )
                    except ValueError as error:
                        raise ValueError(
                            f"insertion trial post-action evidence is inconsistent: {session.session_id}"
                        ) from error
                    if not np.isclose(
                        result.post_action.maximum_joint_tracking_error_rad,
                        final_joint_error,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise ValueError(
                            f"insertion trial final joint tracking is inconsistent: {session.session_id}"
                        )
                    expected_outcome = (
                        ControlResultStatus.from_insertion_rollback_reason(
                            binding.trial_policy.rollback_reason(
                                trial_evidence,
                                InsertionTrialOutcomeObservation(
                                    final_joint_error,
                                    result.post_action.tracking.passed,
                                    result.post_action.command_realization is not None
                                    and result.post_action.command_realization.passed,
                                    max(
                                        result.post_action.contact_force_newtons,
                                        result.execution_interlock.maximum_contact_force_newtons,
                                    ),
                                    result.post_action.collision_detected
                                    or result.execution_interlock.collision_detected,
                                    result.post_action.plug_attached,
                                    limits.maximum_contact_force_newtons,
                                    state.plug_attached,
                                ),
                            )
                        )
                    )
                    if (
                        result.status is not expected_outcome
                        and not (
                            expected_outcome is not ControlResultStatus.APPLIED
                            and result.status is ControlResultStatus.ROLLBACK_FAILED
                            and isinstance(
                                result.insertion_trial_rollback,
                                InsertionTrialRollbackFailure,
                            )
                        )
                    ):
                        raise ValueError(
                            f"insertion trial status is inconsistent: {session.session_id}"
                        )
            if result.insertion_trial_settlement_failure is not None:
                if binding.trial_policy is None:
                    raise ValueError(
                        f"legacy insertion trial has unexpected settlement failure: {session.session_id}"
                    )
                selected_attempt = next(
                    attempt
                    for attempt in result.projection_attempts
                    if attempt.scale == result.selected_action_scale
                    and attempt.gate.passed
                )
                result.insertion_trial_settlement_failure.validate(
                    binding.trial_policy.joint_settlement,
                    expected_requested_motion_radians=(
                        selected_attempt.maximum_joint_delta_rad
                    ),
                    expected_target_joint_positions=(
                        selected_attempt.proposed_joint_positions
                    ),
                )
            if binding.trial_policy is None:
                if result.insertion_trial_rollback is not None:
                    raise ValueError(
                        f"legacy insertion trial has unexpected rollback evidence: {session.session_id}"
                    )
            elif isinstance(
                result.insertion_trial_rollback,
                InsertionTrialRollbackEvidence,
            ):
                result.insertion_trial_rollback.validate(
                    binding.trial_policy.joint_settlement,
                    expected_target_joint_positions=tuple(
                        result.insertion_trial_refresh.live_state.joint_positions
                    ),
                    expected_drive_target=(
                        result.insertion_trial_drive.active_target
                        if result.insertion_trial_drive is not None
                        else None
                    ),
                    expected_attachment=state.plug_attached,
                    expected_target_gripper_width_meters=(
                        result.insertion_trial_refresh.live_state.gripper_width_m
                    ),
                    expected_gripper_error_meters=(
                        binding.trial_policy.rollback_gripper_error_meters
                    ),
                )
            elif isinstance(
                result.insertion_trial_rollback,
                InsertionTrialRollbackFailure,
            ):
                if result.status is not ControlResultStatus.ROLLBACK_FAILED:
                    raise ValueError(
                        f"insertion trial rollback failure status is invalid: {session.session_id}"
                    )
                result.insertion_trial_rollback.validate(
                    binding.trial_policy.joint_settlement,
                    expected_target_joint_positions=tuple(
                        result.insertion_trial_refresh.live_state.joint_positions
                    ),
                    expected_drive_target=(
                        result.insertion_trial_drive.active_target
                        if result.insertion_trial_drive is not None
                        else None
                    ),
                    expected_attachment=state.plug_attached,
                    maximum_contact_force_newtons=(
                        limits.maximum_contact_force_newtons
                    ),
                    expected_target_gripper_width_meters=(
                        result.insertion_trial_refresh.live_state.gripper_width_m
                    ),
                    expected_gripper_error_meters=(
                        binding.trial_policy.rollback_gripper_error_meters
                    ),
                )
            elif result.status in (
                ControlResultStatus.ROLLED_BACK_TRACKING,
                ControlResultStatus.ROLLED_BACK_CONTACT,
                ControlResultStatus.ROLLED_BACK_ATTACHMENT,
                ControlResultStatus.ROLLED_BACK_PROGRESS,
                ControlResultStatus.ROLLED_BACK_EXECUTION,
                ControlResultStatus.ROLLBACK_FAILED,
            ):
                raise ValueError(
                    f"insertion trial rollback settlement is missing: {session.session_id}"
                )
        elif reset_trial_candidate:
            session.load_candidate_binding(response)
            refresh = result.insertion_trial_refresh
            if (
                refresh is None
                or refresh.live_pose is None
                or result.execution_interlock is None
            ):
                raise ValueError(
                    f"reset candidate refresh evidence is missing: {session.session_id}"
                )
            observation, response = refresh.authorize(
                observation,
                response,
                state.require_safety_snapshot(),
            )
            if (
                result.insertion_trial_drive is not None
                or result.insertion_trial_rollback is not None
                or result.insertion_trial_settlement_failure is not None
                or (
                    result.post_action is not None
                    and result.post_action.insertion_trial is not None
                )
            ):
                raise ValueError(
                    f"reset candidate has insertion-trial evidence: {session.session_id}"
                )
        elif contact_grasp_execution:
            contact_grasp_policy = (
                state.require_current_contact_grasp_policy()
            )
            execution_action = contact_grasp_policy.action_for_execution(
                response.actions,
                plug_attached=state.plug_attached,
            )
            expected_projection_policy = contact_grasp_action_scales(
                execution_action,
                attachment_acquired=state.plug_attached,
                require_directional_transport_progress=(
                    contact_grasp_policy.requires_directional_transport_progress
                ),
                require_resolvable_transport=(
                    contact_grasp_policy.uses_horizon_transport_action
                ),
                require_axis_resolvable_transport=(
                    contact_grasp_policy.requires_axis_resolvable_transport
                ),
                coarse_acquisition=contact_grasp_policy.uses_coarse_acquisition_action(
                    observation.target_frame,
                    plug_attached=state.plug_attached,
                ),
                maximum_coarse_translation_command_meters=(
                    contact_grasp_policy.coarse_acquisition_maximum_translation_meters
                ),
                require_resolvable_rotation=(
                    contact_grasp_policy.requires_resolvable_rotation
                ),
                exact_coarse_translation_projection=(
                    contact_grasp_policy.uses_exact_coarse_translation_projection
                ),
                coarse_orientation_hold_fallback=(
                    contact_grasp_policy.uses_coarse_orientation_hold_fallback
                ),
                minimum_coarse_translation_command_meters=(
                    contact_grasp_policy.minimum_coarse_translation_command_meters
                ),
                resolution_floored_acquisition=(
                    contact_grasp_policy.uses_resolution_floored_acquisition_action(
                        observation.target_frame,
                        plug_attached=state.plug_attached,
                    )
                ),
                maximum_resolution_floored_translation_command_meters=(
                    contact_grasp_policy.fine_acquisition_maximum_translation_meters
                ),
            )
            attempted_projection_policy = tuple(
                attempt.scale for attempt in result.projection_attempts
            )
            if (
                not attempted_projection_policy
                or not _projection_scale_policy_matches(
                    attempted_projection_policy,
                    expected_projection_policy[: len(attempted_projection_policy)],
                )
            ):
                raise ValueError(
                    f"contact grasp projection phase is invalid: {session.session_id}"
                )
            if (
                result.insertion_trial_refresh is None
                or result.insertion_trial_refresh.live_pose is None
            ):
                raise ValueError(
                    f"contact grasp live-pose refresh is missing: {session.session_id}"
                )
            if state.active_drive_target is None:
                raise ValueError(
                    f"contact grasp active drive target is missing: {session.session_id}"
                )
            if state.previous_session_id is None:
                observation, response = (
                    result.insertion_trial_refresh.authorize_initial_contact_grasp(
                        observation,
                        response,
                        state.require_safety_snapshot(),
                        MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
                    )
                )
            else:
                observation, response = (
                    result.insertion_trial_refresh.authorize_target_relative(
                        observation,
                        response,
                        state.require_safety_snapshot(),
                        state.active_drive_target,
                        MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
                    )
                )
            if (
                result.insertion_trial_drive is not None
                or result.insertion_trial_rollback is not None
                or result.insertion_trial_settlement_failure is not None
                or (
                    result.post_action is not None
                    and result.post_action.insertion_trial is not None
                )
            ):
                raise ValueError(
                    f"contact grasp has insertion-trial evidence: {session.session_id}"
                )
        elif (
            result.insertion_trial_refresh is not None
            or result.insertion_trial_drive is not None
            or result.insertion_trial_rollback is not None
            or result.insertion_trial_settlement_failure is not None
            or (
                result.post_action is not None
                and result.post_action.insertion_trial is not None
            )
        ):
            raise ValueError(
                f"non-insertion result has insertion evidence: {session.session_id}"
            )
        timing = ControlStepTiming.from_step(observation, response, result)
        if (
            response.observation_id != observation.observation_id
            or response.proposal != observation.expected_proposal
            or result.gate.observation_id != observation.observation_id
            or (
                result.status != ControlResultStatus.BLOCKED
                and (
                    timing.observation_age_seconds
                    > limits.maximum_observation_age_seconds
                    or timing.command_age_seconds
                    > limits.maximum_command_age_seconds
                )
            )
            or response.created_at_unix_seconds < observation.captured_at_unix_seconds
            or response.created_at_unix_seconds
            > observation.captured_at_unix_seconds
            + timing.observation_age_seconds
            + 1e-6
        ):
            raise ValueError(
                f"control step identity or freshness is invalid: {session.session_id}"
            )
        if result.selected_action_scale is not None:
            response_action = (
                state.require_current_contact_grasp_policy().action_for_execution(
                    response.actions,
                    plug_attached=state.plug_attached,
                )
                if contact_grasp_execution
                else response.first_action
            )
            commanded = result.selected_action_scale.apply(response_action)
            expected_pose = observation.pose.applied(commanded)
            if not np.allclose(
                expected_pose.values,
                result.gate.next_pose.values,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"control gate pose is not bound to its response: {session.session_id}"
                )
            if result.post_action is not None and (
                result.post_action.raw_proposed_action != response_action
                or result.post_action.commanded_action != commanded
            ):
                raise ValueError(
                    f"post-action evidence is not bound to its response: {session.session_id}"
                )
            if result.post_action is not None:
                actual = action_between(observation.pose, result.post_action.pose)
                if (
                    not np.allclose(
                        actual.values,
                        result.post_action.actual_action.values,
                        rtol=0.0,
                        atol=1e-10,
                    )
                ):
                    raise ValueError(
                        f"post-action realization is inconsistent: {session.session_id}"
                    )
                try:
                    result.post_action.validate_derived_evidence(
                        commanded,
                        actual,
                        state.execution_policy,
                    )
                except ValueError as error:
                    raise ValueError(
                        f"post-action realization is inconsistent: {session.session_id}"
                    ) from error
        shadow = session.load_shadow() if session.shadow_path.is_file() else None
        shadow_safety = (
            session.load_shadow_safety()
            if session.shadow_safety_path.is_file()
            else None
        )
        return cls(
            state,
            captured_observation,
            captured_response,
            result,
            shadow,
            shadow_safety,
        )

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def status(self) -> ControlResultStatus:
        return self.result.status

    @property
    def is_applied(self) -> bool:
        return self.status is ControlResultStatus.APPLIED and self.post_action_pose is not None

    @property
    def post_action_pose(self) -> DroidPose | None:
        return self.result.post_action.pose if self.result.post_action else None

    @property
    def timing(self) -> ControlStepTiming:
        observation = self.observation
        response = self.response
        if self.result.insertion_trial_refresh is not None:
            if (
                self.state.execution_policy is ControlExecutionPolicy.DIRECT
                and self.state.insertion_target_policy is None
                and self.state.active_drive_target is not None
            ):
                if self.state.previous_session_id is None:
                    observation, response = (
                        self.result.insertion_trial_refresh.authorize_initial_contact_grasp(
                            observation,
                            response,
                            self.state.require_safety_snapshot(),
                            MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
                        )
                    )
                else:
                    observation, response = (
                        self.result.insertion_trial_refresh.authorize_target_relative(
                            observation,
                            response,
                            self.state.require_safety_snapshot(),
                            self.state.active_drive_target,
                            MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
                        )
                    )
            else:
                observation, response = self.result.insertion_trial_refresh.authorize(
                    observation,
                    response,
                    self.state.require_safety_snapshot(),
                )
        return ControlStepTiming.from_step(
            observation,
            response,
            self.result,
        )

    def to_dict(self) -> dict[str, Any]:
        tracking = self.result.post_action.tracking if self.result.post_action else None
        return {
            "session": self.session_id,
            "observation_id": self.observation.observation_id,
            "status": self.status.value,
            "timing": self.timing.to_dict(),
            "selected_action_scale": (
                self.result.selected_action_scale.to_dict()
                if self.result.selected_action_scale is not None
                else None
            ),
            "translation_cosine": (
                tracking.translation_cosine if tracking is not None else None
            ),
            "rotation_cosine": tracking.rotation_cosine if tracking is not None else None,
            "contact_force_newtons": (
                self.result.post_action.contact_force_newtons
                if self.result.post_action is not None
                else None
            ),
            "plug_position": (
                list(self.result.post_action.plug_position)
                if self.result.post_action is not None
                and self.result.post_action.plug_position is not None
                else None
            ),
            "plug_attached": (
                self.result.post_action.plug_attached
                if self.result.post_action is not None
                else None
            ),
            "execution_error": self.result.execution_error,
            "execution_interlock": (
                self.result.execution_interlock.to_dict()
                if self.result.execution_interlock is not None
                else None
            ),
            "shadow_search": self.shadow.to_dict() if self.shadow is not None else None,
            "shadow_safety": (
                self.shadow_safety.to_dict()
                if self.shadow_safety is not None
                else None
            ),
        }


@dataclass(frozen=True)
class IncompleteControlStepSummary:
    session_id: str
    failure: OrchestrationFailure

    def __post_init__(self) -> None:
        validate_recording_id(self.session_id)

    @property
    def status(self) -> IncompleteStepStatus:
        return IncompleteStepStatus.ORCHESTRATION_FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session_id,
            "observation_id": None,
            "status": self.status.value,
            "timing": None,
            "selected_action_scale": None,
            "translation_cosine": None,
            "rotation_cosine": None,
            "contact_force_newtons": None,
            "failure": self.failure.to_dict(),
        }


RolloutStep = Union[ControlStepSummary, IncompleteControlStepSummary]


def _reference_target_pose(
    data_root: Path,
    reference_recording: str,
    observation: ControlObservation,
) -> DroidPose:
    target = observation.target_frame
    if (
        target.is_absolute()
        or len(target.parts) != 4
        or target.parts[:3] != ("recordings", reference_recording, "wrist")
        or target.suffix != ".png"
        or not target.stem.startswith("frame_")
    ):
        raise ValueError("control target frame path is invalid")
    try:
        target_index = int(target.stem.removeprefix("frame_"))
    except ValueError as error:
        raise ValueError("control target frame path is invalid") from error
    steps_path = data_root / "recordings" / reference_recording / "steps.jsonl"
    reference_pose = None
    for line in steps_path.read_text().splitlines():
        payload = json.loads(line)
        if payload.get("index") == target_index:
            reference_pose = DroidPose(tuple(payload["end_effector_pose"]))
            break
    if reference_pose is None:
        raise ValueError("control target pose is missing from recording telemetry")
    return reference_pose


def _target_pose(
    data_root: Path,
    reference_recording: str,
    observation: ControlObservation,
) -> DroidPose:
    # The frame authenticates the reference target selection. The pose in the
    # observation is the exact target actually presented to the controller and
    # may include a validated task-space offset from that reference frame.
    _reference_target_pose(data_root, reference_recording, observation)
    if observation.target_pose is None:
        raise ValueError("control observation exact target pose is missing")
    return observation.target_pose


def _contact_grasp_target_policy(
    steps: Sequence[ControlStepSummary],
) -> ContactGraspTargetPolicy | None:
    policies = tuple(step.state.contact_grasp_target_policy for step in steps)
    if not any(policy is not None for policy in policies):
        return None
    if any(policy is None for policy in policies) or len(set(policies)) != 1:
        raise ValueError("contact-grasp target policy changed")
    policy = policies[0]
    if not isinstance(policy, ContactGraspTargetPolicy):
        raise ValueError("contact-grasp target policy is invalid")
    return policy


def _contact_grasp_target_steps(
    steps: Sequence[ControlStepSummary],
) -> tuple[ContactGraspTargetStep, ...]:
    return tuple(
        ContactGraspTargetStep(
            step.observation,
            step.state.plug_attached,
        )
        for step in steps
    )


def _contact_grasp_retained_direction(
    steps: Sequence[ControlStepSummary],
) -> tuple[float, float, float] | None:
    policy = _contact_grasp_target_policy(steps)
    if policy is None or not policy.requires_directional_transport_progress:
        return None
    acquisition_step = next(
        (
            step
            for step in steps
            if not step.state.plug_attached
            and step.result.post_action is not None
            and step.result.post_action.plug_attached
        ),
        None,
    )
    if acquisition_step is None:
        return (0.0, 0.0, 0.0)
    acquisition_target = acquisition_step.observation.target_pose
    retained_target = steps[-1].observation.target_pose
    if acquisition_target is None or retained_target is None:
        return (0.0, 0.0, 0.0)
    return tuple(
        retained_target.values[axis] - acquisition_target.values[axis]
        for axis in range(3)
    )


@dataclass(frozen=True)
class ControlRolloutReport:
    rollout_id: str
    reference_recording: str
    seed: int
    proposal: Path
    requested_steps: int
    steps: tuple[RolloutStep, ...]
    target_pose: DroidPose | None
    orchestration_failure: OrchestrationFailure | None = None
    reference_task: str | None = None
    predecessor_session_id: str | None = None
    insertion_target: InsertionTarget | None = None

    def __post_init__(self) -> None:
        validate_recording_id(self.rollout_id)
        validate_recording_id(self.reference_recording)
        if self.predecessor_session_id is not None:
            validate_recording_id(self.predecessor_session_id)
        if (
            self.seed < 0
            or not self.proposal.is_absolute()
            or isinstance(self.requested_steps, bool)
            or not len(self.steps)
            <= self.requested_steps
            <= MAX_CONTROL_ROLLOUT_STEPS
            or len({step.session_id for step in self.steps}) != len(self.steps)
        ):
            raise ValueError("control rollout contract is invalid")
        incomplete = [
            index
            for index, step in enumerate(self.steps)
            if isinstance(step, IncompleteControlStepSummary)
        ]
        if incomplete and (
            incomplete != [len(self.steps) - 1]
            or self.orchestration_failure is None
            or self.steps[incomplete[0]].failure != self.orchestration_failure
        ):
            raise ValueError("control rollout has an invalid incomplete step")
        complete = self.complete_steps
        if not complete:
            if self.target_pose is not None:
                raise ValueError("control rollout target has no complete observation")
            return
        if self.target_pose is None:
            raise ValueError("control rollout target pose is missing")
        for index, step in enumerate(complete):
            expected_previous = (
                self.steps[index - 1].session_id
                if index
                else self.predecessor_session_id
            )
            if (
                step.state.previous_session_id != expected_previous
                or step.state.reference_recording != self.reference_recording
                or step.state.seed != self.seed
                or step.observation.expected_proposal != self.proposal
            ):
                raise ValueError("control rollout session chain or provenance is invalid")
        observations = tuple(step.observation for step in complete)
        if len({observation.observation_id for observation in observations}) != len(
            observations
        ):
            raise ValueError("control rollout observation IDs must be unique")
        target_frames = tuple(observation.target_frame for observation in observations)
        direct_contact_grasp = (
            self.reference_task == INSERTION_TASK_ID
            and all(
                step.state.execution_policy is ControlExecutionPolicy.DIRECT
                and step.state.insertion_target_policy is None
                and step.state.active_drive_target is not None
                for step in complete
            )
        )
        insertion_positions = (
            tuple(
                step.state.resolved_insertion_rollout_position()
                for step in complete
            )
            if self.reference_task == INSERTION_TASK_ID
            and all(
                step.state.insertion_target_policy is not None
                and is_insertion_rollout_policy(step.state.execution_policy)
                for step in complete
            )
            else ()
        )
        bounded_insertion_rollout = bool(insertion_positions)
        if bounded_insertion_rollout and (
            tuple(position.step_index for position in insertion_positions)
            != tuple(range(1, len(insertion_positions) + 1))
            or any(
                position.maximum_steps != self.requested_steps
                for position in insertion_positions
            )
        ):
            raise ValueError("control rollout insertion positions are invalid")
        if direct_contact_grasp:
            maximum_steps = MAXIMUM_CONTACT_GRASP_ACTIONS
        elif bounded_insertion_rollout:
            maximum_steps = self.requested_steps
        else:
            maximum_steps = STANDARD_MAX_CONTROL_ROLLOUT_STEPS
        if self.requested_steps > maximum_steps:
            raise ValueError("control rollout exceeds its task-specific action cap")
        if direct_contact_grasp:
            for previous, current in zip(complete, complete[1:]):
                expected_target = previous.contact_grasp_drive_target()
                previous_post_action = previous.result.post_action
                if (
                    previous_post_action is None
                    or current.state.active_drive_target != expected_target
                ):
                    raise ValueError(
                        "contact-grasp follow-up drive target is invalid"
                    )
                current.state.require_safety_snapshot().validate_followup_continuity(
                    previous_post_action.require_safety_snapshot(),
                    expected_target,
                    maximum_gripper_error_meters=(
                        MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
                    ),
                )
        contact_grasp_policy = _contact_grasp_target_policy(complete)
        if contact_grasp_policy is not None and not direct_contact_grasp:
            raise ValueError("contact-grasp target policy is outside its task")
        if contact_grasp_policy is not None:
            current_contact_grasp = True
            contact_grasp_policy.validate_schedule(
                _contact_grasp_target_steps(complete),
                require_initial=self.predecessor_session_id is None,
            )
        else:
            current_contact_grasp = False
        if not current_contact_grasp:
            if self.reference_task == GRASP_TASK_ID or direct_contact_grasp:
                target_indices = tuple(
                    int(frame.stem.removeprefix("frame_"))
                    for frame in target_frames
                )
                if target_indices != tuple(
                    range(
                        target_indices[0],
                        target_indices[0] + len(target_indices),
                    )
                ):
                    raise ValueError("control rollout target schedule is invalid")
            elif (
                not (
                    self.reference_task == INSERTION_TASK_ID
                    and all(
                        step.state.insertion_target_policy is not None
                        for step in complete
                    )
                )
                and len(set(target_frames)) != 1
            ):
                raise ValueError("control rollout changed its target frame")
        if not current_contact_grasp:
            initial_warmup = observations[0].warmup_frames
            if initial_warmup < 4 or any(
                observation.warmup_frames != initial_warmup + index
                for index, observation in enumerate(observations)
            ):
                raise ValueError("control rollout warm-up sequence is invalid")
        if any(
            current.captured_at_unix_seconds <= previous.captured_at_unix_seconds
            for previous, current in zip(observations, observations[1:])
        ):
            raise ValueError("control rollout capture times are not ordered")
        for previous, current in zip(observations, observations[1:]):
            expected_previous_action = action_between(previous.pose, current.pose)
            if not np.allclose(
                current.previous_action.values,
                expected_previous_action.values,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("control rollout previous-action chain is invalid")
        non_applied = [
            index
            for index, step in enumerate(complete)
            if step.status != ControlResultStatus.APPLIED
        ]
        if non_applied and non_applied != [len(self.steps) - 1]:
            raise ValueError("control rollout continued after a non-applied step")
        shadow_adapters = {
            step.shadow.adapter for step in complete if step.shadow is not None
        }
        if len(shadow_adapters) > 1:
            raise ValueError("control rollout changed its shadow action adapter")

    @property
    def complete_steps(self) -> tuple[ControlStepSummary, ...]:
        return tuple(
            step for step in self.steps if isinstance(step, ControlStepSummary)
        )

    @property
    def applied_steps(self) -> tuple[ControlStepSummary, ...]:
        return tuple(
            step
            for step in self.complete_steps
            if step.status == ControlResultStatus.APPLIED
        )

    @property
    def all_steps_applied(self) -> bool:
        return (
            len(self.applied_steps) == self.requested_steps
            and self.orchestration_failure is None
        )

    def require_all_steps_applied(self) -> None:
        """Fail unless every requested control step reconstructed as applied."""

        if not self.all_steps_applied:
            raise ValueError("control rollout did not apply every requested step")

    @property
    def initial_goal_error(self) -> PoseError | None:
        if not self.complete_steps or self.target_pose is None:
            return None
        return PoseError.between(self.complete_steps[0].observation.pose, self.target_pose)

    @property
    def final_goal_error(self) -> PoseError | None:
        initial = self.initial_goal_error
        if initial is None or self.target_pose is None:
            return None
        final_pose = (
            self.applied_steps[-1].post_action_pose
            if self.applied_steps
            else self.complete_steps[0].observation.pose
        )
        if final_pose is None:
            raise AssertionError("validated applied step has no post-action pose")
        return PoseError.between(final_pose, self.target_pose)

    @property
    def current_wire_authenticated(self) -> bool:
        return bool(self.complete_steps) and not any(
            step.observation.schema != CONTROL_SCHEMA
            or step.response.schema != CONTROL_SCHEMA
            for step in self.complete_steps
        )

    @property
    def reach_and_grasp(self) -> ReachAndGraspDecision | None:
        if not self.current_wire_authenticated:
            return None
        initial_state = self.complete_steps[0].state
        if initial_state.plug_position is None:
            return None
        evidence = [
            GraspTaskStep(
                tuple(initial_state.plug_position),
                initial_state.plug_attached,
                True,
                initial_state.collision_detected,
                initial_state.contact_force_newtons,
            )
        ]
        for step in self.complete_steps:
            post_action = step.result.post_action
            if post_action is None or post_action.plug_position is None:
                return None
            evidence.append(
                GraspTaskStep(
                    tuple(post_action.plug_position),
                    post_action.plug_attached,
                    post_action.tracking.passed
                    and post_action.command_realization is not None
                    and post_action.command_realization.passed,
                    post_action.collision_detected,
                    post_action.contact_force_newtons,
                )
            )
        retained_direction = _contact_grasp_retained_direction(
            self.complete_steps
        )
        return (
            evaluate_reach_and_grasp(
                tuple(evidence),
                retained_direction=retained_direction,
            )
            if evidence
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        initial = self.initial_goal_error
        final = self.final_goal_error
        timings = [step.timing for step in self.complete_steps]
        shadows = [step.shadow for step in self.complete_steps if step.shadow is not None]
        shadow_safety = [
            step.shadow_safety
            for step in self.complete_steps
            if step.shadow_safety is not None
        ]
        applied_count = len(self.applied_steps)
        grasp = self.reach_and_grasp
        payload = {
            "schema": ROLLOUT_SCHEMA,
            "rollout_id": self.rollout_id,
            "reference_recording": self.reference_recording,
            "reference_task": self.reference_task,
            "seed": self.seed,
            "proposal": str(self.proposal),
            "requested_steps": self.requested_steps,
            "attempted_steps": len(self.steps),
            "complete_steps": len(self.complete_steps),
            "applied_steps": applied_count,
            "all_steps_applied": self.all_steps_applied,
            "orchestration_failure": (
                self.orchestration_failure.to_dict()
                if self.orchestration_failure is not None
                else None
            ),
            "mean_observation_age_seconds": (
                fsum(timing.observation_age_seconds for timing in timings)
                / len(timings)
                if timings
                else None
            ),
            "maximum_observation_age_seconds": (
                max(timing.observation_age_seconds for timing in timings)
                if timings
                else None
            ),
            "mean_inference_latency_seconds": (
                fsum(timing.inference_latency_seconds for timing in timings)
                / len(timings)
                if timings
                else None
            ),
            "mean_command_age_seconds": (
                fsum(timing.command_age_seconds for timing in timings) / len(timings)
                if timings
                else None
            ),
            "shadow_searches": len(shadows),
            "shadow_adapter": str(shadows[0].adapter) if shadows else None,
            "shadow_gate_passes": sum(shadow.passes_shadow_gate for shadow in shadows),
            "shadow_safety_evaluations": len(shadow_safety),
            "shadow_safety_passes": sum(evidence.passed for evidence in shadow_safety),
            "mean_shadow_energy_improvement": (
                fsum(shadow.energy_improvement for shadow in shadows) / len(shadows)
                if shadows
                else None
            ),
            "mean_shadow_planning_seconds": (
                fsum(shadow.planning_seconds for shadow in shadows) / len(shadows)
                if shadows
                else None
            ),
            "initial_goal_error": initial.to_dict() if initial else None,
            "final_goal_error": final.to_dict() if final else None,
            "translation_progress_meters": (
                initial.translation_meters - final.translation_meters
                if initial and final
                else None
            ),
            "rotation_progress_radians": (
                initial.rotation_radians - final.rotation_radians
                if initial and final
                else None
            ),
            "reach_and_grasp": grasp.to_dict() if grasp is not None else None,
            "insertion_target": (
                self.insertion_target.to_dict()
                if self.insertion_target is not None
                else None
            ),
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.predecessor_session_id is not None:
            payload["predecessor_session_id"] = self.predecessor_session_id
        return payload

    @classmethod
    def from_sessions(
        cls,
        data_root: Path,
        rollout_id: str,
        session_ids: Sequence[str],
        *,
        reference_recording: str,
        seed: int,
        proposal: Path,
        requested_steps: int,
        orchestration_failure: OrchestrationFailure | None = None,
        predecessor_session_id: str | None = None,
    ) -> ControlRolloutReport:
        if not session_ids:
            raise ValueError("control rollout sessions must be non-empty")
        summaries: list[RolloutStep] = []
        for index, session_id in enumerate(session_ids):
            session = ControlSession.at(data_root / "control_sessions", session_id)
            try:
                summaries.append(ControlStepSummary.from_session(session))
            except ValueError:
                if orchestration_failure is None or index != len(session_ids) - 1:
                    raise
                summaries.append(
                    IncompleteControlStepSummary(session_id, orchestration_failure)
                )
        complete = tuple(
            step for step in summaries if isinstance(step, ControlStepSummary)
        )
        reference_task = None
        if complete:
            manifest = json.loads(
                (
                    data_root
                    / "recordings"
                    / reference_recording
                    / "manifest.json"
                ).read_text()
            )
            metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
            reference_task = metadata.get("task") if isinstance(metadata, dict) else None
            if reference_task == INSERTION_TASK_ID:
                policy = _contact_grasp_target_policy(complete)
                if policy is not None:
                    recording = data_root / "recordings" / reference_recording
                    previous_target_step = None
                    if predecessor_session_id is not None:
                        predecessor = ControlStepSummary.from_session(
                            ControlSession.at(
                                data_root / "control_sessions",
                                predecessor_session_id,
                            )
                        )
                        if (
                            predecessor.state.reference_recording
                            != reference_recording
                            or predecessor.state.seed != seed
                            or predecessor.observation.expected_proposal != proposal
                        ):
                            raise ValueError(
                                "control rollout predecessor provenance is invalid"
                            )
                        previous_target_step = ContactGraspTargetStep(
                            predecessor.observation,
                            predecessor.state.plug_attached,
                        )
                    policy.validate_reference_schedule(
                        _contact_grasp_target_steps(complete),
                        recording,
                        frame_root=data_root,
                        previous_step=previous_target_step,
                    )
        target_observation = (
            None
            if not complete
            else complete[-1]
            if reference_task in (GRASP_TASK_ID, INSERTION_TASK_ID)
            else complete[0]
        )
        target = (
            _target_pose(data_root, reference_recording, target_observation.observation)
            if target_observation is not None
            else None
        )
        insertion_target = None
        if reference_task == INSERTION_TASK_ID:
            recording_path = data_root / "recordings" / reference_recording
            manifest = json.loads((recording_path / "manifest.json").read_text())
            target_payload = manifest.get("metadata", {}).get("insertion_target")
            if isinstance(target_payload, dict):
                insertion_target = InsertionTarget(
                    tuple(float(value) for value in target_payload["socket_position"]),
                    tuple(float(value) for value in target_payload["insertion_axis"]),
                )
            applied_post_actions = tuple(
                step.result.post_action
                for step in complete
                if step.result is not None
                and step.result.status is ControlResultStatus.APPLIED
                and step.result.post_action is not None
            )
            live_targets = tuple(
                post_action.insertion_target
                for post_action in applied_post_actions
                if post_action.insertion_target is not None
            )
            if live_targets:
                if (
                    len(live_targets) != len(applied_post_actions)
                    or len(set(live_targets)) != 1
                ):
                    raise ValueError("live insertion target changed during rollout")
                if insertion_target is None or target_observation is None:
                    raise ValueError("live insertion target has no reference target")
                reference_target_pose = _reference_target_pose(
                    data_root,
                    reference_recording,
                    target_observation.observation,
                )
                if target is None:
                    raise AssertionError("validated rollout has no exact target")
                insertion_target = insertion_target.bind_live_target(
                    live_targets[0],
                    tuple(
                        target.values[axis] - reference_target_pose.values[axis]
                        for axis in range(3)
                    ),
                )
        return cls(
            rollout_id=rollout_id,
            reference_recording=reference_recording,
            seed=seed,
            proposal=proposal,
            requested_steps=requested_steps,
            steps=tuple(summaries),
            target_pose=target,
            orchestration_failure=orchestration_failure,
            reference_task=(reference_task if isinstance(reference_task, str) else None),
            predecessor_session_id=predecessor_session_id,
            insertion_target=insertion_target,
        )
