"""Gate, execute, and measure one JEPA-WM simulator action."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Callable

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import (
    DROID_FPS,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_policy import (
    ControlExecutionPolicy,
    is_insertion_trial_execution_policy,
)
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlInterlockEvidence,
    ControlGateDecision,
    ControlGateReason,
    ProjectedTargetProgressPolicy,
    SimulatorControlGate,
    SimulatorSafetyLimits,
    SimulatorSafetyState,
    SafetyProjectionAttempt,
    contact_grasp_action_scales,
)
from jepa_wm.control_tracking import (
    evaluate_action_tracking,
    tracking_limits_for_policy,
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.joint_settlement import (
    GripperSettlementCriterion,
    GripperSettlementMeasurement,
    GripperSettlementTrace,
    GripperTrackedJointSettlementAttempt,
    GripperTrackedJointSettlementEvidence,
    JointSettlementAttempt,
    JointSettlementEvidence,
    TrackedJointSettlementPolicy,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.insertion_trial import (
    InsertionTrialDriveEvidence,
    InsertionTrialOutcomeObservation,
    InsertionTrialPostActionEvidence,
    InsertionTrialRollbackEvidence,
    InsertionTrialRollbackFailure,
    InsertionTrialRollbackFailureReason,
    InsertionTrialRollbackOutcome,
)
from jepa_wm.insertion_refresh import (
    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
    InsertionEvaluationRefresh,
)
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    CONTROL_ROOT,
    PostActionEvidence,
)
from sim.demo_sequence import Phase
from sim.control_context import recording_task
from sim.isaac_control_runtime import (
    LiveContactInterlock,
    LiveInsertionInterlock,
    MAXIMUM_INSERTION_GRIPPER_SETTLEMENT_UPDATES,
    bind_live_runtime,
    contact_sensor,
    live_runtime_for,
    pause_control_timeline,
    read_control_contact,
    synchronized_contact_grasp_execution_runtime,
    synchronized_initial_contact_grasp_execution_runtime,
    synchronized_insertion_execution_runtime,
)
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, capture_camera_frame
from sim.grasp_task import evaluate_grasp_acquisition
from sim.isaac_demo_kinematics import SolvedPose, solve_droid_pose, solve_waypoints
from sim.isaac_demo_runtime import (
    Actuators,
    ContactReading,
    JointCommand,
    PlugAttachment,
    create_actuators,
    move_joint_command,
    prepare_plug,
    recording_snapshot,
    resume_live_simulation,
)
from sim.isaac_demo_scene import ROBOT_PATH, world_pose
from sim.recording import RecordingLabel, RecordingMoment, RecordingSnapshot


CONTACT_GRASP_SETTLEMENT_MAXIMUM_ARM_ERROR_RADIANS = 5e-3


def requires_synchronized_evaluation_refresh(
    execution_policy: ControlExecutionPolicy,
    *,
    contact_grasp_execution: bool,
) -> bool:
    """Identify policies reauthorized only after live continuity is proven."""

    return (
        is_insertion_trial_execution_policy(execution_policy)
        or contact_grasp_execution
        or execution_policy is ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
    )


@dataclass(frozen=True)
class ExecutionSafetyContext:
    observation: ControlObservation
    current: JointCommand
    observed_joint_positions: tuple[float, ...]
    contact_force_newtons: float
    collision_detected: bool
    limits: SimulatorSafetyLimits
    control_period_seconds: float = 1.0 / DROID_FPS

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.control_period_seconds)
            or self.control_period_seconds <= 0.0
        ):
            raise ValueError("execution control period is invalid")

    def evaluate(
        self,
        candidate: ProposedControl,
        proposed_joint_positions: tuple[float, ...],
        *,
        now_unix_seconds: float | None = None,
    ) -> ControlGateDecision:
        return SimulatorControlGate(self.limits).evaluate(
            self.observation,
            candidate,
            SimulatorSafetyState(
                observed_joint_positions=self.observed_joint_positions,
                current_joint_positions=tuple(self.current.arm_positions),
                proposed_joint_positions=proposed_joint_positions,
                control_period_seconds=self.control_period_seconds,
                contact_force_newtons=self.contact_force_newtons,
                collision_detected=self.collision_detected,
            ),
            now_unix_seconds=now_unix_seconds,
        )


@dataclass(frozen=True)
class SafeProjection:
    proposal: ProposedControl
    solved_pose: SolvedPose
    decision: ControlGateDecision


@dataclass(frozen=True)
class CapturedPostActionState:
    frame: dict[str, Any]
    command: JointCommand
    collision_detected: bool
    contact_force_newtons: float
    snapshot: RecordingSnapshot


class UnsettledJointCommand(RuntimeError):
    def __init__(
        self,
        attempt: JointSettlementAttempt | GripperTrackedJointSettlementAttempt,
    ) -> None:
        super().__init__("joint command did not settle within its bounded timeout")
        self.attempt = attempt


class InsertionTrialRollbackFailed(RuntimeError):
    def __init__(self, evidence: InsertionTrialRollbackFailure) -> None:
        super().__init__(evidence.error)
        self.evidence = evidence


async def capture_synchronized_post_action(
    actuators: Actuators,
    attachment: PlugAttachment,
    sensor: Any,
    destination: Path,
    *,
    observe_safety: Callable[[], ContactReading] | None = None,
) -> CapturedPostActionState:
    """Capture RGB first, then read telemetry from the resulting physics tick."""

    frame = await capture_camera_frame(
        JEPA_WM_CAMERA_SPECS[0],
        destination,
        observe_safety=observe_safety,
    )
    command = actuators.actual_command()
    collision_detected, contact_force_newtons = read_control_contact(sensor)
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.MOTION, Phase.READY),
        ObservationStage.APPROACHING_CABLE,
        command,
        attachment,
    )
    return CapturedPostActionState(
        frame,
        command,
        collision_detected,
        contact_force_newtons,
        snapshot,
    )


async def settle_joint_command(
    actuators: Actuators,
    target_arm_positions: np.ndarray,
    advance: Callable[[], Any],
    *,
    observe_safety: Callable[[], ContactReading] | None = None,
    maximum_updates: int = 8,
    maximum_arm_error_radians: float = 0.01,
    gripper: GripperSettlementCriterion | None = None,
) -> None:
    """Settle toward a target while polling the interlock after every update."""

    if maximum_updates <= 0 or maximum_arm_error_radians <= 0.0:
        raise ValueError("settling update count must be positive")
    for _ in range(maximum_updates):
        actual = actuators.actual_command()
        arm_settled = (
            np.max(np.abs(actual.arm_positions - target_arm_positions))
            <= maximum_arm_error_radians
        )
        gripper_settled = (
            gripper is None
            or gripper.error(actual.gripper_width_m) <= gripper.maximum_error_meters
        )
        if arm_settled and gripper_settled:
            return
        await advance()
        if observe_safety is not None:
            observe_safety()
    if gripper is not None:
        error_meters = gripper.error(actuators.actual_command().gripper_width_m)
        raise RuntimeError(
            "gripper did not settle within its bounded timeout: "
            f"error_meters={error_meters:.9f}"
        )


async def settle_tracked_joint_command(
    actuators: Actuators,
    start_arm_positions: np.ndarray,
    target_arm_positions: np.ndarray,
    advance: Callable[[], Any],
    policy: TrackedJointSettlementPolicy,
    *,
    observe_safety: Callable[[], ContactReading] | None = None,
) -> JointSettlementEvidence:
    """Settle one tracked command relative to its exact live start."""

    requested_motion = float(np.max(np.abs(target_arm_positions - start_arm_positions)))
    return await _settle_tracked_joint_command(
        actuators,
        target_arm_positions,
        requested_motion,
        advance,
        policy,
        observe_safety=observe_safety,
    )


async def _settle_tracked_joint_command(
    actuators: Actuators,
    target_arm_positions: np.ndarray,
    requested_motion_radians: float,
    advance: Callable[[], Any],
    policy: TrackedJointSettlementPolicy,
    *,
    observe_safety: Callable[[], ContactReading] | None = None,
    gripper: GripperSettlementCriterion | None = None,
) -> JointSettlementEvidence | GripperTrackedJointSettlementEvidence:
    """Own bounded consecutive tracking for forward and rollback motion."""

    required_error = policy.maximum_tracking_error(requested_motion_radians)
    passing_errors: list[float] = []
    tracking_errors: list[float] = []
    gripper_errors: list[float] | None = [] if gripper is not None else None
    passing_gripper_errors: list[float] | None = [] if gripper is not None else None
    for update_count in range(1, policy.maximum_updates + 1):
        await advance()
        if observe_safety is not None:
            observe_safety()
        actual = actuators.actual_command()
        tracking_error = float(
            np.max(np.abs(actual.arm_positions - target_arm_positions))
        )
        tracking_errors.append(tracking_error)
        gripper_error = (
            gripper.error(actual.gripper_width_m) if gripper is not None else None
        )
        if gripper_errors is not None:
            gripper_errors.append(gripper_error)
        if tracking_error <= required_error and (
            gripper_error is None or gripper_error <= gripper.maximum_error_meters
        ):
            passing_errors.append(tracking_error)
            if passing_gripper_errors is not None:
                passing_gripper_errors.append(gripper_error)
        else:
            passing_errors.clear()
            if passing_gripper_errors is not None:
                passing_gripper_errors.clear()
        if len(passing_errors) >= policy.required_consecutive_updates:
            joint_evidence = JointSettlementEvidence(
                requested_motion_radians,
                required_error,
                update_count,
                tuple(passing_errors),
            )
            if gripper is None or passing_gripper_errors is None:
                return joint_evidence
            return GripperTrackedJointSettlementEvidence(
                joint_evidence,
                GripperSettlementMeasurement(
                    gripper.target_width_meters,
                    actual.gripper_width_m,
                    GripperSettlementTrace(
                        tuple(passing_gripper_errors),
                        gripper.maximum_error_meters,
                    ),
                ),
            )
    final = actuators.actual_command()
    attempt_arguments = (
        requested_motion_radians,
        required_error,
        tuple(tracking_errors),
        tuple(float(value) for value in final.arm_positions),
    )
    if gripper_errors is not None and gripper is not None:
        attempt = GripperTrackedJointSettlementAttempt(
            *attempt_arguments,
            GripperSettlementMeasurement(
                gripper.target_width_meters,
                final.gripper_width_m,
                GripperSettlementTrace(
                    tuple(gripper_errors),
                    gripper.maximum_error_meters,
                ),
            ),
        )
    else:
        attempt = JointSettlementAttempt(*attempt_arguments)
    raise UnsettledJointCommand(attempt)


async def _settle_tracked_joint_and_gripper_command(
    actuators: Actuators,
    target: JointCommand,
    requested_motion_radians: float,
    advance: Callable[[], Any],
    policy: TrackedJointSettlementPolicy,
    gripper: GripperSettlementCriterion,
    *,
    observe_safety: Callable[[], ContactReading],
) -> GripperTrackedJointSettlementEvidence:
    evidence = await _settle_tracked_joint_command(
        actuators,
        target.arm_positions,
        requested_motion_radians,
        advance,
        policy,
        observe_safety=observe_safety,
        gripper=gripper,
    )
    if not isinstance(evidence, GripperTrackedJointSettlementEvidence):
        raise RuntimeError("rollback gripper settlement evidence is missing")
    return evidence


@dataclass(frozen=True)
class RollbackSettlementPolicy:
    maximum_arm_error_radians: float = 1e-3
    maximum_gripper_error_meters: float = 1e-3
    required_consecutive_updates: int = 2
    maximum_updates: int = 32

    def __post_init__(self) -> None:
        if (
            self.maximum_arm_error_radians <= 0.0
            or self.maximum_gripper_error_meters <= 0.0
            or self.required_consecutive_updates <= 0
            or self.maximum_updates < self.required_consecutive_updates
        ):
            raise ValueError("rollback settlement policy is invalid")


async def rollback_control_command(
    actuators: Actuators,
    target: JointCommand,
    attachment: PlugAttachment,
    advance: Callable[[], Any],
    *,
    expected_attachment: bool,
    observe_safety: Callable[[], ContactReading] | None = None,
    settlement: RollbackSettlementPolicy = RollbackSettlementPolicy(),
) -> None:
    """Drive back to reset and require bounded consecutive tracking passes."""

    actuators.apply_drive_command(target)
    arm_limit = settlement.maximum_arm_error_radians
    required_updates = settlement.required_consecutive_updates
    maximum_updates = settlement.maximum_updates
    consecutive = 0
    arm_error = float("inf")
    gripper_error = float("inf")
    for _ in range(maximum_updates):
        await advance()
        if observe_safety is not None:
            observe_safety()
        if attachment.attached is not expected_attachment:
            raise RuntimeError(
                "rollback attachment state did not match its captured reset"
            )
        actual = actuators.actual_command()
        arm_error = float(np.max(np.abs(actual.arm_positions - target.arm_positions)))
        gripper_error = abs(actual.gripper_width_m - target.gripper_width_m)
        if (
            arm_error <= arm_limit
            and gripper_error <= settlement.maximum_gripper_error_meters
        ):
            consecutive += 1
            if consecutive >= required_updates:
                return
        else:
            consecutive = 0
    raise RuntimeError(
        "rollback command did not settle: "
        f"arm_error={arm_error:.6f} rad, "
        f"gripper_error={gripper_error:.6f} m"
    )


async def rollback_insertion_trial_command(
    actuators: Actuators,
    drive_target: JointCommand,
    attachment: PlugAttachment,
    advance: Callable[[], Any],
    policy: TrackedJointSettlementPolicy,
    *,
    settlement_target: JointCommand,
    expected_attachment: bool,
    observe_safety: Callable[[], ContactReading],
    interlock_evidence: Callable[[], ControlInterlockEvidence],
    maximum_contact_force_newtons: float,
    maximum_gripper_error_meters: float,
) -> InsertionTrialRollbackEvidence:
    """Restore one stable loaded reset through its distinct controller target."""

    start = actuators.actual_command()
    requested_motion = float(
        np.max(np.abs(start.arm_positions - settlement_target.arm_positions))
    )
    drive_command_accepted = False
    try:
        actuators.apply_drive_command(drive_target)
        drive_command_accepted = True
        evidence = await _settle_tracked_joint_and_gripper_command(
            actuators,
            settlement_target,
            requested_motion,
            advance,
            policy,
            GripperSettlementCriterion(
                settlement_target.gripper_width_m,
                maximum_gripper_error_meters,
            ),
            observe_safety=observe_safety,
        )
        actual = actuators.actual_command()
        if attachment.attached is not expected_attachment:
            raise RuntimeError(
                "rollback attachment state did not match its captured reset"
            )
    except Exception as error:
        actual = actuators.actual_command()
        interlock = interlock_evidence()
        settlement_attempt = (
            error.attempt if isinstance(error, UnsettledJointCommand) else None
        )
        reason = InsertionTrialRollbackFailureReason.from_evidence(
            settlement_attempt=settlement_attempt,
            plug_attached=attachment.attached,
            expected_attachment=expected_attachment,
            interlock=interlock,
            maximum_contact_force_newtons=maximum_contact_force_newtons,
            drive_command_accepted=drive_command_accepted,
        )
        failure = InsertionTrialRollbackFailure(
            tuple(float(value) for value in start.arm_positions),
            tuple(float(value) for value in settlement_target.arm_positions),
            tuple(float(value) for value in actual.arm_positions),
            attachment.attached,
            reason,
            interlock,
            drive_command_accepted,
            f"{type(error).__name__}: {error}",
            settlement_attempt,
            JointDriveTarget(
                tuple(float(value) for value in drive_target.arm_positions),
                drive_target.gripper_width_m,
            ),
        )
        raise InsertionTrialRollbackFailed(failure) from error
    return InsertionTrialRollbackEvidence(
        tuple(float(value) for value in start.arm_positions),
        tuple(float(value) for value in settlement_target.arm_positions),
        tuple(float(value) for value in actual.arm_positions),
        evidence,
        attachment.attached,
        JointDriveTarget(
            tuple(float(value) for value in drive_target.arm_positions),
            drive_target.gripper_width_m,
        ),
    )


def project_control_candidate(
    context: ExecutionSafetyContext,
    proposal: ProposedControl,
    scale: DroidActionScale,
    *,
    solve: Callable[[DroidPose, np.ndarray], SolvedPose] = solve_droid_pose,
    now_unix_seconds: float | None = None,
    target_progress: ProjectedTargetProgressPolicy | None = None,
) -> tuple[SafetyProjectionAttempt, SafeProjection | None]:
    candidate_action = scale.apply(proposal.first_action)
    candidate = proposal.with_actions((candidate_action, *proposal.actions[1:]))
    current_joints = tuple(context.current.arm_positions)
    try:
        candidate_pose = context.observation.pose.applied(candidate_action)
    except ValueError:
        decision = context.evaluate(
            candidate, current_joints, now_unix_seconds=now_unix_seconds
        )
        return SafetyProjectionAttempt(scale, decision, 0.0, current_joints), None

    preliminary = context.evaluate(
        candidate, current_joints, now_unix_seconds=now_unix_seconds
    )
    if target_progress is not None:
        preliminary = target_progress.apply(
            preliminary,
            context.observation.pose,
            context.observation.target_pose,
        )
    if not preliminary.passed:
        return SafetyProjectionAttempt(scale, preliminary, 0.0, current_joints), None
    try:
        solved = solve(candidate_pose, context.current.arm_positions)
    except (RuntimeError, ValueError):
        decision = ControlGateDecision(
            context.observation.observation_id,
            candidate_pose,
            (ControlGateReason.IK_SOLUTION_FAILED,),
        )
        return SafetyProjectionAttempt(scale, decision, 0.0, current_joints), None
    decision = context.evaluate(
        candidate,
        tuple(solved.arm_positions),
        now_unix_seconds=now_unix_seconds,
    )
    maximum_joint_delta = float(
        np.max(np.abs(solved.arm_positions - context.current.arm_positions))
    )
    attempt = SafetyProjectionAttempt(
        scale,
        decision,
        maximum_joint_delta,
        tuple(solved.arm_positions),
    )
    return attempt, (
        SafeProjection(candidate, solved, decision) if decision.passed else None
    )


def select_safe_projection(
    context: ExecutionSafetyContext,
    proposal: ProposedControl,
    *,
    solve: Callable[[DroidPose, np.ndarray], SolvedPose] = solve_droid_pose,
    now_unix_seconds: float | None = None,
    action_scales: tuple[DroidActionScale, ...] = ACTION_SCALES,
    target_progress: ProjectedTargetProgressPolicy | None = None,
) -> tuple[tuple[SafetyProjectionAttempt, ...], SafeProjection | None]:
    attempts = []
    if not action_scales:
        raise ValueError("safety projection requires at least one action scale")
    for action_scale in action_scales:
        attempt, selected = project_control_candidate(
            context,
            proposal,
            action_scale,
            solve=solve,
            now_unix_seconds=now_unix_seconds,
            target_progress=target_progress,
        )
        attempts.append(attempt)
        if selected is not None:
            return tuple(attempts), selected
    return tuple(attempts), None


async def synchronized_actual_command(
    actuators: Actuators,
    timeline: Any,
    advance: Any,
) -> JointCommand:
    """Refresh a stale paused physics tensor before reading articulation state."""

    if not actuators.articulation.is_physics_tensor_entity_valid():
        try:
            resume_live_simulation(timeline)
            await advance()
            return actuators.actual_command()
        finally:
            timeline.pause()
    return actuators.actual_command()


async def apply_control_response(session_id: str) -> dict[str, Any]:
    """Gate and apply only the first response action, then observe and pause."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager

    session = ControlSession.at(CONTROL_ROOT, session_id)
    observation, proposal, persisted_state = session.load()
    insertion_trial_execution = is_insertion_trial_execution_policy(
        persisted_state.execution_policy
    )
    binding = None
    trial_policy = None
    action_scales = ACTION_SCALES
    target_progress = None
    control_period_seconds = 1.0 / DROID_FPS
    if insertion_trial_execution:
        binding = session.load_insertion_trial_binding(proposal)
        trial_policy = binding.require_current_execution()
        action_scales = binding.allowed_projection_scales
        target_progress = trial_policy.projected_progress
        control_period_seconds = trial_policy.control_period_seconds
    contact_insertion_execution = (
        recording_task(
            CONTROL_ROOT.parent / "recordings" / persisted_state.reference_recording
        )
        == INSERTION_TASK_ID
    )
    contact_grasp_execution = (
        contact_insertion_execution
        and persisted_state.execution_policy is ControlExecutionPolicy.DIRECT
        and persisted_state.insertion_target_policy is None
    )
    requires_evaluation_refresh = requires_synchronized_evaluation_refresh(
        persisted_state.execution_policy,
        contact_grasp_execution=contact_grasp_execution,
    )
    if contact_grasp_execution:
        contact_grasp_policy = persisted_state.require_current_contact_grasp_policy()
        execution_action = contact_grasp_policy.action_for_execution(
            proposal.actions,
            plug_attached=persisted_state.plug_attached,
        )
        proposal = proposal.with_actions((execution_action, *proposal.actions[1:]))
        action_scales = contact_grasp_action_scales(
            execution_action,
            attachment_acquired=persisted_state.plug_attached,
            require_directional_transport_progress=(
                contact_grasp_policy.requires_directional_transport_progress
            ),
            require_resolvable_transport=(
                contact_grasp_policy.uses_horizon_transport_action
            ),
        )
    from sim.isaac_unknown_start_shadow import (
        reauthenticate_unknown_start_shadow_session,
    )

    reauthenticate_unknown_start_shadow_session(session_id)
    session.claim_execution()
    stage = omni.usd.get_context().get_stage()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    timeline = omni.timeline.get_timeline_interface()
    runtime = live_runtime_for(session_id, stage)
    if runtime is None:
        if contact_insertion_execution:
            raise RuntimeError("live insertion runtime was lost before execution")
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        attachment = prepare_plug(stage)
        sensor = contact_sensor(stage, create=False)
        bind_live_runtime(session_id, stage, actuators, attachment, sensor)
    else:
        actuators = runtime.actuators
        attachment = runtime.attachment
        sensor = runtime.sensor
    limits = SimulatorSafetyLimits()
    insertion_trial_refresh = None
    insertion_trial_active_drive_target = None
    if contact_insertion_execution:
        if runtime is None:
            raise RuntimeError("live insertion runtime was lost before execution")
        if contact_grasp_execution:
            if persisted_state.active_drive_target is None:
                raise RuntimeError("captured contact-grasp drive target is missing")
            synchronize_contact_grasp = (
                synchronized_initial_contact_grasp_execution_runtime
                if persisted_state.previous_session_id is None
                else synchronized_contact_grasp_execution_runtime
            )
            synchronized = await synchronize_contact_grasp(
                runtime,
                timeline,
                omni.kit.app.get_app().next_update_async,
                persisted_state.require_safety_snapshot(),
                limits,
                expected_active_drive_target=persisted_state.active_drive_target,
                operation="contact-grasp execution synchronization",
                maximum_gripper_error_meters=(
                    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
                ),
            )
        else:
            synchronized = await synchronized_insertion_execution_runtime(
                runtime,
                timeline,
                omni.kit.app.get_app().next_update_async,
                persisted_state.require_safety_snapshot(),
                limits,
                operation="insertion trial synchronization",
            )
        try:
            runtime = synchronized.runtime
            actuators = runtime.actuators
            attachment = runtime.attachment
            sensor = runtime.sensor
            live_state = synchronized.safety
            current = JointCommand(
                np.asarray(live_state.joint_positions),
                live_state.gripper_width_m,
            )
            collision_detected = live_state.collision_detected
            contact_force = live_state.contact_force_newtons
            if synchronized.pose is None:
                raise RuntimeError("live insertion pose was not refreshed")
            refreshed_drive_target = synchronized.active_drive_target
            if refreshed_drive_target is None:
                raise RuntimeError("live insertion drive target was not refreshed")
            if persisted_state.active_drive_target is None:
                raise RuntimeError("captured insertion drive target is missing")
            persisted_state.active_drive_target.validate_active(
                refreshed_drive_target.joint_positions,
                refreshed_drive_target.gripper_width_m,
            )
            if requires_evaluation_refresh:
                insertion_trial_active_drive_target = (
                    persisted_state.active_drive_target
                )
                insertion_trial_refresh = InsertionEvaluationRefresh(
                    time(),
                    live_state,
                    synchronized.pose,
                )
                if contact_grasp_execution:
                    if persisted_state.previous_session_id is None:
                        observation, proposal = (
                            insertion_trial_refresh.authorize_initial_contact_grasp(
                                observation,
                                proposal,
                                persisted_state.require_safety_snapshot(),
                                MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
                            )
                        )
                    else:
                        observation, proposal = (
                            insertion_trial_refresh.authorize_target_relative(
                                observation,
                                proposal,
                                persisted_state.require_safety_snapshot(),
                                persisted_state.active_drive_target,
                                MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
                            )
                        )
                else:
                    observation, proposal = insertion_trial_refresh.authorize(
                        observation,
                        proposal,
                        persisted_state.require_safety_snapshot(),
                    )
        except Exception:
            await pause_control_timeline(
                timeline,
                omni.kit.app.get_app().next_update_async,
            )
            raise
    else:
        current = await synchronized_actual_command(
            actuators,
            timeline,
            omni.kit.app.get_app().next_update_async,
        )
        collision_detected, contact_force = read_control_contact(sensor)
    expected_current = np.asarray(
        persisted_state.current_joint_positions, dtype=np.float64
    )
    try:
        # Capture leaves the stage paused. Insertion trials resume one
        # fully interlocked update and rebind the complete live state before
        # projection; other policies retain their historical command refresh.
        safety = ExecutionSafetyContext(
            observation,
            current,
            tuple(expected_current),
            contact_force,
            collision_detected,
            limits,
            control_period_seconds,
        )
        attempts, selected = select_safe_projection(
            safety,
            proposal,
            action_scales=action_scales,
            target_progress=target_progress,
        )

        candidate = None
        if selected is None:
            solved = None
            decision = attempts[-1].gate
            selected_scale = None
        else:
            candidate = selected.proposal
            solved = selected.solved_pose
            decision = selected.decision
            selected_scale = attempts[-1].scale

        pre_action_time = time()
        pre_action_age_seconds = pre_action_time - observation.captured_at_unix_seconds
        if candidate is not None:
            decision = safety.evaluate(
                candidate,
                tuple(solved.arm_positions),
                now_unix_seconds=pre_action_time,
            )
            if not decision.passed:
                selected_scale = None

        insertion_drive_target = None
        if (
            candidate is not None
            and decision.passed
            and trial_policy is not None
            and insertion_trial_active_drive_target is not None
        ):
            try:
                insertion_drive_target = trial_policy.forward_drive_target(
                    tuple(float(value) for value in solved.arm_positions),
                    solved.gripper_width_m,
                    insertion_trial_active_drive_target,
                    tuple(float(value) for value in current.arm_positions),
                    limits,
                )
            except ValueError:
                decision = ControlGateDecision(
                    decision.observation_id,
                    decision.next_pose,
                    (ControlGateReason.DRIVE_TARGET_INVALID,),
                )
                selected_scale = None
                candidate = None

        status = ControlResultStatus.BLOCKED
        post_action = None
        execution_error = None
        execution_interlock = None
        insertion_trial_rollback = None
        insertion_trial_drive = (
            InsertionTrialDriveEvidence(
                insertion_trial_active_drive_target,
                insertion_drive_target,
            )
            if insertion_trial_active_drive_target is not None
            and insertion_drive_target is not None
            else None
        )
        insertion_trial_settlement_failure = None
        if decision.passed and candidate is not None:
            try:
                if (
                    insertion_trial_execution
                    and insertion_trial_active_drive_target is None
                ):
                    raise RuntimeError("insertion active drive target is missing")
                preserve_contact_grasp_target = (
                    contact_grasp_execution
                    and persisted_state.plug_attached
                    and insertion_trial_active_drive_target is not None
                )
                active_drive_target = (
                    JointCommand(
                        np.asarray(insertion_trial_active_drive_target.joint_positions),
                        insertion_trial_active_drive_target.gripper_width_m,
                    )
                    if (insertion_trial_execution or preserve_contact_grasp_target)
                    and insertion_trial_active_drive_target is not None
                    else current
                )
                live_interlock = LiveInsertionInterlock(
                    LiveContactInterlock(
                        sensor,
                        limits.maximum_contact_force_newtons,
                        "insertion trial",
                        ContactReading(collision_detected, contact_force),
                    ),
                    attachment,
                    persisted_state.plug_attached,
                    "insertion trial",
                )

                async def rollback_current_command() -> (
                    InsertionTrialRollbackOutcome | None
                ):
                    if trial_policy is not None:
                        return await rollback_insertion_trial_command(
                            actuators,
                            active_drive_target,
                            attachment,
                            omni.kit.app.get_app().next_update_async,
                            trial_policy.joint_settlement,
                            settlement_target=current,
                            expected_attachment=persisted_state.plug_attached,
                            observe_safety=live_interlock.observe,
                            interlock_evidence=lambda: live_interlock.evidence,
                            maximum_contact_force_newtons=(
                                limits.maximum_contact_force_newtons
                            ),
                            maximum_gripper_error_meters=(
                                trial_policy.rollback_gripper_error_meters
                            ),
                        )
                    await rollback_control_command(
                        actuators,
                        current,
                        attachment,
                        omni.kit.app.get_app().next_update_async,
                        expected_attachment=persisted_state.plug_attached,
                        observe_safety=(
                            live_interlock.observe
                            if contact_insertion_execution
                            else None
                        ),
                        settlement=(
                            RollbackSettlementPolicy(
                                maximum_updates=(
                                    MAXIMUM_INSERTION_GRIPPER_SETTLEMENT_UPDATES
                                )
                            )
                            if contact_grasp_execution
                            else RollbackSettlementPolicy()
                        ),
                    )
                    return None

                resume_live_simulation(timeline)
                target = (
                    JointCommand(
                        np.asarray(insertion_drive_target.joint_positions),
                        insertion_drive_target.gripper_width_m,
                    )
                    if insertion_drive_target is not None
                    else JointCommand(
                        solved.arm_positions,
                        (
                            insertion_trial_active_drive_target.gripper_width_m
                            if preserve_contact_grasp_target
                            else solved.gripper_width_m
                        ),
                    )
                )
                await move_joint_command(
                    actuators,
                    current,
                    target,
                    attachment,
                    frame_count=1,
                    phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                    stage=ObservationStage.APPROACHING_CABLE,
                    recorder=None,
                    sample_period_seconds=control_period_seconds,
                    observe_safety=(
                        live_interlock.observe if contact_insertion_execution else None
                    ),
                )
                if trial_policy is not None:
                    settlement = await settle_tracked_joint_command(
                        actuators,
                        current.arm_positions,
                        solved.arm_positions,
                        omni.kit.app.get_app().next_update_async,
                        trial_policy.joint_settlement,
                        observe_safety=live_interlock.observe,
                    )
                else:
                    await settle_joint_command(
                        actuators,
                        solved.arm_positions,
                        omni.kit.app.get_app().next_update_async,
                        observe_safety=(
                            live_interlock.observe
                            if contact_insertion_execution
                            else None
                        ),
                        maximum_updates=(
                            MAXIMUM_INSERTION_GRIPPER_SETTLEMENT_UPDATES
                            if contact_grasp_execution
                            else 8
                        ),
                        maximum_arm_error_radians=(
                            CONTACT_GRASP_SETTLEMENT_MAXIMUM_ARM_ERROR_RADIANS
                            if contact_grasp_execution
                            else 0.01
                        ),
                        gripper=(
                            GripperSettlementCriterion(
                                target.gripper_width_m,
                                MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
                            )
                            if contact_grasp_execution
                            else None
                        ),
                    )
                    settlement = None
                captured = await capture_synchronized_post_action(
                    actuators,
                    attachment,
                    sensor,
                    session.path / "post_action.png",
                    observe_safety=(
                        live_interlock.observe if contact_insertion_execution else None
                    ),
                )
                actual = captured.command
                post_collision = captured.collision_detected or (
                    contact_insertion_execution
                    and live_interlock.evidence.collision_detected
                )
                post_force = (
                    max(
                        captured.contact_force_newtons,
                        live_interlock.evidence.maximum_contact_force_newtons,
                    )
                    if contact_insertion_execution
                    else captured.contact_force_newtons
                )
                post_snapshot = captured.snapshot
                actual_action = action_between(
                    observation.pose, post_snapshot.end_effector_pose
                )
                tracking = evaluate_action_tracking(
                    candidate.first_action,
                    actual_action,
                    tracking_limits_for_policy(persisted_state.execution_policy),
                )
                joint_tracking_error = float(
                    np.max(np.abs(actual.arm_positions - solved.arm_positions))
                )
                realized_progress = (
                    trial_policy.realized_progress.evaluate(
                        observation.pose,
                        observation.target_pose,
                        post_snapshot.end_effector_pose,
                    )
                    if trial_policy is not None and observation.target_pose is not None
                    else None
                )
                acquisition_passed = True
                if not insertion_trial_execution:
                    acquisition = evaluate_grasp_acquisition(
                        world_pose(attachment.hand_prim)[0],
                        solve_waypoints()[2].hand_position,
                        actual.gripper_width_m,
                    )
                    acquisition_passed = acquisition.passed
                if (
                    acquisition_passed
                    and joint_tracking_error <= 0.01
                    and tracking.passed
                    and not post_collision
                    and post_force <= limits.maximum_contact_force_newtons
                ):
                    if not insertion_trial_execution:
                        attachment.attach(world_pose(attachment.hand_prim)[0])
                        post_snapshot = recording_snapshot(
                            RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP),
                            ObservationStage.CABLE_GRASPED,
                            actual,
                            attachment,
                        )
                post_action = PostActionEvidence(
                    raw_proposed_action=proposal.first_action,
                    commanded_action=candidate.first_action,
                    actual_action=actual_action,
                    tracking=tracking,
                    pose=post_snapshot.end_effector_pose,
                    joint_positions=tuple(actual.arm_positions),
                    maximum_joint_tracking_error_rad=joint_tracking_error,
                    contact_force_newtons=post_force,
                    collision_detected=post_collision,
                    frame=captured.frame,
                    plug_position=tuple(
                        float(value) for value in post_snapshot.plug_position
                    ),
                    plug_attached=post_snapshot.plug_attached,
                    insertion_trial=(
                        InsertionTrialPostActionEvidence(
                            settlement,
                            realized_progress,
                        )
                        if settlement is not None and realized_progress is not None
                        else None
                    ),
                )
                status = ControlResultStatus.APPLIED
            except Exception as error:
                execution_error = f"{type(error).__name__}: {error}"
                if isinstance(error, UnsettledJointCommand):
                    insertion_trial_settlement_failure = error.attempt
                try:
                    insertion_trial_rollback = await rollback_current_command()
                    status = ControlResultStatus.ROLLED_BACK_EXECUTION
                except Exception as rollback_error:
                    status = ControlResultStatus.ROLLBACK_FAILED
                    if isinstance(rollback_error, InsertionTrialRollbackFailed):
                        insertion_trial_rollback = rollback_error.evidence
                    execution_error += (
                        "; rollback verification failed: "
                        f"{type(rollback_error).__name__}: {rollback_error}"
                    )
                if contact_insertion_execution:
                    execution_interlock = live_interlock.evidence
            if post_action is not None:
                if trial_policy is not None and post_action.insertion_trial is not None:
                    rollback_reason = trial_policy.rollback_reason(
                        post_action.insertion_trial,
                        InsertionTrialOutcomeObservation(
                            joint_tracking_error,
                            tracking.passed,
                            post_force,
                            post_collision,
                            post_snapshot.plug_attached,
                            limits.maximum_contact_force_newtons,
                            persisted_state.plug_attached,
                        ),
                    )
                    outcome_status = ControlResultStatus.from_insertion_rollback_reason(
                        rollback_reason
                    )
                    rollback_status = (
                        None
                        if outcome_status is ControlResultStatus.APPLIED
                        else outcome_status
                    )
                else:
                    rollback_status = None
                    if joint_tracking_error > 0.01 or not tracking.passed:
                        rollback_status = ControlResultStatus.ROLLED_BACK_TRACKING
                    elif (
                        post_collision
                        or post_force > limits.maximum_contact_force_newtons
                    ):
                        rollback_status = ControlResultStatus.ROLLED_BACK_CONTACT
                if rollback_status is not None:
                    try:
                        insertion_trial_rollback = await rollback_current_command()
                        status = rollback_status
                    except Exception as rollback_error:
                        status = ControlResultStatus.ROLLBACK_FAILED
                        if isinstance(rollback_error, InsertionTrialRollbackFailed):
                            insertion_trial_rollback = rollback_error.evidence
                        execution_error = (
                            "rollback verification failed: "
                            f"{type(rollback_error).__name__}: {rollback_error}"
                        )
                if contact_insertion_execution:
                    execution_interlock = live_interlock.evidence

        result = ControlResult(
            status=status,
            session_id=session_id,
            gate=decision,
            projection_attempts=tuple(attempts),
            selected_action_scale=selected_scale,
            observation_age_seconds=pre_action_age_seconds,
            ik_position_error_m=(
                solved.position_error_m if solved is not None else None
            ),
            ik_orientation_error_rad=(
                solved.orientation_error_rad if solved is not None else None
            ),
            pre_action_contact_force_newtons=contact_force,
            post_action=post_action,
            execution_error=execution_error,
            execution_interlock=execution_interlock,
            insertion_trial_refresh=insertion_trial_refresh,
            insertion_trial_drive=insertion_trial_drive,
            insertion_trial_rollback=insertion_trial_rollback,
            insertion_trial_settlement_failure=(insertion_trial_settlement_failure),
        )
        session.write_result(result)
        return result.to_dict()
    finally:
        await pause_control_timeline(
            timeline,
            omni.kit.app.get_app().next_update_async,
        )
