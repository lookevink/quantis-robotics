"""Reset-repeatable live measurement of sub-millimeter insertion control."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import time
from typing import Any, Callable

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_resolution import (
    CONTROL_RESOLUTION_PROTOCOL,
    ControlResolutionResetPhase,
    ControlResolutionFailureEvidence,
    ControlResolutionProtocol,
    ControlResolutionReport,
    ControlResolutionSample,
    ControlResolutionSettlement,
    ControlResolutionSettlementAttempt,
    ControlResolutionSettlementEvidence,
    ControlResolutionSettlementFailure,
    ControlResolutionSettlementTimeoutTrace,
    ControlResolutionMotionTimeout,
    ControlResolutionRollbackTimeout,
    ControlResolutionProbeKind,
    ControlResolutionProbePlan,
    ControlResolutionProbeExecution,
    ControlResolutionProjectionFailure,
    ControlResolutionForwardEvidence,
    ControlResolutionRollbackFailure,
    ControlResolutionRollbackSuccess,
    TrackedErrorSettlement,
    ControlResolutionEndpoint,
    ControlResolutionBaselineEvidence,
    ControlResolutionBaselineAttempt,
    ControlResolutionBaselinePolicy,
    ControlResolutionBaselineTrace,
    ControlResolutionDriveTarget,
    ControlResolutionCaptureIdentity,
    ControlResolutionLoad,
    ControlResolutionMotionTiming,
    ControlResolutionDriveCommand,
    DriveCommandApplied,
    RejectedControlResolutionReset,
    maximum_joint_position_delta,
)
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.control_resolution_baseline import ControlResolutionCaptureSourceIdentity
from jepa_wm.control_safety import SafetyProjectionAttempt
from jepa_wm.direct_safety import ControlSafetySnapshot
from jepa_wm.persistence import write_json_atomic
from jepa_wm.trial_equivalence import (
    ResetEquivalenceMeasurement,
    ResetEquivalenceTolerances,
    TrialResetState,
    validate_reset_equivalence,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from sim.control_context import recording_task
from sim.control_session import CONTROL_ROOT, RECORDING_ROOT, ControlSession
from sim.demo_sequence import Phase
from sim.isaac_control_runtime import (
    LiveContactInterlock,
    LiveControlRuntime,
    live_runtime_for,
    read_control_contact,
    synchronized_insertion_resolution_runtime,
)
from sim.isaac_control_execution import ExecutionSafetyContext, project_control_candidate
from sim.isaac_demo_runtime import (
    ContactReading,
    JointCommand,
    advance_physics_updates,
    advance_simulation_period,
    move_joint_command,
    recording_snapshot,
)
from sim.recording import RecordingLabel, RecordingMoment


def _reset_state(
    pose: Any,
    command: JointCommand,
    runtime: LiveControlRuntime,
    contact: ContactReading,
) -> TrialResetState:
    plug_position, _ = runtime.attachment.world_pose()
    return TrialResetState(
        pose=pose,
        joint_positions=tuple(float(value) for value in command.arm_positions),
        collision_detected=contact.collision_detected,
        contact_force_newtons=contact.force_newtons,
        plug_position=tuple(float(value) for value in plug_position),
        plug_attached=runtime.attachment.attached,
    )


def _capture_reset_state(
    runtime: LiveControlRuntime,
) -> tuple[JointCommand, TrialResetState]:
    """Capture one raw live reset before applying equivalence policy."""

    contact = ContactReading(*read_control_contact(runtime.sensor))
    command = runtime.actuators.actual_command()
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.INITIAL),
        ObservationStage.CABLE_GRASPED,
        command,
        runtime.attachment,
    )
    reset = _reset_state(
        snapshot.end_effector_pose,
        command,
        runtime,
        contact,
    )
    return command, reset


class ControlResolutionResetMismatch(ValueError):
    def __init__(
        self,
        message: str,
        evidence: RejectedControlResolutionReset,
    ) -> None:
        super().__init__(message)
        self.evidence = evidence


def _require_resolution_reset(
    reference: TrialResetState,
    candidate: TrialResetState,
    tolerances: ResetEquivalenceTolerances,
    phase: ControlResolutionResetPhase,
    sample_index: int | None = None,
) -> None:
    try:
        validate_reset_equivalence(reference, candidate, tolerances=tolerances)
    except ValueError as error:
        raise ControlResolutionResetMismatch(
            str(error),
            RejectedControlResolutionReset(
                phase=phase,
                sample_index=sample_index,
                reference=reference,
                candidate=candidate,
                tolerances=tolerances,
            ),
        ) from error


@dataclass
class ResolutionControlInterlock:
    """Abort if contact is unsafe or the configured load state changes."""

    contact: LiveContactInterlock
    runtime: LiveControlRuntime
    expected_attachment: bool

    def observe(self) -> Any:
        reading = self.contact.observe()
        if self.runtime.attachment.attached is not self.expected_attachment:
            raise RuntimeError("insertion control resolution load state changed")
        return reading


class UnstableControlResolutionBaseline(RuntimeError):
    def __init__(self, attempt: ControlResolutionBaselineAttempt) -> None:
        super().__init__("insertion control resolution baseline did not stabilize")
        self.attempt = attempt


class UnsettledControlResolutionTarget(RuntimeError):
    def __init__(self, attempt: ControlResolutionSettlementAttempt) -> None:
        super().__init__(
            "insertion control resolution did not settle within its bounded timeout"
        )
        self.attempt = attempt


class ControlResolutionSettlementTimeout(RuntimeError):
    def __init__(self, evidence: ControlResolutionSettlementFailure) -> None:
        super().__init__(
            "insertion control resolution did not settle within its bounded timeout"
        )
        self.evidence = evidence


class ControlResolutionProjectionRejected(RuntimeError):
    def __init__(self, evidence: ControlResolutionProjectionFailure) -> None:
        reasons = ",".join(
            reason.value for reason in evidence.projection.gate.reasons
        )
        super().__init__(
            "insertion control resolution probe failed safety projection: "
            + reasons
        )
        self.evidence = evidence


def resolution_failure_evidence(
    session_id: str,
    protocol: ControlResolutionProtocol,
    reference_reset: TrialResetState | None,
    samples: tuple[ControlResolutionSample, ...],
    error: Exception,
    load: ControlResolutionLoad,
    baseline: ControlResolutionBaselineEvidence | None,
    capture_identity: ControlResolutionCaptureIdentity,
) -> ControlResolutionFailureEvidence:
    """Bind a runtime failure to its exact acquired resolution evidence."""

    return ControlResolutionFailureEvidence(
        session_id=session_id,
        failed_at_unix_seconds=time(),
        protocol=protocol,
        reference_reset=reference_reset,
        completed_samples=samples,
        error=f"{type(error).__name__}: {error}",
        rejected_reset=(
            error.evidence
            if isinstance(error, ControlResolutionResetMismatch)
            else None
        ),
        load=load,
        baseline=baseline,
        baseline_attempt=(
            error.attempt
            if isinstance(error, UnstableControlResolutionBaseline)
            else None
        ),
        capture_identity=capture_identity,
        settlement_failure=(
            error.evidence
            if isinstance(error, ControlResolutionSettlementTimeout)
            else None
        ),
        projection_failure=(
            error.evidence
            if isinstance(error, ControlResolutionProjectionRejected)
            else None
        ),
    )


async def stabilize_resolution_baseline(
    runtime: LiveControlRuntime,
    interlock: ResolutionControlInterlock,
    policy: ControlResolutionBaselinePolicy,
    simulation_time_seconds: Callable[[], float],
    load: ControlResolutionLoad = ControlResolutionLoad.ATTACHED,
    safety_limits: SimulatorSafetyLimits = SimulatorSafetyLimits(),
) -> tuple[JointCommand, ControlResolutionBaselineEvidence]:
    """Require consecutive stable no-command intervals before the first probe."""

    active_target = runtime.actuators.current_command()
    drive_target = ControlResolutionDriveTarget(
        tuple(float(value) for value in active_target.arm_positions),
        active_target.gripper_width_m,
    )
    interlock.observe()
    command, initial = _capture_reset_state(runtime)
    states = [initial]
    intervals = []
    previous_time = simulation_time_seconds()
    for _ in range(policy.maximum_intervals):
        await advance_simulation_period(
            policy.observation_period_seconds,
            interlock.observe,
        )
        current_time = simulation_time_seconds()
        intervals.append(current_time - previous_time)
        previous_time = current_time
        command, current = _capture_reset_state(runtime)
        states.append(current)
        trace = ControlResolutionBaselineTrace(
            tuple(states),
            tuple(intervals),
            interlock.contact.evidence,
            drive_target,
        )
        if trace.first_qualifying_end(
            policy,
            load,
            safety_limits,
        ) is not None:
            evidence = ControlResolutionBaselineEvidence(
                trace
            )
            return command, evidence
    attempt = ControlResolutionBaselineAttempt(
        ControlResolutionBaselineTrace(
            tuple(states),
            tuple(intervals),
            interlock.contact.evidence,
            drive_target,
        )
    )
    raise UnstableControlResolutionBaseline(attempt)


async def settle_resolution_target(
    runtime: LiveControlRuntime,
    start: JointCommand,
    target: JointCommand,
    interlock: ResolutionControlInterlock,
    policy: TrackedErrorSettlement,
    tracking_error_cap_radians: float | None = None,
) -> ControlResolutionSettlementEvidence:
    """Wait for consecutive command-relative tracking passes within a bound."""

    requested_motion = maximum_joint_position_delta(
        tuple(float(value) for value in start.arm_positions),
        tuple(float(value) for value in target.arm_positions),
    )
    required_error = policy.maximum_tracking_error(
        requested_motion,
        tracking_error_cap_radians,
    )
    passing_errors: list[float] = []
    tracking_errors: list[float] = []
    for update_count in range(1, policy.maximum_updates + 1):
        await advance_physics_updates(1, interlock.observe)
        actual = runtime.actuators.actual_command()
        tracking_error = maximum_joint_position_delta(
            tuple(float(value) for value in actual.arm_positions),
            tuple(float(value) for value in target.arm_positions),
        )
        tracking_errors.append(tracking_error)
        if tracking_error <= required_error:
            passing_errors.append(tracking_error)
        else:
            passing_errors.clear()
        if len(passing_errors) >= policy.required_consecutive_updates:
            return ControlResolutionSettlementEvidence(
                requested_joint_motion_radians=requested_motion,
                required_tracking_error_radians=required_error,
                updates_used=update_count,
                passing_tracking_errors_radians=tuple(passing_errors),
            )
    raise UnsettledControlResolutionTarget(
        ControlResolutionSettlementAttempt(
            requested_joint_motion_radians=requested_motion,
            required_tracking_error_radians=required_error,
            tracking_errors_radians=tuple(tracking_errors),
            final_joint_positions=tuple(
                float(value) for value in actual.arm_positions
            ),
        )
    )


async def settle_resolution_motion(
    runtime: LiveControlRuntime,
    start: JointCommand,
    target: JointCommand,
    interlock: ResolutionControlInterlock,
    settlement: ControlResolutionSettlement,
    tracking_error_cap_radians: float | None = None,
) -> ControlResolutionSettlementEvidence | None:
    if isinstance(settlement, TrackedErrorSettlement):
        return await settle_resolution_target(
            runtime,
            start,
            target,
            interlock,
            settlement,
            tracking_error_cap_radians,
        )
    if tracking_error_cap_radians is not None:
        raise ValueError("fixed settlement cannot carry a tracking error cap")
    await advance_physics_updates(settlement.updates, interlock.observe)
    return None


async def execute_resolution_probe_motion(
    runtime: LiveControlRuntime,
    start: JointCommand,
    target: JointCommand,
    probe: ControlResolutionProbePlan,
    motion_period_seconds: float,
    interlock: ResolutionControlInterlock,
) -> None:
    """Advance a true zero probe or apply one drive-only nonzero command."""

    if not probe.applies_drive_command:
        await advance_simulation_period(
            motion_period_seconds,
            interlock.observe,
        )
        return
    await move_joint_command(
        runtime.actuators,
        start,
        target,
        runtime.attachment,
        frame_count=1,
        phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
        stage=ObservationStage.CABLE_GRASPED,
        recorder=None,
        sample_period_seconds=motion_period_seconds,
        observe_safety=interlock.observe,
    )


@dataclass(frozen=True)
class ResolutionDriveTargetRecovery:
    protocol: ControlResolutionProtocol
    probe: ControlResolutionProbePlan
    drive_target: ControlResolutionDriveTarget
    reference_reset: TrialResetState

    @property
    def command(self) -> JointCommand:
        return JointCommand(
            np.asarray(self.drive_target.joint_positions, dtype=np.float64),
            self.drive_target.gripper_width_m,
        )

    @property
    def settlement_target(self) -> JointCommand:
        return JointCommand(
            np.asarray(
                self.probe.rollback_joint_target(
                    self.drive_target,
                    self.reference_reset,
                ),
                dtype=np.float64,
            ),
            self.drive_target.gripper_width_m,
        )

    def drive_command(self, start: JointCommand) -> ControlResolutionDriveCommand:
        return self.probe.drive_command(
            self.protocol.safe_joint_motion_period(
                tuple(float(value) for value in start.arm_positions),
                self.drive_target.joint_positions,
                self.protocol.motion_period_for(
                    self.probe.requested_translation_meters
                ),
            )
            if self.probe.applies_drive_command
            else None
        )


async def recover_resolution_drive_target(
    runtime: LiveControlRuntime,
    start: JointCommand,
    interlock: ResolutionControlInterlock,
    context: ResolutionDriveTargetRecovery,
) -> ControlResolutionRollbackSuccess | ControlResolutionRollbackFailure:
    """Restore the persisted drive target after an interrupted probe."""

    start_positions = tuple(float(value) for value in start.arm_positions)
    drive_target = context.command
    settlement_target = context.settlement_target
    drive_command_evidence = context.drive_command(start)
    try:
        if isinstance(drive_command_evidence, DriveCommandApplied):
            await move_joint_command(
                runtime.actuators,
                start,
                drive_target,
                runtime.attachment,
                frame_count=1,
                phase=RecordingLabel(RecordingMoment.SETTLE, Phase.READY),
                stage=ObservationStage.CABLE_GRASPED,
                recorder=None,
                sample_period_seconds=drive_command_evidence.period_seconds,
                observe_safety=interlock.observe,
            )
        else:
            require_resolution_drive_target(runtime, context.drive_target)
        settlement_evidence = await settle_resolution_motion(
            runtime,
            start,
            settlement_target,
            interlock,
            context.protocol.settlement,
            (
                context.protocol.settlement.rollback_tracking_error_cap_radians
                if isinstance(
                    context.protocol.settlement,
                    TrackedErrorSettlement,
                )
                else None
            ),
        )
        if settlement_evidence is None:
            raise RuntimeError("tracked rollback produced no settlement evidence")
        require_resolution_drive_target(
            runtime,
            context.drive_target,
        )
        _, reset = _capture_reset_state(runtime)
        _require_resolution_reset(
            context.reference_reset,
            reset,
            context.protocol.reset_tolerances,
            ControlResolutionResetPhase.ROLLBACK,
            context.probe.sample_index,
        )
        return ControlResolutionRollbackSuccess(
            start_positions,
            drive_command_evidence,
            settlement_evidence,
            interlock.contact.evidence,
            reset,
        )
    except Exception as error:
        try:
            require_resolution_drive_target(
                runtime,
                context.drive_target,
            )
            target_error = ""
        except Exception as verification_error:
            target_error = (
                "; drive target verification failed: "
                f"{type(verification_error).__name__}: {verification_error}"
            )
        return ControlResolutionRollbackFailure(
            start_positions,
            drive_command_evidence,
            (
                error.attempt
                if isinstance(error, UnsettledControlResolutionTarget)
                else None
            ),
            interlock.contact.evidence,
            f"{type(error).__name__}: {error}{target_error}",
        )


def _capture_endpoint(
    runtime: LiveControlRuntime,
) -> tuple[JointCommand, ControlResolutionEndpoint]:
    collision_detected, contact_force = read_control_contact(runtime.sensor)
    command = runtime.actuators.actual_command()
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.SETTLE, Phase.READY),
        ObservationStage.CABLE_GRASPED,
        command,
        runtime.attachment,
    )
    plug_position, _ = runtime.attachment.world_pose()
    return command, ControlResolutionEndpoint(
        snapshot.end_effector_pose,
        ControlSafetySnapshot(
            joint_positions=tuple(float(value) for value in command.arm_positions),
            gripper_width_m=command.gripper_width_m,
            plug_position=tuple(float(value) for value in plug_position),
            contact_force_newtons=contact_force,
            collision_detected=collision_detected,
            plug_attached=runtime.attachment.attached,
        ),
    )


def resolution_settlement_target(
    probe: ControlResolutionProbePlan,
    selected: Any | None,
    live_start: JointCommand,
) -> JointCommand:
    """Resolve the joint reference used to evaluate probe settlement."""

    selected_positions = (
        tuple(float(value) for value in selected.solved_pose.arm_positions)
        if selected is not None
        else None
    )
    positions = probe.settlement_joint_target(
        tuple(float(value) for value in live_start.arm_positions),
        selected_positions,
    )
    return JointCommand(
        np.asarray(positions, dtype=np.float64),
        (
            selected.solved_pose.gripper_width_m
            if selected is not None
            else live_start.gripper_width_m
        ),
    )


def resolution_probe_observation(
    captured: ControlObservation,
    live_pose: DroidPose,
    captured_at_unix_seconds: float,
) -> ControlObservation:
    """Bind one probe safety decision to its live pose and timestamp."""

    return replace(
        captured,
        captured_at_unix_seconds=captured_at_unix_seconds,
        pose=live_pose,
    )


def require_resolution_drive_target(
    runtime: LiveControlRuntime,
    drive_target: ControlResolutionDriveTarget,
) -> JointCommand:
    """Read and verify the active controller target without changing it."""

    active = runtime.actuators.current_command()
    drive_target.validate_active(
        tuple(float(value) for value in active.arm_positions),
        active.gripper_width_m,
    )
    return active


async def measure_insertion_control_resolution(
    session_id: str,
    load: ControlResolutionLoad | str = ControlResolutionLoad.ATTACHED,
    protocol: ControlResolutionProtocol = CONTROL_RESOLUTION_PROTOCOL,
) -> dict[str, Any]:
    """Run bounded retreat probes, returning to one verified reset after each."""

    load = ControlResolutionLoad(load)

    import omni.kit.app
    import omni.timeline
    import omni.usd

    session = ControlSession.at(CONTROL_ROOT, session_id)
    observation, state = session.load_capture()
    if (
        state.execution_policy
        is not ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT
        or recording_task(RECORDING_ROOT / state.reference_recording)
        != INSERTION_TASK_ID
        or observation.target_pose is None
        or session.response_path.exists()
        or session.execution_path.exists()
        or session.result_path.exists()
    ):
        raise ValueError("control session is not a fresh resolution measurement")
    report_path = session.path / "control_resolution.json"
    started_path = session.path / "control_resolution_started.json"
    failure_path = session.path / "control_resolution_failure.json"
    if report_path.exists() or failure_path.exists():
        raise ValueError("control resolution measurement already finished")
    try:
        with started_path.open("x", encoding="utf-8") as output:
            output.write('{"diagnostic_only":true,"production_authority_granted":false}\n')
    except FileExistsError as error:
        raise ValueError("control resolution measurement already started") from error

    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(session_id, stage)
    if runtime is None:
        raise RuntimeError("live insertion runtime was lost before measurement")
    timeline = omni.timeline.get_timeline_interface()
    advance = omni.kit.app.get_app().next_update_async
    limits = protocol.safety_limits
    samples: list[ControlResolutionSample] = []
    reference_reset: TrialResetState | None = None
    baseline: ControlResolutionBaselineEvidence | None = None
    try:
        synchronized = await synchronized_insertion_resolution_runtime(
            runtime,
            timeline,
            advance,
            state.require_safety_snapshot(),
            limits,
            operation="insertion control resolution synchronization",
        )
        runtime = synchronized.runtime
        if load is ControlResolutionLoad.UNLOADED:
            runtime.attachment.remove_load_for_diagnostic()
        timeline.play()
        if protocol.baseline_policy is None:
            raise RuntimeError("current control resolution has no baseline policy")
        baseline_interlock = ResolutionControlInterlock(
            LiveContactInterlock(
                runtime.sensor,
                limits.maximum_contact_force_newtons,
                "insertion control resolution baseline",
            ),
            runtime,
            load.plug_attached,
        )
        _, baseline = await stabilize_resolution_baseline(
            runtime,
            baseline_interlock,
            protocol.baseline_policy,
            timeline.get_current_time,
            load,
            limits,
        )
        baseline.validate(protocol.baseline_policy, load, limits)
        if baseline.drive_target is None:
            raise RuntimeError("control resolution baseline lost its drive target")
        baseline_command = JointCommand(
            np.asarray(baseline.drive_target.joint_positions, dtype=np.float64),
            baseline.drive_target.gripper_width_m,
        )
        reference_reset = baseline.reference_reset
        captured_reset = ControlSession.trial_context(observation, state).reset
        if load is ControlResolutionLoad.UNLOADED:
            captured_reset = replace(captured_reset, plug_attached=False)
        _require_resolution_reset(
            captured_reset,
            baseline.initial_reset,
            protocol.capture_tolerances,
            ControlResolutionResetPhase.CAPTURE_TO_BASELINE,
        )

        for probe in protocol.probe_plans:
            index = probe.sample_index
            magnitude = probe.requested_translation_meters
            require_resolution_drive_target(runtime, baseline.drive_target)
            motion_period_seconds = protocol.motion_period_for(magnitude)
            start_command, start_reset = _capture_reset_state(
                runtime,
            )
            _require_resolution_reset(
                reference_reset,
                start_reset,
                protocol.reset_tolerances,
                ControlResolutionResetPhase.SAMPLE_START,
                index,
            )
            commanded = protocol.probe_action(
                start_reset.pose,
                observation.target_pose,
                magnitude,
            )
            sample_time = time()
            sample_observation = resolution_probe_observation(
                observation,
                start_reset.pose,
                sample_time,
            )
            proposal = ProposedControl(
                observation_id=observation.observation_id,
                created_at_unix_seconds=sample_time,
                actions=(commanded, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
                proposal=observation.expected_proposal,
            )
            safety = ExecutionSafetyContext(
                sample_observation,
                start_command,
                start_reset.joint_positions,
                start_reset.contact_force_newtons,
                start_reset.collision_detected,
                limits,
                control_period_seconds=motion_period_seconds,
            )
            if probe.kind is ControlResolutionProbeKind.HOLD:
                zero_joint_positions = tuple(
                    float(value) for value in start_command.arm_positions
                )
                gate = safety.evaluate(
                    proposal,
                    zero_joint_positions,
                    now_unix_seconds=sample_time,
                )
                maximum_joint_delta = maximum_joint_position_delta(
                    zero_joint_positions,
                    start_reset.joint_positions,
                )
                projection = SafetyProjectionAttempt(
                    DroidActionScale.uniform(1.0),
                    gate,
                    maximum_joint_delta,
                    zero_joint_positions,
                )
                selected = None
            else:
                projection, selected = project_control_candidate(
                    safety,
                    proposal,
                    DroidActionScale.uniform(1.0),
                    now_unix_seconds=sample_time,
                )
                if selected is None:
                    raise ControlResolutionProjectionRejected(
                        ControlResolutionProjectionFailure(
                            probe=probe,
                            recorded_target_pose=observation.target_pose,
                            start_reset=start_reset,
                            commanded_action=commanded,
                            target_pose=start_reset.pose.applied(commanded),
                            projection=projection,
                            motion_period_seconds=motion_period_seconds,
                        )
                    )
            if not projection.gate.passed:
                raise RuntimeError(
                    "insertion control resolution probe failed safety projection: "
                    + ",".join(reason.value for reason in projection.gate.reasons)
                )
            target = resolution_settlement_target(
                probe,
                selected,
                start_command,
            )
            execution = ControlResolutionProbeExecution(
                probe=probe,
                recorded_target_pose=observation.target_pose,
                start_reset=start_reset,
                commanded_action=commanded,
                target_pose=start_reset.pose.applied(commanded),
                projection=projection,
            )
            execution.validate(
                protocol,
                observation.observation_id,
            )

            sample_interlock = ResolutionControlInterlock(
                LiveContactInterlock(
                    runtime.sensor,
                    limits.maximum_contact_force_newtons,
                    f"insertion control resolution sample {index}",
                ),
                runtime,
                load.plug_attached,
            )
            motion_started_at_sim_seconds = timeline.get_current_time()
            await execute_resolution_probe_motion(
                runtime,
                start_command,
                target,
                probe,
                motion_period_seconds,
                sample_interlock,
            )
            try:
                motion_settlement = await settle_resolution_motion(
                    runtime,
                    start_command,
                    target,
                    sample_interlock,
                    protocol.settlement,
                )
            except UnsettledControlResolutionTarget as error:
                motion_failed_at_sim_seconds = timeline.get_current_time()
                failed_actual = runtime.actuators.actual_command()
                recovery_interlock = ResolutionControlInterlock(
                    LiveContactInterlock(
                        runtime.sensor,
                        limits.maximum_contact_force_newtons,
                        f"insertion control resolution recovery {index}",
                    ),
                    runtime,
                    load.plug_attached,
                )
                rollback_outcome = await recover_resolution_drive_target(
                    runtime,
                    failed_actual,
                    recovery_interlock,
                    ResolutionDriveTargetRecovery(
                        protocol,
                        probe,
                        baseline.drive_target,
                        reference_reset,
                    ),
                )
                raise ControlResolutionSettlementTimeout(
                    ControlResolutionMotionTimeout(
                        ControlResolutionSettlementTimeoutTrace(
                            execution=execution,
                            start_joint_positions=tuple(
                                float(value)
                                for value in start_command.arm_positions
                            ),
                            target_joint_positions=tuple(
                                float(value) for value in target.arm_positions
                            ),
                            attempt=error.attempt,
                            interlock=sample_interlock.contact.evidence,
                            drive_command=probe.drive_command(
                                motion_period_seconds
                                if probe.applies_drive_command
                                else None
                            ),
                            timing=ControlResolutionMotionTiming(
                                motion_started_at_sim_seconds,
                                motion_failed_at_sim_seconds,
                            ),
                        ),
                        rollback_outcome,
                    )
                ) from error
            actual, endpoint = _capture_endpoint(runtime)
            motion_settled_at_sim_seconds = timeline.get_current_time()
            sample_interlock.observe()
            if motion_settlement is None:
                raise RuntimeError("tracked motion produced no settlement evidence")
            forward = ControlResolutionForwardEvidence(
                endpoint=endpoint,
                settlement=motion_settlement,
                interlock=sample_interlock.contact.evidence,
                timing=ControlResolutionMotionTiming(
                    motion_started_at_sim_seconds,
                    motion_settled_at_sim_seconds,
                ),
            )
            forward.validate(protocol, execution, load.plug_attached)

            rollback_interlock = ResolutionControlInterlock(
                LiveContactInterlock(
                    runtime.sensor,
                    limits.maximum_contact_force_newtons,
                    f"insertion control resolution rollback {index}",
                ),
                runtime,
                load.plug_attached,
            )
            rollback_target = JointCommand(
                np.asarray(
                    probe.rollback_joint_target(
                        baseline.drive_target,
                        reference_reset,
                    ),
                    dtype=np.float64,
                ),
                baseline.drive_target.gripper_width_m,
            )
            rollback_drive_command = probe.drive_command(
                protocol.safe_joint_motion_period(
                    tuple(float(value) for value in actual.arm_positions),
                    baseline.drive_target.joint_positions,
                    motion_period_seconds,
                )
                if probe.applies_drive_command
                else None
            )
            rollback_started_at_sim_seconds = timeline.get_current_time()
            if isinstance(rollback_drive_command, DriveCommandApplied):
                await move_joint_command(
                    runtime.actuators,
                    actual,
                    baseline_command,
                    runtime.attachment,
                    frame_count=1,
                    phase=RecordingLabel(RecordingMoment.SETTLE, Phase.READY),
                    stage=ObservationStage.CABLE_GRASPED,
                    recorder=None,
                    sample_period_seconds=rollback_drive_command.period_seconds,
                    observe_safety=rollback_interlock.observe,
                )
            try:
                rollback_settlement = await settle_resolution_motion(
                    runtime,
                    actual,
                    rollback_target,
                    rollback_interlock,
                    protocol.settlement,
                    (
                        protocol.settlement.rollback_tracking_error_cap_radians
                        if isinstance(
                            protocol.settlement,
                            TrackedErrorSettlement,
                        )
                        else None
                    ),
                )
            except UnsettledControlResolutionTarget as error:
                rollback_failed_at_sim_seconds = timeline.get_current_time()
                raise ControlResolutionSettlementTimeout(
                    ControlResolutionRollbackTimeout(
                        ControlResolutionSettlementTimeoutTrace(
                            execution=execution,
                            start_joint_positions=tuple(
                                float(value) for value in actual.arm_positions
                            ),
                            target_joint_positions=tuple(
                                float(value)
                                for value in rollback_target.arm_positions
                            ),
                            attempt=error.attempt,
                            interlock=rollback_interlock.contact.evidence,
                            drive_command=rollback_drive_command,
                            timing=ControlResolutionMotionTiming(
                                rollback_started_at_sim_seconds,
                                rollback_failed_at_sim_seconds,
                            ),
                        ),
                        forward,
                    )
                ) from error
            if rollback_settlement is None:
                raise RuntimeError("tracked rollback produced no settlement evidence")
            require_resolution_drive_target(runtime, baseline.drive_target)
            _, rollback_reset = _capture_reset_state(
                runtime,
            )
            _require_resolution_reset(
                reference_reset,
                rollback_reset,
                protocol.reset_tolerances,
                ControlResolutionResetPhase.ROLLBACK,
                index,
            )
            tracked_settlement = protocol.settlement.complete_evidence(
                motion_settlement,
                rollback_settlement,
                rollback_interlock.contact.evidence,
            )
            if tracked_settlement is not None:
                tracked_settlement = replace(
                    tracked_settlement,
                    rollback_drive_command=rollback_drive_command,
                )
            samples.append(
                ControlResolutionSample(
                    index=index,
                    requested_translation_meters=magnitude,
                    start_reset=start_reset,
                    commanded_action=commanded,
                    target_pose=start_reset.pose.applied(commanded),
                    projection=projection,
                    endpoint=endpoint,
                    interlock=sample_interlock.contact.evidence,
                    rollback_reset=rollback_reset,
                    tracked_settlement=tracked_settlement,
                    motion_timing=forward.timing,
                )
            )

        report = ControlResolutionReport(
            session_id=session_id,
            reference_recording=state.reference_recording,
            seed=state.seed,
            context_index=observation.warmup_frames,
            observation_id=observation.observation_id,
            captured_pose=observation.pose,
            recorded_target_pose=observation.target_pose,
            reference_reset=reference_reset,
            samples=tuple(samples),
            protocol=protocol,
            load=load,
            baseline=baseline,
        )
        write_json_atomic(report_path, report.to_dict())
        return report.to_dict()
    except Exception as error:
        failure = resolution_failure_evidence(
            session_id,
            protocol,
            reference_reset,
            tuple(samples),
            error,
            load,
            baseline,
            ControlResolutionCaptureIdentity(
                ControlResolutionCaptureSourceIdentity(
                    state.reference_recording,
                    state.seed,
                    observation.warmup_frames,
                ),
                observation.observation_id,
            ),
        )
        write_json_atomic(failure_path, failure.to_dict())
        raise
    finally:
        timeline.pause()
