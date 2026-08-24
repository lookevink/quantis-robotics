"""Gate, execute, and measure one JEPA-WM simulator action."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Callable

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DROID_FPS, DroidActionScale, DroidPose, action_between
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
    SimulatorControlGate,
    SimulatorSafetyLimits,
    SimulatorSafetyState,
    SafetyProjectionAttempt,
)
from jepa_wm.control_tracking import evaluate_action_tracking, tracking_limits_for_policy
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    CONTROL_ROOT,
    PostActionEvidence,
)
from sim.demo_sequence import Phase
from sim.isaac_control_runtime import contact_sensor, read_contact
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, capture_camera_frame
from sim.isaac_demo_kinematics import SolvedPose, solve_droid_pose
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    move_joint_command,
    prepare_plug,
    recording_snapshot,
)
from sim.isaac_demo_scene import ROBOT_PATH
from sim.recording import RecordingLabel, RecordingMoment


@dataclass(frozen=True)
class ExecutionSafetyContext:
    observation: ControlObservation
    current: JointCommand
    observed_joint_positions: tuple[float, ...]
    contact_force_newtons: float
    collision_detected: bool
    limits: SimulatorSafetyLimits

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
                control_period_seconds=1.0 / DROID_FPS,
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


def project_control_candidate(
    context: ExecutionSafetyContext,
    proposal: ProposedControl,
    scale: DroidActionScale,
    *,
    solve: Callable[[DroidPose, np.ndarray], SolvedPose] = solve_droid_pose,
    now_unix_seconds: float | None = None,
) -> tuple[SafetyProjectionAttempt, SafeProjection | None]:
    candidate_action = scale.apply(proposal.first_action)
    candidate = ProposedControl(
        observation_id=proposal.observation_id,
        created_at_unix_seconds=proposal.created_at_unix_seconds,
        actions=(candidate_action, *proposal.actions[1:]),
        proposal=proposal.proposal,
    )
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
    return attempt, SafeProjection(candidate, solved, decision) if decision.passed else None


def select_safe_projection(
    context: ExecutionSafetyContext,
    proposal: ProposedControl,
    *,
    solve: Callable[[DroidPose, np.ndarray], SolvedPose] = solve_droid_pose,
    now_unix_seconds: float | None = None,
) -> tuple[tuple[SafetyProjectionAttempt, ...], SafeProjection | None]:
    attempts = []
    for action_scale in ACTION_SCALES:
        attempt, selected = project_control_candidate(
            context,
            proposal,
            action_scale,
            solve=solve,
            now_unix_seconds=now_unix_seconds,
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
        timeline.play()
        try:
            await advance()
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
    session.claim_execution()
    stage = omni.usd.get_context().get_stage()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    actuators = create_actuators(stage, Articulation(ROBOT_PATH))
    timeline = omni.timeline.get_timeline_interface()
    current = await synchronized_actual_command(
        actuators,
        timeline,
        omni.kit.app.get_app().next_update_async,
    )
    expected_current = np.asarray(
        persisted_state.current_joint_positions, dtype=np.float64
    )
    sensor = contact_sensor(stage, create=False)
    try:
        # Follow-up capture leaves the stage synchronized and paused. Read the
        # live articulation/contact state without spending a render tick from
        # the command freshness budget; start time only after the final gate.
        collision_detected, contact_force = read_contact(sensor)
        limits = SimulatorSafetyLimits()
        safety = ExecutionSafetyContext(
            observation,
            current,
            tuple(expected_current),
            contact_force,
            collision_detected,
            limits,
        )
        attempts, selected = select_safe_projection(safety, proposal)

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
        pre_action_age_seconds = (
            pre_action_time - observation.captured_at_unix_seconds
        )
        if candidate is not None:
            decision = safety.evaluate(
                candidate,
                tuple(solved.arm_positions),
                now_unix_seconds=pre_action_time,
            )
            if not decision.passed:
                selected_scale = None

        status = ControlResultStatus.BLOCKED
        post_action = None
        execution_error = None
        if decision.passed and candidate is not None:
            try:
                timeline.play()
                attachment = prepare_plug(stage)
                target = JointCommand(solved.arm_positions, solved.gripper_width_m)
                await move_joint_command(
                    actuators,
                    current,
                    target,
                    attachment,
                    frame_count=1,
                    phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                    stage=ObservationStage.APPROACHING_CABLE,
                    recorder=None,
                    sample_period_seconds=1.0 / DROID_FPS,
                )
                for _ in range(8):
                    actual = actuators.actual_command()
                    if np.max(
                        np.abs(actual.arm_positions - solved.arm_positions)
                    ) <= 0.01:
                        break
                    await omni.kit.app.get_app().next_update_async()
                actual = actuators.actual_command()
                post_collision, post_force = read_contact(sensor)
                post_snapshot = recording_snapshot(
                    RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                    ObservationStage.APPROACHING_CABLE,
                    actual,
                    attachment,
                )
                actual_action = action_between(
                    observation.pose, post_snapshot.end_effector_pose
                )
                tracking = evaluate_action_tracking(
                    candidate.first_action,
                    actual_action,
                    tracking_limits_for_policy(persisted_state.execution_policy),
                )
                capture = await capture_camera_frame(
                    JEPA_WM_CAMERA_SPECS[0], session.path / "post_action.png"
                )
                joint_tracking_error = float(
                    np.max(np.abs(actual.arm_positions - solved.arm_positions))
                )
                post_action = PostActionEvidence(
                    proposal.first_action,
                    candidate.first_action,
                    actual_action,
                    tracking,
                    post_snapshot.end_effector_pose,
                    tuple(actual.arm_positions),
                    joint_tracking_error,
                    post_force,
                    post_collision,
                    capture,
                )
                status = ControlResultStatus.APPLIED
            except Exception as error:
                actuators.apply(current)
                status = ControlResultStatus.ROLLED_BACK_EXECUTION
                execution_error = f"{type(error).__name__}: {error}"
            if post_action is not None:
                if joint_tracking_error > 0.01 or not tracking.passed:
                    actuators.apply(current)
                    await omni.kit.app.get_app().next_update_async()
                    status = ControlResultStatus.ROLLED_BACK_TRACKING
                elif post_collision or post_force > limits.maximum_contact_force_newtons:
                    actuators.apply(current)
                    await omni.kit.app.get_app().next_update_async()
                    status = ControlResultStatus.ROLLED_BACK_CONTACT

        result = ControlResult(
            status=status,
            session_id=session_id,
            gate=decision,
            projection_attempts=tuple(attempts),
            selected_action_scale=selected_scale,
            observation_age_seconds=pre_action_age_seconds,
            ik_position_error_m=(solved.position_error_m if solved is not None else None),
            ik_orientation_error_rad=(
                solved.orientation_error_rad if solved is not None else None
            ),
            pre_action_contact_force_newtons=contact_force,
            post_action=post_action,
            execution_error=execution_error,
        )
        session.write_result(result)
        return result.to_dict()
    finally:
        timeline.pause()
