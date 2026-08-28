"""Authenticate fail-closed insertion outcomes as transition supervision."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jepa_wm.action import DroidPose
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.control_safety import ControlGateReason, SimulatorSafetyLimits
from jepa_wm.insertion_transition import (
    InsertionTransitionExample,
    InsertionTransitionSupervisionPolicy,
)
from jepa_wm.insertion_trial import InsertionTrialRollbackFailureReason
from jepa_wm.target_progress import RealizedTargetProgressReason
from jepa_wm.training_artifact import ArtifactIdentity
from sim.control_session import ControlResultStatus, ControlSession, ControlSessionState

if TYPE_CHECKING:
    from jepa_wm.control_rollout import ControlStepSummary


def _transition_example(
    session: ControlSession,
    observation: ControlObservation,
    state: ControlSessionState,
    source_proposal: ArtifactIdentity,
    context_pose: DroidPose,
    target_pose: DroidPose,
) -> InsertionTransitionExample:
    supervision = InsertionTransitionSupervisionPolicy()
    return InsertionTransitionExample(
        source_session_id=session.session_id,
        reference_recording=state.reference_recording,
        seed=state.seed,
        observation_id=observation.observation_id,
        context_frame=observation.context_frame,
        target_frame=observation.target_frame,
        context_pose=context_pose,
        target_pose=target_pose,
        previous_action=observation.previous_action,
        task_context_index=observation.warmup_frames,
        source_proposal=source_proposal,
        actions=supervision.actions(context_pose, target_pose),
        supervision=supervision,
    )


def _no_actuation_example(
    session: ControlSession,
) -> InsertionTransitionExample:
    observation, response, state = session.load()
    safety = session.load_direct_safety()
    target_pose = observation.target_pose
    if (
        target_pose is None
        or safety.live_pose is None
        or safety.passed
        or safety.selected_action_scale is not None
        or not state.plug_attached
        or safety.live_state.plug_attached is not True
        or any(
            attempt.gate.reasons
            != (ControlGateReason.TARGET_PROGRESS_INSUFFICIENT,)
            for attempt in safety.attempts
        )
        or safety.proposed_actions != response.actions
    ):
        raise ValueError(
            "transition training requires an attached fail-closed insertion session"
        )
    return _transition_example(
        session,
        observation,
        state,
        safety.proposal,
        safety.live_pose,
        target_pose,
    )


def _contact_interlock_failure_example(
    session: ControlSession,
    summary: ControlStepSummary,
) -> InsertionTransitionExample:
    result = summary.result
    target_pose = summary.observation.target_pose
    refresh = result.insertion_trial_refresh
    rollback = result.insertion_trial_rollback
    interlock = result.execution_interlock
    limits = SimulatorSafetyLimits()
    if (
        target_pose is None
        or summary.response.proposal_fingerprint is None
        or refresh is None
        or refresh.live_pose is None
        or result.post_action is not None
        or rollback is None
        or rollback.reason
        is not InsertionTrialRollbackFailureReason.SAFETY_INTERLOCK
        or not rollback.plug_attached
        or not rollback.drive_command_accepted
        or rollback.interlock.collision_detected
        or rollback.interlock.maximum_contact_force_newtons
        <= limits.maximum_contact_force_newtons
        or interlock is None
        or interlock.collision_detected
        or interlock.maximum_contact_force_newtons
        != rollback.interlock.maximum_contact_force_newtons
        or not summary.state.plug_attached
    ):
        raise ValueError(
            "transition training requires an attached contact-interlock failure"
        )
    return _transition_example(
        session,
        summary.observation,
        summary.state,
        ArtifactIdentity(
            summary.response.proposal,
            summary.response.proposal_fingerprint,
        ),
        refresh.live_pose,
        target_pose,
    )


def _safe_progress_rollback_example(
    session: ControlSession,
    summary: ControlStepSummary,
) -> InsertionTransitionExample:
    observation = summary.observation
    response = summary.response
    state = summary.state
    result = summary.result
    target_pose = observation.target_pose
    refresh = result.insertion_trial_refresh
    post_action = result.post_action
    insertion_trial = post_action.insertion_trial if post_action is not None else None
    progress = (
        insertion_trial.realized_target_progress
        if insertion_trial is not None
        else None
    )
    interlock = result.execution_interlock
    limits = SimulatorSafetyLimits()
    if (
        target_pose is None
        or response.proposal_fingerprint is None
        or result.status is not ControlResultStatus.ROLLED_BACK_PROGRESS
        or refresh is None
        or refresh.live_pose is None
        or post_action is None
        or not post_action.tracking.passed
        or post_action.collision_detected
        or post_action.contact_force_newtons
        > limits.maximum_contact_force_newtons
        or not post_action.plug_attached
        or progress is None
        or progress.reasons
        != (RealizedTargetProgressReason.TRANSLATION_PROGRESS,)
        or result.insertion_trial_rollback is None
        or interlock is None
        or interlock.collision_detected
        or interlock.maximum_contact_force_newtons
        > limits.maximum_contact_force_newtons
        or not state.plug_attached
    ):
        raise ValueError(
            "transition training requires a safe settled target-progress rollback"
        )
    return _transition_example(
        session,
        observation,
        state,
        ArtifactIdentity(response.proposal, response.proposal_fingerprint),
        refresh.live_pose,
        target_pose,
    )


def _rollback_example(session: ControlSession) -> InsertionTransitionExample:
    # Local import avoids making the dependency-light session domain depend on
    # rollout reconstruction while still authenticating the complete result.
    from jepa_wm.control_rollout import ControlStepSummary

    summary = ControlStepSummary.from_session(session)
    if summary.result.status is ControlResultStatus.ROLLBACK_FAILED:
        return _contact_interlock_failure_example(session, summary)
    return _safe_progress_rollback_example(session, summary)


def transition_example_from_session(
    data_root: Path,
    session_id: str,
) -> InsertionTransitionExample:
    """Reconstruct one authenticated fail-closed insertion training example."""

    session = ControlSession.at(
        data_root.resolve() / "control_sessions",
        session_id,
    )
    if session.direct_safety_path.is_file():
        return _no_actuation_example(session)
    return _rollback_example(session)
