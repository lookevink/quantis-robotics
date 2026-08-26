"""Persist one reset-bound insertion response inside the resident runtime."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from jepa_wm.insertion_trial import (
    InsertionTrialSourceEvidence,
    build_insertion_trial_response,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation
from sim.control_identity import observation_id_for_session
from sim.control_session import (
    CONTROL_ROOT,
    ControlSession,
    ControlSessionState,
)
from sim.trial_source_cache import ResidentTrialSourceCache


_PREPARED_SOURCES = ResidentTrialSourceCache[InsertionTrialSourceEvidence]()


def _persist_bound_insertion_trial_response(
    session: ControlSession,
    source_session_id: str,
    source_evidence: InsertionTrialSourceEvidence,
) -> dict[str, Any]:
    observation, state = session.load_capture()
    binding, response = build_insertion_trial_response(
        execution_session_id=session.session_id,
        source_session_id=source_session_id,
        execution=session.trial_context(observation, state),
        source=source_evidence,
    )
    session.write_insertion_trial_binding(binding, source_evidence)
    session.write_response(response)
    return {"binding": binding.to_dict(), "response": response.to_dict()}


def build_insertion_followup_execution_capture(
    session_id: str,
    source_observation: ControlObservation,
    source_state: ControlSessionState,
) -> tuple[ControlObservation, ControlSessionState]:
    """Clone one authenticated no-actuation capture without replay or reset."""

    if (
        source_state.execution_policy
        is not ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION
        or source_state.previous_session_id is None
        or source_state.insertion_target_policy is None
        or source_state.active_drive_target is None
        or not source_state.plug_attached
    ):
        raise ValueError("insertion follow-up source is incomplete")
    return (
        replace(
            source_observation,
            observation_id=observation_id_for_session(session_id),
        ),
        replace(
            source_state,
            session_id=session_id,
            execution_policy=ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
        ),
    )


def prepare_insertion_trial_source(
    source_session_id: str,
    *,
    control_root: Path = CONTROL_ROOT,
) -> dict[str, Any]:
    """Validate and retain one immutable no-actuation source before reset."""

    source_session = ControlSession.at(control_root, source_session_id)
    evidence = _PREPARED_SOURCES.prepare(
        source_session,
        source_session.load_insertion_trial_source_evidence,
    )
    if not evidence.safety.passed:
        raise ValueError("insertion trial source did not pass no-actuation safety")
    return {
        "source_session_id": source_session_id,
        "observation_id": evidence.safety.observation_id,
        "safety_passed": evidence.safety.passed,
        "proposal": evidence.safety.proposal.to_dict(),
        "selected_action_scale": evidence.safety.selected_action_scale.to_dict(),
    }


def persist_insertion_trial_response(
    session_id: str,
    source_session_id: str,
    *,
    control_root: Path = CONTROL_ROOT,
) -> dict[str, Any]:
    """Revalidate source/reset evidence and timestamp the bounded response."""

    session = ControlSession.at(control_root, session_id)
    source_session = ControlSession.at(control_root, source_session_id)
    source_evidence = _PREPARED_SOURCES.consume(
        source_session,
        source_session.load_insertion_trial_source_evidence,
    )
    return _persist_bound_insertion_trial_response(
        session,
        source_session_id,
        source_evidence,
    )


def persist_insertion_followup_response(
    session_id: str,
    source_session_id: str,
    *,
    control_root: Path = CONTROL_ROOT,
) -> dict[str, Any]:
    """Rebind one follow-up safety source without resetting the live stage."""

    import omni.usd

    from sim.isaac_control_runtime import bind_live_runtime, live_runtime_for

    source_session = ControlSession.at(control_root, source_session_id)
    source_evidence = _PREPARED_SOURCES.consume(
        source_session,
        source_session.load_insertion_trial_source_evidence,
    )
    source_observation, source_state = source_session.load_capture()
    observation, state = build_insertion_followup_execution_capture(
        session_id,
        source_observation,
        source_state,
    )
    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(source_session_id, stage)
    if runtime is None:
        raise RuntimeError("live insertion runtime was lost before follow-up binding")

    session = ControlSession.at(control_root, session_id)
    session.write_capture(observation, state)
    persisted = _persist_bound_insertion_trial_response(
        session,
        source_session_id,
        source_evidence,
    )
    bind_live_runtime(
        session_id,
        stage,
        runtime.actuators,
        runtime.attachment,
        runtime.sensor,
    )
    return persisted
