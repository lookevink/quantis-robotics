"""Persist one reset-bound insertion response inside the resident runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jepa_wm.insertion_trial import (
    InsertionTrialSourceEvidence,
    build_insertion_trial_response,
)
from sim.control_session import CONTROL_ROOT, ControlSession
from sim.trial_source_cache import ResidentTrialSourceCache


_PREPARED_SOURCES = ResidentTrialSourceCache[InsertionTrialSourceEvidence]()


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
    observation, state = session.load_capture()
    source_session = ControlSession.at(control_root, source_session_id)
    source_evidence = _PREPARED_SOURCES.consume(
        source_session,
        source_session.load_insertion_trial_source_evidence,
    )
    execution = session.trial_context(observation, state)
    binding, response = build_insertion_trial_response(
        execution_session_id=session_id,
        source_session_id=source_session_id,
        execution=execution,
        source=source_evidence,
    )
    session.write_insertion_trial_binding(binding, source_evidence)
    session.write_response(response)
    return {"binding": binding.to_dict(), "response": response.to_dict()}
