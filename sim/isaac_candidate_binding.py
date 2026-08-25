"""Persist a reset-trial candidate response inside Isaac's resident runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jepa_wm.experimental_candidate import build_experimental_candidate_response
from jepa_wm.experimental_candidate import CandidateSourceEvidence
from sim.control_session import CONTROL_ROOT, ControlSession
from sim.trial_source_cache import ResidentTrialSourceCache


_PREPARED_SOURCES = ResidentTrialSourceCache[CandidateSourceEvidence]()


def prepare_experimental_candidate_source(
    source_session_id: str,
    *,
    control_root: Path = CONTROL_ROOT,
) -> dict[str, Any]:
    """Warm and validate immutable source evidence before live capture begins."""

    source_session = ControlSession.at(control_root, source_session_id)
    evidence = _PREPARED_SOURCES.prepare(
        source_session,
        source_session.load_candidate_source_evidence,
    )
    shadow = evidence.shadow
    safety = evidence.safety
    if not shadow.passes_shadow_gate or not safety.passed:
        raise ValueError("candidate source did not pass shadow search and safety")
    return {
        "source_session_id": source_session_id,
        "observation_id": shadow.observation_id,
        "shadow_gate_passed": shadow.passes_shadow_gate,
        "safety_passed": safety.passed,
    }


def persist_experimental_candidate_response(
    session_id: str,
    source_session_id: str,
    *,
    control_root: Path = CONTROL_ROOT,
) -> dict[str, Any]:
    """Validate immutable source evidence and timestamp the response afterward."""

    session = ControlSession.at(control_root, session_id)
    observation, _ = session.load_capture()
    source_session = ControlSession.at(control_root, source_session_id)
    source_evidence = _PREPARED_SOURCES.consume(
        source_session,
        source_session.load_candidate_source_evidence,
    )
    shadow = source_evidence.shadow
    safety = source_evidence.safety
    binding, response = build_experimental_candidate_response(
        execution_session_id=session_id,
        source_session_id=source_session_id,
        observation=observation,
        shadow=shadow,
        safety=safety,
    )
    session.write_candidate_binding(binding, source_evidence)
    session.write_response(response)
    return {"binding": binding.to_dict(), "response": response.to_dict()}
