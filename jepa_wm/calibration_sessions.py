"""Validated live-session evidence for action-response calibration."""

from __future__ import annotations

from pathlib import Path

from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_rollout import ControlStepSummary
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.objective_calibration import ActionResponseTrial
from sim.control_session import ControlResultStatus, ControlSession, ControlSessionState
from sim.exploration import DatasetSplit


def validated_calibration_recording(
    data_root: Path,
    state: ControlSessionState,
) -> DomainRecording:
    """Bind calibration evidence to one training-only collection session."""

    if state.execution_policy is not ControlExecutionPolicy.CALIBRATION_COLLECTION:
        raise ValueError("calibration evidence requires calibration_collection policy")
    recording = DomainRecording.from_path(
        data_root / "recordings" / state.reference_recording,
        expected_split=DatasetSplit.TRAIN,
    )
    if recording.seed != state.seed:
        raise ValueError("calibration recording seed does not match the live session")
    return recording


def calibration_trial_from_session(
    data_root: Path,
    session_id: str,
) -> ActionResponseTrial:
    """Reconstruct one calibration trial from raw, policy-bound live evidence."""

    session = ControlSession.at(data_root / "control_sessions", session_id)
    step = ControlStepSummary.from_session(session)
    state = step.state
    validated_calibration_recording(data_root, state)
    response = step.response
    result = step.result
    post_action = result.post_action
    if (
        result.status is not ControlResultStatus.APPLIED
        or post_action is None
        or not post_action.tracking.passed
        or post_action.raw_proposed_action != response.first_action
        or post_action.collision_detected
        or post_action.contact_force_newtons > 0.0
    ):
        raise ValueError(
            f"session is not valid calibration evidence: {session_id}"
        )
    return ActionResponseTrial(
        session_id,
        state.seed,
        post_action.raw_proposed_action,
        post_action.actual_action,
    )
