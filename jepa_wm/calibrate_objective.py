"""Build a fail-closed action-response calibration from realized candidate trials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_rollout import ControlStepSummary
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
    ActionResponseTrial,
)
from jepa_wm.persistence import write_json_atomic
from sim.control_session import ControlResultStatus, ControlSession


def calibrate_sessions(
    data_root: Path,
    session_ids: Sequence[str],
) -> ActionResponseCalibration:
    identifiers = tuple(session_ids)
    if len(identifiers) < 3 or len(set(identifiers)) != len(identifiers):
        raise ValueError("calibration requires three unique candidate sessions")
    trials = []
    for session_id in identifiers:
        session = ControlSession.at(data_root / "control_sessions", session_id)
        step = ControlStepSummary.from_session(session)
        state = step.state
        response = step.response
        if state.execution_policy is ControlExecutionPolicy.RESET_TRIAL_CANDIDATE:
            expected_action = session.load_candidate_binding(response).actions[0]
        elif state.execution_policy in (
            ControlExecutionPolicy.DIRECT,
            ControlExecutionPolicy.SCRIPTED_BASELINE,
        ):
            expected_action = response.first_action
        else:
            raise ValueError(
                f"control policy has no active calibration action: {session_id}"
            )
        result = step.result
        post_action = result.post_action
        if (
            result.status is not ControlResultStatus.APPLIED
            or post_action is None
            or not post_action.tracking.passed
            or post_action.raw_proposed_action != expected_action
            or post_action.collision_detected
            or post_action.contact_force_newtons > 0.0
        ):
            raise ValueError(
                f"candidate session is not valid calibration evidence: {session_id}"
            )
        trials.append(
            ActionResponseTrial(
                session_id,
                state.seed,
                post_action.raw_proposed_action,
                post_action.actual_action,
            )
        )
    return ActionResponseCalibration.fit(trials)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibration = calibrate_sessions(args.data_root, args.session)
    write_json_atomic(args.output, calibration.to_dict())
    print(args.output)
    print(calibration.to_dict())


if __name__ == "__main__":
    main()
