"""Build a fail-closed calibration from realized collection trials."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from jepa_wm.calibration_sessions import calibration_trial_from_session
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
)
from jepa_wm.persistence import write_json_atomic


def calibrate_sessions(
    data_root: Path,
    session_ids: Sequence[str],
) -> ActionResponseCalibration:
    identifiers = tuple(session_ids)
    if len(identifiers) < 3 or len(set(identifiers)) != len(identifiers):
        raise ValueError("calibration requires three unique collection sessions")
    trials = tuple(
        calibration_trial_from_session(data_root, session_id)
        for session_id in identifiers
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
