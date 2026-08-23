"""Persist one strict realized experimental-candidate comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.candidate_trial import CandidateTrialReport
from jepa_wm.persistence import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-experiment-id", required=True)
    parser.add_argument("--candidate-session", required=True)
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = CandidateTrialReport.from_sessions(
        args.data_root,
        args.experiment_id,
        args.baseline_experiment_id,
        args.candidate_session,
        args.source_session,
    )
    payload = report.to_dict()
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
