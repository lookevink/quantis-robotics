"""Build the strict two-seed live reach-and-grasp readiness artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.grasp_control_readiness import GraspControlReadinessSummary
from jepa_wm.persistence import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--baseline-experiments")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--readiness-id")
    parser.add_argument("--fingerprint-only", action="store_true")
    args = parser.parse_args()
    if args.readiness_id is not None:
        if args.baseline_experiments is not None or args.output is not None:
            raise ValueError("readiness verification does not accept summary outputs")
        summary = GraspControlReadinessSummary.load_verified(
            args.data_root,
            args.readiness_id,
        )
        if not summary.filming_readiness_passed:
            raise SystemExit(2)
        if args.fingerprint_only:
            print(summary.evidence[0].proposal.fingerprint)
        else:
            print(json.dumps(summary.to_dict(), indent=2))
        return
    if args.fingerprint_only:
        raise ValueError("fingerprint output requires readiness verification")
    if args.baseline_experiments is None or args.output is None:
        raise ValueError("readiness summary requires baselines and output")
    experiment_ids = tuple(args.baseline_experiments.split(","))
    if not experiment_ids or any(not value for value in experiment_ids):
        raise ValueError("baseline experiment identifiers must be non-empty")
    summary = GraspControlReadinessSummary.from_persisted(
        args.data_root,
        experiment_ids,
    )
    payload = summary.to_dict()
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))
    if not summary.filming_readiness_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
