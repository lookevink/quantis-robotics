"""Revalidate and summarize strict candidate trials across whole seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.candidate_readiness import (
    CandidateReadinessEvidence,
    CandidateReadinessSummary,
)
from jepa_wm.persistence import write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = CandidateReadinessSummary.from_evidence(
        tuple(
            CandidateReadinessEvidence.from_persisted(
                args.data_root, experiment_id
            )
            for experiment_id in args.experiment
        )
    )
    payload = summary.to_dict()
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
