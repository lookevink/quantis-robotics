"""Build one strict realized direct/zero/scripted control comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.control_baselines import RealizedBaselineReport, RolloutArtifact
from jepa_wm.persistence import write_json_atomic


def _artifact(args: argparse.Namespace, role: str) -> RolloutArtifact:
    return RolloutArtifact(
        getattr(args, f"{role}_rollout"),
        tuple(getattr(args, f"{role}_sessions").split(",")),
        getattr(args, f"{role}_proposal"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--reference-recording", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--requested-steps", type=int, required=True)
    for role in ("direct", "zero", "scripted"):
        parser.add_argument(f"--{role}-rollout", required=True)
        parser.add_argument(f"--{role}-sessions", required=True)
        parser.add_argument(f"--{role}-proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = RealizedBaselineReport.from_sessions(
        args.data_root,
        args.experiment_id,
        reference_recording=args.reference_recording,
        seed=args.seed,
        requested_steps=args.requested_steps,
        direct=_artifact(args, "direct"),
        zero=_artifact(args, "zero"),
        scripted=_artifact(args, "scripted"),
    )
    payload = report.to_dict()
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
