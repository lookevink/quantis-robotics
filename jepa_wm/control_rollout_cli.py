"""Command-line persistence boundary for simulator control rollout reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jepa_wm.control_rollout import ControlRolloutReport
from sim.control_session import ControlSession


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--data-root", type=Path, required=True)
    status.add_argument("--session", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--data-root", type=Path, required=True)
    report.add_argument("--rollout-id", required=True)
    report.add_argument("--reference-recording", required=True)
    report.add_argument("--seed", type=int, required=True)
    report.add_argument("--proposal", type=Path, required=True)
    report.add_argument("--sessions", required=True)
    report.add_argument("--requested-steps", type=int, required=True)
    report.add_argument("--orchestration-error")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        session = ControlSession.at(args.data_root / "control_sessions", args.session)
        print(session.load_result().status.value)
        return
    result = ControlRolloutReport.from_sessions(
        args.data_root,
        args.rollout_id,
        tuple(args.sessions.split(",")),
        reference_recording=args.reference_recording,
        seed=args.seed,
        proposal=args.proposal,
        requested_steps=args.requested_steps,
        orchestration_error=args.orchestration_error,
    )
    payload = result.to_dict()
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
