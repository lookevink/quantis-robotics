"""Command-line persistence boundary for simulator control rollout reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.control_rollout import ControlRolloutReport, OrchestrationFailure
from jepa_wm.persistence import write_json_atomic
from sim.control_session import ControlSession


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("status")
    status.add_argument("--data-root", type=Path, required=True)
    status.add_argument("--session", required=True)
    grasp_status = subparsers.add_parser("reach-and-grasp-status")
    grasp_status.add_argument("--data-root", type=Path, required=True)
    grasp_status.add_argument("--rollout-id", required=True)
    grasp_status.add_argument("--reference-recording", required=True)
    grasp_status.add_argument("--seed", type=int, required=True)
    grasp_status.add_argument("--proposal", type=Path, required=True)
    grasp_status.add_argument("--sessions", required=True)
    grasp_status.add_argument("--requested-steps", type=int, required=True)
    grasp_status.add_argument("--predecessor-session")
    report = subparsers.add_parser("report")
    report.add_argument("--data-root", type=Path, required=True)
    report.add_argument("--rollout-id", required=True)
    report.add_argument("--reference-recording", required=True)
    report.add_argument("--seed", type=int, required=True)
    report.add_argument("--proposal", type=Path, required=True)
    report.add_argument("--sessions", required=True)
    report.add_argument("--requested-steps", type=int, required=True)
    report.add_argument("--orchestration-failure")
    report.add_argument("--predecessor-session")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        session = ControlSession.at(args.data_root / "control_sessions", args.session)
        print(session.load_result().status.value)
        return
    if args.command == "reach-and-grasp-status":
        rollout = ControlRolloutReport.from_sessions(
            args.data_root,
            args.rollout_id,
            tuple(args.sessions.split(",")),
            reference_recording=args.reference_recording,
            seed=args.seed,
            proposal=args.proposal,
            requested_steps=args.requested_steps,
            predecessor_session_id=args.predecessor_session,
        )
        decision = rollout.reach_and_grasp
        print("ready" if decision is not None and decision.passed else "pending")
        return
    result = ControlRolloutReport.from_sessions(
        args.data_root,
        args.rollout_id,
        tuple(args.sessions.split(",")),
        reference_recording=args.reference_recording,
        seed=args.seed,
        proposal=args.proposal,
        requested_steps=args.requested_steps,
        orchestration_failure=(
            OrchestrationFailure.parse(args.orchestration_failure)
            if args.orchestration_failure is not None
            else None
        ),
        predecessor_session_id=args.predecessor_session,
    )
    payload = result.to_dict()
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
