"""Frozen continuation from the passed unknown-start action to retained grasp."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.persistence import write_json_atomic
from jepa_wm.training_artifact import artifact_fingerprint


EXPERIMENT_ID = "unknown-start-grasp-continuation-v2"
ROLLOUT_ID = "unknown-start-e2e-v2-62605-grasp"
PREDECESSOR_SESSION_ID = "unknown-start-live-action-v7-62605"
REFERENCE_RECORDING = "contact-insertion-v10-drive-slow-2600-held-00"
REFERENCE_SEED = 12600
PROPOSAL_NAME = "contact-grasp-v10-drive-slow-2600_task12_h256_s3000"
WORKER_IDENTITY = "contact-insertion-v10-unknown-start-shadow-canary-v5"
MAXIMUM_CONTINUATION_ACTIONS = 51
OUTPUT_DIRECTORY = "unknown_start_grasp_continuation_v2"

SOURCE_FINGERPRINTS = {
    "terminal": "14dcf231b035a6481e1c79b1e538358383d3c2f0246d7e983ad0e7eae92efeb2",
    "evaluation": "181e742c3b7c2785dd8eb86690a9a1e3b3f54bc4a54981638bb2e8b6cadd56e2",
    "request": "f81e2c2237a309328722c9dfe84e7c53cd7bba534f9255121a627117397b1e0a",
    "state": "6fe844dd049fed7c214a01a5306fcca02b964d884496dad0a422e9329fc5cc3e",
    "response": "3a0101e72999f3c394cb6ee53eaae52bbc715ee662667c8c1ef8dea1b3f4d386",
    "candidate": "8a4061736693cf66c69d02b89f3c508090abda085f7c5cb53343a2495fb3369c",
    "result": "8d602d168d1812c367808893656345efeffc5907ab8e4d317d670dfec428c7ff",
    "handoff": "30bba34e99eae680d007f540d2a10ff7c0cb87b34078984a019c7801ee84f38a",
}

RUNTIME_FILES = (
    "jepa_wm/unknown_start_grasp_continuation.py",
    "ops/aws.sh",
    "ops/run_unknown_start_grasp_continuation.sh",
    "ops/shell_helpers.sh",
    "sim/control_session.py",
    "sim/isaac_control_execution.py",
    "sim/isaac_control_followup.py",
    "sim/isaac_control_runtime.py",
    "sim/isaac_demo.py",
)


def runtime_fingerprint(repository: Path | None = None) -> str:
    root = repository or Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative in RUNTIME_FILES:
        encoded = relative.encode()
        contents = (root / relative).read_bytes()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def paths(checkpoint_root: Path) -> tuple[Path, Path, Path, Path]:
    root = checkpoint_root / OUTPUT_DIRECTORY
    return (
        root / "CLAIM.json",
        root / "EVALUATION.json",
        root / "RESULT.json",
        root / "FAILURE.json",
    )


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("unknown-start grasp continuation was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _source_paths(checkpoint_root: Path, data_root: Path) -> dict[str, Path]:
    session = data_root / "control_sessions" / PREDECESSOR_SESSION_ID
    return {
        "terminal": checkpoint_root / "unknown_start_live_action_v7" / "RESULT.json",
        "evaluation": checkpoint_root
        / "unknown_start_live_action_v7"
        / "EVALUATION.json",
        "request": session / "request.json",
        "state": session / "state.json",
        "response": session / "response.json",
        "candidate": session / "experimental_candidate.json",
        "result": session / "result.json",
        "handoff": session / "unknown_start_handoff.json",
    }


def authenticate_source(checkpoint_root: Path, data_root: Path) -> None:
    source_paths = _source_paths(checkpoint_root, data_root)
    if any(
        artifact_fingerprint(source_paths[name]) != expected
        for name, expected in SOURCE_FINGERPRINTS.items()
    ):
        raise ValueError("unknown-start grasp predecessor changed")
    terminal = json.loads(source_paths["terminal"].read_text())
    result = json.loads(source_paths["result"].read_text())
    if (
        terminal.get("passed") is not True
        or terminal.get("recovery_verified") is not True
        or terminal.get("applied_model_actions") != 1
        or result.get("status") != "applied"
        or result.get("post_action_plug_attached") is not False
        or result.get("post_action_collision_detected") is not False
        or result.get("post_action_contact_force_newtons", float("inf")) > 2.0
        or result.get("action_tracking", {}).get("passed") is not True
    ):
        raise ValueError("unknown-start grasp predecessor did not pass")


def claim(
    checkpoint_root: Path,
    data_root: Path,
    source_revision: str,
    expected_runtime_fingerprint: str,
) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if any(path.exists() for path in (evaluation_path, result_path, failure_path)):
        raise ValueError("unknown-start grasp continuation is already terminal")
    actual_runtime = runtime_fingerprint()
    if len(source_revision) != 40 or actual_runtime != expected_runtime_fingerprint:
        raise ValueError("unknown-start grasp continuation runtime changed")
    authenticate_source(checkpoint_root, data_root)
    payload = {
        "schema": "quantis.unknown_start_grasp_continuation_claim.v1",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "rollout_id": ROLLOUT_ID,
        "predecessor_session_id": PREDECESSOR_SESSION_ID,
        "reference_recording": REFERENCE_RECORDING,
        "reference_seed": REFERENCE_SEED,
        "proposal_name": PROPOSAL_NAME,
        "worker_identity": WORKER_IDENTITY,
        "maximum_continuation_actions": MAXIMUM_CONTINUATION_ACTIONS,
        "maximum_total_grasp_actions": MAXIMUM_CONTINUATION_ACTIONS + 1,
        "source_revision": source_revision,
        "runtime_fingerprint": actual_runtime,
        "source_fingerprints": SOURCE_FINGERPRINTS,
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    _write_exclusive(claim_path, payload)
    return payload


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("unknown-start grasp continuation claim is invalid")
    authenticate_source(checkpoint_root, data_root)
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    report = json.loads(report_path.read_text())
    decision = report.get("reach_and_grasp")
    passed = (
        report.get("rollout_id") == ROLLOUT_ID
        and report.get("reference_recording") == REFERENCE_RECORDING
        and report.get("seed") == REFERENCE_SEED
        and report.get("requested_steps") == MAXIMUM_CONTINUATION_ACTIONS
        and report.get("predecessor_session_id") == PREDECESSOR_SESSION_ID
        and report.get("orchestration_failure") is None
        and isinstance(decision, dict)
        and decision.get("passed") is True
        and report.get("applied_steps") == report.get("complete_steps")
        and 1 <= report.get("applied_steps", 0) <= MAXIMUM_CONTINUATION_ACTIONS
    )
    payload = {
        "schema": "quantis.unknown_start_grasp_continuation_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "evaluation_passed": passed,
        "recovery_verified": False,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "report_fingerprint": artifact_fingerprint(report_path),
        "applied_continuation_actions": report.get("applied_steps"),
        "reach_and_grasp": decision,
        "terminal_session_id": (
            report.get("steps", [{}])[-1].get("session")
            if report.get("steps")
            else None
        ),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(evaluation_path, payload)
    if not passed:
        raise ValueError("unknown-start grasp continuation did not reach retained grasp")
    return payload


def finalize(
    checkpoint_root: Path,
    recovery_checkpoint_root: Path,
    data_root: Path,
    recovery_data_root: Path,
) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    recovery_claim, recovery_evaluation, recovery_result, _ = paths(
        recovery_checkpoint_root
    )
    if result_path.exists() or failure_path.exists():
        raise ValueError("unknown-start grasp continuation is already terminal")
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    recovery_report = (
        recovery_data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    )
    for primary, recovery in (
        (claim_path, recovery_claim),
        (evaluation_path, recovery_evaluation),
        (report_path, recovery_report),
    ):
        if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
            raise ValueError("unknown-start grasp recovery changed")
    evaluation = json.loads(evaluation_path.read_text())
    report = json.loads(report_path.read_text())
    steps = report.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("unknown-start grasp recovery roster is invalid")
    for step in steps:
        session_id = step.get("session") if isinstance(step, dict) else None
        if not isinstance(session_id, str):
            raise ValueError("unknown-start grasp recovery roster is invalid")
        primary_session = data_root / "control_sessions" / session_id
        recovery_session = recovery_data_root / "control_sessions" / session_id
        for name in (
            "request.json",
            "state.json",
            "response.json",
            "execution_started.json",
            "result.json",
            "context.png",
            "post_action.png",
        ):
            if artifact_fingerprint(primary_session / name) != artifact_fingerprint(
                recovery_session / name
            ):
                raise ValueError(
                    f"unknown-start grasp session recovery changed: {session_id}/{name}"
                )
    passed = evaluation.get("evaluation_passed") is True
    payload = {
        "schema": "quantis.unknown_start_grasp_continuation_terminal.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "recovery_verified": True,
        "evaluation_fingerprint": artifact_fingerprint(evaluation_path),
        "report_fingerprint": artifact_fingerprint(report_path),
        "applied_continuation_actions": evaluation.get(
            "applied_continuation_actions"
        ),
        "terminal_session_id": evaluation.get("terminal_session_id"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(result_path, payload)
    write_json_atomic(recovery_result, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("unknown-start grasp terminal recovery changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or failure_path.exists() or not claim_path.is_file():
        raise ValueError("unknown-start grasp continuation failure is invalid")
    payload = {
        "schema": "quantis.unknown_start_grasp_continuation_failure.v1",
        "status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "retry_authorized": False,
        "filming_authorized": False,
    }
    write_json_atomic(failure_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("fingerprint", "claim", "evaluate", "finalize", "failure"),
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--recovery-checkpoint-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--recovery-data-root", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--runtime-fingerprint")
    parser.add_argument("--error")
    args = parser.parse_args(argv)
    if args.command == "fingerprint":
        payload: Any = runtime_fingerprint()
    elif args.command == "claim":
        payload = claim(
            args.checkpoint_root,
            args.data_root,
            args.source_revision,
            args.runtime_fingerprint,
        )
    elif args.command == "evaluate":
        payload = evaluate(args.checkpoint_root, args.data_root)
    elif args.command == "finalize":
        payload = finalize(
            args.checkpoint_root,
            args.recovery_checkpoint_root,
            args.data_root,
            args.recovery_data_root,
        )
    else:
        payload = failure(args.checkpoint_root, args.error)
    print(payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
