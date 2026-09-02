"""Frozen V7 acquisition authority after the V6 tracking rollback."""

from __future__ import annotations

import argparse
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jepa_wm.contact_grasp_acquisition_continuation import (
    PROPOSAL_FINGERPRINT,
    PROPOSAL_NAME,
    READINESS_FINGERPRINT,
    READINESS_NAME,
    REFERENCE_RECORDING,
    REFERENCE_SEED,
    REPLAY_FINGERPRINT,
    REPLAY_NAME,
    SOURCE_FINGERPRINTS,
    SOURCE_SESSION_ID,
    WORKER_FINGERPRINT,
    WORKER_NAME,
    _validate_source,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.runtime_fingerprint import runtime_source_files, runtime_source_fingerprint
from jepa_wm.training_artifact import artifact_fingerprint


HANDOFF_SCHEMA = "quantis.contact_grasp_acquisition_hold.v4"
V6_ROLLOUT_ID = "unknown-start-e2e-v6-62605-grasp"
V6_REPORT_FINGERPRINT = (
    "2b1952ebf834a28465f1a3d5f4b5d3fe0df365ddfded63e8f3a762113cc3c9b9"
)
V6_CLAIM_FINGERPRINT = (
    "4136fe3bec2d4ba5e77b7ce686975efcd10cdc3213c419a4cffc9bda8598660d"
)
V6_FAILURE_FINGERPRINT = (
    "6d9a0bb7c30c0f27cca73a7452c3770d7e1755bd254f531fe3e76b096f5c2799"
)
V6_SESSION_ID = "unknown-start-e2e-v6-62605-grasp-01"
V6_SESSION_FINGERPRINTS = {
    "request.json": "b289241d5733397ece4f319d8bc4f00c2ec8e0620a085f2dd10f099eb823cd17",
    "state.json": "f2648675978925502b872276a990d2d62768b57b5a3e0284a654cde77807fe97",
    "response.json": "10000b449c74ac4495dfaac1eb9421dbe0b6ff4ac92b97660ce267c62bbf9743",
    "execution_started.json": "0c19c026d389c44d43d690785a0a5e9998cbfe5b9b49484bab59ff8665b5029a",
    "result.json": "3cde463fdef17d68f7b5480e3c9e91cd5de073921004f002e72a48ab233fcf4d",
    "context.png": "0e41788975b6e76a112efff57eba076115251835f7411bf9e7fc6a1a928d66ee",
    "post_action.png": "cad1e85c5a9f4fbe1bf8922658a8a106aff78ad08c70f9b935dcd3bd5410ef8a",
}
V7_CLAIM_FINGERPRINT = (
    "b4ee5a15938ad3e005367b24e2c79671da6d8005d09f42d65eeb9baf8a07b46f"
)
V7_FAILURE_FINGERPRINT = (
    "f1772f8004bd5a65c963dd581bfccf6d8c7409bfc4e8cd1446cf6d23e9ff6bd8"
)
V8_CLAIM_FINGERPRINT = (
    "aca940403e3755d979f5a5977fcb6f0b22e0f86272629f502412ec06acc67cd2"
)
V8_FAILURE_FINGERPRINT = (
    "ebfba0979be2533a9fd5a4b11c926b90bce1b427aa896de84471e42b1ac2d1d4"
)
V9_CLAIM_FINGERPRINT = (
    "d363d5e284d57eb72bb09c7f94917e41b87fae547d364239d68a7210b2add4f5"
)
V9_FAILURE_FINGERPRINT = (
    "608bfa1811d44355d916c5f0fc8bda0713afaf91d4e6df8516c3e8695b0d9cf5"
)
EXPERIMENT_DIRECTORY = "unknown_start_acquisition_hold_v10"
ROLLOUT_ID = "unknown-start-e2e-v10-62605-grasp"
MAXIMUM_ACTIONS = 52


RUNTIME_FILES = (
    "jepa_wm/action.py",
    "jepa_wm/contact_grasp_acquisition_continuation.py",
    "jepa_wm/contact_grasp_acquisition_hold.py",
    "jepa_wm/contact_grasp_target.py",
    "jepa_wm/control_protocol.py",
    "jepa_wm/control_rollout.py",
    "jepa_wm/control_safety.py",
    "jepa_wm/control_worker.py",
    "ops/aws.sh",
    "ops/jepa_wm.sh",
    "ops/run_unknown_start_acquisition_hold.sh",
    "ops/shell_helpers.sh",
    "sim/control_session.py",
    "sim/isaac_control_execution.py",
    "sim/isaac_control_followup.py",
    "sim/isaac_control_bridge.py",
    "sim/isaac_control_runtime.py",
    "sim/isaac_demo.py",
    "sim/isaac_demo_kinematics.py",
    "sim/runtime_loader.py",
)
RUNTIME_FILES = runtime_source_files(Path(__file__).resolve().parents[1])


def runtime_fingerprint(repository: Path | None = None) -> str:
    root = repository or Path(__file__).resolve().parents[1]
    return runtime_source_fingerprint(root)


@dataclass(frozen=True)
class ContactGraspAcquisitionHold:
    followup_session_id: str
    runtime_fingerprint: str
    source_revision: str
    schema: str = HANDOFF_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANDOFF_SCHEMA
            or not self.followup_session_id
            or len(self.runtime_fingerprint) != 64
            or len(self.source_revision) != 40
        ):
            raise ValueError("contact-grasp acquisition hold is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_session_id": SOURCE_SESSION_ID,
            "followup_session_id": self.followup_session_id,
            "reference_recording": REFERENCE_RECORDING,
            "reference_seed": REFERENCE_SEED,
            "proposal_name": PROPOSAL_NAME,
            "proposal_fingerprint": PROPOSAL_FINGERPRINT,
            "worker_name": WORKER_NAME,
            "worker_fingerprint": WORKER_FINGERPRINT,
            "readiness_name": READINESS_NAME,
            "readiness_fingerprint": READINESS_FINGERPRINT,
            "replay_name": REPLAY_NAME,
            "replay_fingerprint": REPLAY_FINGERPRINT,
            "source_fingerprints": SOURCE_FINGERPRINTS,
            "v6_rollout_id": V6_ROLLOUT_ID,
            "v6_report_fingerprint": V6_REPORT_FINGERPRINT,
            "v6_claim_fingerprint": V6_CLAIM_FINGERPRINT,
            "v6_failure_fingerprint": V6_FAILURE_FINGERPRINT,
            "v6_session_id": V6_SESSION_ID,
            "v6_session_fingerprints": V6_SESSION_FINGERPRINTS,
            "v7_claim_fingerprint": V7_CLAIM_FINGERPRINT,
            "v7_failure_fingerprint": V7_FAILURE_FINGERPRINT,
            "v8_claim_fingerprint": V8_CLAIM_FINGERPRINT,
            "v8_failure_fingerprint": V8_FAILURE_FINGERPRINT,
            "v9_claim_fingerprint": V9_CLAIM_FINGERPRINT,
            "v9_failure_fingerprint": V9_FAILURE_FINGERPRINT,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_revision": self.source_revision,
            "simulator_action_authorized": True,
            "filming_authorized": False,
            "production_authority_granted": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContactGraspAcquisitionHold:
        try:
            instance = cls(
                followup_session_id=str(payload["followup_session_id"]),
                runtime_fingerprint=str(payload["runtime_fingerprint"]),
                source_revision=str(payload["source_revision"]),
                schema=str(payload["schema"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("contact-grasp acquisition hold is incomplete") from error
        if dict(payload) != instance.to_dict():
            raise ValueError("contact-grasp acquisition hold changed")
        return instance


def _validate_v6(checkpoint_root: Path, data_root: Path) -> None:
    _validate_source(checkpoint_root, data_root)
    report_path = data_root / "control_rollouts" / V6_ROLLOUT_ID / "report.json"
    claim_path = checkpoint_root / "unknown_start_acquisition_continuation_v6" / "CLAIM.json"
    failure_path = checkpoint_root / "unknown_start_acquisition_continuation_v6" / "FAILURE.json"
    session = data_root / "control_sessions" / V6_SESSION_ID
    if (
        artifact_fingerprint(report_path) != V6_REPORT_FINGERPRINT
        or artifact_fingerprint(claim_path) != V6_CLAIM_FINGERPRINT
        or artifact_fingerprint(failure_path) != V6_FAILURE_FINGERPRINT
        or any(
            artifact_fingerprint(session / name) != expected
            for name, expected in V6_SESSION_FINGERPRINTS.items()
        )
    ):
        raise ValueError("terminal V6 evidence changed")
    report = json.loads(report_path.read_text())
    result = json.loads((session / "result.json").read_text())
    failure = json.loads(failure_path.read_text())
    if (
        report.get("complete_steps") != 1
        or report.get("applied_steps") != 0
        or report.get("steps", [{}])[-1].get("session") != V6_SESSION_ID
        or result.get("status") != "rolled_back_after_tracking_failure"
        or result.get("action_tracking", {}).get("reasons") != ["rotation_direction"]
        or result.get("post_action_contact_force_newtons") != 0.0
        or result.get("post_action_collision_detected") is not False
        or result.get("post_action_plug_attached") is not False
        or result.get("selected_action_scale")
        != {"translation": 0.03125, "rotation": 0.0625, "gripper": 0.125}
        or failure.get("retry_authorized") is not False
    ):
        raise ValueError("V6 was not the exact safe tracking rollback")
    v7_root = checkpoint_root / "unknown_start_acquisition_hold_v7"
    v7_claim = v7_root / "CLAIM.json"
    v7_failure = v7_root / "FAILURE.json"
    if (
        artifact_fingerprint(v7_claim) != V7_CLAIM_FINGERPRINT
        or artifact_fingerprint(v7_failure) != V7_FAILURE_FINGERPRINT
    ):
        raise ValueError("terminal V7 evidence changed")
    v7_failure_payload = json.loads(v7_failure.read_text())
    v7_claim_payload = json.loads(v7_claim.read_text())
    if (
        v7_claim_payload.get("followup_session_id")
        != "unknown-start-e2e-v7-62605-grasp-01"
        or v7_failure_payload.get("error") != "capture_01:exit_1"
        or v7_failure_payload.get("retry_authorized") is not False
        or (
            data_root
            / "control_sessions"
            / "unknown-start-e2e-v7-62605-grasp-01"
        ).exists()
        or (data_root / "control_rollouts" / "unknown-start-e2e-v7-62605-grasp").exists()
    ):
        raise ValueError("V7 was not the exact no-capture runtime-owner failure")
    v8_root = checkpoint_root / "unknown_start_acquisition_hold_v8"
    v8_claim = v8_root / "CLAIM.json"
    v8_failure = v8_root / "FAILURE.json"
    if (
        artifact_fingerprint(v8_claim) != V8_CLAIM_FINGERPRINT
        or artifact_fingerprint(v8_failure) != V8_FAILURE_FINGERPRINT
    ):
        raise ValueError("terminal V8 evidence changed")
    v8_claim_payload = json.loads(v8_claim.read_text())
    v8_failure_payload = json.loads(v8_failure.read_text())
    if (
        v8_claim_payload.get("followup_session_id")
        != "unknown-start-e2e-v8-62605-grasp-01"
        or v8_failure_payload.get("error") != "capture_01:exit_1"
        or v8_failure_payload.get("retry_authorized") is not False
        or (
            data_root
            / "control_sessions"
            / "unknown-start-e2e-v8-62605-grasp-01"
        ).exists()
        or (data_root / "control_rollouts" / "unknown-start-e2e-v8-62605-grasp").exists()
    ):
        raise ValueError("V8 was not the exact no-capture reload failure")
    v9_root = checkpoint_root / "unknown_start_acquisition_hold_v9"
    v9_claim = v9_root / "CLAIM.json"
    v9_failure = v9_root / "FAILURE.json"
    if (
        artifact_fingerprint(v9_claim) != V9_CLAIM_FINGERPRINT
        or artifact_fingerprint(v9_failure) != V9_FAILURE_FINGERPRINT
    ):
        raise ValueError("terminal V9 evidence changed")
    v9_claim_payload = json.loads(v9_claim.read_text())
    v9_failure_payload = json.loads(v9_failure.read_text())
    if (
        v9_claim_payload.get("followup_session_id")
        != "unknown-start-e2e-v9-62605-grasp-01"
        or v9_failure_payload.get("error") != "capture_01:exit_1"
        or v9_failure_payload.get("retry_authorized") is not False
        or (
            data_root
            / "control_sessions"
            / "unknown-start-e2e-v9-62605-grasp-01"
        ).exists()
        or (data_root / "control_rollouts" / "unknown-start-e2e-v9-62605-grasp").exists()
    ):
        raise ValueError("V9 was not the exact no-capture drive-target failure")


def paths(checkpoint_root: Path) -> tuple[Path, Path, Path, Path]:
    root = checkpoint_root / EXPERIMENT_DIRECTORY
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
        raise ValueError("contact-grasp acquisition hold was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def claim(
    checkpoint_root: Path,
    data_root: Path,
    followup_session_id: str,
    source_revision: str,
    expected_runtime_fingerprint: str,
) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if any(path.exists() for path in (evaluation_path, result_path, failure_path)):
        raise ValueError("contact-grasp acquisition hold is already terminal")
    _validate_v6(checkpoint_root, data_root)
    handoff = ContactGraspAcquisitionHold(
        followup_session_id,
        runtime_fingerprint(),
        source_revision,
    )
    if handoff.runtime_fingerprint != expected_runtime_fingerprint:
        raise ValueError("contact-grasp acquisition hold runtime changed")
    payload = handoff.to_dict()
    _write_exclusive(claim_path, payload)
    return payload


def encode_claim(checkpoint_root: Path) -> str:
    claim_path, _, _, _ = paths(checkpoint_root)
    payload = json.loads(claim_path.read_text())
    ContactGraspAcquisitionHold.from_dict(payload)
    return b64encode(json.dumps(payload, sort_keys=True).encode()).decode()


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("contact-grasp acquisition hold claim is invalid")
    _validate_v6(checkpoint_root, data_root)
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    report = json.loads(report_path.read_text())
    decision = report.get("reach_and_grasp")
    passed = (
        report.get("rollout_id") == ROLLOUT_ID
        and report.get("reference_recording") == REFERENCE_RECORDING
        and report.get("seed") == REFERENCE_SEED
        and report.get("requested_steps") == MAXIMUM_ACTIONS
        and report.get("predecessor_session_id") == SOURCE_SESSION_ID
        and report.get("orchestration_failure") is None
        and isinstance(decision, dict)
        and decision.get("passed") is True
        and report.get("applied_steps") == report.get("complete_steps")
        and 1 <= report.get("applied_steps", 0) <= MAXIMUM_ACTIONS
    )
    payload = {
        "schema": "quantis.contact_grasp_acquisition_hold_evaluation.v4",
        "status": "evaluated_pending_recovery",
        "evaluation_passed": passed,
        "recovery_verified": False,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "report_fingerprint": artifact_fingerprint(report_path),
        "applied_actions": report.get("applied_steps"),
        "reach_and_grasp": decision,
        "terminal_session_id": report.get("steps", [{}])[-1].get("session"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(evaluation_path, payload)
    if not passed:
        raise ValueError("contact-grasp acquisition hold did not retain grasp")
    return payload


def finalize(
    checkpoint_root: Path,
    recovery_checkpoint_root: Path,
    data_root: Path,
    recovery_data_root: Path,
) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    recovery_claim, recovery_evaluation, recovery_result, _ = paths(recovery_checkpoint_root)
    if result_path.exists() or failure_path.exists():
        raise ValueError("contact-grasp acquisition hold is already terminal")
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    recovery_report = recovery_data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    for primary, recovery in (
        (claim_path, recovery_claim),
        (evaluation_path, recovery_evaluation),
        (report_path, recovery_report),
    ):
        if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
            raise ValueError("contact-grasp acquisition hold backup changed")
    evaluation = json.loads(evaluation_path.read_text())
    report = json.loads(report_path.read_text())
    for step in report.get("steps", ()):
        session_id = step.get("session") if isinstance(step, dict) else None
        if not isinstance(session_id, str):
            raise ValueError("contact-grasp acquisition hold roster is invalid")
        for name in (
            "request.json", "state.json", "response.json", "execution_started.json",
            "result.json", "context.png", "post_action.png",
        ):
            primary = data_root / "control_sessions" / session_id / name
            recovery = recovery_data_root / "control_sessions" / session_id / name
            if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
                raise ValueError(f"acquisition hold changed: {session_id}/{name}")
    first_session = report["steps"][0]["session"]
    handoff = data_root / "control_sessions" / SOURCE_SESSION_ID / f"acquisition_handoff_{first_session}.json"
    recovery_handoff = recovery_data_root / "control_sessions" / SOURCE_SESSION_ID / handoff.name
    if artifact_fingerprint(handoff) != artifact_fingerprint(recovery_handoff):
        raise ValueError("contact-grasp acquisition hold handoff backup changed")
    payload = {
        "schema": "quantis.contact_grasp_acquisition_hold_terminal.v4",
        "status": "passed",
        "passed": True,
        "recovery_verified": True,
        "evaluation_fingerprint": artifact_fingerprint(evaluation_path),
        "report_fingerprint": artifact_fingerprint(report_path),
        "applied_actions": evaluation.get("applied_actions"),
        "terminal_session_id": evaluation.get("terminal_session_id"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    if evaluation.get("evaluation_passed") is not True:
        raise ValueError("contact-grasp acquisition hold evaluation failed")
    write_json_atomic(result_path, payload)
    write_json_atomic(recovery_result, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("contact-grasp acquisition hold terminal backup changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or failure_path.exists() or not claim_path.is_file():
        raise ValueError("contact-grasp acquisition hold failure is invalid")
    payload = {
        "schema": "quantis.contact_grasp_acquisition_hold_failure.v4",
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
    parser.add_argument("command", choices=("fingerprint", "claim", "encode", "evaluate", "finalize", "failure"))
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--recovery-checkpoint-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--recovery-data-root", type=Path)
    parser.add_argument("--followup-session")
    parser.add_argument("--source-revision")
    parser.add_argument("--runtime-fingerprint")
    parser.add_argument("--error")
    args = parser.parse_args(argv)
    if args.command == "fingerprint":
        payload: Any = runtime_fingerprint()
    elif args.command == "claim":
        payload = claim(args.checkpoint_root, args.data_root, args.followup_session, args.source_revision, args.runtime_fingerprint)
    elif args.command == "encode":
        payload = encode_claim(args.checkpoint_root)
    elif args.command == "evaluate":
        payload = evaluate(args.checkpoint_root, args.data_root)
    elif args.command == "finalize":
        payload = finalize(args.checkpoint_root, args.recovery_checkpoint_root, args.data_root, args.recovery_data_root)
    else:
        payload = failure(args.checkpoint_root, args.error)
    print(payload if isinstance(payload, str) else json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
