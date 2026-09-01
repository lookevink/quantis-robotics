"""Authenticated V11 continuation after the bounded V10 acquisition negative."""

from __future__ import annotations

import argparse
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
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
    WORKER_FINGERPRINT,
    WORKER_NAME,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.training_artifact import artifact_fingerprint


HANDOFF_SCHEMA = "quantis.contact_grasp_acquisition_resolution.v1"
EXPERIMENT_DIRECTORY = "unknown_start_acquisition_resolution_v11"
ROLLOUT_ID = "unknown-start-e2e-v11-62605-grasp"
SOURCE_ROLLOUT_ID = "unknown-start-e2e-v10-62605-grasp"
SOURCE_SESSION_ID = f"{SOURCE_ROLLOUT_ID}-52"
MAXIMUM_ACTIONS = 96
SOURCE_MAXIMUM_ACTIONS = 52
SOURCE_CLAIM_FINGERPRINT = (
    "2a1e7feba1a1e1f299f3bbcc884f55a838f5b5129fbc290cd62c1d721d784cda"
)
SOURCE_FAILURE_FINGERPRINT = (
    "732f772516c2f7407ca9643b292081498d926fad27f0b96b9b46d440e4b31c48"
)
SOURCE_REPORT_FINGERPRINT = (
    "22d266bb7aedc156499e1b405dc2c73d86c86dc2123d78b9de0993a4395346b4"
)
SOURCE_ROSTER_FINGERPRINT = (
    "561076bc5ce667d297ed9ba45623c7dc204747c683d12b2ba6ad84c8c4e7f1c7"
)
SESSION_FILES = (
    "request.json",
    "state.json",
    "response.json",
    "execution_started.json",
    "result.json",
    "context.png",
    "post_action.png",
)
RUNTIME_FILES = (
    "jepa_wm/action.py",
    "jepa_wm/contact_grasp_acquisition_resolution.py",
    "jepa_wm/contact_grasp_target.py",
    "jepa_wm/control_protocol.py",
    "jepa_wm/control_rollout.py",
    "jepa_wm/control_safety.py",
    "jepa_wm/control_worker.py",
    "ops/aws.sh",
    "ops/jepa_wm.sh",
    "ops/run_unknown_start_acquisition_resolution.sh",
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


def runtime_fingerprint(repository: Path | None = None) -> str:
    root = repository or Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative in RUNTIME_FILES:
        name = relative.encode()
        contents = (root / relative).read_bytes()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def source_roster_fingerprint(data_root: Path) -> str:
    digest = sha256()
    for index in range(1, 53):
        session_id = f"{SOURCE_ROLLOUT_ID}-{index:02d}"
        for filename in SESSION_FILES:
            relative = f"{session_id}/{filename}".encode()
            contents = (data_root / "control_sessions" / session_id / filename).read_bytes()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(contents).to_bytes(8, "big"))
            digest.update(contents)
    return digest.hexdigest()


@dataclass(frozen=True)
class ContactGraspAcquisitionResolution:
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
            raise ValueError("contact-grasp acquisition resolution is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_rollout_id": SOURCE_ROLLOUT_ID,
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
            "source_claim_fingerprint": SOURCE_CLAIM_FINGERPRINT,
            "source_failure_fingerprint": SOURCE_FAILURE_FINGERPRINT,
            "source_report_fingerprint": SOURCE_REPORT_FINGERPRINT,
            "source_roster_fingerprint": SOURCE_ROSTER_FINGERPRINT,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_revision": self.source_revision,
            "no_actuation_diagnostic_required": True,
            "simulator_action_authorized": True,
            "filming_authorized": False,
            "production_authority_granted": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ContactGraspAcquisitionResolution:
        try:
            instance = cls(
                followup_session_id=str(payload["followup_session_id"]),
                runtime_fingerprint=str(payload["runtime_fingerprint"]),
                source_revision=str(payload["source_revision"]),
                schema=str(payload["schema"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("contact-grasp acquisition resolution is incomplete") from error
        if dict(payload) != instance.to_dict():
            raise ValueError("contact-grasp acquisition resolution changed")
        return instance


def paths(checkpoint_root: Path) -> tuple[Path, Path, Path, Path]:
    root = checkpoint_root / EXPERIMENT_DIRECTORY
    return root / "CLAIM.json", root / "EVALUATION.json", root / "RESULT.json", root / "FAILURE.json"


def diagnostic_path(data_root: Path, followup_session_id: str) -> Path:
    return (
        data_root
        / "control_sessions"
        / SOURCE_SESSION_ID
        / f"acquisition_resolution_{followup_session_id}.json"
    )


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("contact-grasp acquisition resolution was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_source(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    from jepa_wm.control_rollout import ControlRolloutReport

    source_root = checkpoint_root / "unknown_start_acquisition_hold_v10"
    report_path = data_root / "control_rollouts" / SOURCE_ROLLOUT_ID / "report.json"
    if (
        artifact_fingerprint(source_root / "CLAIM.json") != SOURCE_CLAIM_FINGERPRINT
        or artifact_fingerprint(source_root / "FAILURE.json")
        != SOURCE_FAILURE_FINGERPRINT
        or artifact_fingerprint(report_path) != SOURCE_REPORT_FINGERPRINT
        or source_roster_fingerprint(data_root) != SOURCE_ROSTER_FINGERPRINT
    ):
        raise ValueError("terminal V10 acquisition evidence changed")
    payload = json.loads(report_path.read_text())
    session_ids = tuple(step["session"] for step in payload.get("steps", ()))
    reconstructed = ControlRolloutReport.from_sessions(
        data_root,
        SOURCE_ROLLOUT_ID,
        session_ids,
        reference_recording=REFERENCE_RECORDING,
        seed=REFERENCE_SEED,
        proposal=Path(payload["proposal"]),
        requested_steps=SOURCE_MAXIMUM_ACTIONS,
        predecessor_session_id=payload.get("predecessor_session_id"),
    )
    failure = json.loads((source_root / "FAILURE.json").read_text())
    decision = payload.get("reach_and_grasp")
    if (
        reconstructed.to_dict() != payload
        or payload.get("applied_steps") != SOURCE_MAXIMUM_ACTIONS
        or payload.get("all_steps_applied") is not True
        or payload.get("orchestration_failure") is not None
        or not isinstance(decision, dict)
        or decision.get("passed") is not False
        or decision.get("failures") != ["no_attachment_transition"]
        or failure.get("error") != "report:exit_1"
        or failure.get("retry_authorized") is not False
    ):
        raise ValueError("V10 was not the exact bounded acquisition negative")
    return payload


def claim(
    checkpoint_root: Path,
    recovery_checkpoint_root: Path,
    data_root: Path,
    recovery_data_root: Path,
    followup_session_id: str,
    source_revision: str,
    expected_runtime_fingerprint: str,
) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if any(path.exists() for path in (evaluation_path, result_path, failure_path)):
        raise ValueError("contact-grasp acquisition resolution is already terminal")
    validate_source(checkpoint_root, data_root)
    validate_source(recovery_checkpoint_root, recovery_data_root)
    handoff = ContactGraspAcquisitionResolution(
        followup_session_id,
        runtime_fingerprint(),
        source_revision,
    )
    if handoff.runtime_fingerprint != expected_runtime_fingerprint:
        raise ValueError("contact-grasp acquisition resolution runtime changed")
    payload = handoff.to_dict()
    _write_exclusive(claim_path, payload)
    return payload


def encode_claim(checkpoint_root: Path) -> str:
    claim_path, _, _, _ = paths(checkpoint_root)
    payload = json.loads(claim_path.read_text())
    ContactGraspAcquisitionResolution.from_dict(payload)
    return b64encode(json.dumps(payload, sort_keys=True).encode()).decode()


def validate_diagnostic_evidence(
    payload: Mapping[str, Any],
    handoff: ContactGraspAcquisitionResolution,
    claim_fingerprint: str,
) -> dict[str, Any]:
    # Resolve after the persistent Isaac server has completed its ordered module
    # reload, rather than retaining a scale roster from the prior generation.
    from jepa_wm.control_safety import CONTACT_GRASP_COARSE_ACTION_SCALES

    selected = payload.get("selected_scale")
    attempts = payload.get("attempts")
    safe_attempts = (
        tuple(
            attempt
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get("passed") is True
        )
        if isinstance(attempts, list)
        else ()
    )
    allowed_scales = tuple(
        scale.to_dict() for scale in CONTACT_GRASP_COARSE_ACTION_SCALES
    )
    if (
        payload.get("schema")
        != "quantis.contact_grasp_acquisition_resolution_diagnostic.v1"
        or payload.get("status") != "passed_no_actuation"
        or payload.get("source_session_id") != SOURCE_SESSION_ID
        or payload.get("followup_session_id") != handoff.followup_session_id
        or payload.get("claim_fingerprint") != claim_fingerprint
        or payload.get("simulator_action_applied") is not False
        or not isinstance(selected, dict)
        or selected not in allowed_scales
        or not safe_attempts
        or safe_attempts[0].get("scale") != selected
        or not isinstance(selected.get("translation"), (int, float))
        or isinstance(selected.get("translation"), bool)
        or selected["translation"] <= 0.125
    ):
        raise ValueError("coarse acquisition diagnostic did not pass")
    return dict(payload)


def validate_diagnostic(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, _, _, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or failure_path.exists():
        raise ValueError("contact-grasp acquisition resolution claim is invalid")
    claim_payload = json.loads(claim_path.read_text())
    handoff = ContactGraspAcquisitionResolution.from_dict(claim_payload)
    payload = json.loads(diagnostic_path(data_root, handoff.followup_session_id).read_text())
    return validate_diagnostic_evidence(
        payload,
        handoff,
        artifact_fingerprint(claim_path),
    )


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("contact-grasp acquisition resolution claim is invalid")
    diagnostic = validate_diagnostic(checkpoint_root, data_root)
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
        "schema": "quantis.contact_grasp_acquisition_resolution_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "evaluation_passed": passed,
        "recovery_verified": False,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "diagnostic_fingerprint": artifact_fingerprint(
            diagnostic_path(data_root, report["steps"][0]["session"])
        ),
        "report_fingerprint": artifact_fingerprint(report_path),
        "applied_actions": report.get("applied_steps"),
        "reach_and_grasp": decision,
        "terminal_session_id": report.get("steps", [{}])[-1].get("session"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(evaluation_path, payload)
    if not passed:
        raise ValueError("contact-grasp acquisition resolution did not retain grasp")
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
        raise ValueError("contact-grasp acquisition resolution is already terminal")
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    recovery_report = recovery_data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    claim_payload = json.loads(claim_path.read_text())
    handoff = ContactGraspAcquisitionResolution.from_dict(claim_payload)
    primary_diagnostic = diagnostic_path(data_root, handoff.followup_session_id)
    recovery_diagnostic = diagnostic_path(recovery_data_root, handoff.followup_session_id)
    for primary, recovery in (
        (claim_path, recovery_claim),
        (evaluation_path, recovery_evaluation),
        (report_path, recovery_report),
        (primary_diagnostic, recovery_diagnostic),
    ):
        if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
            raise ValueError("contact-grasp acquisition resolution backup changed")
    evaluation = json.loads(evaluation_path.read_text())
    report = json.loads(report_path.read_text())
    for step in report.get("steps", ()):
        session_id = step.get("session") if isinstance(step, dict) else None
        if not isinstance(session_id, str):
            raise ValueError("contact-grasp acquisition resolution roster is invalid")
        for name in SESSION_FILES:
            primary = data_root / "control_sessions" / session_id / name
            recovery = recovery_data_root / "control_sessions" / session_id / name
            if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
                raise ValueError(f"acquisition resolution changed: {session_id}/{name}")
    payload = {
        "schema": "quantis.contact_grasp_acquisition_resolution_terminal.v1",
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
        raise ValueError("contact-grasp acquisition resolution evaluation failed")
    write_json_atomic(result_path, payload)
    write_json_atomic(recovery_result, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("contact-grasp acquisition resolution terminal backup changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or failure_path.exists() or not claim_path.is_file():
        raise ValueError("contact-grasp acquisition resolution failure is invalid")
    payload = {
        "schema": "quantis.contact_grasp_acquisition_resolution_failure.v1",
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
        choices=(
            "fingerprint",
            "claim",
            "encode",
            "validate-diagnostic",
            "evaluate",
            "finalize",
            "failure",
        ),
    )
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
        payload = claim(
            args.checkpoint_root,
            args.recovery_checkpoint_root,
            args.data_root,
            args.recovery_data_root,
            args.followup_session,
            args.source_revision,
            args.runtime_fingerprint,
        )
    elif args.command == "encode":
        payload = encode_claim(args.checkpoint_root)
    elif args.command == "validate-diagnostic":
        payload = validate_diagnostic(args.checkpoint_root, args.data_root)
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
