"""Frozen evidence for continuing after the terminal V5 no-motion IK block."""

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

from jepa_wm.contact_grasp_acquisition_handoff import (
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


HANDOFF_SCHEMA = "quantis.contact_grasp_acquisition_continuation.v1"
SOURCE_SESSION_ID = "unknown-start-e2e-v5-62605-grasp-23"
SOURCE_FINGERPRINTS = {
    "request.json": "ce5efd51bb06ed6310fe0ebb955083c1d0eaef4b4d15d7173c6de68fde00f4e1",
    "state.json": "9112fd3215bbb7b23b6fe1edca490db930379e91af51aa6827c3c6e6caa5e999",
    "response.json": "eeaa843d19e22bee79980cf5269b47b4f72d18450920be272fc85500f404cd6d",
    "execution_started.json": "592fb5b75d25c30a91f1c10e7114ddee61c50c8f5bd8b9b282271c0cbda40a74",
    "result.json": "aec27a720d948fee10d005d6d44cb0d34dfe38cdc9976ecf55a2638532b17d41",
    "context.png": "da2db99a5e02e04746a3e2c00e51531260338f483e67a0720dc440e72507b0d0",
}
PRIOR_ROLLOUT_ID = "unknown-start-e2e-v5-62605-grasp"
PRIOR_REPORT_FINGERPRINT = (
    "4796ebd06957deba06d0bc31435eb0c7a8d3c5cc668b9b5ef5814a1cb236dcec"
)
PRIOR_CLAIM_FINGERPRINT = (
    "43df308bcbe72589edc3386a8b7d11859dc7fe06d122d2302f6b7547f59028f2"
)
PRIOR_FAILURE_FINGERPRINT = (
    "b5c4bc6a66ebdc662bad1b45db7d51df32eb46c9149cae8bbce2ffe355d149d0"
)
EXPERIMENT_DIRECTORY = "unknown_start_acquisition_continuation_v6"
ROLLOUT_ID = "unknown-start-e2e-v6-62605-grasp"
MAXIMUM_ACTIONS = 52


RUNTIME_FILES = (
    "jepa_wm/action.py",
    "jepa_wm/contact_grasp_acquisition_handoff.py",
    "jepa_wm/contact_grasp_acquisition_continuation.py",
    "jepa_wm/contact_grasp_target.py",
    "jepa_wm/control_protocol.py",
    "jepa_wm/control_rollout.py",
    "jepa_wm/control_safety.py",
    "jepa_wm/control_server.py",
    "jepa_wm/control_worker.py",
    "jepa_wm/proposal.py",
    "jepa_wm/worker_artifacts.py",
    "ops/aws.sh",
    "ops/jepa_wm.sh",
    "ops/run_unknown_start_acquisition_continuation.sh",
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
        encoded = relative.encode()
        contents = (root / relative).read_bytes()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


@dataclass(frozen=True)
class ContactGraspAcquisitionContinuation:
    """Exact authority for one new experiment after V5 ended terminally."""

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
            raise ValueError("contact-grasp acquisition continuation is invalid")

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
            "prior_rollout_id": PRIOR_ROLLOUT_ID,
            "prior_report_fingerprint": PRIOR_REPORT_FINGERPRINT,
            "prior_claim_fingerprint": PRIOR_CLAIM_FINGERPRINT,
            "prior_failure_fingerprint": PRIOR_FAILURE_FINGERPRINT,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_revision": self.source_revision,
            "simulator_action_authorized": True,
            "filming_authorized": False,
            "production_authority_granted": False,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> ContactGraspAcquisitionContinuation:
        try:
            instance = cls(
                followup_session_id=str(payload["followup_session_id"]),
                runtime_fingerprint=str(payload["runtime_fingerprint"]),
                source_revision=str(payload["source_revision"]),
                schema=str(payload["schema"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "contact-grasp acquisition continuation is incomplete"
            ) from error
        if dict(payload) != instance.to_dict():
            raise ValueError("contact-grasp acquisition continuation changed")
        return instance


def _validate_source(checkpoint_root: Path, data_root: Path) -> None:
    source = data_root / "control_sessions" / SOURCE_SESSION_ID
    if any(
        artifact_fingerprint(source / name) != expected
        for name, expected in SOURCE_FINGERPRINTS.items()
    ):
        raise ValueError("contact-grasp continuation source changed")
    state = json.loads((source / "state.json").read_text())
    result = json.loads((source / "result.json").read_text())
    refresh = result.get("insertion_trial_refresh")
    live = refresh.get("live_state") if isinstance(refresh, dict) else None
    if (
        state.get("session_id") != SOURCE_SESSION_ID
        or state.get("reference_recording") != REFERENCE_RECORDING
        or state.get("seed") != REFERENCE_SEED
        or state.get("collision_detected") is not False
        or state.get("contact_force_newtons") != 0.0
        or state.get("plug_attached") is not False
        or state.get("active_drive_target") is None
        or result.get("status") != "blocked"
        or result.get("selected_action_scale") is not None
        or result.get("gate", {}).get("reasons")
        != ["joint_velocity_violation"]
        or "post_action_pose" in result
        or not isinstance(refresh.get("live_pose") if isinstance(refresh, dict) else None, list)
        or not isinstance(live, dict)
        or live.get("contact_force_newtons") != 0.0
        or live.get("collision_detected") is not False
        or live.get("plug_attached") is not False
    ):
        raise ValueError("continuation source was not a safe no-motion IK block")

    prior_report = data_root / "control_rollouts" / PRIOR_ROLLOUT_ID / "report.json"
    prior_claim = checkpoint_root / "unknown_start_acquisition_recovery_v5" / "CLAIM.json"
    prior_failure = checkpoint_root / "unknown_start_acquisition_recovery_v5" / "FAILURE.json"
    if (
        artifact_fingerprint(prior_report) != PRIOR_REPORT_FINGERPRINT
        or artifact_fingerprint(prior_claim) != PRIOR_CLAIM_FINGERPRINT
        or artifact_fingerprint(prior_failure) != PRIOR_FAILURE_FINGERPRINT
    ):
        raise ValueError("terminal V5 evidence changed")
    report = json.loads(prior_report.read_text())
    failure = json.loads(prior_failure.read_text())
    if (
        report.get("rollout_id") != PRIOR_ROLLOUT_ID
        or report.get("applied_steps") != 22
        or report.get("complete_steps") != 23
        or report.get("steps", [{}])[-1].get("session") != SOURCE_SESSION_ID
        or report.get("steps", [{}])[-1].get("status") != "blocked"
        or failure.get("status") != "failed"
        or failure.get("retry_authorized") is not False
    ):
        raise ValueError("V5 was not terminal at the exact continuation source")

    artifacts = (
        (checkpoint_root / f"{PROPOSAL_NAME}.pth", PROPOSAL_FINGERPRINT),
        (checkpoint_root / WORKER_NAME, WORKER_FINGERPRINT),
        (checkpoint_root / "experiments" / READINESS_NAME, READINESS_FINGERPRINT),
        (checkpoint_root / "experiments" / REPLAY_NAME, REPLAY_FINGERPRINT),
    )
    if any(artifact_fingerprint(path) != expected for path, expected in artifacts):
        raise ValueError("authenticated acquisition artifact changed")


def build_handoff(
    *,
    checkpoint_root: Path,
    data_root: Path,
    followup_session_id: str,
    source_revision: str,
) -> ContactGraspAcquisitionContinuation:
    _validate_source(checkpoint_root, data_root)
    return ContactGraspAcquisitionContinuation(
        followup_session_id=followup_session_id,
        runtime_fingerprint=runtime_fingerprint(),
        source_revision=source_revision,
    )


def paths(checkpoint_root: Path) -> tuple[Path, Path, Path, Path]:
    root = checkpoint_root / EXPERIMENT_DIRECTORY
    return tuple(
        root / name for name in ("CLAIM.json", "EVALUATION.json", "RESULT.json", "FAILURE.json")
    )


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("contact-grasp continuation was already claimed") from error
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
        raise ValueError("contact-grasp continuation is already terminal")
    handoff = build_handoff(
        checkpoint_root=checkpoint_root,
        data_root=data_root,
        followup_session_id=followup_session_id,
        source_revision=source_revision,
    )
    if handoff.runtime_fingerprint != expected_runtime_fingerprint:
        raise ValueError("contact-grasp continuation runtime changed")
    payload = handoff.to_dict()
    _write_exclusive(claim_path, payload)
    return payload


def encode_claim(checkpoint_root: Path) -> str:
    claim_path, _, _, _ = paths(checkpoint_root)
    payload = json.loads(claim_path.read_text())
    ContactGraspAcquisitionContinuation.from_dict(payload)
    return b64encode(json.dumps(payload, sort_keys=True).encode()).decode()


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("contact-grasp continuation claim is invalid")
    handoff = ContactGraspAcquisitionContinuation.from_dict(
        json.loads(claim_path.read_text())
    )
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
        "schema": "quantis.contact_grasp_acquisition_continuation_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "evaluation_passed": passed,
        "recovery_verified": False,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "report_fingerprint": artifact_fingerprint(report_path),
        "applied_actions": report.get("applied_steps"),
        "reach_and_grasp": decision,
        "terminal_session_id": (
            report.get("steps", [{}])[-1].get("session")
            if report.get("steps")
            else handoff.followup_session_id
        ),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(evaluation_path, payload)
    if not passed:
        raise ValueError("contact-grasp continuation did not retain grasp")
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
        raise ValueError("contact-grasp continuation is already terminal")
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    recovery_report = recovery_data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    for primary, recovery in (
        (claim_path, recovery_claim),
        (evaluation_path, recovery_evaluation),
        (report_path, recovery_report),
    ):
        if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
            raise ValueError("contact-grasp continuation backup changed")
    evaluation = json.loads(evaluation_path.read_text())
    report = json.loads(report_path.read_text())
    for step in report.get("steps", ()):
        session_id = step.get("session") if isinstance(step, dict) else None
        if not isinstance(session_id, str):
            raise ValueError("contact-grasp continuation roster is invalid")
        for name in (
            "request.json",
            "state.json",
            "response.json",
            "execution_started.json",
            "result.json",
            "context.png",
            "post_action.png",
        ):
            primary = data_root / "control_sessions" / session_id / name
            recovery = recovery_data_root / "control_sessions" / session_id / name
            if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
                raise ValueError(f"continuation changed: {session_id}/{name}")
    first_session = report["steps"][0]["session"]
    handoff = (
        data_root
        / "control_sessions"
        / SOURCE_SESSION_ID
        / f"acquisition_handoff_{first_session}.json"
    )
    recovery_handoff = (
        recovery_data_root
        / "control_sessions"
        / SOURCE_SESSION_ID
        / handoff.name
    )
    if artifact_fingerprint(handoff) != artifact_fingerprint(recovery_handoff):
        raise ValueError("contact-grasp continuation handoff backup changed")
    passed = evaluation.get("evaluation_passed") is True
    payload = {
        "schema": "quantis.contact_grasp_acquisition_continuation_terminal.v1",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "recovery_verified": True,
        "evaluation_fingerprint": artifact_fingerprint(evaluation_path),
        "report_fingerprint": artifact_fingerprint(report_path),
        "applied_actions": evaluation.get("applied_actions"),
        "terminal_session_id": evaluation.get("terminal_session_id"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(result_path, payload)
    write_json_atomic(recovery_result, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("contact-grasp continuation terminal backup changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or failure_path.exists() or not claim_path.is_file():
        raise ValueError("contact-grasp continuation failure is invalid")
    payload = {
        "schema": "quantis.contact_grasp_acquisition_continuation_failure.v1",
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
        choices=("fingerprint", "claim", "encode", "evaluate", "finalize", "failure"),
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
            args.data_root,
            args.followup_session,
            args.source_revision,
            args.runtime_fingerprint,
        )
    elif args.command == "encode":
        payload = encode_claim(args.checkpoint_root)
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
