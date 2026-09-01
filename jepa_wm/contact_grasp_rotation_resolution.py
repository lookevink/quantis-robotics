"""Authenticated V18 handoff after the V15 rotation-resolution rollback."""

from __future__ import annotations

import argparse
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
    REFERENCE_RECORDING,
    REFERENCE_SEED,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.training_artifact import artifact_fingerprint


HANDOFF_SCHEMA = "quantis.contact_grasp_rotation_resolution.v3"
EXPERIMENT_DIRECTORY = "unknown_start_rotation_resolution_v18"
ROLLOUT_ID = "unknown-start-e2e-v18-62605-grasp"
SOURCE_ROLLOUT_ID = "unknown-start-e2e-v15-62605-grasp"
SOURCE_SESSION_ID = f"{SOURCE_ROLLOUT_ID}-36"
ROLLED_BACK_SESSION_ID = f"{SOURCE_ROLLOUT_ID}-37"
SOURCE_PREDECESSOR_SESSION_ID = "unknown-start-e2e-v10-62605-grasp-52"
MAXIMUM_ACTIONS = 52
SOURCE_REPORT_FINGERPRINT = (
    "7ab01b20c3b4ded1b045a43392611d9a113966c3943781efbbd373b5b354afa3"
)
SOURCE_CLAIM_FINGERPRINT = (
    "0586509485913386506aa7804aaeb66af744343e8e081b6577ab5e21a99da7ff"
)
SOURCE_FAILURE_FINGERPRINT = (
    "4e2ce2af3e80d2af876e3c475bd31ba59373f8e94eb60e7e00392f3d918ee370"
)
SOURCE_ROSTER_FINGERPRINT = (
    "f6d3d15ae888732238568112c4b482ee1a7fde35ea03eb9469ff778ad6a44a0a"
)
V16_EXPERIMENT_DIRECTORY = "unknown_start_rotation_resolution_v16"
V16_ROLLOUT_ID = "unknown-start-e2e-v16-62605-grasp"
V16_CLAIM_FINGERPRINT = (
    "5786115a58256886d630001ac3bc5f24ae605b41532172e31ae2c64e42768b33"
)
V16_FAILURE_FINGERPRINT = (
    "3aaca80b884739a3f028a0482934c3500c34a8ccbc77bcf0fda261b99a9ca54c"
)
V17_EXPERIMENT_DIRECTORY = "unknown_start_rotation_resolution_v17"
V17_ROLLOUT_ID = "unknown-start-e2e-v17-62605-grasp"
V17_CLAIM_FINGERPRINT = (
    "ae02bbe88eebe60909600676894815f88be8d102be39aa2e622bb93c36451ad2"
)
V17_FAILURE_FINGERPRINT = (
    "6e2babc73c6ac1ec8b63d644f9e12e45f8c80b855f3e1a50fdfec6868e13f90c"
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
    "jepa_wm/contact_grasp_rotation_resolution.py",
    "jepa_wm/contact_grasp_target.py",
    "jepa_wm/control_protocol.py",
    "jepa_wm/control_rollout.py",
    "jepa_wm/control_safety.py",
    "jepa_wm/control_worker.py",
    "ops/aws.sh",
    "ops/jepa_wm.sh",
    "ops/run_unknown_start_rotation_resolution.sh",
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
    for index in range(1, 38):
        session = data_root / "control_sessions" / f"{SOURCE_ROLLOUT_ID}-{index:02d}"
        for filename in SESSION_FILES:
            digest.update((artifact_fingerprint(session / filename) + "\n").encode())
    return digest.hexdigest()


def handoff_path(data_root: Path, followup_session_id: str) -> Path:
    return (
        data_root
        / "control_sessions"
        / SOURCE_SESSION_ID
        / f"acquisition_handoff_{followup_session_id}.json"
    )


def rollback_drive_target(data_root: Path):
    """Canonicalize the exact pre-action command retained by the V15 rollback."""

    from jepa_wm.joint_drive import JointDriveTarget
    from sim.control_session import ControlSession

    session = ControlSession.at(
        data_root / "control_sessions",
        ROLLED_BACK_SESSION_ID,
    )
    result = session.load_result()
    refresh = result.insertion_trial_refresh
    if refresh is None:
        raise ValueError("V15 rollback pre-action refresh is missing")
    return JointDriveTarget.for_command(
        tuple(refresh.live_state.joint_positions),
        refresh.live_state.gripper_width_m,
    )


@dataclass(frozen=True)
class ContactGraspRotationResolution:
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
            raise ValueError("contact-grasp rotation resolution is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_rollout_id": SOURCE_ROLLOUT_ID,
            "source_session_id": SOURCE_SESSION_ID,
            "rolled_back_session_id": ROLLED_BACK_SESSION_ID,
            "followup_session_id": self.followup_session_id,
            "reference_recording": REFERENCE_RECORDING,
            "reference_seed": REFERENCE_SEED,
            "proposal_name": PROPOSAL_NAME,
            "proposal_fingerprint": PROPOSAL_FINGERPRINT,
            "source_report_fingerprint": SOURCE_REPORT_FINGERPRINT,
            "source_claim_fingerprint": SOURCE_CLAIM_FINGERPRINT,
            "source_failure_fingerprint": SOURCE_FAILURE_FINGERPRINT,
            "source_roster_fingerprint": SOURCE_ROSTER_FINGERPRINT,
            "v16_claim_fingerprint": V16_CLAIM_FINGERPRINT,
            "v16_failure_fingerprint": V16_FAILURE_FINGERPRINT,
            "v16_simulator_action_applied": False,
            "v17_claim_fingerprint": V17_CLAIM_FINGERPRINT,
            "v17_failure_fingerprint": V17_FAILURE_FINGERPRINT,
            "v17_simulator_action_applied": False,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_revision": self.source_revision,
            "simulator_action_authorized": True,
            "filming_authorized": False,
            "production_authority_granted": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContactGraspRotationResolution:
        try:
            instance = cls(
                str(payload["followup_session_id"]),
                str(payload["runtime_fingerprint"]),
                str(payload["source_revision"]),
                str(payload["schema"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("contact-grasp rotation resolution is incomplete") from error
        if dict(payload) != instance.to_dict():
            raise ValueError("contact-grasp rotation resolution changed")
        return instance


def paths(checkpoint_root: Path) -> tuple[Path, Path, Path, Path]:
    root = checkpoint_root / EXPERIMENT_DIRECTORY
    return root / "CLAIM.json", root / "EVALUATION.json", root / "RESULT.json", root / "FAILURE.json"


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("contact-grasp rotation resolution was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_source(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    from jepa_wm.control_rollout import ControlRolloutReport, ControlStepSummary
    from sim.control_session import ControlResultStatus, ControlSession

    source_root = checkpoint_root / "unknown_start_acquisition_resolution_v15"
    report_path = data_root / "control_rollouts" / SOURCE_ROLLOUT_ID / "report.json"
    if (
        artifact_fingerprint(source_root / "CLAIM.json") != SOURCE_CLAIM_FINGERPRINT
        or artifact_fingerprint(source_root / "FAILURE.json") != SOURCE_FAILURE_FINGERPRINT
        or artifact_fingerprint(report_path) != SOURCE_REPORT_FINGERPRINT
        or source_roster_fingerprint(data_root) != SOURCE_ROSTER_FINGERPRINT
    ):
        raise ValueError("terminal V15 rotation evidence changed")
    payload = json.loads(report_path.read_text())
    session_ids = tuple(step["session"] for step in payload.get("steps", ()))
    reconstructed = ControlRolloutReport.from_sessions(
        data_root,
        SOURCE_ROLLOUT_ID,
        session_ids,
        reference_recording=REFERENCE_RECORDING,
        seed=REFERENCE_SEED,
        proposal=Path(payload["proposal"]),
        requested_steps=MAXIMUM_ACTIONS,
        predecessor_session_id=SOURCE_PREDECESSOR_SESSION_ID,
    )
    source = ControlStepSummary.from_session(
        ControlSession.at(data_root / "control_sessions", SOURCE_SESSION_ID)
    )
    failed = ControlStepSummary.from_session(
        ControlSession.at(data_root / "control_sessions", ROLLED_BACK_SESSION_ID)
    )
    failure = json.loads((source_root / "FAILURE.json").read_text())
    post = failed.result.post_action
    if (
        reconstructed.to_dict() != payload
        or payload.get("applied_steps") != 36
        or payload.get("complete_steps") != 37
        or payload.get("translation_progress_meters", 0.0) <= 0.04
        or source.result.status is not ControlResultStatus.APPLIED
        or failed.result.status is not ControlResultStatus.ROLLED_BACK_TRACKING
        or failed.state.previous_session_id != SOURCE_SESSION_ID
        or post is None
        or tuple(reason.value for reason in post.tracking.reasons)
        != ("rotation_direction",)
        or post.contact_force_newtons != 0.0
        or post.collision_detected
        or post.plug_attached
        or failure.get("error") != "report:exit_1"
        or failure.get("retry_authorized") is not False
    ):
        raise ValueError("V15 was not the exact safe rotation rollback")
    return payload


def validate_v16_no_action(checkpoint_root: Path, data_root: Path) -> None:
    root = checkpoint_root / V16_EXPERIMENT_DIRECTORY
    claim_path = root / "CLAIM.json"
    failure_path = root / "FAILURE.json"
    if (
        artifact_fingerprint(claim_path) != V16_CLAIM_FINGERPRINT
        or artifact_fingerprint(failure_path) != V16_FAILURE_FINGERPRINT
    ):
        raise ValueError("terminal no-action V16 evidence changed")
    failure_payload = json.loads(failure_path.read_text())
    if (
        failure_payload.get("error") != "restore:exit_1"
        or failure_payload.get("retry_authorized") is not False
        or (data_root / "control_rollouts" / V16_ROLLOUT_ID).exists()
        or any(
            (data_root / "control_sessions" / f"{V16_ROLLOUT_ID}-{index:02d}").exists()
            for index in range(1, MAXIMUM_ACTIONS + 1)
        )
    ):
        raise ValueError("V16 was not the exact no-action ownership failure")


def validate_v17_no_action(checkpoint_root: Path, data_root: Path) -> None:
    root = checkpoint_root / V17_EXPERIMENT_DIRECTORY
    claim_path = root / "CLAIM.json"
    failure_path = root / "FAILURE.json"
    if (
        artifact_fingerprint(claim_path) != V17_CLAIM_FINGERPRINT
        or artifact_fingerprint(failure_path) != V17_FAILURE_FINGERPRINT
    ):
        raise ValueError("terminal no-action V17 evidence changed")
    failure_payload = json.loads(failure_path.read_text())
    if (
        failure_payload.get("error") != "capture_01:exit_1"
        or failure_payload.get("retry_authorized") is not False
        or (data_root / "control_rollouts" / V17_ROLLOUT_ID).exists()
        or any(
            (data_root / "control_sessions" / f"{V17_ROLLOUT_ID}-{index:02d}").exists()
            for index in range(1, MAXIMUM_ACTIONS + 1)
        )
    ):
        raise ValueError("V17 was not the exact no-action handoff failure")


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
        raise ValueError("contact-grasp rotation resolution is already terminal")
    validate_source(checkpoint_root, data_root)
    validate_source(recovery_checkpoint_root, recovery_data_root)
    validate_v16_no_action(checkpoint_root, data_root)
    validate_v16_no_action(recovery_checkpoint_root, recovery_data_root)
    validate_v17_no_action(checkpoint_root, data_root)
    validate_v17_no_action(recovery_checkpoint_root, recovery_data_root)
    handoff = ContactGraspRotationResolution(
        followup_session_id,
        runtime_fingerprint(),
        source_revision,
    )
    if handoff.runtime_fingerprint != expected_runtime_fingerprint:
        raise ValueError("contact-grasp rotation-resolution runtime changed")
    payload = handoff.to_dict()
    _write_exclusive(claim_path, payload)
    return payload


def encode_claim(checkpoint_root: Path) -> str:
    from base64 import b64encode

    claim_path, _, _, _ = paths(checkpoint_root)
    payload = json.loads(claim_path.read_text())
    ContactGraspRotationResolution.from_dict(payload)
    return b64encode(json.dumps(payload, sort_keys=True).encode()).decode()


def validate_handoff(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, _, _, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or failure_path.exists():
        raise ValueError("contact-grasp rotation-resolution claim is invalid")
    handoff = ContactGraspRotationResolution.from_dict(
        json.loads(claim_path.read_text())
    )
    payload = json.loads(handoff_path(data_root, handoff.followup_session_id).read_text())
    if payload != handoff.to_dict():
        raise ValueError("contact-grasp rotation-resolution handoff changed")
    return payload


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    from jepa_wm.control_rollout import ControlRolloutReport

    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("contact-grasp rotation-resolution claim is invalid")
    handoff = ContactGraspRotationResolution.from_dict(json.loads(claim_path.read_text()))
    validate_handoff(checkpoint_root, data_root)
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    report = json.loads(report_path.read_text())
    sessions = tuple(step["session"] for step in report.get("steps", ()))
    reconstructed = ControlRolloutReport.from_sessions(
        data_root,
        ROLLOUT_ID,
        sessions,
        reference_recording=REFERENCE_RECORDING,
        seed=REFERENCE_SEED,
        proposal=Path(report["proposal"]),
        requested_steps=MAXIMUM_ACTIONS,
        predecessor_session_id=SOURCE_SESSION_ID,
    )
    decision = report.get("reach_and_grasp")
    passed = (
        reconstructed.to_dict() == report
        and report.get("rollout_id") == ROLLOUT_ID
        and report.get("predecessor_session_id") == SOURCE_SESSION_ID
        and report.get("orchestration_failure") is None
        and isinstance(decision, dict)
        and decision.get("passed") is True
        and report.get("applied_steps") == report.get("complete_steps")
        and 1 <= report.get("applied_steps", 0) <= MAXIMUM_ACTIONS
    )
    payload = {
        "schema": "quantis.contact_grasp_rotation_resolution_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "evaluation_passed": passed,
        "recovery_verified": False,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "handoff_fingerprint": artifact_fingerprint(
            handoff_path(data_root, handoff.followup_session_id)
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
        raise ValueError("contact-grasp rotation resolution did not retain grasp")
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
        raise ValueError("contact-grasp rotation resolution is already terminal")
    handoff = ContactGraspRotationResolution.from_dict(json.loads(claim_path.read_text()))
    report_path = data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    recovery_report = recovery_data_root / "control_rollouts" / ROLLOUT_ID / "report.json"
    for primary, recovery in (
        (claim_path, recovery_claim),
        (evaluation_path, recovery_evaluation),
        (report_path, recovery_report),
        (
            handoff_path(data_root, handoff.followup_session_id),
            handoff_path(recovery_data_root, handoff.followup_session_id),
        ),
    ):
        if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
            raise ValueError("contact-grasp rotation-resolution backup changed")
    report = json.loads(report_path.read_text())
    for step in report.get("steps", ()):
        session_id = step.get("session")
        if not isinstance(session_id, str):
            raise ValueError("contact-grasp rotation-resolution roster is invalid")
        for filename in SESSION_FILES:
            primary = data_root / "control_sessions" / session_id / filename
            recovery = recovery_data_root / "control_sessions" / session_id / filename
            if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
                raise ValueError(f"rotation-resolution changed: {session_id}/{filename}")
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation.get("evaluation_passed") is not True:
        raise ValueError("contact-grasp rotation-resolution evaluation failed")
    payload = {
        "schema": "quantis.contact_grasp_rotation_resolution_terminal.v1",
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
    write_json_atomic(result_path, payload)
    write_json_atomic(recovery_result, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("contact-grasp rotation-resolution result backup changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or failure_path.exists() or not claim_path.is_file():
        raise ValueError("contact-grasp rotation-resolution failure is invalid")
    payload = {
        "schema": "quantis.contact_grasp_rotation_resolution_failure.v1",
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
        choices=("fingerprint", "claim", "encode", "validate-handoff", "evaluate", "finalize", "failure"),
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
    elif args.command == "validate-handoff":
        payload = validate_handoff(args.checkpoint_root, args.data_root)
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
