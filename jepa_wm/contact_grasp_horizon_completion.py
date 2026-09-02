"""Authenticated V35 continuation after exact local-IK branch recovery."""

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

from jepa_wm.persistence import write_json_atomic
from jepa_wm.training_artifact import artifact_fingerprint


HANDOFF_SCHEMA = "quantis.contact_grasp_horizon_completion.v16"
FAILURE_SCHEMA = "quantis.contact_grasp_horizon_completion_failure.v16"
EXPERIMENT_DIRECTORY = "unknown_start_horizon_completion_v35"
ROLLOUT_ID = "unknown-start-e2e-v35-62605-grasp"
SOURCE_ROLLOUT_ID = "unknown-start-e2e-v34-62605-grasp"
SOURCE_SESSION_ID = f"{SOURCE_ROLLOUT_ID}-001"
RUNTIME_OWNER_SESSION_ID = SOURCE_SESSION_ID
SOURCE_PREDECESSOR_SESSION_ID = "unknown-start-e2e-v33-62605-grasp-002"
SOURCE_ENDPOINT_STATUS = "blocked"
SOURCE_TRACKING_REASONS = ("joint_velocity_violation",)
SOURCE_EXECUTION_ERROR = None
SOURCE_CLAIM_EXECUTION_ERROR = (
    "RuntimeError: contact-grasp command did not satisfy its tracking gates "
    "within its bounded timeout: tracking_reasons=['translation_direction'], "
    "translation_error_meters=0.000466207, "
    "rotation_error_radians=0.001737717, joint_error_radians=0.001047902, "
    "gripper_error_meters=0.000114093"
)
REFERENCE_RECORDING = "contact-insertion-v10-drive-slow-2600-held-00"
REFERENCE_SEED = 12600
PROPOSAL_NAME = (
    "contact-grasp-acquisition-v10-drive-slow-2600_"
    "task12_h256_s3000_cfopen-v3-retained"
)
PROPOSAL_FINGERPRINT = (
    "0275f70f90dad8126f31046a4d153a1aaaf0f2ea105a085fee10485ae1f98dbe"
)
READINESS_FINGERPRINT = (
    "957da24f472d332eec54cc1f4eef6ecfcb0e45b58853fb47153b4e75d113163b"
)
WORKER_IDENTITY = "contact-insertion-v10-unknown-start-acquisition-v3-retained"
WORKER_FINGERPRINT = (
    "6e34cf0f1cd6ad3a894d18fe2f157b3a33802e4a3a19d45350e019a6b86401ed"
)
SOURCE_CLAIM_FINGERPRINT = (
    "9cb9634b37a18ffad1e0a71b8593e63d67bdacc667d02685fdd6e3ede168881c"
)
SOURCE_FAILURE_FINGERPRINT = (
    "c79979a1cc08a04bb481b34c1c9c1ba4920914b00b94d1d75da2e58fe90b17b9"
)
SOURCE_REPORT_FINGERPRINT = (
    "d600b31ba213a6ef0f6dcbd821ceddd5f1cb5a05b08cae979f246ae5d0dca4ac"
)
SOURCE_ROSTER_FINGERPRINT = (
    "c0dee97c8771e4bc4dea4f6fa5610b9c5e8930a3d38415f6ece1b16401f7a9b1"
)
SOURCE_RUNTIME_FINGERPRINT = (
    "9f31895062c747fdf857e16c9566d3d7858febbc6e6047c4b3e5a0bb60f3ac08"
)
SOURCE_REVISION = "f888f27fcb4a16ddbf58c3723f0adc9c58d44208"
SOURCE_SESSION_COUNT = 1
SOURCE_APPLIED_ACTIONS = 0
SOURCE_CUMULATIVE_APPLIED_ACTIONS = 188
SOURCE_HORIZON_ACTIONS = 147
MAXIMUM_ACTIONS = 45
SESSION_FILES = (
    "request.json",
    "state.json",
    "response.json",
    "execution_started.json",
    "result.json",
    "context.png",
    "post_action.png",
)
SOURCE_TERMINAL_SESSION_FILES = SESSION_FILES[:-1]
RUNTIME_FILES = (
    "jepa_wm/action.py",
    "jepa_wm/contact_grasp_horizon_completion.py",
    "jepa_wm/contact_grasp_target.py",
    "jepa_wm/control_protocol.py",
    "jepa_wm/control_rollout.py",
    "jepa_wm/control_safety.py",
    "jepa_wm/control_worker.py",
    "jepa_wm/grasp_task.py",
    "jepa_wm/task_windows.py",
    "ops/aws.sh",
    "ops/jepa_wm.sh",
    "ops/run_unknown_start_horizon_completion.sh",
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
    for index in range(1, SOURCE_SESSION_COUNT + 1):
        session = (
            data_root
            / "control_sessions"
            / f"{SOURCE_ROLLOUT_ID}-{index:03d}"
        )
        filenames = (
            SESSION_FILES
            if index <= SOURCE_APPLIED_ACTIONS
            else SOURCE_TERMINAL_SESSION_FILES
        )
        for filename in filenames:
            digest.update((artifact_fingerprint(session / filename) + "\n").encode())
    return digest.hexdigest()


def proposal_path(checkpoint_root: Path) -> Path:
    return checkpoint_root / f"{PROPOSAL_NAME}.pth"


def readiness_path(checkpoint_root: Path) -> Path:
    return (
        checkpoint_root
        / "experiments"
        / f"{PROPOSAL_NAME}_contact_grasp_acquisition_readiness.json"
    )


def worker_path(checkpoint_root: Path) -> Path:
    return checkpoint_root / f"{WORKER_IDENTITY}.worker.json"


def handoff_path(data_root: Path, followup_session_id: str) -> Path:
    return (
        data_root
        / "control_sessions"
        / SOURCE_SESSION_ID
        / f"acquisition_handoff_{followup_session_id}.json"
    )


def rollback_drive_target(result: Any):
    """Rebuild the target from the synchronized pre-action refresh."""

    from jepa_wm.joint_drive import JointDriveTarget

    refresh = result.insertion_trial_refresh
    if refresh is None:
        raise ValueError("contact-grasp rollback refresh is missing")
    return JointDriveTarget.for_command(
        tuple(refresh.live_state.joint_positions),
        refresh.live_state.gripper_width_m,
    )


def retained_drive_target(data_root: Path):
    """Return the unchanged active drive command captured by V34."""

    from sim.control_session import ControlSession

    _, state = ControlSession.at(
        data_root / "control_sessions", SOURCE_SESSION_ID
    ).load_capture()
    if state.active_drive_target is None:
        raise ValueError("contact-grasp blocked drive target is missing")
    return state.active_drive_target


@dataclass(frozen=True)
class ContactGraspHorizonCompletion:
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
            raise ValueError("contact-grasp horizon completion is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_rollout_id": SOURCE_ROLLOUT_ID,
            "source_session_id": SOURCE_SESSION_ID,
            "runtime_owner_session_id": RUNTIME_OWNER_SESSION_ID,
            "source_endpoint_status": SOURCE_ENDPOINT_STATUS,
            "source_tracking_reasons": list(SOURCE_TRACKING_REASONS),
            "source_execution_error": SOURCE_EXECUTION_ERROR,
            "followup_session_id": self.followup_session_id,
            "reference_recording": REFERENCE_RECORDING,
            "reference_seed": REFERENCE_SEED,
            "proposal_name": PROPOSAL_NAME,
            "proposal_fingerprint": PROPOSAL_FINGERPRINT,
            "readiness_fingerprint": READINESS_FINGERPRINT,
            "worker_identity": WORKER_IDENTITY,
            "worker_fingerprint": WORKER_FINGERPRINT,
            "source_claim_fingerprint": SOURCE_CLAIM_FINGERPRINT,
            "source_failure_fingerprint": SOURCE_FAILURE_FINGERPRINT,
            "source_report_fingerprint": SOURCE_REPORT_FINGERPRINT,
            "source_roster_fingerprint": SOURCE_ROSTER_FINGERPRINT,
            "source_attempted_actions": SOURCE_SESSION_COUNT,
            "source_applied_actions": SOURCE_APPLIED_ACTIONS,
            "source_cumulative_applied_actions": SOURCE_CUMULATIVE_APPLIED_ACTIONS,
            "source_horizon_actions": SOURCE_HORIZON_ACTIONS,
            "maximum_actions": MAXIMUM_ACTIONS,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_revision": self.source_revision,
            "simulator_action_authorized": True,
            "filming_authorized": False,
            "production_authority_granted": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContactGraspHorizonCompletion:
        try:
            instance = cls(
                str(payload["followup_session_id"]),
                str(payload["runtime_fingerprint"]),
                str(payload["source_revision"]),
                str(payload["schema"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("contact-grasp horizon completion is incomplete") from error
        if dict(payload) != instance.to_dict():
            raise ValueError("contact-grasp horizon completion changed")
        return instance


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
        raise ValueError("contact-grasp horizon completion was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def validate_model(checkpoint_root: Path) -> dict[str, Any]:
    proposal = proposal_path(checkpoint_root)
    readiness = readiness_path(checkpoint_root)
    if (
        artifact_fingerprint(proposal) != PROPOSAL_FINGERPRINT
        or artifact_fingerprint(readiness) != READINESS_FINGERPRINT
        or artifact_fingerprint(worker_path(checkpoint_root)) != WORKER_FINGERPRINT
    ):
        raise ValueError("contact-grasp horizon model artifact changed")
    payload = json.loads(readiness.read_text())
    aggregate = payload.get("aggregate")
    if (
        payload.get("passed") is not True
        or payload.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or not isinstance(aggregate, dict)
        or aggregate.get("rollouts") != 252
        or aggregate.get("first_action_gate_pass_rate") != 1.0
        or aggregate.get("active_first_action_direction_pass_rate") != 1.0
    ):
        raise ValueError("contact-grasp horizon model did not pass readiness")
    return {
        "proposal_fingerprint": PROPOSAL_FINGERPRINT,
        "readiness_fingerprint": READINESS_FINGERPRINT,
        "worker_fingerprint": WORKER_FINGERPRINT,
    }


def validate_source(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    from jepa_wm.control_rollout import ControlRolloutReport
    from sim.control_session import ControlResultStatus

    source_root = checkpoint_root / "unknown_start_horizon_completion_v34"
    report_path = data_root / "control_rollouts" / SOURCE_ROLLOUT_ID / "report.json"
    if (
        artifact_fingerprint(source_root / "CLAIM.json")
        != SOURCE_CLAIM_FINGERPRINT
        or artifact_fingerprint(source_root / "FAILURE.json")
        != SOURCE_FAILURE_FINGERPRINT
        or artifact_fingerprint(report_path) != SOURCE_REPORT_FINGERPRINT
        or source_roster_fingerprint(data_root) != SOURCE_ROSTER_FINGERPRINT
    ):
        raise ValueError("V34 terminal evidence changed")
    claim = json.loads((source_root / "CLAIM.json").read_text())
    failure = json.loads((source_root / "FAILURE.json").read_text())
    report = json.loads(report_path.read_text())
    sessions = tuple(step["session"] for step in report.get("steps", ()))
    reconstructed = ControlRolloutReport.from_sessions(
        data_root,
        SOURCE_ROLLOUT_ID,
        sessions,
        reference_recording=REFERENCE_RECORDING,
        seed=REFERENCE_SEED,
        proposal=(
            Path("/home/ubuntu/docker/jepa-wm/checkpoints")
            / f"{PROPOSAL_NAME}.pth"
        ),
        requested_steps=46,
        predecessor_session_id=SOURCE_PREDECESSOR_SESSION_ID,
    )
    step = reconstructed.complete_steps[-1]
    refresh = step.result.insertion_trial_refresh
    attempts = step.result.projection_attempts
    if (
        claim.get("schema") != "quantis.contact_grasp_horizon_completion.v15"
        or claim.get("source_rollout_id")
        != "unknown-start-e2e-v33-62605-grasp"
        or claim.get("source_session_id") != SOURCE_PREDECESSOR_SESSION_ID
        or claim.get("runtime_owner_session_id") != SOURCE_PREDECESSOR_SESSION_ID
        or claim.get("source_endpoint_status")
        != "rolled_back_after_execution_failure"
        or claim.get("source_tracking_reasons") != []
        or claim.get("source_execution_error") != SOURCE_CLAIM_EXECUTION_ERROR
        or claim.get("followup_session_id")
        != "unknown-start-e2e-v34-62605-grasp-001"
        or claim.get("maximum_actions") != 46
        or claim.get("runtime_fingerprint") != SOURCE_RUNTIME_FINGERPRINT
        or claim.get("source_revision") != SOURCE_REVISION
        or claim.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or reconstructed.to_dict() != report
        or sessions
        != tuple(
            f"{SOURCE_ROLLOUT_ID}-{index:03d}"
            for index in range(1, SOURCE_SESSION_COUNT + 1)
        )
        or report.get("complete_steps") != SOURCE_SESSION_COUNT
        or report.get("attempted_steps") != SOURCE_SESSION_COUNT
        or report.get("applied_steps") != SOURCE_APPLIED_ACTIONS
        or report.get("orchestration_failure") is not None
        or reconstructed.reach_and_grasp is not None
        or step.result.status is not ControlResultStatus.BLOCKED
        or step.state.previous_session_id != SOURCE_PREDECESSOR_SESSION_ID
        or str(step.observation.target_frame)
        != "recordings/contact-insertion-v10-drive-slow-2600-held-00/wrist/frame_000113.png"
        or step.result.post_action is not None
        or step.result.execution_error != SOURCE_EXECUTION_ERROR
        or step.result.selected_action_scale is not None
        or tuple(reason.value for reason in step.result.gate.reasons)
        != SOURCE_TRACKING_REASONS
        or len(attempts) != 4
        or tuple(
            (
                attempt.scale.translation,
                attempt.scale.rotation,
                attempt.scale.gripper,
                tuple(reason.value for reason in attempt.gate.reasons),
                attempt.maximum_joint_delta_rad,
            )
            for attempt in attempts
        )
        != (
            (
                0.30668039348262693,
                0.0,
                0.0625,
                SOURCE_TRACKING_REASONS,
                1.715502774471806,
            ),
            (
                0.30668039348262693,
                0.0,
                0.0625,
                SOURCE_TRACKING_REASONS,
                1.715502774471806,
            ),
            (
                0.20445359565508464,
                0.0,
                0.0625,
                SOURCE_TRACKING_REASONS,
                1.7168039041760328,
            ),
            (
                0.20445359565508464,
                0.0,
                0.0625,
                SOURCE_TRACKING_REASONS,
                1.7168039041760328,
            ),
        )
        or refresh is None
        or refresh.live_state.plug_attached
        or refresh.live_state.collision_detected
        or refresh.live_state.contact_force_newtons != 0.0
        or retained_drive_target(data_root) != step.state.active_drive_target
        or failure.get("error") != "report:exit_1"
        or failure.get("claim_fingerprint") != SOURCE_CLAIM_FINGERPRINT
        or failure.get("retry_authorized") is not False
    ):
        raise ValueError("V34 was not the exact no-motion local-IK block")
    validate_model(checkpoint_root)
    return {
        "source_session_id": SOURCE_SESSION_ID,
        "source_roster_fingerprint": SOURCE_ROSTER_FINGERPRINT,
        "source_attempted_actions": SOURCE_SESSION_COUNT,
        "source_applied_actions": SOURCE_APPLIED_ACTIONS,
        "source_cumulative_applied_actions": SOURCE_CUMULATIVE_APPLIED_ACTIONS,
        "source_horizon_actions": SOURCE_HORIZON_ACTIONS,
        "source_target_frame": str(step.observation.target_frame),
    }


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
        raise ValueError("contact-grasp horizon completion is already terminal")
    validate_source(checkpoint_root, data_root)
    validate_source(recovery_checkpoint_root, recovery_data_root)
    handoff = ContactGraspHorizonCompletion(
        followup_session_id,
        runtime_fingerprint(),
        source_revision,
    )
    if handoff.runtime_fingerprint != expected_runtime_fingerprint:
        raise ValueError("contact-grasp horizon runtime changed")
    payload = handoff.to_dict()
    _write_exclusive(claim_path, payload)
    return payload


def encode_claim(checkpoint_root: Path) -> str:
    claim_path, _, _, _ = paths(checkpoint_root)
    payload = json.loads(claim_path.read_text())
    ContactGraspHorizonCompletion.from_dict(payload)
    return b64encode(json.dumps(payload, sort_keys=True).encode()).decode()


def validate_handoff(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, _, _, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or failure_path.exists():
        raise ValueError("contact-grasp horizon claim is invalid")
    handoff = ContactGraspHorizonCompletion.from_dict(
        json.loads(claim_path.read_text())
    )
    payload = json.loads(handoff_path(data_root, handoff.followup_session_id).read_text())
    if payload != handoff.to_dict():
        raise ValueError("contact-grasp horizon handoff changed")
    return payload


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    from jepa_wm.control_rollout import ControlRolloutReport

    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("contact-grasp horizon claim is invalid")
    handoff = ContactGraspHorizonCompletion.from_dict(json.loads(claim_path.read_text()))
    validate_handoff(checkpoint_root, data_root)
    validate_model(checkpoint_root)
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
        "schema": "quantis.contact_grasp_horizon_completion_evaluation.v16",
        "status": "evaluated_pending_recovery",
        "evaluation_passed": passed,
        "recovery_verified": False,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "handoff_fingerprint": artifact_fingerprint(
            handoff_path(data_root, handoff.followup_session_id)
        ),
        "report_fingerprint": artifact_fingerprint(report_path),
        "proposal_fingerprint": PROPOSAL_FINGERPRINT,
        "applied_actions": report.get("applied_steps"),
        "reach_and_grasp": decision,
        "terminal_session_id": report.get("steps", [{}])[-1].get("session"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(evaluation_path, payload)
    if not passed:
        raise ValueError("contact-grasp horizon completion did not retain grasp")
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
        raise ValueError("contact-grasp horizon completion is already terminal")
    handoff = ContactGraspHorizonCompletion.from_dict(json.loads(claim_path.read_text()))
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
        (proposal_path(checkpoint_root), proposal_path(recovery_checkpoint_root)),
        (readiness_path(checkpoint_root), readiness_path(recovery_checkpoint_root)),
        (worker_path(checkpoint_root), worker_path(recovery_checkpoint_root)),
    ):
        if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
            raise ValueError("contact-grasp horizon backup changed")
    report = json.loads(report_path.read_text())
    for step in report.get("steps", ()):
        session_id = step.get("session")
        if not isinstance(session_id, str):
            raise ValueError("contact-grasp horizon roster is invalid")
        for filename in SESSION_FILES:
            primary = data_root / "control_sessions" / session_id / filename
            recovery = recovery_data_root / "control_sessions" / session_id / filename
            if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
                raise ValueError(f"contact-grasp horizon changed: {session_id}/{filename}")
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation.get("evaluation_passed") is not True:
        raise ValueError("contact-grasp horizon evaluation failed")
    payload = {
        "schema": "quantis.contact_grasp_horizon_completion_terminal.v16",
        "status": "passed",
        "passed": True,
        "recovery_verified": True,
        "evaluation_fingerprint": artifact_fingerprint(evaluation_path),
        "report_fingerprint": artifact_fingerprint(report_path),
        "proposal_fingerprint": PROPOSAL_FINGERPRINT,
        "applied_actions": evaluation.get("applied_actions"),
        "terminal_session_id": evaluation.get("terminal_session_id"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(result_path, payload)
    write_json_atomic(recovery_result, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("contact-grasp horizon result backup changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or not claim_path.is_file():
        raise ValueError("contact-grasp horizon failure is invalid")
    claim_fingerprint = artifact_fingerprint(claim_path)
    if failure_path.exists():
        existing = json.loads(failure_path.read_text())
        if (
            existing.get("schema") == FAILURE_SCHEMA
            and existing.get("status") == "failed"
            and existing.get("error") == error
            and existing.get("claim_fingerprint") == claim_fingerprint
            and existing.get("retry_authorized") is False
            and existing.get("filming_authorized") is False
        ):
            return existing
        raise ValueError("contact-grasp horizon failure is invalid")
    payload = {
        "schema": FAILURE_SCHEMA,
        "status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "claim_fingerprint": claim_fingerprint,
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
            "validate-source",
            "claim",
            "encode",
            "validate-handoff",
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
    elif args.command == "validate-source":
        payload = validate_source(args.checkpoint_root, args.data_root)
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
