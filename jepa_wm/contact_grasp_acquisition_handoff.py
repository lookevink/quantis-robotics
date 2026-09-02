"""Frozen evidence for recovering the V4 contact-grasp acquisition failure."""

from __future__ import annotations

import argparse
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jepa_wm.persistence import write_json_atomic
from jepa_wm.runtime_fingerprint import runtime_source_files, runtime_source_fingerprint
from jepa_wm.training_artifact import artifact_fingerprint


HANDOFF_SCHEMA = "quantis.contact_grasp_acquisition_handoff.v1"
SOURCE_SESSION_ID = "unknown-start-e2e-v4-62605-grasp-03"
REFERENCE_RECORDING = "contact-insertion-v10-drive-slow-2600-held-00"
REFERENCE_SEED = 12600
PROPOSAL_NAME = (
    "contact-grasp-acquisition-v10-drive-slow-2600_task12_h256_s3000_cfopen-v2"
)
WORKER_NAME = "contact-insertion-v10-unknown-start-acquisition-v2.worker.json"
READINESS_NAME = (
    "contact-grasp-acquisition-v10-drive-slow-2600_task12_h256_s3000_cfopen-v2_"
    "contact_grasp_acquisition_readiness.json"
)
REPLAY_NAME = (
    "contact-grasp-acquisition-v10-drive-slow-2600_task12_h256_s3000_cfopen-v2_"
    "v4_failure_replay.json"
)

SOURCE_FINGERPRINTS = {
    "request.json": "7d74adb159ec0e781b04204bb58dbfdea2718ac4e57914c9ee397021c3810394",
    "state.json": "736c5250b5ecfd052e68ef3d1f74ca1626ee4fa00c2b5f798b0fd5b992342509",
    "response.json": "716046e8c67571b72a1f9ae1be0b0befdfaf6d7bb2484a62b2ecb8d5c7b1cd64",
    "execution_started.json": "fb9cc3068341a697319917a236151fa2a1d394c43de89dd7695345cc65eb1a8a",
    "result.json": "480921a5daee6a0bd53d9b3e3b98922f1d1423bee80c157e4af2af85cdb9589a",
    "context.png": "9e32575a43ab6202eb95eafb40d9e776bf78e9d480fdcba07d1e0378c48c14b4",
}
PROPOSAL_FINGERPRINT = (
    "16cdb2a36f1d80d8e07b321c9607a6a91747429a2e73895b22bc0e7e4e2f4dfa"
)
WORKER_FINGERPRINT = (
    "1cd9d7adee65af5817bcebb7abd03e5ba82a12d95b1674fe5138c74256b8c51d"
)
READINESS_FINGERPRINT = (
    "1cf2d752e17325ed737c5c761de06fd2934a1ada06c703bc0a2e43372bcdfc4a"
)
REPLAY_FINGERPRINT = (
    "7e48b5450a1b798033f307c24472bb08ab61fc51cf08d53ef43793c29565dc31"
)
EXPERIMENT_DIRECTORY = "unknown_start_acquisition_recovery_v5"
ROLLOUT_ID = "unknown-start-e2e-v5-62605-grasp"
MAXIMUM_ACTIONS = 52


RUNTIME_FILES = (
    "jepa_wm/action.py",
    "jepa_wm/contact_grasp_acquisition_handoff.py",
    "jepa_wm/contact_grasp_target.py",
    "jepa_wm/control_protocol.py",
    "jepa_wm/control_rollout.py",
    "jepa_wm/control_server.py",
    "jepa_wm/control_worker.py",
    "jepa_wm/proposal.py",
    "jepa_wm/worker_artifacts.py",
    "ops/aws.sh",
    "ops/jepa_wm.sh",
    "ops/run_unknown_start_acquisition_recovery.sh",
    "ops/shell_helpers.sh",
    "sim/control_session.py",
    "sim/isaac_control_execution.py",
    "sim/isaac_control_followup.py",
    "sim/isaac_control_bridge.py",
    "sim/isaac_control_runtime.py",
    "sim/isaac_demo.py",
    "sim/runtime_loader.py",
)
RUNTIME_FILES = runtime_source_files(Path(__file__).resolve().parents[1])


def runtime_fingerprint(repository: Path | None = None) -> str:
    root = repository or Path(__file__).resolve().parents[1]
    return runtime_source_fingerprint(root)


@dataclass(frozen=True)
class ContactGraspAcquisitionHandoff:
    """Exact offline evidence allowed to replace the failed V4 proposal."""

    followup_session_id: str
    runtime_fingerprint: str
    source_revision: str
    replay_fingerprint: str = REPLAY_FINGERPRINT
    schema: str = HANDOFF_SCHEMA

    def __post_init__(self) -> None:
        if (
            self.schema != HANDOFF_SCHEMA
            or not self.followup_session_id
            or len(self.runtime_fingerprint) != 64
            or len(self.source_revision) != 40
            or self.replay_fingerprint != REPLAY_FINGERPRINT
        ):
            raise ValueError("contact-grasp acquisition handoff is invalid")

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
            "replay_fingerprint": self.replay_fingerprint,
            "source_fingerprints": SOURCE_FINGERPRINTS,
            "runtime_fingerprint": self.runtime_fingerprint,
            "source_revision": self.source_revision,
            "simulator_action_authorized": True,
            "filming_authorized": False,
            "production_authority_granted": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ContactGraspAcquisitionHandoff:
        try:
            instance = cls(
                followup_session_id=str(payload["followup_session_id"]),
                runtime_fingerprint=str(payload["runtime_fingerprint"]),
                source_revision=str(payload["source_revision"]),
                replay_fingerprint=str(payload["replay_fingerprint"]),
                schema=str(payload["schema"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("contact-grasp acquisition handoff is incomplete") from error
        if dict(payload) != instance.to_dict():
            raise ValueError("contact-grasp acquisition handoff changed")
        return instance


def build_handoff(
    *,
    checkpoint_root: Path,
    data_root: Path,
    followup_session_id: str,
    source_revision: str,
) -> ContactGraspAcquisitionHandoff:
    """Authenticate the frozen artifacts and construct one live handoff."""

    source = data_root / "control_sessions" / SOURCE_SESSION_ID
    if any(
        artifact_fingerprint(source / name) != expected
        for name, expected in SOURCE_FINGERPRINTS.items()
    ):
        raise ValueError("contact-grasp acquisition source changed")
    request_payload = json.loads((source / "request.json").read_text())
    state_payload = json.loads((source / "state.json").read_text())
    result_payload = json.loads((source / "result.json").read_text())
    refresh = result_payload.get("insertion_trial_refresh")
    if (
        state_payload.get("session_id") != SOURCE_SESSION_ID
        or state_payload.get("reference_recording") != REFERENCE_RECORDING
        or state_payload.get("seed") != REFERENCE_SEED
        or state_payload.get("collision_detected") is not False
        or state_payload.get("contact_force_newtons") != 0.0
        or state_payload.get("plug_attached") is not False
        or state_payload.get("active_drive_target") is None
        or result_payload.get("status") != "blocked"
        or result_payload.get("selected_action_scale") is not None
        or result_payload.get("gate", {}).get("reasons") != ["gripper_violation"]
        or "post_action_pose" in result_payload
        or not isinstance(refresh, dict)
        or refresh.get("live_pose") != request_payload.get("pose")
        or refresh.get("live_state", {}).get("joint_positions")
        != state_payload.get("current_joint_positions")
        or refresh.get("live_state", {}).get("gripper_width_m")
        != state_payload.get("current_gripper_width_m")
        or refresh.get("live_state", {}).get("plug_position")
        != state_payload.get("plug_position")
        or refresh.get("live_state", {}).get("contact_force_newtons")
        != state_payload.get("contact_force_newtons")
        or refresh.get("live_state", {}).get("collision_detected")
        is not state_payload.get("collision_detected")
        or refresh.get("live_state", {}).get("plug_attached")
        is not state_payload.get("plug_attached")
    ):
        raise ValueError("contact-grasp acquisition source was not a no-motion block")
    proposal = checkpoint_root / f"{PROPOSAL_NAME}.pth"
    worker = checkpoint_root / WORKER_NAME
    readiness = checkpoint_root / "experiments" / READINESS_NAME
    replay = checkpoint_root / "experiments" / REPLAY_NAME
    if artifact_fingerprint(proposal) != PROPOSAL_FINGERPRINT:
        raise ValueError("contact-grasp acquisition proposal changed")
    if artifact_fingerprint(worker) != WORKER_FINGERPRINT:
        raise ValueError("contact-grasp acquisition worker changed")
    if artifact_fingerprint(readiness) != READINESS_FINGERPRINT:
        raise ValueError("contact-grasp acquisition readiness changed")
    readiness_payload = json.loads(readiness.read_text())
    replay_payload = json.loads(replay.read_text())
    if (
        readiness_payload.get("passed") is not True
        or replay_payload.get("passed") is not True
        or replay_payload.get("simulator_action_authorized") is not False
        or replay_payload.get("proposal_fingerprint") != PROPOSAL_FINGERPRINT
        or replay_payload.get("worker_fingerprint") != WORKER_FINGERPRINT
        or replay_payload.get("readiness_fingerprint") != READINESS_FINGERPRINT
    ):
        raise ValueError("contact-grasp acquisition offline gate did not pass")
    return ContactGraspAcquisitionHandoff(
        followup_session_id=followup_session_id,
        runtime_fingerprint=runtime_fingerprint(),
        source_revision=source_revision,
        replay_fingerprint=artifact_fingerprint(replay),
    )


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
        raise ValueError("contact-grasp acquisition recovery was already claimed") from error
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
        raise ValueError("contact-grasp acquisition recovery is already terminal")
    handoff = build_handoff(
        checkpoint_root=checkpoint_root,
        data_root=data_root,
        followup_session_id=followup_session_id,
        source_revision=source_revision,
    )
    if handoff.runtime_fingerprint != expected_runtime_fingerprint:
        raise ValueError("contact-grasp acquisition recovery runtime changed")
    payload = handoff.to_dict()
    _write_exclusive(claim_path, payload)
    return payload


def encode_claim(checkpoint_root: Path) -> str:
    claim_path, _, _, _ = paths(checkpoint_root)
    payload = json.loads(claim_path.read_text())
    ContactGraspAcquisitionHandoff.from_dict(payload)
    return b64encode(json.dumps(payload, sort_keys=True).encode()).decode()


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("contact-grasp acquisition recovery claim is invalid")
    claim_payload = json.loads(claim_path.read_text())
    handoff = ContactGraspAcquisitionHandoff.from_dict(claim_payload)
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
        "schema": "quantis.contact_grasp_acquisition_recovery_evaluation.v1",
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
        raise ValueError("contact-grasp acquisition recovery did not retain grasp")
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
        raise ValueError("contact-grasp acquisition recovery is already terminal")
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
            raise ValueError("contact-grasp acquisition recovery backup changed")
    evaluation = json.loads(evaluation_path.read_text())
    report = json.loads(report_path.read_text())
    for step in report.get("steps", ()):
        session_id = step.get("session") if isinstance(step, dict) else None
        if not isinstance(session_id, str):
            raise ValueError("contact-grasp acquisition recovery roster is invalid")
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
                raise ValueError(
                    f"contact-grasp acquisition recovery changed: {session_id}/{name}"
                )
    source_handoff = (
        data_root
        / "control_sessions"
        / SOURCE_SESSION_ID
        / f"acquisition_handoff_{report['steps'][0]['session']}.json"
    )
    recovery_handoff = (
        recovery_data_root
        / "control_sessions"
        / SOURCE_SESSION_ID
        / source_handoff.name
    )
    if artifact_fingerprint(source_handoff) != artifact_fingerprint(recovery_handoff):
        raise ValueError("contact-grasp acquisition handoff recovery changed")
    passed = evaluation.get("evaluation_passed") is True
    payload = {
        "schema": "quantis.contact_grasp_acquisition_recovery_terminal.v1",
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
        raise ValueError("contact-grasp acquisition terminal recovery changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or failure_path.exists() or not claim_path.is_file():
        raise ValueError("contact-grasp acquisition recovery failure is invalid")
    payload = {
        "schema": "quantis.contact_grasp_acquisition_recovery_failure.v1",
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
