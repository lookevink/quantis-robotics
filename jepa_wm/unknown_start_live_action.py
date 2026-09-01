"""One recovery-gated live action from a passed unknown-start shadow source."""

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


EXPERIMENT_ID = "unknown-start-live-action-v5"
EXECUTION_SESSION_ID = "unknown-start-live-action-v5-62605"
PREDECESSOR_SESSION_ID = "unknown-start-live-action-v4-62605"
SOURCE_SESSION_ID = "unknown-start-shadow-canary-v5-62605"
RESET_RECORDING_ID = "unknown-start-reset-v6-62605"
RESET_RESULT_FINGERPRINT = (
    "70a8fba8022e687c2fc9ecd78f8d63924a8a5840497af9249c60bb781a0a6d58"
)
SOURCE_TERMINAL_FINGERPRINT = (
    "4f1ba5c510d4cc9da7f0cb05f03b9ec24ca06e58d4556135b2d8bfc0e0b3c542"
)
SOURCE_EVALUATION_FINGERPRINT = (
    "cf083d1c0d84256e3616d7260f505d24991e1630343c90be80c7a06b8ca22bef"
)
SOURCE_SHADOW_FINGERPRINT = (
    "75b77d011c314db3118993723755ea1524ec2eab22bd141d98543d1162475ce2"
)
SOURCE_SAFETY_FINGERPRINT = (
    "467a62dae7ed8728e536e81f3b6b58dad3aa3702b12477ab6cdeac52b7506bae"
)
SOURCE_RESPONSE_FINGERPRINT = (
    "9fa33499d1db898a3c695d3395c7830d15126f04b09c808a75113d8a7e8f0d04"
)
SOURCE_HANDOFF_FINGERPRINT = (
    "296000b85c090b161966b95559451a3e3bc96b0b6929222e1154cd1701102d73"
)
REFERENCE_RECORDING = "contact-insertion-v10-drive-slow-2600-held-00"
REFERENCE_SEED = 12600
OUTPUT_DIRECTORY = "unknown_start_live_action_v5"

RUNTIME_FILES = (
    "jepa_wm/action.py",
    "jepa_wm/contact_grasp_target.py",
    "jepa_wm/control_policy.py",
    "jepa_wm/control_protocol.py",
    "jepa_wm/control_safety.py",
    "jepa_wm/control_tracking.py",
    "jepa_wm/experimental_candidate.py",
    "jepa_wm/identifiers.py",
    "jepa_wm/insertion_contract.py",
    "jepa_wm/insertion_refresh.py",
    "jepa_wm/insertion_rollout.py",
    "jepa_wm/insertion_trial.py",
    "jepa_wm/joint_drive.py",
    "jepa_wm/joint_settlement.py",
    "jepa_wm/physical_observation.py",
    "jepa_wm/shadow_planning.py",
    "jepa_wm/shadow_safety.py",
    "jepa_wm/target_progress.py",
    "jepa_wm/trajectory.py",
    "jepa_wm/trial_equivalence.py",
    "jepa_wm/unknown_start_live_action.py",
    "ops/aws.sh",
    "ops/run_unknown_start_live_action.sh",
    "ops/shell_helpers.sh",
    "sim/control_context.py",
    "sim/control_identity.py",
    "sim/control_session.py",
    "sim/demo_sequence.py",
    "sim/grasp_task.py",
    "sim/isaac_candidate_binding.py",
    "sim/isaac_control_capture.py",
    "sim/isaac_control_execution.py",
    "sim/isaac_control_runtime.py",
    "sim/isaac_demo.py",
    "sim/isaac_demo_camera.py",
    "sim/isaac_demo_kinematics.py",
    "sim/isaac_demo_runtime.py",
    "sim/isaac_demo_scene.py",
    "sim/isaac_unknown_start_shadow.py",
    "sim/recording.py",
    "sim/runtime_loader.py",
    "sim/trial_source_cache.py",
    "sim/unknown_start_reset.py",
    "sim/unknown_start_shadow.py",
)


def runtime_fingerprint(repository: Path | None = None) -> str:
    root = repository or Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative in RUNTIME_FILES:
        path = root / relative
        encoded = relative.encode()
        contents = path.read_bytes()
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("unknown-start live action was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def paths(checkpoint_root: Path) -> tuple[Path, Path, Path, Path]:
    root = checkpoint_root / OUTPUT_DIRECTORY
    return (
        root / "CLAIM.json",
        root / "EVALUATION.json",
        root / "RESULT.json",
        root / "FAILURE.json",
    )


def predecessor_recovery_fingerprint(data_root: Path) -> str:
    path = (
        data_root
        / "control_sessions"
        / PREDECESSOR_SESSION_ID
        / "rollback_recovery.json"
    )
    payload = json.loads(path.read_text())
    if (
        payload.get("schema") != "quantis.unknown_start_rollback_recovery.v1"
        or payload.get("session_id") != PREDECESSOR_SESSION_ID
        or payload.get("recovered") is not True
        or payload.get("applied_model_actions") != 0
        or payload.get("collision_detected") is not False
        or payload.get("plug_attached") is not False
        or payload.get("timeline_playing") is not False
        or payload.get("contact_force_newtons", float("inf")) > 2.0
    ):
        raise ValueError("unknown-start predecessor recovery is invalid")
    return artifact_fingerprint(path)


def claim(
    checkpoint_root: Path,
    source_revision: str,
    expected_runtime_fingerprint: str,
    data_root: Path,
    expected_predecessor_recovery_fingerprint: str,
) -> dict[str, Any]:
    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if any(path.exists() for path in (evaluation_path, result_path, failure_path)):
        raise ValueError("unknown-start live action is already terminal")
    actual_runtime_fingerprint = runtime_fingerprint()
    if actual_runtime_fingerprint != expected_runtime_fingerprint:
        raise ValueError("unknown-start live action runtime changed")
    if len(source_revision) != 40:
        raise ValueError("unknown-start live action revision is invalid")
    recovery_fingerprint = predecessor_recovery_fingerprint(data_root)
    if recovery_fingerprint != expected_predecessor_recovery_fingerprint:
        raise ValueError("unknown-start predecessor recovery changed")
    payload = {
        "schema": "quantis.unknown_start_live_action_claim.v1",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": EXPERIMENT_ID,
        "execution_session_id": EXECUTION_SESSION_ID,
        "predecessor_session_id": PREDECESSOR_SESSION_ID,
        "predecessor_recovery_fingerprint": recovery_fingerprint,
        "source_session_id": SOURCE_SESSION_ID,
        "reset_recording_id": RESET_RECORDING_ID,
        "reset_result_fingerprint": RESET_RESULT_FINGERPRINT,
        "source_terminal_fingerprint": SOURCE_TERMINAL_FINGERPRINT,
        "source_evaluation_fingerprint": SOURCE_EVALUATION_FINGERPRINT,
        "source_shadow_fingerprint": SOURCE_SHADOW_FINGERPRINT,
        "source_safety_fingerprint": SOURCE_SAFETY_FINGERPRINT,
        "source_response_fingerprint": SOURCE_RESPONSE_FINGERPRINT,
        "source_handoff_fingerprint": SOURCE_HANDOFF_FINGERPRINT,
        "source_revision": source_revision,
        "runtime_fingerprint": actual_runtime_fingerprint,
        "maximum_model_actions": 1,
        "filming_authorized": False,
    }
    _write_exclusive(claim_path, payload)
    return payload


def _authenticate_source(checkpoint_root: Path, data_root: Path) -> None:
    from sim.control_session import ControlSession

    model_root = checkpoint_root / "quantis_physical_state_residual_v1"
    terminal_path = model_root / "unknown-start-shadow-canary-v5.json"
    evaluation_path = model_root / "unknown-start-shadow-canary-v5-evaluation.json"
    if (
        artifact_fingerprint(terminal_path) != SOURCE_TERMINAL_FINGERPRINT
        or artifact_fingerprint(evaluation_path) != SOURCE_EVALUATION_FINGERPRINT
    ):
        raise ValueError("unknown-start live action source identity changed")
    terminal = json.loads(terminal_path.read_text())
    evaluation = json.loads(evaluation_path.read_text())
    source = ControlSession.at(data_root / "control_sessions", SOURCE_SESSION_ID)
    shadow = source.load_shadow()
    safety = source.load_shadow_safety()
    source_identities = (
        ("shadow", source.shadow_path, SOURCE_SHADOW_FINGERPRINT),
        ("shadow_safety", source.shadow_safety_path, SOURCE_SAFETY_FINGERPRINT),
        ("direct_response", source.response_path, SOURCE_RESPONSE_FINGERPRINT),
        (
            "unknown_start_handoff",
            source.path / "unknown_start_handoff.json",
            SOURCE_HANDOFF_FINGERPRINT,
        ),
    )
    if (
        terminal.get("passed") is not True
        or terminal.get("recovery_verified") is not True
        or terminal.get("evaluation", {}).get("fingerprint")
        != SOURCE_EVALUATION_FINGERPRINT
        or evaluation.get("evaluation_passed") is not True
        or evaluation.get("apply_action") is not False
        or evaluation.get("session_id") != SOURCE_SESSION_ID
        or any(
            evaluation.get(label, {}).get("fingerprint") != expected
            or artifact_fingerprint(path) != expected
            for label, path, expected in source_identities
        )
        or not shadow.passes_shadow_gate
        or not safety.passed
    ):
        raise ValueError("unknown-start live action source did not pass")


def evaluate(checkpoint_root: Path, data_root: Path) -> dict[str, Any]:
    from jepa_wm.control_policy import ControlExecutionPolicy
    from sim.control_session import ControlResultStatus, ControlSession
    from sim.unknown_start_shadow import UnknownStartControlHandoff

    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    if not claim_path.is_file() or result_path.exists() or failure_path.exists():
        raise ValueError("unknown-start live action claim is invalid")
    _authenticate_source(checkpoint_root, data_root)
    session = ControlSession.at(data_root / "control_sessions", EXECUTION_SESSION_ID)
    observation, state = session.load_capture()
    response = session.load_response()
    binding = session.load_candidate_binding(response)
    result = session.load_result()
    handoff_path = session.path / "unknown_start_handoff.json"
    handoff = UnknownStartControlHandoff.from_dict(json.loads(handoff_path.read_text()))
    post = result.post_action
    evaluation_passed = (
        state.execution_policy is ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
        and state.recording == RESET_RECORDING_ID
        and state.reference_recording == REFERENCE_RECORDING
        and state.seed == REFERENCE_SEED
        and binding.source_session_id == SOURCE_SESSION_ID
        and binding.execution_session_id == EXECUTION_SESSION_ID
        and handoff.session_id == EXECUTION_SESSION_ID
        and handoff.reset_recording_id == RESET_RECORDING_ID
        and handoff.reset_result_fingerprint == RESET_RESULT_FINGERPRINT
        and response.actions == binding.actions
        and result.status is ControlResultStatus.APPLIED
        and result.gate.passed
        and post is not None
        and post.tracking.passed
        and not post.collision_detected
        and post.contact_force_newtons <= 2.0
        and result.execution_error is None
    )
    payload = {
        "schema": "quantis.unknown_start_live_action_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "evaluation_passed": evaluation_passed,
        "recovery_verified": False,
        "experiment_id": EXPERIMENT_ID,
        "execution_session_id": EXECUTION_SESSION_ID,
        "source_session_id": SOURCE_SESSION_ID,
        "claim_fingerprint": artifact_fingerprint(claim_path),
        "result_status": result.status.value,
        "result_fingerprint": artifact_fingerprint(session.result_path),
        "candidate_binding_fingerprint": artifact_fingerprint(
            session.candidate_binding_path
        ),
        "handoff_fingerprint": artifact_fingerprint(handoff_path),
        "selected_action_scale": (
            result.selected_action_scale.to_dict()
            if result.selected_action_scale is not None
            else None
        ),
        "post_action": post.to_dict() if post is not None else None,
        "model_action_attempts": (1 if result.selected_action_scale is not None else 0),
        "applied_model_actions": (1 if result.selected_action_scale is not None else 0),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(evaluation_path, payload)
    return payload


def finalize(
    checkpoint_root: Path,
    recovery_checkpoint_root: Path,
    data_root: Path,
    recovery_data_root: Path,
) -> dict[str, Any]:
    from sim.control_session import ControlSession

    claim_path, evaluation_path, result_path, failure_path = paths(checkpoint_root)
    recovery_claim, recovery_evaluation, recovery_result, _ = paths(
        recovery_checkpoint_root
    )
    if failure_path.exists() or result_path.exists():
        raise ValueError("unknown-start live action is already terminal")
    for primary, recovery in (
        (claim_path, recovery_claim),
        (evaluation_path, recovery_evaluation),
    ):
        if artifact_fingerprint(primary) != artifact_fingerprint(recovery):
            raise ValueError("unknown-start live action recovery changed")
    claim_payload = json.loads(claim_path.read_text())
    predecessor_recovery = (
        data_root
        / "control_sessions"
        / PREDECESSOR_SESSION_ID
        / "rollback_recovery.json"
    )
    recovery_predecessor_recovery = (
        recovery_data_root
        / "control_sessions"
        / PREDECESSOR_SESSION_ID
        / "rollback_recovery.json"
    )
    if artifact_fingerprint(predecessor_recovery) != claim_payload.get(
        "predecessor_recovery_fingerprint"
    ) or artifact_fingerprint(predecessor_recovery) != artifact_fingerprint(
        recovery_predecessor_recovery
    ):
        raise ValueError("unknown-start predecessor recovery backup changed")
    session = ControlSession.at(data_root / "control_sessions", EXECUTION_SESSION_ID)
    recovery_session = ControlSession.at(
        recovery_data_root / "control_sessions", EXECUTION_SESSION_ID
    )
    for name in (
        "request.json",
        "state.json",
        "response.json",
        "experimental_candidate.json",
        "unknown_start_handoff.json",
        "unknown_start_routing_target.json",
        "unknown_start_routing_step.json",
        "execution_started.json",
        "result.json",
        "context.png",
    ):
        if artifact_fingerprint(session.path / name) != artifact_fingerprint(
            recovery_session.path / name
        ):
            raise ValueError(f"unknown-start live action recovery changed: {name}")
    post_action_path = session.path / "post_action.png"
    recovery_post_action_path = recovery_session.path / "post_action.png"
    if post_action_path.exists() != recovery_post_action_path.exists() or (
        post_action_path.exists()
        and artifact_fingerprint(post_action_path)
        != artifact_fingerprint(recovery_post_action_path)
    ):
        raise ValueError("unknown-start live action recovery changed: post_action.png")
    evaluation = json.loads(evaluation_path.read_text())
    if (
        evaluation.get("schema") != "quantis.unknown_start_live_action_evaluation.v1"
        or evaluation.get("status") != "evaluated_pending_recovery"
        or evaluation.get("experiment_id") != EXPERIMENT_ID
        or evaluation.get("execution_session_id") != EXECUTION_SESSION_ID
        or evaluation.get("source_session_id") != SOURCE_SESSION_ID
        or evaluation.get("passed") is not False
        or not isinstance(evaluation.get("evaluation_passed"), bool)
        or evaluation.get("recovery_verified") is not False
        or evaluation.get("model_action_attempts") not in (0, 1)
        or evaluation.get("applied_model_actions")
        != evaluation.get("model_action_attempts")
        or evaluation.get("filming_authorized") is not False
        or evaluation.get("production_authority_granted") is not False
    ):
        raise ValueError("unknown-start live action evaluation is invalid")
    passed = evaluation.get("evaluation_passed") is True
    payload = {
        "schema": "quantis.unknown_start_live_action_terminal.v1",
        "status": "passed" if passed else "failed_execution_gate",
        "passed": passed,
        "evaluation_fingerprint": artifact_fingerprint(evaluation_path),
        "recovery_verified": True,
        "applied_model_actions": evaluation.get("applied_model_actions"),
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(result_path, payload)
    write_json_atomic(recovery_result, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("unknown-start live action terminal recovery changed")
    return payload


def failure(checkpoint_root: Path, error: str) -> dict[str, Any]:
    claim_path, _, result_path, failure_path = paths(checkpoint_root)
    if result_path.exists() or failure_path.exists() or not claim_path.is_file():
        raise ValueError("unknown-start live action failure is invalid")
    payload = {
        "schema": "quantis.unknown_start_live_action_failure.v1",
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
            "recovery-fingerprint",
            "claim",
            "evaluate",
            "finalize",
            "failure",
        ),
    )
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--recovery-checkpoint-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--recovery-data-root", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--runtime-fingerprint")
    parser.add_argument("--predecessor-recovery-fingerprint")
    parser.add_argument("--error")
    args = parser.parse_args(argv)
    if args.command == "fingerprint":
        payload: Any = runtime_fingerprint()
    elif args.command == "recovery-fingerprint":
        payload = predecessor_recovery_fingerprint(args.data_root)
    elif args.command == "claim":
        payload = claim(
            args.checkpoint_root,
            args.source_revision,
            args.runtime_fingerprint,
            args.data_root,
            args.predecessor_recovery_fingerprint,
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
    print(
        payload
        if isinstance(payload, str)
        else json.dumps(payload, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
