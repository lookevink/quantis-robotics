"""Authenticated V15 continuation after the bounded V10 acquisition negative."""

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


HANDOFF_SCHEMA = "quantis.contact_grasp_acquisition_resolution.v5"
EXPERIMENT_DIRECTORY = "unknown_start_acquisition_resolution_v15"
ROLLOUT_ID = "unknown-start-e2e-v15-62605-grasp"
SOURCE_ROLLOUT_ID = "unknown-start-e2e-v10-62605-grasp"
SOURCE_SESSION_ID = f"{SOURCE_ROLLOUT_ID}-52"
MAXIMUM_ACTIONS = 52
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
V11_CLAIM_FINGERPRINT = (
    "3e345a656e255c21af42eccd81496ad53f0816084eab352068a933178b424eac"
)
V11_FAILURE_FINGERPRINT = (
    "ef6157b3816e536f0b9abf3b6824538ad49a3f5c655fa7b7362ecde1946d8912"
)
V11_DIAGNOSTIC_FINGERPRINT = (
    "b32aedf987f5634ff5ad9cbdac2944a54bd94afae5fe7a0b4c79bc50f6e74f30"
)
V11_EXPERIMENT_DIRECTORY = "unknown_start_acquisition_resolution_v11"
V11_ROLLOUT_ID = "unknown-start-e2e-v11-62605-grasp"
V11_SESSION_ID = f"{V11_ROLLOUT_ID}-01"
V12_CLAIM_FINGERPRINT = (
    "7ae213360600a8bdc0befd5b11b62f39530350004f93cc6bd9f06734019277f8"
)
V12_FAILURE_FINGERPRINT = (
    "baf4bf5eca0912490ec337e6e5649c2ab2bb7a4644e22e38679a5f22d6ce43ad"
)
V12_DIAGNOSTIC_FINGERPRINT = (
    "016a804de71f2bd48f47871f3e2fc8ff2563036c937fe2cbc2de756fb7831c43"
)
V12_EXPERIMENT_DIRECTORY = "unknown_start_acquisition_resolution_v12"
V12_ROLLOUT_ID = "unknown-start-e2e-v12-62605-grasp"
V12_SESSION_ID = f"{V12_ROLLOUT_ID}-01"
RUNTIME_OWNER_SESSION_ID = V12_SESSION_ID
V12_SESSION_FINGERPRINTS = {
    "request.json": "514888eb69a17938444575d16990f4f2a97366bf9626b2c93c03ddea374750d0",
    "state.json": "c025c1e3c4a477ac3339fdcb82f4b80707765b224dffdbe1b0ac4c43f9ca661e",
    "response.json": "abb0ef3a7ed8d39af8338d4f78702f024a3b300e1e7f3a065461bf921ef9301d",
    "execution_started.json": "9adfbb01f6301b4953594586f3ab3ad6c3d98dbe834033334b6a366c1e5e1505",
    "result.json": "2716e34a13cc8a9d3e54fdd6f17cfd890546f5018538d911c7701c8d26f42232",
    "context.png": "d8dabdd0715c500399f8cab8ba85acd923c8a4770cd6db761ad32ab182ff33f8",
    "post_action.png": "6db31f040e0d1f8d387ace3b7883a8247927870d6ca9c39b2733f44c6545b9e2",
}
V13_CLAIM_FINGERPRINT = (
    "c090f910fb50969def39c603120d63ca1b0f89bb7c34cdb02e3347ca5224ac89"
)
V13_FAILURE_FINGERPRINT = (
    "925a7d4f7646a238db433aa042d1f253a1082b50bfd7ef6628d7af9fc2c3cfb8"
)
V13_DIAGNOSTIC_FINGERPRINT = (
    "e32ed2335eff625334efc529f9bcb5ce6efa7cb33d7487ef8d88193a90028995"
)
V13_EXPERIMENT_DIRECTORY = "unknown_start_acquisition_resolution_v13"
V13_ROLLOUT_ID = "unknown-start-e2e-v13-62605-grasp"
V13_SESSION_ID = f"{V13_ROLLOUT_ID}-01"
V14_CLAIM_FINGERPRINT = (
    "bda86bbe875774d7ad47e4ef2c3da03a2c03d49bf4a1c756269f8292fda27ace"
)
V14_FAILURE_FINGERPRINT = (
    "07b0b21a5248ac31ce530dbde3a064314ec180947c35d8e9b980fdb0e114f8a8"
)
V14_EXPERIMENT_DIRECTORY = "unknown_start_acquisition_resolution_v14"
V14_ROLLOUT_ID = "unknown-start-e2e-v14-62605-grasp"
V14_SESSION_ID = f"{V14_ROLLOUT_ID}-01"
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
            "v11_claim_fingerprint": V11_CLAIM_FINGERPRINT,
            "v11_failure_fingerprint": V11_FAILURE_FINGERPRINT,
            "v11_diagnostic_fingerprint": V11_DIAGNOSTIC_FINGERPRINT,
            "v11_simulator_action_applied": False,
            "v12_claim_fingerprint": V12_CLAIM_FINGERPRINT,
            "v12_failure_fingerprint": V12_FAILURE_FINGERPRINT,
            "v12_diagnostic_fingerprint": V12_DIAGNOSTIC_FINGERPRINT,
            "v12_session_id": V12_SESSION_ID,
            "v12_session_fingerprints": V12_SESSION_FINGERPRINTS,
            "v12_runtime_owner_session_id": RUNTIME_OWNER_SESSION_ID,
            "v12_rollback_verified": True,
            "v13_claim_fingerprint": V13_CLAIM_FINGERPRINT,
            "v13_failure_fingerprint": V13_FAILURE_FINGERPRINT,
            "v13_diagnostic_fingerprint": V13_DIAGNOSTIC_FINGERPRINT,
            "v13_simulator_action_applied": False,
            "v14_claim_fingerprint": V14_CLAIM_FINGERPRINT,
            "v14_failure_fingerprint": V14_FAILURE_FINGERPRINT,
            "v14_diagnostic_persisted": False,
            "v14_simulator_action_applied": False,
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


def validate_v11_no_action(checkpoint_root: Path, data_root: Path) -> None:
    root = checkpoint_root / V11_EXPERIMENT_DIRECTORY
    claim_path = root / "CLAIM.json"
    failure_path = root / "FAILURE.json"
    diagnostic = diagnostic_path(data_root, V11_SESSION_ID)
    if (
        artifact_fingerprint(claim_path) != V11_CLAIM_FINGERPRINT
        or artifact_fingerprint(failure_path) != V11_FAILURE_FINGERPRINT
        or artifact_fingerprint(diagnostic) != V11_DIAGNOSTIC_FINGERPRINT
    ):
        raise ValueError("terminal no-action V11 evidence changed")
    claim_payload = json.loads(claim_path.read_text())
    failure_payload = json.loads(failure_path.read_text())
    diagnostic_payload = json.loads(diagnostic.read_text())
    if (
        claim_payload.get("followup_session_id") != V11_SESSION_ID
        or failure_payload.get("error") != "diagnostic_validation:exit_1"
        or failure_payload.get("retry_authorized") is not False
        or diagnostic_payload.get("status") != "passed_no_actuation"
        or diagnostic_payload.get("simulator_action_applied") is not False
        or (data_root / "control_sessions" / V11_SESSION_ID).exists()
        or (data_root / "control_rollouts" / V11_ROLLOUT_ID).exists()
    ):
        raise ValueError("V11 was not the exact validator-only failure")


def validate_v12_tracking_rollback(checkpoint_root: Path, data_root: Path) -> None:
    from jepa_wm.action import action_between
    from jepa_wm.control_tracking import ActionTrackingLimits
    from jepa_wm.control_rollout import ControlStepSummary
    from jepa_wm.insertion_refresh import (
        MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
    )
    from sim.control_session import ControlSession

    root = checkpoint_root / V12_EXPERIMENT_DIRECTORY
    claim_path = root / "CLAIM.json"
    failure_path = root / "FAILURE.json"
    diagnostic = diagnostic_path(data_root, V12_SESSION_ID)
    session_root = data_root / "control_sessions"
    session = session_root / V12_SESSION_ID
    if (
        artifact_fingerprint(claim_path) != V12_CLAIM_FINGERPRINT
        or artifact_fingerprint(failure_path) != V12_FAILURE_FINGERPRINT
        or artifact_fingerprint(diagnostic) != V12_DIAGNOSTIC_FINGERPRINT
        or any(
            artifact_fingerprint(session / name) != expected
            for name, expected in V12_SESSION_FINGERPRINTS.items()
        )
    ):
        raise ValueError("terminal V12 tracking evidence changed")
    source = ControlStepSummary.from_session(
        ControlSession.at(session_root, SOURCE_SESSION_ID)
    )
    step = ControlStepSummary.from_session(
        ControlSession.at(session_root, V12_SESSION_ID)
    )
    source_post = source.result.post_action
    post = step.result.post_action
    refresh = step.result.insertion_trial_refresh
    failure_payload = json.loads(failure_path.read_text())
    if (
        source_post is None
        or post is None
        or step.state.previous_session_id != SOURCE_SESSION_ID
        or step.result.status.value != "rolled_back_after_tracking_failure"
        or tuple(reason.value for reason in post.tracking.reasons)
        != ("translation_error",)
        or step.result.selected_action_scale is None
        or step.result.selected_action_scale.translation != 1.0
        or post.contact_force_newtons != 0.0
        or post.collision_detected
        or post.plug_attached
        or refresh is None
        or tuple(step.state.current_joint_positions)
        != tuple(refresh.live_state.joint_positions)
        or step.state.current_gripper_width_m
        != refresh.live_state.gripper_width_m
        or failure_payload.get("error") != "report:exit_1"
        or failure_payload.get("retry_authorized") is not False
        or (data_root / "control_rollouts" / V12_ROLLOUT_ID).exists()
    ):
        raise ValueError("V12 was not the exact safe tracking rollback")
    source_post.require_safety_snapshot().validate_followup_continuity(
        refresh.live_state,
        source.contact_grasp_drive_target(),
        maximum_gripper_error_meters=(
            MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
        ),
    )
    drift = action_between(source_post.pose, refresh.live_pose)
    limits = ActionTrackingLimits()
    if (
        sum(value * value for value in drift.values[:3])
        > limits.maximum_translation_error_meters**2
        or sum(value * value for value in drift.values[3:6])
        > limits.maximum_rotation_error_radians**2
        or abs(drift.values[6]) > limits.maximum_gripper_error
    ):
        raise ValueError("V12 rollback did not return to the bounded source state")


def v12_rollback_drive_target(data_root: Path):
    from jepa_wm.joint_drive import JointDriveTarget
    from sim.control_session import ControlSession

    _, state = ControlSession.at(
        data_root / "control_sessions",
        V12_SESSION_ID,
    ).load_capture()
    return JointDriveTarget.for_command(
        tuple(state.current_joint_positions),
        state.current_gripper_width_m,
    )


def validate_v13_no_action(checkpoint_root: Path, data_root: Path) -> None:
    root = checkpoint_root / V13_EXPERIMENT_DIRECTORY
    claim_path = root / "CLAIM.json"
    failure_path = root / "FAILURE.json"
    diagnostic = diagnostic_path(data_root, V13_SESSION_ID)
    if (
        artifact_fingerprint(claim_path) != V13_CLAIM_FINGERPRINT
        or artifact_fingerprint(failure_path) != V13_FAILURE_FINGERPRINT
        or artifact_fingerprint(diagnostic) != V13_DIAGNOSTIC_FINGERPRINT
    ):
        raise ValueError("terminal no-action V13 evidence changed")
    failure_payload = json.loads(failure_path.read_text())
    diagnostic_payload = json.loads(diagnostic.read_text())
    if (
        failure_payload.get("error") != "capture_01:exit_1"
        or failure_payload.get("retry_authorized") is not False
        or diagnostic_payload.get("status") != "passed_no_actuation"
        or diagnostic_payload.get("simulator_action_applied") is not False
        or (data_root / "control_sessions" / V13_SESSION_ID).exists()
        or (data_root / "control_rollouts" / V13_ROLLOUT_ID).exists()
    ):
        raise ValueError("V13 was not the exact no-capture target-owner failure")


def validate_v14_no_action(checkpoint_root: Path, data_root: Path) -> None:
    root = checkpoint_root / V14_EXPERIMENT_DIRECTORY
    claim_path = root / "CLAIM.json"
    failure_path = root / "FAILURE.json"
    diagnostic = diagnostic_path(data_root, V14_SESSION_ID)
    if (
        artifact_fingerprint(claim_path) != V14_CLAIM_FINGERPRINT
        or artifact_fingerprint(failure_path) != V14_FAILURE_FINGERPRINT
    ):
        raise ValueError("terminal no-action V14 evidence changed")
    failure_payload = json.loads(failure_path.read_text())
    if (
        failure_payload.get("error") != "diagnostic:exit_1"
        or failure_payload.get("retry_authorized") is not False
        or diagnostic.exists()
        or (data_root / "control_sessions" / V14_SESSION_ID).exists()
        or (data_root / "control_rollouts" / V14_ROLLOUT_ID).exists()
    ):
        raise ValueError("V14 was not the exact no-diagnostic target failure")


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
    validate_v11_no_action(checkpoint_root, data_root)
    validate_v11_no_action(recovery_checkpoint_root, recovery_data_root)
    validate_v12_tracking_rollback(checkpoint_root, data_root)
    validate_v12_tracking_rollback(recovery_checkpoint_root, recovery_data_root)
    validate_v13_no_action(checkpoint_root, data_root)
    validate_v13_no_action(recovery_checkpoint_root, recovery_data_root)
    validate_v14_no_action(checkpoint_root, data_root)
    validate_v14_no_action(recovery_checkpoint_root, recovery_data_root)
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
    from jepa_wm.action import DroidAction
    from jepa_wm.control_safety import contact_grasp_action_scales

    selected = payload.get("selected_scale")
    attempts = payload.get("attempts")
    action_values = payload.get("action")
    try:
        action = DroidAction(tuple(action_values))
        expected_scales = contact_grasp_action_scales(
            action,
            coarse_acquisition=True,
        )
    except (TypeError, ValueError):
        expected_scales = ()
    safe_attempts = (
        tuple(
            attempt
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get("passed") is True
        )
        if isinstance(attempts, list)
        else ()
    )
    expected_scale_payloads = tuple(scale.to_dict() for scale in expected_scales)
    attempted_scale_payloads = (
        tuple(
            attempt.get("scale") if isinstance(attempt, dict) else None
            for attempt in attempts
        )
        if isinstance(attempts, list)
        else ()
    )
    if (
        payload.get("schema")
        != "quantis.contact_grasp_acquisition_resolution_diagnostic.v2"
        or payload.get("status") != "passed_no_actuation"
        or payload.get("source_session_id") != SOURCE_SESSION_ID
        or payload.get("followup_session_id") != handoff.followup_session_id
        or payload.get("claim_fingerprint") != claim_fingerprint
        or payload.get("simulator_action_applied") is not False
        or payload.get("runtime_owner_session_id") != RUNTIME_OWNER_SESSION_ID
        or not isinstance(payload.get("active_drive_target"), dict)
        or not isinstance(selected, dict)
        or not expected_scale_payloads
        or attempted_scale_payloads != expected_scale_payloads
        or selected not in expected_scale_payloads
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
    validated = validate_diagnostic_evidence(
        payload,
        handoff,
        artifact_fingerprint(claim_path),
    )
    if validated["active_drive_target"] != v12_rollback_drive_target(
        data_root
    ).to_dict():
        raise ValueError("coarse acquisition rollback drive target changed")
    return validated


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
