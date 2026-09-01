"""One-shot terminal evidence for the physical residual shadow canary."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from jepa_wm.persistence import write_json_atomic
from jepa_wm.action import DroidAction
from jepa_wm.physical_observation import PhysicalRoutingObservation
from jepa_wm.physical_shadow_canary_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as V1_CONFIG_FINGERPRINT,
)
from jepa_wm.physical_shadow_canary_v2_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as V2_CONFIG_FINGERPRINT,
)
from jepa_wm.physical_shadow_canary_v3_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as V3_CONFIG_FINGERPRINT,
)
from jepa_wm.physical_shadow_canary_v4_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as V4_CONFIG_FINGERPRINT,
)
from jepa_wm.physical_shadow_canary_v5_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as V5_CONFIG_FINGERPRINT,
)
from jepa_wm.planner import CEMConfig
from jepa_wm.shadow_planning import CandidateAuthority
from jepa_wm.training_artifact import ArtifactIdentity, artifact_fingerprint
from jepa_wm.worker_artifacts import ControlWorkerArtifacts
from sim.control_session import ControlSession
from sim.unknown_start_shadow import UnknownStartControlHandoff
from sim.unknown_start_reset import UnknownStartResetEvidence


EXPERIMENT_SCHEMA = "quantis.jepa_wm_physical_shadow_canary_experiment.v1"
EXPERIMENT_SCHEMA_V2 = "quantis.jepa_wm_physical_shadow_canary_experiment.v2"
EXPERIMENT_SCHEMA_V3 = "quantis.jepa_wm_physical_shadow_canary_experiment.v3"
EXPERIMENT_SCHEMA_V4 = "quantis.jepa_wm_physical_shadow_canary_experiment.v4"
EXPERIMENT_SCHEMA_V5 = "quantis.jepa_wm_physical_shadow_canary_experiment.v5"


def _serialized_action_scale(scale: Any) -> dict[str, Any] | None:
    return scale.to_dict() if scale is not None else None


def load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    fingerprint = sha256(encoded).hexdigest()
    payload = json.loads(encoded)
    expected_fingerprint = {
        EXPERIMENT_SCHEMA: V1_CONFIG_FINGERPRINT,
        EXPERIMENT_SCHEMA_V2: V2_CONFIG_FINGERPRINT,
        EXPERIMENT_SCHEMA_V3: V3_CONFIG_FINGERPRINT,
        EXPERIMENT_SCHEMA_V4: V4_CONFIG_FINGERPRINT,
        EXPERIMENT_SCHEMA_V5: V5_CONFIG_FINGERPRINT,
    }.get(payload.get("schema"))
    if expected_fingerprint is None or (
        expected_fingerprint != "PENDING_CHECKPOINT"
        and fingerprint != expected_fingerprint
    ):
        raise ValueError("physical shadow canary configuration changed")
    execution = payload.get("execution", {})
    gate = payload.get("gate", {})
    if (
        payload.get("schema")
        not in (
            EXPERIMENT_SCHEMA,
            EXPERIMENT_SCHEMA_V2,
            EXPERIMENT_SCHEMA_V3,
            EXPERIMENT_SCHEMA_V4,
            EXPERIMENT_SCHEMA_V5,
        )
        or not isinstance(payload.get("session_id"), str)
        or not payload["session_id"]
        or gate
        != {
            "require_shadow_gate": True,
            "require_counterfactual_safety": True,
            "require_physical_router_input": True,
            "require_zero_actuation": True,
        }
        or execution
        != {
            "evaluations": 1,
            "apply_action": False,
            "train": False,
            "film": False,
            "hardware": False,
            "production": False,
        }
    ):
        raise ValueError("physical shadow canary contract is invalid")
    if payload.get("schema") in (
        EXPERIMENT_SCHEMA_V3,
        EXPERIMENT_SCHEMA_V4,
        EXPERIMENT_SCHEMA_V5,
    ):
        unknown_start = payload.get("unknown_start")
        if (
            not isinstance(unknown_start, Mapping)
            or set(unknown_start)
            != {
                "recording_id",
                "seed",
                "result_fingerprint",
                "evidence_fingerprint",
                "contract_fingerprint",
            }
            or not isinstance(unknown_start["recording_id"], str)
            or not unknown_start["recording_id"]
            or not isinstance(unknown_start["seed"], int)
            or unknown_start["seed"] < 0
            or any(
                not isinstance(unknown_start[field], str)
                or len(unknown_start[field]) != 64
                for field in (
                    "result_fingerprint",
                    "evidence_fingerprint",
                    "contract_fingerprint",
                )
            )
        ):
            raise ValueError("unknown-start shadow canary reset binding is invalid")
    terminal_paths(payload)
    return payload


def terminal_paths(experiment: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    output = Path(str(experiment.get("output", "")))
    if (
        not output.is_absolute()
        or not output.stem.startswith(
            (
                "known-start-shadow-canary-v",
                "unknown-start-shadow-canary-v",
            )
        )
    ):
        raise ValueError("physical shadow canary output is invalid")
    stem = output.stem
    return (
        output,
        output.with_name(f"{stem}-claim.json"),
        output.with_name(f"{stem}-failure.json"),
        output.with_name(f"{stem}-evaluation.json"),
    )


def claim_canary(
    path: Path,
    session_id: str,
    experiment_config_fingerprint: str,
) -> dict[str, Any]:
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_canary_claim.v1",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "experiment_config_fingerprint": experiment_config_fingerprint,
        "evaluations_claimed": 1,
        "apply_action": False,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("physical shadow canary was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def _identity(source: Mapping[str, str], label: str) -> ArtifactIdentity:
    identity = ArtifactIdentity.from_artifact(Path(source["path"]))
    if identity.fingerprint != source["fingerprint"]:
        raise ValueError(f"{label} identity changed")
    return identity


@dataclass(frozen=True)
class AuthenticatedCanaryArtifacts:
    proposal: ArtifactIdentity
    proposal_report: ArtifactIdentity
    readiness: ArtifactIdentity
    action_model: ArtifactIdentity
    held_out_gate: ArtifactIdentity


def authenticate_runtime_sources(experiment: Mapping[str, Any]) -> Path:
    repository = Path(__file__).resolve().parents[1]
    sources = experiment.get("runtime_sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("physical shadow canary runtime sources are invalid")
    for relative, expected in sources.items():
        path = repository / str(relative)
        if not path.is_file() or artifact_fingerprint(path) != expected:
            raise ValueError(f"physical shadow canary runtime source changed: {relative}")
    return repository


def authenticated_deployment_revision(
    experiment: Mapping[str, Any], claimed_revision: str
) -> str:
    authenticate_runtime_sources(experiment)
    implementation = str(experiment["evaluator"]["implementation_revision"])
    revisions = (implementation, claimed_revision)
    if any(
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        for revision in revisions
    ):
        raise ValueError("physical shadow canary deployment revision changed")
    return claimed_revision


def prepare_worker(
    experiment: Mapping[str, Any], output: Path, recording_root: Path
) -> ControlWorkerArtifacts:
    """Write the one worker manifest named by the authenticated canary contract."""

    authenticated = _authenticate_frozen_artifacts(experiment)
    start = experiment["known_start"]
    manifest = recording_root / start["reference"] / "manifest.json"
    if artifact_fingerprint(manifest) != start["manifest_fingerprint"]:
        raise ValueError("known-start recording identity changed")
    planner = CEMConfig(**experiment["worker"]["planner"])
    artifacts = ControlWorkerArtifacts(
        proposal=authenticated.proposal.path,
        adapter=authenticated.action_model.path,
        planner=planner,
    )
    artifacts.write(output)
    return artifacts


def _authenticate_frozen_artifacts(
    experiment: Mapping[str, Any],
) -> AuthenticatedCanaryArtifacts:
    proposal = _identity(experiment["proposal"], "proposal")
    proposal_report = ArtifactIdentity.from_artifact(
        proposal.path.with_suffix(proposal.path.suffix + ".json")
    )
    if (
        proposal_report.fingerprint
        != experiment["proposal"]["training_report_fingerprint"]
    ):
        raise ValueError("proposal training report identity changed")
    readiness = _identity(experiment["proposal"]["readiness"], "proposal readiness")
    readiness_payload = json.loads(readiness.path.read_text())
    if (
        readiness_payload.get("passed") is not True
        or readiness_payload.get("proposal_fingerprint") != proposal.fingerprint
        or len(readiness_payload.get("held_out_evaluations", ())) != 2
    ):
        raise ValueError("proposal readiness is invalid")
    action_model = _identity(experiment["action_model"], "action model")
    held_out = _identity(
        experiment["action_model"]["held_out_gate"], "held-out gate"
    )
    if json.loads(held_out.path.read_text()).get("passed") is not True:
        raise ValueError("action model held-out gate did not pass")
    return AuthenticatedCanaryArtifacts(
        proposal,
        proposal_report,
        readiness,
        action_model,
        held_out,
    )


def evaluate_canary(
    experiment_config: Path,
    session_path: Path,
    output: Path,
    deployed_revision_claim: str,
) -> dict[str, Any]:
    experiment = load_experiment_config(experiment_config)
    frozen_output, claim_path, failure_path, evaluation_path = terminal_paths(experiment)
    if (
        output.resolve() != frozen_output
        or output.exists()
        or failure_path.exists()
        or evaluation_path.exists()
    ):
        raise ValueError("physical shadow canary is already terminal")
    evaluator = experiment["evaluator"]
    evaluator_path = Path(__file__).resolve()
    deployed_revision = authenticated_deployment_revision(
        experiment, deployed_revision_claim
    )
    if (
        evaluator.get("path") != "jepa_wm/physical_shadow_canary.py"
        or artifact_fingerprint(evaluator_path) != evaluator.get("fingerprint")
    ):
        raise ValueError("physical shadow canary evaluator identity changed")
    claim = json.loads(claim_path.read_text())
    if (
        claim.get("schema")
        != "quantis.jepa_wm_physical_shadow_canary_claim.v1"
        or claim.get("session_id") != session_path.name
        or claim.get("experiment_config_fingerprint")
        != artifact_fingerprint(experiment_config)
        or claim.get("evaluations_claimed") != 1
        or claim.get("apply_action") is not False
    ):
        raise ValueError("physical shadow canary claim is invalid")

    authenticated = _authenticate_frozen_artifacts(experiment)
    proposal = authenticated.proposal
    proposal_report = authenticated.proposal_report
    readiness = authenticated.readiness
    action_model = authenticated.action_model
    held_out = authenticated.held_out_gate

    session = ControlSession(session_path.resolve(), session_path.resolve().parent.parent)
    observation, state = session.load_capture()
    response = session.load_response()
    shadow = session.load_shadow()
    safety = session.load_shadow_safety()
    start = experiment["known_start"]
    unknown_start = experiment.get("unknown_start")
    handoff_path = session.path / "unknown_start_handoff.json"
    reference_manifest = (
        session.data_root / "recordings" / start["reference"] / "manifest.json"
    )
    forbidden = (
        session.result_path,
        session.execution_path,
        session.direct_safety_path,
        session.candidate_binding_path,
        session.insertion_trial_binding_path,
    )
    if (
        state.reference_recording != start["reference"]
        or state.seed != start["seed"]
        or artifact_fingerprint(reference_manifest)
        != start["manifest_fingerprint"]
        or observation.warmup_frames != start["context_index"]
        or observation.physical_routing is None
        or state.contact_grasp_target_policy is None
        or state.collision_detected
        or state.contact_force_newtons > 2.0
        or observation.expected_proposal.resolve() != proposal.path
        or response.proposal.resolve() != proposal.path
        or response.proposal_fingerprint != proposal.fingerprint
        or shadow.adapter.resolve() != action_model.path
        or shadow.authority is not CandidateAuthority.SHADOW_ONLY
        or safety.authority is not CandidateAuthority.SHADOW_ONLY
        or any(path.exists() for path in forbidden)
    ):
        raise ValueError("physical shadow canary gate failed")
    handoff = None
    if experiment["schema"] in (EXPERIMENT_SCHEMA_V3, EXPERIMENT_SCHEMA_V4):
        if not handoff_path.is_file() or not isinstance(unknown_start, Mapping):
            raise ValueError("unknown-start shadow canary handoff is missing")
        handoff = UnknownStartControlHandoff.from_dict(
            json.loads(handoff_path.read_text())
        )
        reset_recording = session.data_root / "recordings" / unknown_start["recording_id"]
        reset_result_path = reset_recording / "RESULT.json"
        reset_evidence_path = reset_recording / "unknown_start_reset_evidence.json"
        routing_target_path = session.path / "unknown_start_routing_target.json"
        routing_step_path = session.path / "unknown_start_routing_step.json"
        reset_evidence = UnknownStartResetEvidence.from_dict(
            json.loads(reset_evidence_path.read_text())
        )
        reference_target = json.loads(reference_manifest.read_text()).get(
            "metadata", {}
        ).get("insertion_target")
        if not isinstance(reference_target, dict):
            raise ValueError("unknown-start shadow reference target is invalid")
        expected_routing_target = {
            **reference_target,
            "socket_position": list(reset_evidence.workspace.socket_position_m),
            "socket_orientation_wxyz": reference_target[
                "socket_orientation_wxyz"
            ],
            "geometry_source": "authenticated_unknown_start_live_state",
            "reset_recording_id": unknown_start["recording_id"],
        }
        reference_scene_offset = json.loads(reference_manifest.read_text()).get(
            "metadata", {}
        ).get("scene_offset_m")
        if (
            not isinstance(reference_scene_offset, list)
            or len(reference_scene_offset) != 3
        ):
            raise ValueError("unknown-start shadow reference offset is invalid")
        expected_scene_translation = tuple(
            reset_value - float(reference_value)
            for reset_value, reference_value in zip(
                reset_evidence.sample.scene_offset_m,
                reference_scene_offset,
            )
        )
        routing_target = json.loads(routing_target_path.read_text())
        routing_step = json.loads(routing_step_path.read_text())
        reset_lines = (reset_recording / "steps.jsonl").read_text().splitlines()
        if len(reset_lines) != 1:
            raise ValueError("unknown-start reset step is invalid")
        reset_step = json.loads(reset_lines[0])
        expected_routing_step = {
            field: reset_step[field]
            for field in (
                "plug_position",
                "plug_orientation_wxyz",
                "end_effector_world_position",
                "gripper_frame_world_position",
                "gripper_width_m",
                "arm_tracking_error_rad",
                "gripper_tracking_error_m",
                "contact_force_newtons",
                "plug_attached",
            )
        }
        expected_handoff = UnknownStartControlHandoff(
            session_id=session.session_id,
            reset_recording_id=unknown_start["recording_id"],
            reset_seed=unknown_start["seed"],
            reset_result_fingerprint=unknown_start["result_fingerprint"],
            reset_evidence_fingerprint=unknown_start["evidence_fingerprint"],
            reset_contract_fingerprint=unknown_start["contract_fingerprint"],
            reference_recording=start["reference"],
            reference_seed=start["seed"],
            context_fingerprint=artifact_fingerprint(
                session.data_root / observation.context_frame
            ),
            routing_target_fingerprint=artifact_fingerprint(routing_target_path),
            routing_step_fingerprint=artifact_fingerprint(routing_step_path),
            request_fingerprint=artifact_fingerprint(session.request_path),
            state_fingerprint=artifact_fingerprint(session.state_path),
        )
        if (
            handoff != expected_handoff
            or artifact_fingerprint(reset_result_path)
            != unknown_start["result_fingerprint"]
            or artifact_fingerprint(reset_evidence_path)
            != unknown_start["evidence_fingerprint"]
            or reset_evidence.sample.seed != unknown_start["seed"]
            or expected_scene_translation
            != state.contact_grasp_target_policy.scene_translation_m
            or routing_target != expected_routing_target
            or routing_step != expected_routing_step
            or observation.physical_routing
            != PhysicalRoutingObservation.from_recorded_step(
                expected_routing_step,
                expected_routing_target,
                DroidAction((0.0,) * 7),
            )
        ):
            raise ValueError("unknown-start shadow canary handoff is inauthentic")
    elif handoff_path.exists():
        raise ValueError("known-start shadow canary has an unexpected reset handoff")

    evaluation_passed = shadow.passes_shadow_gate and safety.passed
    report = {
        "schema": "quantis.jepa_wm_physical_shadow_canary_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "evaluation_passed": evaluation_passed,
        "recovery_verified": False,
        "outcome": (
            "physical_residual_live_shadow_candidate"
            if evaluation_passed
            else "authenticated_model_shadow_rejection"
        ),
        "experiment_config": ArtifactIdentity.from_artifact(experiment_config).to_dict(),
        "evaluator": {
            "path": str(evaluator_path),
            "fingerprint": evaluator["fingerprint"],
            "implementation_revision": evaluator["implementation_revision"],
            "deployed_revision": deployed_revision,
        },
        "claim": ArtifactIdentity.from_artifact(claim_path).to_dict(),
        "proposal": proposal.to_dict(),
        "proposal_training_report": proposal_report.to_dict(),
        "proposal_readiness": readiness.to_dict(),
        "action_model": action_model.to_dict(),
        "held_out_gate": held_out.to_dict(),
        "session_id": session.session_id,
        "known_start": start,
        "unknown_start": unknown_start,
        "unknown_start_handoff": (
            ArtifactIdentity.from_artifact(handoff_path).to_dict()
            if handoff is not None
            else None
        ),
        "physical_routing": observation.physical_routing.to_dict(),
        "direct_response": ArtifactIdentity.from_artifact(session.response_path).to_dict(),
        "shadow": ArtifactIdentity.from_artifact(session.shadow_path).to_dict(),
        "shadow_safety": ArtifactIdentity.from_artifact(session.shadow_safety_path).to_dict(),
        "shadow_gate_passed": shadow.passes_shadow_gate,
        "counterfactual_safety_passed": safety.passed,
        "selected_action_scale": _serialized_action_scale(
            safety.selected_action_scale
        ),
        "apply_action": False,
        "execution_started": False,
        "trained": False,
        "filming_authorized": False,
        "hardware_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(evaluation_path, report)
    return report


def finalize_recovery(
    experiment_config: Path,
    recovery_checkpoint_root: Path,
    claimed_revision: str,
) -> dict[str, Any]:
    experiment = load_experiment_config(experiment_config)
    deployed_revision = authenticated_deployment_revision(
        experiment, claimed_revision
    )
    output, _, failure_path, evaluation_path = terminal_paths(experiment)
    checkpoint_root = Path(experiment["action_model"]["path"]).resolve().parents[1]
    recovery_evaluation = recovery_checkpoint_root / evaluation_path.relative_to(
        checkpoint_root
    )
    recovery_output = recovery_checkpoint_root / output.relative_to(checkpoint_root)
    if (
        output.exists()
        or failure_path.exists()
        or not evaluation_path.is_file()
        or not recovery_evaluation.is_file()
        or artifact_fingerprint(evaluation_path)
        != artifact_fingerprint(recovery_evaluation)
    ):
        raise ValueError("physical shadow canary recovery is invalid")
    evaluation = json.loads(evaluation_path.read_text())
    if (
        evaluation.get("schema")
        != "quantis.jepa_wm_physical_shadow_canary_evaluation.v1"
        or evaluation.get("status") != "evaluated_pending_recovery"
        or evaluation.get("passed") is not False
        or evaluation.get("recovery_verified") is not False
        or evaluation.get("apply_action") is not False
        or not isinstance(evaluation.get("evaluation_passed"), bool)
    ):
        raise ValueError("physical shadow canary evaluation is invalid")
    evaluation_passed = evaluation["evaluation_passed"]
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_canary_terminal.v1",
        "status": "passed" if evaluation_passed else "failed_model_gate",
        "passed": evaluation_passed,
        "model_gate_adjudicated": True,
        "evaluation": ArtifactIdentity.from_artifact(evaluation_path).to_dict(),
        "recovery_evaluation": ArtifactIdentity.from_artifact(
            recovery_evaluation
        ).to_dict(),
        "recovery_verified": True,
        "deployed_revision": deployed_revision,
        "apply_action": False,
        "execution_started": False,
        "trained": False,
        "filming_authorized": False,
        "hardware_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(recovery_output, payload)
    write_json_atomic(output, payload)
    if artifact_fingerprint(output) != artifact_fingerprint(recovery_output):
        raise ValueError("physical shadow canary terminal recovery changed")
    return payload


def write_failure(
    error: str, session_id: str, experiment: Mapping[str, Any]
) -> dict[str, Any]:
    output, claim_path, failure_path, _ = terminal_paths(experiment)
    if output.exists() or failure_path.exists():
        raise ValueError("physical shadow canary is already terminal")
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_canary_failure.v1",
        "status": "failed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "error": error,
        "claim": ArtifactIdentity.from_artifact(claim_path).to_dict(),
        "retry_authorized": False,
        "apply_action": False,
        "training_authorized": False,
        "filming_authorized": False,
    }
    write_json_atomic(failure_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    claim = subparsers.add_parser("claim")
    claim.add_argument("--config", type=Path, required=True)
    claim.add_argument("--session", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--session-path", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--deployed-revision", required=True)
    failure = subparsers.add_parser("failure")
    failure.add_argument("--config", type=Path, required=True)
    failure.add_argument("--session")
    failure.add_argument("--error", required=True)
    prepare = subparsers.add_parser("prepare-worker")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--recording-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize-recovery")
    finalize.add_argument("--config", type=Path, required=True)
    finalize.add_argument("--recovery-checkpoint-root", type=Path, required=True)
    finalize.add_argument("--deployed-revision", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-worker":
        experiment = load_experiment_config(args.config)
        authenticate_runtime_sources(experiment)
        payload = prepare_worker(
            experiment, args.output, args.recording_root
        ).to_dict()
    elif args.command == "finalize-recovery":
        payload = finalize_recovery(
            args.config,
            args.recovery_checkpoint_root,
            args.deployed_revision,
        )
    elif args.command == "claim":
        fingerprint = sha256(args.config.resolve().read_bytes()).hexdigest()
        experiment = load_experiment_config(args.config)
        _, claim_path, _, _ = terminal_paths(experiment)
        payload = claim_canary(claim_path, args.session, fingerprint)
    elif args.command == "failure":
        experiment = load_experiment_config(args.config)
        payload = write_failure(
            args.error, args.session or str(experiment["session_id"]), experiment
        )
    else:
        payload = evaluate_canary(
            args.config,
            args.session_path,
            args.output,
            args.deployed_revision,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
