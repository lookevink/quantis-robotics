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
from jepa_wm.physical_shadow_canary_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)
from jepa_wm.planner import CEMConfig
from jepa_wm.shadow_planning import CandidateAuthority
from jepa_wm.training_artifact import ArtifactIdentity, artifact_fingerprint
from jepa_wm.worker_artifacts import ControlWorkerArtifacts
from sim.control_session import ControlSession


EXPERIMENT_SCHEMA = "quantis.jepa_wm_physical_shadow_canary_experiment.v1"


def load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    fingerprint = sha256(encoded).hexdigest()
    if (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT != "PENDING_CHECKPOINT"
        and fingerprint != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
    ):
        raise ValueError("physical shadow canary configuration changed")
    payload = json.loads(encoded)
    execution = payload.get("execution", {})
    gate = payload.get("gate", {})
    if (
        payload.get("schema") != EXPERIMENT_SCHEMA
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
    terminal_paths(payload)
    return payload


def terminal_paths(experiment: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    output = Path(str(experiment.get("output", "")))
    if not output.is_absolute() or output.name != "known-start-shadow-canary-v1.json":
        raise ValueError("physical shadow canary output is invalid")
    return (
        output,
        output.with_name("known-start-shadow-canary-v1-claim.json"),
        output.with_name("known-start-shadow-canary-v1-failure.json"),
        output.with_name("known-start-shadow-canary-v1-evaluation.json"),
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
        or not shadow.passes_shadow_gate
        or not safety.passed
        or any(path.exists() for path in forbidden)
    ):
        raise ValueError("physical shadow canary gate failed")

    report = {
        "schema": "quantis.jepa_wm_physical_shadow_canary_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "passed": False,
        "evaluation_passed": True,
        "recovery_verified": False,
        "outcome": "physical_residual_live_shadow_candidate",
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
        "physical_routing": observation.physical_routing.to_dict(),
        "direct_response": ArtifactIdentity.from_artifact(session.response_path).to_dict(),
        "shadow": ArtifactIdentity.from_artifact(session.shadow_path).to_dict(),
        "shadow_safety": ArtifactIdentity.from_artifact(session.shadow_safety_path).to_dict(),
        "shadow_gate_passed": shadow.passes_shadow_gate,
        "counterfactual_safety_passed": safety.passed,
        "selected_action_scale": safety.selected_action_scale.to_dict(),
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
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_canary_terminal.v1",
        "status": "passed",
        "passed": True,
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
