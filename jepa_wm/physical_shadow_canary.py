"""One-shot terminal evidence for the physical residual shadow canary."""

from __future__ import annotations

import argparse
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
OUTPUT_PATH = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/quantis_physical_state_residual_v1/"
    "known-start-shadow-canary-v1.json"
)
CLAIM_PATH = OUTPUT_PATH.with_name("known-start-shadow-canary-v1-claim.json")
FAILURE_PATH = OUTPUT_PATH.with_name("known-start-shadow-canary-v1-failure.json")


def load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    fingerprint = sha256(encoded).hexdigest()
    if (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT != "PENDING_CHECKPOINT"
        and fingerprint != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
    ):
        raise ValueError("physical shadow canary configuration changed")
    payload = json.loads(encoded)
    start = payload.get("known_start", {})
    execution = payload.get("execution", {})
    gate = payload.get("gate", {})
    if (
        payload.get("schema") != EXPERIMENT_SCHEMA
        or start.get("reference")
        != "contact-insertion-v10-drive-slow-2600-held-01"
        or start.get("seed") != 12601
        or start.get("context_index") != 110
        or start.get("context_purpose") != "contact_grasp"
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
        or payload.get("output") != str(OUTPUT_PATH)
    ):
        raise ValueError("physical shadow canary contract is invalid")
    return payload


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


def prepare_worker(
    experiment: Mapping[str, Any], output: Path, recording_root: Path
) -> ControlWorkerArtifacts:
    """Write the one worker manifest named by the authenticated canary contract."""

    proposal, action_model = _authenticate_frozen_artifacts(experiment)
    start = experiment["known_start"]
    manifest = recording_root / start["reference"] / "manifest.json"
    if artifact_fingerprint(manifest) != start["manifest_fingerprint"]:
        raise ValueError("known-start recording identity changed")
    planner = CEMConfig(**experiment["worker"]["planner"])
    artifacts = ControlWorkerArtifacts(
        proposal=proposal.path,
        adapter=action_model.path,
        planner=planner,
    )
    artifacts.write(output)
    return artifacts


def _authenticate_frozen_artifacts(
    experiment: Mapping[str, Any],
) -> tuple[ArtifactIdentity, ArtifactIdentity]:
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
    return proposal, action_model


def evaluate_canary(
    experiment_config: Path,
    session_path: Path,
    output: Path,
    evaluator_revision: str,
) -> dict[str, Any]:
    if output.resolve() != OUTPUT_PATH or output.exists() or FAILURE_PATH.exists():
        raise ValueError("physical shadow canary is already terminal")
    experiment = load_experiment_config(experiment_config)
    evaluator = experiment["evaluator"]
    evaluator_path = Path(__file__).resolve()
    if (
        evaluator.get("path") != "jepa_wm/physical_shadow_canary.py"
        or artifact_fingerprint(evaluator_path) != evaluator.get("fingerprint")
        or evaluator_revision != evaluator.get("implementation_revision")
    ):
        raise ValueError("physical shadow canary evaluator identity changed")
    claim = json.loads(CLAIM_PATH.read_text())
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

    proposal, action_model = _authenticate_frozen_artifacts(experiment)
    proposal_report = ArtifactIdentity.from_artifact(
        proposal.path.with_suffix(proposal.path.suffix + ".json")
    )
    readiness = _identity(experiment["proposal"]["readiness"], "proposal readiness")
    held_out = _identity(
        experiment["action_model"]["held_out_gate"], "held-out gate"
    )

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
        "status": "evaluated",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "outcome": "physical_residual_live_shadow_candidate",
        "experiment_config": ArtifactIdentity.from_artifact(experiment_config).to_dict(),
        "evaluator": {
            "path": str(evaluator_path),
            "fingerprint": evaluator["fingerprint"],
            "implementation_revision": evaluator_revision,
        },
        "claim": ArtifactIdentity.from_artifact(CLAIM_PATH).to_dict(),
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
    write_json_atomic(output, report)
    return report


def write_failure(error: str, session_id: str) -> dict[str, Any]:
    if OUTPUT_PATH.exists() or FAILURE_PATH.exists():
        raise ValueError("physical shadow canary is already terminal")
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_canary_failure.v1",
        "status": "failed",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "error": error,
        "claim": ArtifactIdentity.from_artifact(CLAIM_PATH).to_dict(),
        "retry_authorized": False,
        "apply_action": False,
        "training_authorized": False,
        "filming_authorized": False,
    }
    write_json_atomic(FAILURE_PATH, payload)
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
    evaluate.add_argument("--evaluator-revision", required=True)
    failure = subparsers.add_parser("failure")
    failure.add_argument("--session", required=True)
    failure.add_argument("--error", required=True)
    prepare = subparsers.add_parser("prepare-worker")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--recording-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-worker":
        experiment = load_experiment_config(args.config)
        payload = prepare_worker(
            experiment, args.output, args.recording_root
        ).to_dict()
    elif args.command == "claim":
        fingerprint = sha256(args.config.resolve().read_bytes()).hexdigest()
        load_experiment_config(args.config)
        payload = claim_canary(CLAIM_PATH, args.session, fingerprint)
    elif args.command == "failure":
        payload = write_failure(args.error, args.session)
    else:
        payload = evaluate_canary(
            args.config,
            args.session_path,
            args.output,
            args.evaluator_revision,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
