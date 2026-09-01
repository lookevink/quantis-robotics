"""One-shot offline adjudication of the corrected physical shadow planner."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.persistence import write_json_atomic
from jepa_wm.physical_shadow_replay_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)
from jepa_wm.planner_readiness import FirstActionReason
from jepa_wm.shadow_planning import ShadowPlanningRequest, ShadowSearchEvidence
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from jepa_wm.training_artifact import ArtifactIdentity, artifact_fingerprint
from jepa_wm.worker_artifacts import ControlWorkerArtifacts


SCHEMA = "quantis.jepa_wm_physical_shadow_replay_experiment.v1"
NON_AUTHORITY = {
    "isaac": False,
    "capture": False,
    "apply_action": False,
    "train": False,
    "film": False,
    "hardware": False,
    "production": False,
}


def load_experiment(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    fingerprint = sha256(encoded).hexdigest()
    if (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT != "PENDING_CHECKPOINT"
        and fingerprint != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
    ):
        raise ValueError("physical shadow replay configuration changed")
    payload = json.loads(encoded)
    execution = dict(payload.get("execution", {}))
    replays = execution.pop("replays", None)
    if (
        payload.get("schema") != SCHEMA
        or replays != 1
        or execution != NON_AUTHORITY
        or not Path(str(payload.get("output", ""))).is_absolute()
    ):
        raise ValueError("physical shadow replay contract is invalid")
    repository = Path(__file__).resolve().parents[1]
    evaluator = payload.get("evaluator", {})
    sources = payload.get("runtime_sources", {})
    if (
        evaluator.get("path") != "jepa_wm/physical_shadow_replay.py"
        or not isinstance(sources, Mapping)
        or not sources
    ):
        raise ValueError("physical shadow replay runtime identity is invalid")
    for relative, expected in sources.items():
        if expected != "PENDING_CHECKPOINT" and artifact_fingerprint(
            repository / relative
        ) != expected:
            raise ValueError(f"physical shadow replay runtime changed: {relative}")
    if (
        evaluator.get("fingerprint") != "PENDING_CHECKPOINT"
        and artifact_fingerprint(repository / evaluator["path"])
        != evaluator["fingerprint"]
    ):
        raise ValueError("physical shadow replay evaluator changed")
    return payload


def paths(experiment: Mapping[str, Any]) -> tuple[Path, Path, Path, Path]:
    root = Path(str(experiment["output"]))
    return root, root / "claim.json", root / "evaluation.json", root / "RESULT.json"


def authenticate_inputs(experiment: Mapping[str, Any]) -> None:
    source = Path(str(experiment["source"]["session"]))
    for name, expected in experiment["source"]["files"].items():
        if artifact_fingerprint(source / name) != expected:
            raise ValueError(f"physical shadow replay source changed: {name}")
    worker = experiment["worker"]
    manifest = Path(str(worker["manifest"]))
    if artifact_fingerprint(manifest) != worker["fingerprint"]:
        raise ValueError("physical shadow replay worker manifest changed")


def claim(experiment: Mapping[str, Any]) -> dict[str, Any]:
    authenticate_inputs(experiment)
    root, claim_path, _, _ = paths(experiment)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError("physical shadow replay was already claimed") from error
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_replay_claim.v1",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "replays_claimed": 1,
        **NON_AUTHORITY,
    }
    write_json_atomic(claim_path, payload)
    return payload


def evaluate(experiment: Mapping[str, Any]) -> dict[str, Any]:
    authenticate_inputs(experiment)
    root, claim_path, evaluation_path, result_path = paths(experiment)
    if not claim_path.is_file() or evaluation_path.exists() or result_path.exists():
        raise ValueError("physical shadow replay is not evaluable")
    source = Path(str(experiment["source"]["session"]))
    source_observation = ControlObservation.from_dict(
        json.loads((source / "request.json").read_text())
    )
    source_direct = ProposedControl.from_dict(
        json.loads((source / "response.json").read_text())
    )
    old = ShadowSearchEvidence.from_dict(
        json.loads((source / "shadow.json").read_text())
    )
    old_safety = ShadowSafetyEvidence.from_dict(
        json.loads((source / "shadow_safety.json").read_text())
    )
    replay_request = ShadowPlanningRequest.from_dict(
        json.loads((root / "shadow_request.json").read_text())
    )
    replay = ShadowSearchEvidence.from_dict(
        json.loads((root / "shadow.json").read_text())
    )
    manifest = ControlWorkerArtifacts.load(Path(experiment["worker"]["manifest"]))
    forbidden = (
        source / "result.json",
        source / "execution.json",
        source / "candidate_binding.json",
        source / "insertion_trial_binding.json",
    )
    if (
        old.passes_shadow_gate
        or old.objective_improvement <= 0.0
        or old.first_action_gate.reasons
        != (FirstActionReason.DIRECTION_MISMATCH,)
        or not old_safety.passed
        or replay_request.observation != source_observation
        or replay_request.direct_control != source_direct
        or replay_request.expected_adapter.resolve() != manifest.adapter
        or replay.observation_id != old.observation_id
        or replay.proposal.resolve() != old.proposal.resolve()
        or replay.adapter.resolve() != old.adapter.resolve()
        or replay.objective_improvement <= 0.0
        or not replay.first_action_gate.passed
        or not replay.passes_shadow_gate
        or any(path.exists() for path in forbidden)
    ):
        raise ValueError("physical shadow corrected replay failed")
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_replay_evaluation.v1",
        "status": "evaluated_pending_recovery",
        "passed": False,
        "adjudication_passed": True,
        "recovery_verified": False,
        "claim": ArtifactIdentity.from_artifact(claim_path).to_dict(),
        "source_shadow": ArtifactIdentity.from_artifact(source / "shadow.json").to_dict(),
        "replay_shadow": ArtifactIdentity.from_artifact(root / "shadow.json").to_dict(),
        "old_first_action_cosine": old.first_action_gate.cosine,
        "replay_first_action_cosine": replay.first_action_gate.cosine,
        "old_objective_improvement": old.objective_improvement,
        "replay_objective_improvement": replay.objective_improvement,
        **NON_AUTHORITY,
    }
    write_json_atomic(evaluation_path, payload)
    return payload


def finalize(experiment: Mapping[str, Any], recovery_checkpoint_root: Path) -> dict[str, Any]:
    root, _, evaluation_path, result_path = paths(experiment)
    checkpoint_root = Path(experiment["worker"]["manifest"]).resolve().parent
    recovery_root = recovery_checkpoint_root / root.relative_to(checkpoint_root)
    recovery_evaluation = recovery_root / evaluation_path.name
    recovery_result = recovery_root / result_path.name
    if (
        result_path.exists()
        or not recovery_evaluation.is_file()
        or artifact_fingerprint(evaluation_path)
        != artifact_fingerprint(recovery_evaluation)
    ):
        raise ValueError("physical shadow replay recovery is invalid")
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_replay_result.v1",
        "status": "passed",
        "passed": True,
        "evaluation": ArtifactIdentity.from_artifact(evaluation_path).to_dict(),
        "recovery_evaluation": ArtifactIdentity.from_artifact(
            recovery_evaluation
        ).to_dict(),
        "recovery_verified": True,
        **NON_AUTHORITY,
    }
    write_json_atomic(recovery_result, payload)
    write_json_atomic(result_path, payload)
    if artifact_fingerprint(result_path) != artifact_fingerprint(recovery_result):
        raise ValueError("physical shadow replay result recovery changed")
    return payload


def failure(experiment: Mapping[str, Any], error: str) -> dict[str, Any]:
    root, claim_path, _, result_path = paths(experiment)
    failure_path = root / "FAILURE.json"
    if result_path.exists() or failure_path.exists():
        raise ValueError("physical shadow replay is already terminal")
    payload = {
        "schema": "quantis.jepa_wm_physical_shadow_replay_failure.v1",
        "status": "failed",
        "error": error,
        "claim": ArtifactIdentity.from_artifact(claim_path).to_dict(),
        "retry_authorized": False,
        **NON_AUTHORITY,
    }
    write_json_atomic(failure_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("claim", "evaluate", "finalize", "failure"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recovery-checkpoint-root", type=Path)
    parser.add_argument("--error")
    args = parser.parse_args(argv)
    experiment = load_experiment(args.config)
    if args.command == "claim":
        payload = claim(experiment)
    elif args.command == "evaluate":
        payload = evaluate(experiment)
    elif args.command == "finalize":
        if args.recovery_checkpoint_root is None:
            parser.error("finalize requires --recovery-checkpoint-root")
        payload = finalize(experiment, args.recovery_checkpoint_root)
    else:
        if not args.error:
            parser.error("failure requires --error")
        payload = failure(experiment, args.error)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
