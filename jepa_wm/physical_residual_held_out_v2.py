"""One-shot v2 held-out gate with complete pre-claim authentication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, Sequence

from jepa_wm.persistence import write_json_atomic
from jepa_wm.physical_residual_held_out import (
    _authenticate_authority,
    _authenticate_recordings,
    _router_population,
    evaluate_population_gate,
)
from jepa_wm.physical_residual_held_out_v2_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)
from jepa_wm.runtime_environment import validate_headless_runtime
from jepa_wm.training_artifact import ArtifactIdentity, artifact_fingerprint


EXPERIMENT_SCHEMA = "quantis.jepa_wm_physical_state_residual_held_out_experiment.v2"
OUTPUT_PATH = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/"
    "quantis_physical_state_residual_v1/canonical-held-out-v2.json"
)
FAILURE_PATH = OUTPUT_PATH.with_name("canonical-held-out-v2-failure.json")
INVOCATION_FAILURE_PATH = OUTPUT_PATH.with_name(
    "canonical-held-out-v2-invocation-failure.json"
)
ACCESS_CLAIM_PATH = OUTPUT_PATH.with_name(
    "canonical-held-out-v2-access-claim.json"
)
TRAIN_EVALUATION_PATH = OUTPUT_PATH.with_name("train-evaluation.json")


def _sha256(path: Path) -> str:
    return artifact_fingerprint(path.resolve())


def load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    fingerprint = sha256(encoded).hexdigest()
    if (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT != "PENDING_CHECKPOINT"
        and fingerprint != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
    ):
        raise ValueError("physical residual held-out v2 configuration changed")
    payload = json.loads(encoded)
    corpus = payload.get("corpus", {})
    recordings = corpus.get("recordings", ())
    execution = payload.get("execution", {})
    recovery = payload.get("recovery", {})
    if (
        payload.get("schema") != EXPERIMENT_SCHEMA
        or corpus.get("split") != "held_out"
        or corpus.get("camera") != "wrist"
        or corpus.get("window") != {"start_index": 113, "count": 168, "stride": 1}
        or corpus.get("action_horizon") != 3
        or [item.get("name") for item in recordings]
        != [
            "contact-insertion-v10-drive-slow-2600-held-00",
            "contact-insertion-v10-drive-slow-2600-held-01",
        ]
        or [item.get("seed") for item in recordings] != [12600, 12601]
        or payload.get("gate", {}).get("populations")
        != ["combined", *(item["name"] for item in recordings)]
        or execution
        != {
            "evaluations": 1,
            "train": False,
            "run_isaac": False,
            "issue_live_action": False,
            "film": False,
            "stop_on_new_failure_class": True,
        }
        or payload.get("output") != str(OUTPUT_PATH)
        or recovery.get("required_terminal_artifacts") != [str(ACCESS_CLAIM_PATH)]
        or recovery.get("terminal_report_alternatives")
        != [str(OUTPUT_PATH), str(FAILURE_PATH)]
    ):
        raise ValueError("physical residual held-out v2 contract is invalid")
    return payload


def _authenticate_identity(identity: Mapping[str, str], label: str) -> Path:
    path = Path(identity["path"]).resolve()
    if not path.is_file() or _sha256(path) != identity["fingerprint"]:
        raise ValueError(f"{label} identity changed")
    return path


def authenticate_prior_evidence(experiment: Mapping[str, Any]) -> dict[str, Any]:
    """Authenticate the v1 no-score negative and deployment remediation."""

    prior = experiment["prior_attempt"]
    claim = _authenticate_identity(prior["access_claim"], "v1 access claim")
    failure = _authenticate_identity(prior["failure"], "v1 failure")
    evaluation = Path(prior["evaluation_report"]).resolve()
    if prior.get("require_no_evaluation_report") is not True or evaluation.exists():
        raise ValueError("v1 produced an evaluation report")
    runtime = experiment["runtime_remediation"]
    runtime_report = _authenticate_identity(
        runtime["report"], "runtime remediation report"
    )
    runtime_claim = _authenticate_identity(
        runtime["claim"], "runtime remediation claim"
    )
    claim_payload = json.loads(claim.read_text())
    failure_payload = json.loads(failure.read_text())
    if (
        claim_payload.get("schema")
        != "quantis.jepa_wm_physical_state_residual_held_out_access.v1"
        or claim_payload.get("evaluations_claimed") != 1
        or failure_payload.get("schema")
        != "quantis.jepa_wm_physical_state_residual_held_out_failure.v1"
        or failure_payload.get("canonical_accessed") is not True
        or failure_payload.get("terminal_experiment_failure") is not True
        or failure_payload.get("retry_authorized") is not False
        or failure_payload.get("retraining_authorized") is not False
        or failure_payload.get("live_action_authorized") is not False
        or failure_payload.get("filming_authorized") is not False
    ):
        raise ValueError("v1 no-score terminal evidence is invalid")
    return {
        "v1_access_claim": ArtifactIdentity.from_artifact(claim).to_dict(),
        "v1_failure": ArtifactIdentity.from_artifact(failure).to_dict(),
        "v1_evaluation_absent": str(evaluation),
        "runtime_report": ArtifactIdentity.from_artifact(runtime_report).to_dict(),
        "runtime_claim": ArtifactIdentity.from_artifact(runtime_claim).to_dict(),
    }


def _authenticate_runtime_evidence(
    experiment: Mapping[str, Any], actual: Mapping[str, str]
) -> None:
    expected = experiment["runtime_remediation"]
    for field in (
        "checkpoint_fingerprint",
        "dinov3_checkpoint_fingerprint",
        "source_revision",
        "dinov3_revision",
    ):
        if actual.get(field) != expected[field]:
            raise ValueError(f"authenticated runtime {field} changed")
    report = json.loads(Path(expected["report"]["path"]).read_text())
    claim = json.loads(Path(expected["claim"]["path"]).read_text())
    if (
        report.get("schema") != "quantis.jepa_wm_model_load_preflight.v1"
        or report.get("status") != "passed"
        or report.get("runtime") != dict(actual)
        or report.get("recordings_loaded") is not False
        or report.get("canonical_accessed") is not False
        or report.get("trained") is not False
        or report.get("live_action_authorized") is not False
        or report.get("claim", {}).get("fingerprint") != expected["claim"]["fingerprint"]
        or claim.get("schema") != "quantis.jepa_wm_model_load_preflight_claim.v1"
        or claim.get("recordings_loaded") is not False
        or claim.get("canonical_accessed") is not False
    ):
        raise ValueError("runtime remediation evidence is invalid")


def _authenticate_evaluator(
    experiment: Mapping[str, Any], implementation_revision: str
) -> dict[str, str]:
    evaluator = experiment["evaluator"]
    path = Path(__file__).resolve()
    if (
        evaluator.get("path") != "jepa_wm/physical_residual_held_out_v2.py"
        or _sha256(path) != evaluator.get("fingerprint")
    ):
        raise ValueError("physical residual held-out v2 evaluator identity changed")
    if implementation_revision != evaluator.get("implementation_revision"):
        raise ValueError("held-out v2 evaluator implementation revision changed")
    return {
        "path": str(path),
        "fingerprint": evaluator["fingerprint"],
        "implementation_revision": implementation_revision,
    }


def _claim_canonical_access(
    claim_path: Path,
    recordings: Sequence[Path],
    experiment_config_fingerprint: str,
) -> dict[str, Any]:
    payload = {
        "schema": "quantis.jepa_wm_physical_state_residual_held_out_access.v2",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config_fingerprint": experiment_config_fingerprint,
        "recordings": [path.name for path in recordings],
        "evaluations_claimed": 1,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(claim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("canonical held-out v2 evaluation was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(claim_path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return payload


def _claim_after_preclaim_authentication(
    authentication_complete: bool,
    claim_path: Path,
    recordings: Sequence[Path],
    experiment_config_fingerprint: str,
) -> dict[str, Any]:
    """Keep the irreversible claim behind one explicit fail-closed seam."""

    if authentication_complete is not True:
        raise ValueError("canonical access requires complete pre-claim authentication")
    return _claim_canonical_access(
        claim_path, recordings, experiment_config_fingerprint
    )


def evaluate(
    source: Path,
    checkpoint: Path,
    recordings: Sequence[Path],
    artifact: Path,
    adjudication: Path,
    train_evaluation: Path,
    output: Path,
    *,
    experiment_config: Path,
    training_experiment_config: Path,
    evaluator_revision: str,
) -> dict[str, Any]:
    import torch

    from jepa_wm.action import ActionSelectionBounds
    from jepa_wm.action_conditioning import (
        LoadedActionConditioning,
        PhysicalStateResidualActionEncoder,
        installed_action_conditioning,
    )
    from jepa_wm.action_routing_experiment import _authenticate_base_model
    from jepa_wm.causal_context_routing_experiment import _selected_corpus_input_fingerprint
    from jepa_wm.frames import encode_clips
    from jepa_wm.model import load_headless_model
    from jepa_wm.observed_context_routing_experiment import _gate_for_context_indices
    from jepa_wm.physical_state_residual_experiment import (
        _applied_residual_ratio_report,
        _authenticate_training_contract,
        _labeling_spec,
        _load_experiment_config as load_training_experiment_config,
        _physical_dataset,
        _score_evaluation_batches,
    )
    from jepa_wm.rollout_protocol import DROID_ROLLOUT_PROTOCOL
    from jepa_wm.rollout_scoring import rollout_action_tensor
    from jepa_wm.rollout_training import RolloutTrainingSelection
    from jepa_wm.training_artifact import load_training_report
    from jepa_wm.trajectory import RolloutWindow

    if not torch.cuda.is_available():
        raise RuntimeError("physical residual held-out v2 evaluation requires CUDA")
    if (
        FAILURE_PATH.exists()
        or ACCESS_CLAIM_PATH.exists()
        or output.resolve() != OUTPUT_PATH
        or output.exists()
    ):
        raise ValueError("physical residual held-out v2 evaluation is already terminal")

    experiment = load_experiment_config(experiment_config)
    config_fingerprint = sha256(experiment_config.resolve().read_bytes()).hexdigest()
    evaluator_identity = _authenticate_evaluator(experiment, evaluator_revision)

    # Everything below this comment must succeed before canonical access is claimed.
    runtime_identity = validate_headless_runtime(source, checkpoint)
    prior_evidence = authenticate_prior_evidence(experiment)
    _authenticate_runtime_evidence(experiment, runtime_identity)
    training_experiment = load_training_experiment_config(training_experiment_config)
    if (
        training_experiment["schema"]
        != "quantis.jepa_wm_physical_state_residual_experiment.v1"
        or experiment["source"]["training_experiment_config_fingerprint"]
        != sha256(training_experiment_config.resolve().read_bytes()).hexdigest()
    ):
        raise ValueError("source training experiment changed")
    base_identity = _authenticate_base_model(training_experiment, source, checkpoint)
    artifact_identity, training_identity, adjudication_identity = _authenticate_authority(
        experiment, artifact, adjudication, train_evaluation
    )
    loaded = LoadedActionConditioning.load(artifact, expected_identity=artifact_identity)
    training_report = load_training_report(artifact)
    _authenticate_training_contract(
        loaded.contract,
        training_report,
        training_experiment,
        source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
    )
    if DROID_ROLLOUT_PROTOCOL.action_horizon != int(experiment["corpus"]["action_horizon"]):
        raise ValueError("frozen held-out action horizon changed")

    access_claim = _claim_after_preclaim_authentication(
        True, ACCESS_CLAIM_PATH, recordings, config_fingerprint
    )
    _authenticate_recordings(recordings, experiment)
    corpus = experiment["corpus"]
    selection = RolloutTrainingSelection.load(
        recordings,
        camera=corpus["camera"],
        bounds=ActionSelectionBounds(minimum_action_norm=0.0),
        window=RolloutWindow(**corpus["window"]),
    )
    expected_examples = len(recordings) * int(corpus["window"]["count"])
    if len(selection.rollouts) != expected_examples:
        raise ValueError("canonical held-out selection is incomplete")
    selected_input_fingerprint = _selected_corpus_input_fingerprint(recordings, selection)
    dataset = _physical_dataset(recordings, selection, _labeling_spec(training_experiment))

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    load_started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    loaded.apply(
        model, expected_source_revision=os.environ.get("JEPA_WM_REVISION", "unknown")
    )
    encoder = installed_action_conditioning(model)
    if not isinstance(encoder, PhysicalStateResidualActionEncoder):
        raise ValueError("held-out artifact installed the wrong encoder")
    model.eval()
    load_seconds = monotonic() - load_started
    with torch.inference_mode():
        route_decision = encoder.route(dataset.features.to(device))
    rollouts = selection.rollouts
    encoding_started = monotonic()
    contexts = encode_clips(model, [rollout.context_paths for rollout in rollouts], batch_size=4)
    targets = encode_clips(model, [rollout.target_clip for rollout in rollouts], batch_size=4)
    encoding_seconds = monotonic() - encoding_started
    actions = rollout_action_tensor(rollouts)
    scoring_started = monotonic()
    energies = _score_evaluation_batches(
        model, contexts, targets, actions, dataset.features, device=device
    )
    torch.cuda.synchronize(device)
    scoring_seconds = monotonic() - scoring_started

    population_indices = {"combined": list(range(expected_examples))}
    offset = 0
    for recording_selection in selection.recordings:
        count = len(recording_selection.context_indices)
        population_indices[recording_selection.recording] = list(range(offset, offset + count))
        offset += count
    populations = {}
    for name in experiment["gate"]["populations"]:
        indices = population_indices[name]
        selected = torch.tensor(indices, dtype=torch.long)
        subset_energies = type(energies)(
            **{
                field: getattr(energies, field)[selected]
                for field in ("recorded", "zero", "x_zero", "x_opposed")
            }
        )
        subset_contexts = [rollouts[index].context[0].index for index in indices]
        residual = _applied_residual_ratio_report(
            encoder,
            actions[:, selected],
            dataset.features[selected],
            tuple(dataset.slices[index] for index in indices),
        )
        aggregate, retained, post, by_segment, _ = _gate_for_context_indices(
            subset_energies,
            subset_contexts,
            maximum_residual_ratio=float(residual["maximum_applied_ratio"]),
        )
        router = _router_population(
            dataset.labels[selected],
            route_decision.routes.cpu()[selected],
            route_decision.failed_closed.cpu()[selected],
            tuple(dataset.slices[index] for index in indices),
        )
        population = {
            "examples": len(indices),
            "aggregate": aggregate,
            "retained": retained,
            "post": post,
            "by_segment": by_segment,
            "router": router,
            "residual_ratios": residual,
            "maximum_residual_ratio": residual["maximum_applied_ratio"],
            "semantic_holds_exact_base": residual["semantic_holds_exact_base"],
        }
        population["gate"] = evaluate_population_gate(population, experiment["gate"])
        populations[name] = population
    passed = all(population["gate"]["passed"] for population in populations.values())
    report = {
        "schema": "quantis.jepa_wm_physical_state_residual_held_out_evaluation.v2",
        "status": "evaluated",
        "scope": "one disjoint offline canonical held-out gate after runtime remediation",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config": ArtifactIdentity.from_artifact(experiment_config).to_dict(),
        "evaluator": evaluator_identity,
        "access_claim": {
            **ArtifactIdentity.from_artifact(ACCESS_CLAIM_PATH).to_dict(),
            "payload": access_claim,
        },
        "runtime": runtime_identity,
        "prior_evidence": prior_evidence,
        "base_model": base_identity,
        "artifact": artifact_identity.to_dict(),
        "training_report": training_identity.to_dict(),
        "train_evaluation": ArtifactIdentity.from_artifact(train_evaluation).to_dict(),
        "adjudication": adjudication_identity.to_dict(),
        "recordings": [
            {**expected, "path": str(path.resolve())}
            for path, expected in zip(recordings, corpus["recordings"])
        ],
        "selection_fingerprint": selection.fingerprint,
        "selected_input_fingerprint": selected_input_fingerprint,
        "populations": populations,
        "passed": passed,
        "outcome": (
            "physical_state_residual_held_out_candidate"
            if passed
            else "physical_state_residual_held_out_failed"
        ),
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "evaluation_seconds": round(scoring_seconds, 3),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device_index) / 2**30, 3),
        "evaluations_consumed": 1,
        "trained": False,
        "isaac_run": False,
        "live_action_authorized": False,
        "filming_authorized": False,
        "hardware_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(output, report)
    return report


def _write_failure(error: Exception) -> None:
    if OUTPUT_PATH.exists():
        return
    canonical_accessed = ACCESS_CLAIM_PATH.exists()
    path = FAILURE_PATH if canonical_accessed else INVOCATION_FAILURE_PATH
    if path.exists() and canonical_accessed:
        return
    write_json_atomic(
        path,
        {
            "schema": "quantis.jepa_wm_physical_state_residual_held_out_failure.v2",
            "status": "failed",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "canonical_accessed": canonical_accessed,
            "terminal_experiment_failure": canonical_accessed,
            "retry_authorized": not canonical_accessed,
            "retraining_authorized": False,
            "live_action_authorized": False,
            "filming_authorized": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recording", dest="recordings", type=Path, action="append", required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--train-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--training-experiment-config", type=Path, required=True)
    parser.add_argument("--evaluator-revision", required=True)
    arguments = parser.parse_args(argv)
    try:
        report = evaluate(**vars(arguments))
    except Exception as error:
        _write_failure(error)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
