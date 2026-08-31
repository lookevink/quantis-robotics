"""One-shot disjoint held-out gate for the frozen physical residual artifact."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from time import monotonic
from typing import Any, Mapping, Sequence

from jepa_wm.persistence import write_json_atomic
from jepa_wm.physical_residual_held_out_contract import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
)
from jepa_wm.readiness import ResidualTrainGate
from jepa_wm.training_artifact import ArtifactIdentity, artifact_fingerprint


EXPERIMENT_SCHEMA = (
    "quantis.jepa_wm_physical_state_residual_held_out_experiment.v1"
)
OUTPUT_PATH = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/"
    "quantis_physical_state_residual_v1/canonical-held-out-v1.json"
)
FAILURE_PATH = OUTPUT_PATH.with_name("canonical-held-out-v1-failure.json")
INVOCATION_FAILURE_PATH = OUTPUT_PATH.with_name(
    "canonical-held-out-v1-invocation-failure.json"
)
ACCESS_CLAIM_PATH = OUTPUT_PATH.with_name(
    "canonical-held-out-v1-access-claim.json"
)
TRAIN_EVALUATION_PATH = OUTPUT_PATH.with_name("train-evaluation.json")


def load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT:
        raise ValueError("physical residual held-out configuration changed")
    payload = json.loads(encoded)
    corpus = payload.get("corpus", {})
    recordings = corpus.get("recordings", ())
    gate = payload.get("gate", {})
    execution = payload.get("execution", {})
    evaluator = payload.get("evaluator", {})
    recovery = payload.get("recovery", {})
    if (
        payload.get("schema") != EXPERIMENT_SCHEMA
        or corpus.get("split") != "held_out"
        or corpus.get("camera") != "wrist"
        or corpus.get("window")
        != {"start_index": 113, "count": 168, "stride": 1}
        or corpus.get("action_horizon") != 3
        or [item.get("name") for item in recordings]
        != [
            "contact-insertion-v10-drive-slow-2600-held-00",
            "contact-insertion-v10-drive-slow-2600-held-01",
        ]
        or [item.get("seed") for item in recordings] != [12600, 12601]
        or gate.get("populations")
        != [
            "combined",
            "contact-insertion-v10-drive-slow-2600-held-00",
            "contact-insertion-v10-drive-slow-2600-held-01",
        ]
        or gate.get("required_signed_segments") != ["retreat", "align", "insert"]
        or set(evaluator) != {"path", "fingerprint", "implementation_revision"}
        or recovery
        != {
            "root": "/mnt/quantis-assets/quantis-state",
            "backup_command": "AWS_PROFILE=quantis ./ops/aws.sh backup-state",
            "require_dedicated_filesystem": True,
            "compare_live_and_recovery_fingerprints": True,
            "required_terminal_artifacts": [
                str(ACCESS_CLAIM_PATH),
            ],
            "terminal_report_alternatives": [
                str(OUTPUT_PATH),
                str(FAILURE_PATH),
            ],
        }
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
    ):
        raise ValueError("physical residual held-out contract is invalid")
    return payload


def _authenticate_evaluator(
    experiment: Mapping[str, Any],
    implementation_revision: str,
) -> dict[str, str]:
    evaluator = experiment["evaluator"]
    path = Path(__file__).resolve()
    if (
        evaluator["path"] != "jepa_wm/physical_residual_held_out.py"
        or _sha256(path) != evaluator["fingerprint"]
    ):
        raise ValueError("physical residual held-out evaluator identity changed")
    if implementation_revision != evaluator["implementation_revision"]:
        raise ValueError("held-out evaluator implementation revision changed")
    return {
        "path": str(path),
        "fingerprint": evaluator["fingerprint"],
        "implementation_revision": implementation_revision,
    }


def _claim_canonical_access(
    claim_path: Path,
    recordings: Sequence[Path],
) -> dict[str, Any]:
    payload = {
        "schema": "quantis.jepa_wm_physical_state_residual_held_out_access.v1",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "recordings": [path.name for path in recordings],
        "evaluations_claimed": 1,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(
            claim_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o644,
        )
    except FileExistsError as error:
        raise ValueError("canonical held-out evaluation was already claimed") from error
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


def evaluate_population_gate(
    population: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply every frozen model, router, bound, and hold predicate."""

    decision = ResidualTrainGate(
        minimum_overall_win_rate=float(gate["minimum_overall_win_rate"]),
        minimum_retained_win_rate=float(gate["minimum_retained_win_rate"]),
        minimum_post_win_rate=float(gate["minimum_post_win_rate"]),
        minimum_signed_order_fraction=float(gate["minimum_signed_order_fraction"]),
        maximum_residual_ratio=float(gate["maximum_residual_ratio"]),
        residual_ratio_tolerance=float(gate["residual_ratio_absolute_tolerance"]),
        required_signed_segments=tuple(gate["required_signed_segments"]),
    ).evaluate(
        aggregate=population["aggregate"],
        retained=population["retained"],
        post=population["post"],
        by_segment=population["by_segment"],
        maximum_residual_ratio=float(population["maximum_residual_ratio"]),
    )
    reasons = [reason.value for reason in decision.reasons]
    router = population["router"]
    router_gate = gate["router"]
    router_metrics = (
        router.get("accuracy"),
        router.get("by_route", {}).get("retreat", {}).get("recall"),
        router.get("by_route", {}).get("advance", {}).get("recall"),
        router.get("grasp_attach_accuracy"),
        router.get("failed_closed_fraction"),
    )
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in router_metrics
    ):
        reasons.append("non_finite_router_metric")
    else:
        if float(router_metrics[0]) < float(router_gate["minimum_accuracy"]):
            reasons.append("insufficient_router_accuracy")
        if float(router_metrics[1]) < float(router_gate["minimum_retreat_recall"]):
            reasons.append("insufficient_router_retreat_recall")
        if float(router_metrics[2]) < float(router_gate["minimum_advance_recall"]):
            reasons.append("insufficient_router_advance_recall")
        if float(router_metrics[3]) < float(
            router_gate["minimum_grasp_attach_accuracy"]
        ):
            reasons.append("insufficient_router_grasp_attach_accuracy")
        if float(router_metrics[4]) > float(
            router_gate["maximum_failed_closed_fraction"]
        ):
            reasons.append("excess_router_failed_closed_fraction")
    hold_activations = router.get(
        "maximum_semantic_hold_owned_route_activations"
    )
    if (
        not isinstance(hold_activations, int)
        or hold_activations
        > int(router_gate["maximum_semantic_hold_owned_route_activations"])
    ):
        reasons.append("semantic_hold_owned_route_activation")
    if population.get("semantic_holds_exact_base") is not True:
        reasons.append("semantic_hold_base_passthrough")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "residual_ratio": decision.to_dict(),
    }


def _sha256(path: Path) -> str:
    return artifact_fingerprint(path.resolve())


def _authenticate_authority(
    experiment: Mapping[str, Any],
    artifact: Path,
    adjudication: Path,
    train_evaluation: Path,
) -> tuple[ArtifactIdentity, ArtifactIdentity, ArtifactIdentity]:
    source = experiment["source"]
    expected_artifact = ArtifactIdentity.from_dict(source["artifact"])
    actual_artifact = ArtifactIdentity.from_artifact(artifact)
    if actual_artifact != expected_artifact:
        raise ValueError("physical residual artifact identity changed")
    training_report = artifact.with_suffix(artifact.suffix + ".json")
    training_identity = ArtifactIdentity.from_artifact(training_report)
    if training_identity.fingerprint != source["training_report_fingerprint"]:
        raise ValueError("physical residual training report identity changed")
    evaluation_identity = ArtifactIdentity.from_artifact(train_evaluation)
    if (
        train_evaluation.resolve() != TRAIN_EVALUATION_PATH
        or evaluation_identity.fingerprint != source["train_evaluation_fingerprint"]
    ):
        raise ValueError("physical residual TRAIN evaluation identity changed")
    expected_adjudication = source["adjudication"]
    adjudication_identity = ArtifactIdentity.from_artifact(adjudication)
    if (
        adjudication.resolve() != Path(expected_adjudication["path"])
        or adjudication_identity.fingerprint != expected_adjudication["fingerprint"]
    ):
        raise ValueError("physical residual adjudication identity changed")
    payload = json.loads(adjudication.read_text())
    if (
        payload.get("status") != "adjudicated"
        or payload.get("passed") is not True
        or payload.get("outcome") != expected_adjudication["required_outcome"]
        or payload.get("eligible_for_separately_frozen_held_out_gate_proposal")
        is not True
        or payload.get("source_evaluation") != evaluation_identity.to_dict()
        or payload.get("model_loaded") is not False
        or payload.get("recordings_loaded") is not False
        or payload.get("rescored") is not False
        or payload.get("trained") is not False
        or payload.get("live_action_authorized") is not False
    ):
        raise ValueError("physical residual adjudication authority is invalid")
    return actual_artifact, training_identity, adjudication_identity


def _authenticate_recordings(
    recordings: Sequence[Path], experiment: Mapping[str, Any]
) -> None:
    from jepa_wm.insertion_recording import ContactInsertionEvidence

    expected = experiment["corpus"]["recordings"]
    if [path.name for path in recordings] != [item["name"] for item in expected]:
        raise ValueError("held-out inputs do not match the exact canonical roster")
    for path, item in zip(recordings, expected):
        if _sha256(path / "manifest.json") != item["manifest_fingerprint"]:
            raise ValueError(f"canonical manifest identity changed: {path.name}")
        ContactInsertionEvidence.from_recording(
            path,
            expected_split="held_out",
            expected_seed=int(item["seed"]),
        )


def _router_population(
    labels: Any,
    routes: Any,
    failed_closed: Any,
    slices: Sequence[str],
) -> dict[str, Any]:
    import torch

    from jepa_wm.route_metrics import route_metrics

    metrics = route_metrics(labels, routes)
    by_slice = {}
    for name in sorted(set(slices)):
        selected = torch.tensor([value == name for value in slices])
        by_slice[name] = route_metrics(labels[selected], routes[selected])
    holds = ("retreat_hold", "align_hold", "seated_hold")
    hold_activations = {
        name: by_slice[name]["predictions"]["retreat"]
        + by_slice[name]["predictions"]["advance"]
        for name in holds
    }
    return {
        **metrics,
        "failed_closed_fraction": float(failed_closed.float().mean()),
        "grasp_attach_accuracy": by_slice["grasp_attach"]["accuracy"],
        "by_slice": by_slice,
        "semantic_hold_owned_route_activations": hold_activations,
        "maximum_semantic_hold_owned_route_activations": max(
            hold_activations.values()
        ),
    }


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
    from jepa_wm.causal_context_routing_experiment import (
        _selected_corpus_input_fingerprint,
    )
    from jepa_wm.frames import encode_clips
    from jepa_wm.model import load_headless_model
    from jepa_wm.observed_context_routing_experiment import (
        _gate_for_context_indices,
    )
    from jepa_wm.physical_state_residual_experiment import (
        _applied_residual_ratio_report,
        _authenticate_training_contract,
        _labeling_spec,
        _load_experiment_config as load_training_experiment_config,
        _physical_dataset,
        _score_evaluation_batches,
    )
    from jepa_wm.rollout_scoring import rollout_action_tensor
    from jepa_wm.rollout_protocol import DROID_ROLLOUT_PROTOCOL
    from jepa_wm.rollout_training import RolloutTrainingSelection
    from jepa_wm.training_artifact import load_training_report
    from jepa_wm.trajectory import RolloutWindow

    if not torch.cuda.is_available():
        raise RuntimeError("physical residual held-out evaluation requires CUDA")
    if (
        FAILURE_PATH.exists()
        or ACCESS_CLAIM_PATH.exists()
        or output.resolve() != OUTPUT_PATH
        or output.exists()
    ):
        raise ValueError("physical residual held-out evaluation is already terminal")
    experiment = load_experiment_config(experiment_config)
    training_experiment = load_training_experiment_config(
        training_experiment_config
    )
    if (
        training_experiment["schema"]
        != "quantis.jepa_wm_physical_state_residual_experiment.v1"
        or experiment["source"]["training_experiment_config_fingerprint"]
        != sha256(training_experiment_config.resolve().read_bytes()).hexdigest()
    ):
        raise ValueError("source training experiment changed")
    base_identity = _authenticate_base_model(
        training_experiment, source, checkpoint
    )
    evaluator_identity = _authenticate_evaluator(
        experiment, evaluator_revision
    )
    artifact_identity, training_identity, adjudication_identity = (
        _authenticate_authority(
            experiment, artifact, adjudication, train_evaluation
        )
    )
    if DROID_ROLLOUT_PROTOCOL.action_horizon != int(
        experiment["corpus"]["action_horizon"]
    ):
        raise ValueError("frozen held-out action horizon changed")
    loaded = LoadedActionConditioning.load(
        artifact, expected_identity=artifact_identity
    )
    training_report = load_training_report(artifact)
    _authenticate_training_contract(
        loaded.contract,
        training_report,
        training_experiment,
        source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
    )
    access_claim = _claim_canonical_access(ACCESS_CLAIM_PATH, recordings)
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
    selected_input_fingerprint = _selected_corpus_input_fingerprint(
        recordings, selection
    )
    dataset = _physical_dataset(
        recordings, selection, _labeling_spec(training_experiment)
    )
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    load_started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    loaded.apply(
        model,
        expected_source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
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
    contexts = encode_clips(
        model, [rollout.context_paths for rollout in rollouts], batch_size=4
    )
    targets = encode_clips(
        model, [rollout.target_clip for rollout in rollouts], batch_size=4
    )
    encoding_seconds = monotonic() - encoding_started
    actions = rollout_action_tensor(rollouts)
    scoring_started = monotonic()
    energies = _score_evaluation_batches(
        model,
        contexts,
        targets,
        actions,
        dataset.features,
        device=device,
    )
    torch.cuda.synchronize(device)
    scoring_seconds = monotonic() - scoring_started
    population_indices = {"combined": list(range(expected_examples))}
    offset = 0
    for recording_selection in selection.recordings:
        count = len(recording_selection.context_indices)
        population_indices[recording_selection.recording] = list(
            range(offset, offset + count)
        )
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
            "semantic_holds_exact_base": residual[
                "semantic_holds_exact_base"
            ],
        }
        population["gate"] = evaluate_population_gate(
            population, experiment["gate"]
        )
        populations[name] = population
    passed = all(population["gate"]["passed"] for population in populations.values())
    report = {
        "schema": "quantis.jepa_wm_physical_state_residual_held_out_evaluation.v1",
        "status": "evaluated",
        "scope": "one disjoint offline canonical held-out gate",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_config": ArtifactIdentity.from_artifact(
            experiment_config
        ).to_dict(),
        "evaluator": evaluator_identity,
        "access_claim": {
            **ArtifactIdentity.from_artifact(ACCESS_CLAIM_PATH).to_dict(),
            "payload": access_claim,
        },
        "base_model": base_identity,
        "artifact": artifact_identity.to_dict(),
        "training_report": training_identity.to_dict(),
        "train_evaluation": ArtifactIdentity.from_artifact(
            train_evaluation
        ).to_dict(),
        "adjudication": adjudication_identity.to_dict(),
        "recordings": [
            {
                **expected,
                "path": str(path.resolve()),
            }
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
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30, 3
        ),
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
            "schema": "quantis.jepa_wm_physical_state_residual_held_out_failure.v1",
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
    parser.add_argument(
        "--recording",
        dest="recordings",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--train-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--training-experiment-config", type=Path, required=True)
    parser.add_argument("--evaluator-revision", required=True)
    arguments = parser.parse_args(argv)
    values = vars(arguments)
    try:
        report = evaluate(**values)
    except Exception as error:
        _write_failure(error)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
