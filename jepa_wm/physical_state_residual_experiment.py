"""Frozen TRAIN-only physical-router and bounded-residual experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch

from jepa_wm.action_conditioning import (
    ACTION_CONDITIONING_SCHEMA,
    ActionConditioningContract,
    ActionConditioningKind,
    ActionConditioningSpec,
    LoadedActionConditioning,
    PhysicalStateResidualActionEncoder,
    install_action_conditioning,
    installed_action_conditioning,
    physical_residual_parameters,
    save_action_conditioning,
)
from jepa_wm.action_conditioning_experiment import (
    EXPERIMENT_WINDOW,
    TRAINING_RECORDINGS,
    EvaluationEnergies,
)
from jepa_wm.action_conditioning_training import (
    AlternatingCommandRouteSampler,
    signed_x_margin_loss,
    signed_x_negatives,
)
from jepa_wm.action_routing_experiment import (
    CONTROL_ARTIFACT_FINGERPRINT,
    _authenticate_base_model,
    _validated_selection,
)
from jepa_wm.adapt_recording import (
    ContrastiveTermConfig,
    _sample_mismatched_indices,
    mismatched_negative_candidates,
)
from jepa_wm.candidate_negatives import (
    CandidateMiningConfig,
    sample_local_candidates,
)
from jepa_wm.causal_context_routing_experiment import (
    _selected_corpus_input_fingerprint,
)
from jepa_wm.causal_routing import RecordedMotionLabelSpec
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import encode_clips
from jepa_wm.model import load_headless_model
from jepa_wm.observed_context_routing_experiment import (
    _gate_for_context_indices,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.physical_observation import (
    PHYSICAL_ROUTING_FEATURE_NAMES,
    PHYSICAL_ROUTING_OBSERVATION_SCHEMA,
)
from jepa_wm.physical_routing import PhysicalStateRoutingSpec
from jepa_wm.physical_routing_training import (
    PhysicalRouterTrainingConfig,
    fit_final_physical_router,
)
from jepa_wm.physical_state_routing_experiment import (
    _authenticate_training_recordings,
    _physical_dataset,
)
from jepa_wm.rollout_scoring import rollout_action_tensor, score_actions
from jepa_wm.route_metrics import route_metrics
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    artifact_fingerprint,
    load_training_report,
    training_configuration_fingerprint,
    training_report_path,
)


EXPERIMENT_SCHEMA = "quantis.jepa_wm_physical_state_residual_experiment.v1"
FROZEN_EXPERIMENT_CONFIG_FINGERPRINT = (
    "b296b7fc064627f13ed87c1baeaf84d4961f1b04db115f9afcc689bf05dda78d"
)
OUTPUT_ROOT = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/quantis_physical_state_residual_v1"
)
ARTIFACT_PATH = OUTPUT_ROOT / "physical_state_residual.pth"
PREFLIGHT_PATH = OUTPUT_ROOT / "preflight.json"
EVALUATION_PATH = OUTPUT_ROOT / "train-evaluation.json"
FAILURE_PATH = OUTPUT_ROOT / "failure.json"
INVOCATION_FAILURE_PATH = OUTPUT_ROOT / "invocation-failure.json"
RUN_STATE_PATH = OUTPUT_ROOT / "run-state.json"
PHYSICAL_TREATMENT = ActionConditioningKind.PHYSICAL_STATE_RESIDUAL.value


def _load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT:
        raise ValueError("physical residual experiment configuration changed")
    payload = json.loads(encoded)
    corpus = payload.get("corpus", {})
    labeling = payload.get("labeling", {})
    router = payload.get("router", {})
    probe = payload.get("passed_router_probe", {})
    evaluation = payload.get("evaluation", {})
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != EXPERIMENT_SCHEMA
        or tuple(corpus.get("training_recordings", ())) != TRAINING_RECORDINGS
        or corpus.get("window") != EXPERIMENT_WINDOW.to_dict()
        or router.get("observation_schema") != PHYSICAL_ROUTING_OBSERVATION_SCHEMA
        or router.get("feature_names") != list(PHYSICAL_ROUTING_FEATURE_NAMES)
        or router.get("hidden_dimensions") != [64, 64]
        or router.get("minimum_route_confidence") != 0.75
        or router.get("maximum_residual_ratio") != 0.15
        or router.get("runtime_inputs") != ["physical_observation"]
        or router.get("candidate_invariant") is not True
        or "candidate_action" not in router.get("forbidden_runtime_inputs", ())
        or "recorded_future_action" not in router.get("forbidden_runtime_inputs", ())
        or sorted(router.get("semantic_hold_segments", ()))
        != sorted(labeling.get("semantic_hold_segments", ()))
        or probe.get("report_fingerprint")
        != "385305e7268e296702f1ecfd5e0104426894a119e15e73567acb03e129801ffa"
        or evaluation.get("residual_ratio_candidates")
        != ["recorded", "zero", "x_zero", "x_opposed"]
        or evaluation.get("require_exact_base_in_semantic_holds") is not True
        or payload.get("stopping_rules", {}).get("one_training_artifact") is not True
        or payload.get("stopping_rules", {}).get("one_train_evaluation") is not True
    ):
        raise ValueError("physical residual experiment contract is invalid")
    return payload


def _routing_spec(experiment: dict[str, Any]) -> PhysicalStateRoutingSpec:
    payload = dict(experiment["router"])
    for name in (
        "runtime_inputs",
        "forbidden_runtime_inputs",
        "candidate_invariant",
        "semantic_hold_segments",
        "fit",
    ):
        payload.pop(name)
    return PhysicalStateRoutingSpec.from_dict(payload)


def _labeling_spec(experiment: dict[str, Any]) -> RecordedMotionLabelSpec:
    payload = dict(experiment["labeling"])
    payload.pop("source")
    payload.pop("semantic_hold_segments")
    return RecordedMotionLabelSpec.from_dict(payload)


PHYSICAL_TREATMENT_SPEC = ActionConditioningSpec(
    ActionConditioningKind.PHYSICAL_STATE_RESIDUAL,
    physical_state_routing=PhysicalStateRoutingSpec(
        hidden_dimensions=(64, 64),
        minimum_route_confidence=0.75,
        maximum_residual_ratio=0.15,
    ),
)


def _validate_output(output: Path, expected: Path) -> None:
    if FAILURE_PATH.exists():
        raise ValueError("physical residual experiment already ended with a failure")
    if output.resolve() != expected or output.exists():
        raise ValueError(f"physical residual output must be new at {expected}")


def _write_run_state(phase: str) -> None:
    if phase not in {
        "training_started",
        "training_completed",
        "evaluation_started",
        "evaluation_completed",
    }:
        raise ValueError("physical residual run phase is invalid")
    write_json_atomic(
        RUN_STATE_PATH,
        {
            "schema": "quantis.jepa_wm_physical_state_residual_run_state.v1",
            "phase": phase,
            "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _command_entered_experiment(command: str) -> bool:
    if not RUN_STATE_PATH.is_file():
        return False
    try:
        state = json.loads(RUN_STATE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return _is_terminal_command_phase(command, state.get("phase"))


def _is_terminal_command_phase(command: str, phase: Any) -> bool:
    return (command, phase) in {
        ("train", "training_started"),
        ("evaluate-train", "evaluation_started"),
    }


def _authenticate_passed_probe(
    path: Path,
    experiment: dict[str, Any],
) -> tuple[ArtifactIdentity, dict[str, Any]]:
    identity = ArtifactIdentity.from_artifact(path)
    expected = experiment["passed_router_probe"]
    if identity.fingerprint != expected["report_fingerprint"]:
        raise ValueError("passed physical router probe identity changed")
    report = json.loads(path.read_text())
    if (
        report.get("status") != "passed"
        or report.get("gate", {}).get("passed") is not True
        or report.get("experiment_config_fingerprint")
        != expected["experiment_config_fingerprint"]
        or report.get("training_selection_fingerprint")
        != expected["training_selection_fingerprint"]
        or report.get("selected_input_fingerprint")
        != expected["selected_input_fingerprint"]
        or report.get("candidate_actions_used_as_router_inputs") is not False
        or report.get("visual_latents_used_as_router_inputs") is not False
        or report.get("held_out_accessed") is not False
        or report.get("canonical_accessed") is not False
    ):
        raise ValueError("physical router probe did not pass its frozen contract")
    return identity, report


def _router_fit_report(
    router: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    slices: Sequence[str],
    training_report: dict[str, object],
    experiment: dict[str, Any],
) -> dict[str, Any]:
    with torch.inference_mode():
        decision = router.decide(features.to(next(router.parameters()).device))
    predictions = decision.routes.cpu()
    by_slice = {
        name: route_metrics(labels[indices], predictions[indices])
        for name in sorted(set(slices))
        if (indices := torch.tensor([value == name for value in slices])).any()
    }
    holds = experiment["router"]["semantic_hold_segments"]
    hold_activations = {
        name: by_slice[name]["predictions"]["retreat"]
        + by_slice[name]["predictions"]["advance"]
        for name in holds
    }
    gate = experiment["router"]["fit"]
    passed = (
        training_report["accuracy"] >= gate["minimum_accuracy"]
        and training_report["by_route"]["retreat"]["recall"]
        >= gate["minimum_retreat_recall"]
        and training_report["by_route"]["advance"]["recall"]
        >= gate["minimum_advance_recall"]
        and by_slice["grasp_attach"]["accuracy"]
        >= gate["minimum_grasp_attach_accuracy"]
        and training_report["failed_closed_fraction"]
        <= gate["maximum_failed_closed_fraction"]
        and max(hold_activations.values())
        <= gate["maximum_semantic_hold_owned_route_activations"]
    )
    return {
        **training_report,
        "by_slice": by_slice,
        "semantic_hold_owned_route_activations": hold_activations,
        "gate_passed": passed,
    }


def preflight(
    recordings: Sequence[Path],
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    route_probe: Path,
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    _validate_output(output, PREFLIGHT_PATH)
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    probe_identity, _ = _authenticate_passed_probe(route_probe, experiment)
    _authenticate_training_recordings(recordings)
    selection = _validated_selection(recordings)
    selected_input_fingerprint = _selected_corpus_input_fingerprint(
        recordings,
        selection,
    )
    expected = experiment["passed_router_probe"]
    if (
        selection.fingerprint != expected["training_selection_fingerprint"]
        or selected_input_fingerprint != expected["selected_input_fingerprint"]
    ):
        raise ValueError("physical residual TRAIN corpus changed after route probe")
    dataset = _physical_dataset(
        recordings,
        selection,
        _labeling_spec(experiment),
    )
    report = {
        "schema": "quantis.jepa_wm_physical_state_residual_preflight.v1",
        "status": "passed",
        "scope": "TRAIN-only physical residual training; no held-out or live action",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "control_artifact": control_identity.to_dict(),
        "passed_router_probe": probe_identity.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "selected_input_fingerprint": selected_input_fingerprint,
        "examples": int(dataset.labels.numel()),
        "semantic_routes": route_metrics(dataset.labels, dataset.labels)["labels"],
        "router_runtime_inputs": ["physical_observation"],
        "candidate_router_inputs": [],
        "held_out_accessed": False,
        "canonical_accessed": False,
        "live_action_authorized": False,
    }
    write_json_atomic(output, report)
    return report


def _score_physical(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    encoder = installed_action_conditioning(model)
    if not isinstance(encoder, PhysicalStateResidualActionEncoder):
        raise ValueError("model has no physical-state action conditioning")
    with encoder.use_physical_observations(features):
        return score_actions(model, context, target, actions)


def _mine_physical_candidates(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    features: torch.Tensor,
    *,
    scoring_batch_size: int,
) -> torch.Tensor:
    if candidates.ndim != 4 or candidates.shape[-1] != 7:
        raise ValueError(
            "candidate actions must have shape [horizon, batch, candidates, 7]"
        )
    horizon, batch, candidate_count, _ = candidates.shape
    if (
        context.shape[0] != batch
        or target.shape[0] != batch
        or features.shape[0] != batch
    ):
        raise ValueError("physical candidate batch is inconsistent")
    flattened = candidates.reshape(horizon, batch * candidate_count, 7)
    repeated_context = context.repeat_interleave(candidate_count, dim=0)
    repeated_target = target.repeat_interleave(candidate_count, dim=0)
    repeated_features = features.repeat_interleave(candidate_count, dim=0)
    with torch.no_grad():
        chunks = []
        for start in range(0, batch * candidate_count, scoring_batch_size):
            stop = start + scoring_batch_size
            chunks.append(
                _score_physical(
                    model,
                    repeated_context[start:stop],
                    repeated_target[start:stop],
                    flattened[:, start:stop],
                    repeated_features[start:stop],
                )
            )
        energies = torch.cat(chunks).reshape(batch, candidate_count)
    selected_indices = energies.argmin(dim=1)
    by_rollout = candidates.permute(1, 2, 0, 3)
    selected = by_rollout[
        torch.arange(batch, device=candidates.device),
        selected_indices,
    ]
    return selected.permute(1, 0, 2).contiguous()


def _training_config(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "treatment": PHYSICAL_TREATMENT,
        "spec": PHYSICAL_TREATMENT_SPEC.to_dict(),
        "control_artifact_fingerprint": CONTROL_ARTIFACT_FINGERPRINT,
        "passed_router_probe_fingerprint": experiment["passed_router_probe"][
            "report_fingerprint"
        ],
        "router_fit": experiment["router"]["fit"],
        "training": experiment["training"],
    }


def _expected_training_metadata(
    experiment: dict[str, Any],
    source_revision: str,
) -> TrainingArtifactMetadata:
    return TrainingArtifactMetadata(
        MODEL_ID,
        source_revision,
        "wrist",
        TRAINING_RECORDINGS,
        int(experiment["training"]["steps"]),
    )


def _authenticate_training_contract(
    contract: ActionConditioningContract,
    report: dict[str, Any],
    experiment: dict[str, Any],
    *,
    source_revision: str,
) -> None:
    config = _training_config(experiment)
    config_fingerprint = training_configuration_fingerprint(config)
    metadata = _expected_training_metadata(experiment, source_revision)
    try:
        control_identity = ArtifactIdentity.from_dict(report["control_artifact"])
        probe_identity = ArtifactIdentity.from_dict(report["passed_router_probe"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("physical residual training provenance is invalid") from error
    if (
        contract.training_config_fingerprint != config_fingerprint
        or contract.experiment_config_fingerprint
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
        or contract.metadata != metadata
        or contract.spec != PHYSICAL_TREATMENT_SPEC
        or report.get("contract") != contract.to_dict()
        or report.get("metadata") != metadata.to_dict()
        or report.get("config") != config
        or report.get("training_config_fingerprint") != config_fingerprint
        or report.get("experiment_config_fingerprint")
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
        or control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT
        or probe_identity.fingerprint
        != experiment["passed_router_probe"]["report_fingerprint"]
    ):
        raise ValueError("physical residual training contract changed")


def train_residual(
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    recordings: Sequence[Path],
    route_probe: Path,
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("physical residual training requires CUDA")
    _validate_output(output, ARTIFACT_PATH)
    if training_report_path(output).exists():
        raise ValueError("physical residual training report already exists")
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    probe_identity, _ = _authenticate_passed_probe(route_probe, experiment)
    if not PREFLIGHT_PATH.is_file():
        raise ValueError("physical residual training requires passed preflight")
    preflight_report = json.loads(PREFLIGHT_PATH.read_text())
    if (
        preflight_report.get("status") != "passed"
        or preflight_report.get("experiment_config_fingerprint")
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
        or preflight_report.get("passed_router_probe", {}).get("fingerprint")
        != probe_identity.fingerprint
    ):
        raise ValueError("physical residual preflight did not pass")
    _authenticate_training_recordings(recordings)
    selection = _validated_selection(recordings)
    selected_input_fingerprint = _selected_corpus_input_fingerprint(
        recordings,
        selection,
    )
    if (
        preflight_report.get("training_selection_fingerprint") != selection.fingerprint
        or preflight_report.get("selected_input_fingerprint")
        != selected_input_fingerprint
    ):
        raise ValueError("physical residual TRAIN corpus changed after preflight")
    dataset = _physical_dataset(
        recordings,
        selection,
        _labeling_spec(experiment),
    )
    _write_run_state("training_started")
    training = experiment["training"]
    router_fit = experiment["router"]["fit"]
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    router_started = monotonic()
    fitted_router, raw_router_report = fit_final_physical_router(
        dataset.features,
        dataset.labels,
        _routing_spec(experiment),
        PhysicalRouterTrainingConfig(
            steps=int(router_fit["steps"]),
            learning_rate=float(router_fit["learning_rate"]),
            weight_decay=float(router_fit["weight_decay"]),
            seed=int(router_fit["seed"]),
        ),
        device=device,
    )
    final_router_report = _router_fit_report(
        fitted_router,
        dataset.features,
        dataset.labels,
        dataset.slices,
        raw_router_report,
        experiment,
    )
    router_seconds = monotonic() - router_started
    if not final_router_report["gate_passed"]:
        raise ValueError("final physical router did not pass its frozen TRAIN gate")

    load_started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=control_adapter,
    )
    encoder = install_action_conditioning(model, PHYSICAL_TREATMENT_SPEC)
    if not isinstance(encoder, PhysicalStateResidualActionEncoder):
        raise ValueError("physical residual encoder installation failed")
    encoder.router.load_state_dict(fitted_router.state_dict(), strict=True)
    encoder.router.eval()
    load_seconds = monotonic() - load_started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = physical_residual_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)
    frozen_base = tuple(
        value.detach().clone() for value in encoder.base.state_dict().values()
    )
    frozen_router = tuple(
        value.detach().clone() for value in encoder.router.state_dict().values()
    )

    rollouts = selection.rollouts
    encoding_started = monotonic()
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=int(training["encoding_batch_size"]),
    )
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=int(training["encoding_batch_size"]),
    )
    encoding_seconds = monotonic() - encoding_started
    actions = rollout_action_tensor(rollouts)
    features = dataset.features
    goal_actions = torch.tensor(
        [rollout.goal_action.values for rollout in rollouts],
        dtype=actions.dtype,
    )
    mismatched_candidates = mismatched_negative_candidates(rollouts)
    candidate_config = CandidateMiningConfig.from_dict(training["candidate_mining"])
    term_names = (
        "zero_negative",
        "mismatched_negative",
        "candidate_negative",
        "signed_x_zero_negative",
        "signed_x_opposed_negative",
        "signed_x_zero_before_opposed",
    )
    terms = {
        name: ContrastiveTermConfig(**training["objective"][name])
        for name in term_names
    }
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sampler = AlternatingCommandRouteSampler(dataset.labels, seed=seed)
    mismatch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    candidate_generator = torch.Generator(device=device).manual_seed(seed)
    minimum_activity = float(training["objective"]["signed_x_activity_threshold"])
    losses = []
    model.eval()
    training_started = monotonic()
    for _ in range(int(training["steps"])):
        indices = torch.tensor((sampler.next_index(),), dtype=torch.long)
        context = contexts[indices].to(device)
        target = targets[indices].to(device)
        action_batch = actions[:, indices].to(device)
        feature_batch = features[indices].to(device)
        negative_indices = _sample_mismatched_indices(
            mismatched_candidates,
            indices,
            mismatch_generator,
        )
        mismatched = actions[:, negative_indices].to(device)
        local_candidates = sample_local_candidates(
            action_batch,
            config=candidate_config,
            generator=candidate_generator,
            goal_actions=goal_actions[indices].to(device),
        )
        mined = _mine_physical_candidates(
            model,
            context,
            target,
            local_candidates,
            feature_batch,
            scoring_batch_size=candidate_config.scoring_batch_size,
        )
        recorded_energy = _score_physical(
            model, context, target, action_batch, feature_batch
        )
        zero_energy = _score_physical(
            model, context, target, torch.zeros_like(action_batch), feature_batch
        )
        mismatched_energy = _score_physical(
            model, context, target, mismatched, feature_batch
        )
        candidate_energy = _score_physical(model, context, target, mined, feature_batch)
        x_zero, x_opposed = signed_x_negatives(action_batch)
        x_zero_energy = _score_physical(model, context, target, x_zero, feature_batch)
        x_opposed_energy = _score_physical(
            model, context, target, x_opposed, feature_batch
        )
        loss = (
            recorded_energy.mean()
            + terms["zero_negative"].loss(recorded_energy, zero_energy)
            + terms["mismatched_negative"].loss(recorded_energy, mismatched_energy)
            + terms["candidate_negative"].loss(recorded_energy, candidate_energy)
            + signed_x_margin_loss(
                recorded_energy,
                x_zero_energy,
                action_batch,
                weight=terms["signed_x_zero_negative"].weight,
                margin=terms["signed_x_zero_negative"].margin,
                minimum_activity=minimum_activity,
            )
            + signed_x_margin_loss(
                recorded_energy,
                x_opposed_energy,
                action_batch,
                weight=terms["signed_x_opposed_negative"].weight,
                margin=terms["signed_x_opposed_negative"].margin,
                minimum_activity=minimum_activity,
            )
            + signed_x_margin_loss(
                x_zero_energy,
                x_opposed_energy,
                action_batch,
                weight=terms["signed_x_zero_before_opposed"].weight,
                margin=terms["signed_x_zero_before_opposed"].margin,
                minimum_activity=minimum_activity,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    training_seconds = monotonic() - training_started
    base_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(frozen_base, encoder.base.state_dict().values())
    )
    router_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(frozen_router, encoder.router.state_dict().values())
    )
    if not base_unchanged or not router_unchanged:
        raise ValueError(
            "frozen base or physical router changed during residual training"
        )

    metadata = _expected_training_metadata(
        experiment,
        os.environ.get("JEPA_WM_REVISION", "unknown"),
    )
    config_payload = _training_config(experiment)
    config_fingerprint = training_configuration_fingerprint(config_payload)
    contract = ActionConditioningContract(
        ACTION_CONDITIONING_SCHEMA,
        metadata,
        selection.fingerprint,
        config_fingerprint,
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        PHYSICAL_TREATMENT_SPEC,
    )
    save_action_conditioning(model, output, contract)
    report = {
        "schema": "quantis.jepa_wm_physical_state_residual_training.v1",
        "status": "trained",
        "treatment": PHYSICAL_TREATMENT,
        "scope": "TRAIN-only physical residual; no held-out or live action",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "artifact": str(output.resolve()),
        "artifact_fingerprint": artifact_fingerprint(output),
        "metadata": metadata.to_dict(),
        "contract": contract.to_dict(),
        "config": config_payload,
        "training_config_fingerprint": config_fingerprint,
        **selection.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "selected_input_fingerprint": selected_input_fingerprint,
        "passed_router_probe": probe_identity.to_dict(),
        "final_router": final_router_report,
        "sampling": sampler.to_dict(),
        "control_artifact": control_identity.to_dict(),
        "base_model": base_identity,
        "base_map_unchanged": base_unchanged,
        "router_unchanged_during_residual_training": router_unchanged,
        "candidate_invariant": True,
        "ordered_pairwise_objective": True,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": float(np.min(losses)),
        "router_fit_seconds": round(router_seconds, 3),
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "held_out_accessed": False,
        "canonical_accessed": False,
        "live_action_authorized": False,
    }
    report["report"] = str(training_report_path(output).resolve())
    write_json_atomic(training_report_path(output), report)
    _write_run_state("training_completed")
    return report


def _score_evaluation_batches(
    model: Any,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 2,
) -> EvaluationEnergies:
    chunks: dict[str, list[torch.Tensor]] = {
        "recorded": [],
        "zero": [],
        "x_zero": [],
        "x_opposed": [],
    }
    x_zero, x_opposed = signed_x_negatives(actions)
    candidates = {
        "recorded": actions,
        "zero": torch.zeros_like(actions),
        "x_zero": x_zero,
        "x_opposed": x_opposed,
    }
    with torch.inference_mode():
        for start in range(0, contexts.shape[0], batch_size):
            stop = start + batch_size
            for name, candidate in candidates.items():
                chunks[name].append(
                    _score_physical(
                        model,
                        contexts[start:stop].to(device),
                        targets[start:stop].to(device),
                        candidate[:, start:stop].to(device),
                        features[start:stop].to(device),
                    ).cpu()
                )
    return EvaluationEnergies(
        **{name: torch.cat(values) for name, values in chunks.items()}
    )


def _applied_residual_ratio_report(
    encoder: PhysicalStateResidualActionEncoder,
    actions: torch.Tensor,
    features: torch.Tensor,
    slices: Sequence[str],
) -> dict[str, Any]:
    x_zero, x_opposed = signed_x_negatives(actions)
    candidates = {
        "recorded": actions,
        "zero": torch.zeros_like(actions),
        "x_zero": x_zero,
        "x_opposed": x_opposed,
    }
    device = encoder.base.weight.device
    feature_batch = features.to(device)
    semantic_holds = torch.tensor(
        [name in {"retreat_hold", "align_hold", "seated_hold"} for name in slices],
        dtype=torch.bool,
        device=device,
    )
    candidate_report = {}
    exact_holds = True
    maxima = []
    for name, candidate in candidates.items():
        batch_actions = candidate.transpose(0, 1).to(device)
        with torch.inference_mode(), encoder.use_physical_observations(feature_batch):
            base = encoder.base(batch_actions)
            output = encoder(batch_actions)
        applied = output - base
        ratios = torch.linalg.vector_norm(applied, dim=-1) / torch.clamp(
            torch.linalg.vector_norm(base, dim=-1),
            min=1e-12,
        )
        maximum = float(ratios.max())
        maxima.append(maximum)
        candidate_exact = bool(
            torch.equal(output[semantic_holds], base[semantic_holds])
        )
        exact_holds = exact_holds and candidate_exact
        candidate_report[name] = {
            "mean_applied_residual_to_base_embedding_ratio": float(ratios.mean()),
            "maximum_applied_residual_to_base_embedding_ratio": maximum,
            "semantic_holds_exact_base": candidate_exact,
        }
    return {
        "candidates": candidate_report,
        "maximum_applied_ratio": max(maxima),
        "semantic_holds_exact_base": exact_holds,
    }


def evaluate_train(
    source: Path,
    checkpoint: Path,
    recordings: Sequence[Path],
    artifact: Path,
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("physical residual evaluation requires CUDA")
    _validate_output(output, EVALUATION_PATH)
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    if artifact.resolve() != ARTIFACT_PATH or not artifact.is_file():
        raise ValueError("physical residual evaluation artifact path changed")
    identity = ArtifactIdentity.from_artifact(artifact)
    loaded = LoadedActionConditioning.load(artifact, expected_identity=identity)
    training_report = load_training_report(artifact)
    _authenticate_training_contract(
        loaded.contract,
        training_report,
        experiment,
        source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
    )
    if (
        training_report.get("artifact_fingerprint") != identity.fingerprint
        or training_report.get("treatment") != PHYSICAL_TREATMENT
        or training_report.get("base_model") != base_identity
        or training_report.get("base_map_unchanged") is not True
        or training_report.get("router_unchanged_during_residual_training") is not True
        or training_report.get("candidate_invariant") is not True
        or training_report.get("ordered_pairwise_objective") is not True
        or training_report.get("final_router", {}).get("gate_passed") is not True
    ):
        raise ValueError("physical residual training report is invalid")
    _authenticate_training_recordings(recordings)
    selection = _validated_selection(recordings)
    if loaded.contract.training_selection_fingerprint != selection.fingerprint:
        raise ValueError("physical residual artifact TRAIN selection changed")
    selected_input_fingerprint = _selected_corpus_input_fingerprint(
        recordings,
        selection,
    )
    if training_report.get("selected_input_fingerprint") != selected_input_fingerprint:
        raise ValueError("physical residual artifact TRAIN contents changed")
    dataset = _physical_dataset(
        recordings,
        selection,
        _labeling_spec(experiment),
    )
    _write_run_state("evaluation_started")
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
        raise ValueError("physical residual artifact installed the wrong encoder")
    model.eval()
    load_seconds = monotonic() - load_started
    with torch.inference_mode():
        decision = encoder.route(dataset.features.to(device))
    final_router_report = _router_fit_report(
        encoder.router,
        dataset.features,
        dataset.labels,
        dataset.slices,
        {
            **route_metrics(dataset.labels, decision.routes.cpu()),
            "failed_closed_fraction": float(decision.failed_closed.float().mean()),
            "mean_confidence": float(decision.confidence.mean()),
            "normalization_fitted": bool(encoder.router.normalization_fitted.item()),
        },
        experiment,
    )
    if not final_router_report["gate_passed"]:
        raise ValueError("serialized physical router no longer passes its gate")
    rollouts = selection.rollouts
    encoding_started = monotonic()
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=4,
    )
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=4,
    )
    encoding_seconds = monotonic() - encoding_started
    actions = rollout_action_tensor(rollouts)
    evaluation_started = monotonic()
    energies = _score_evaluation_batches(
        model,
        contexts,
        targets,
        actions,
        dataset.features,
        device=device,
    )
    torch.cuda.synchronize(device)
    evaluation_seconds = monotonic() - evaluation_started
    residual_ratios = _applied_residual_ratio_report(
        encoder,
        actions,
        dataset.features,
        dataset.slices,
    )
    aggregate, retained, post, by_segment, base_gate_passed = _gate_for_context_indices(
        energies,
        tuple(rollout.context[0].index for rollout in rollouts),
        maximum_residual_ratio=float(residual_ratios["maximum_applied_ratio"]),
    )
    evaluation = experiment["evaluation"]
    passed = (
        base_gate_passed
        and final_router_report["gate_passed"]
        and residual_ratios["semantic_holds_exact_base"]
        and residual_ratios["maximum_applied_ratio"]
        <= evaluation["maximum_applied_residual_to_base_embedding_ratio"] + 1e-6
    )
    report = {
        "schema": "quantis.jepa_wm_physical_state_residual_train_evaluation.v1",
        "status": "evaluated",
        "scope": "TRAIN optimization-contract gate; not generalization evidence",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "artifact": identity.to_dict(),
        "training_report": ArtifactIdentity.from_artifact(
            training_report_path(artifact)
        ).to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "selected_input_fingerprint": selected_input_fingerprint,
        "final_router": final_router_report,
        "candidate_invariant": True,
        "router_inputs": ["physical_observation"],
        "aggregate": aggregate,
        "retained": retained,
        "post": post,
        "by_segment": by_segment,
        "residual_ratios": residual_ratios,
        "experimental_gate": {
            "passed": passed,
            "minimum_overall_win_rate": evaluation["minimum_overall_win_rate"],
            "minimum_retained_win_rate": evaluation["minimum_retained_win_rate"],
            "minimum_post_win_rate": evaluation["minimum_post_win_rate"],
            "minimum_signed_order_fraction": {
                "retreat": evaluation["minimum_retreat_signed_order_fraction"],
                "align": evaluation["minimum_alignment_signed_order_fraction"],
                "insert": evaluation["minimum_insertion_signed_order_fraction"],
            },
            "requires_positive_mean_each_segment": evaluation[
                "require_positive_mean_improvement_each_semantic_segment"
            ],
            "maximum_applied_residual_to_base_embedding_ratio": evaluation[
                "maximum_applied_residual_to_base_embedding_ratio"
            ],
            "requires_exact_base_in_semantic_holds": evaluation[
                "require_exact_base_in_semantic_holds"
            ],
        },
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "outcome": (
            "physical_state_residual_train_candidate"
            if passed
            else "physical_state_residual_train_failed"
        ),
        "held_out_gate_authorized": False,
        "held_out_accessed": False,
        "canonical_accessed": False,
        "live_action_authorized": False,
    }
    write_json_atomic(output, report)
    _write_run_state("evaluation_completed")
    return report


def _add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)


def _failure_argument(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (list, tuple)):
        return [_failure_argument(item) for item in value]
    return value


def _write_failure_record(
    path: Path,
    command: str,
    arguments: dict[str, Any],
    error: Exception,
    *,
    terminal: bool,
) -> None:
    """Preserve the first reconstructible negative for this frozen experiment."""

    if path.exists():
        return
    artifacts = {}
    for name, artifact in (
        ("preflight", PREFLIGHT_PATH),
        ("training_artifact", ARTIFACT_PATH),
        ("training_report", training_report_path(ARTIFACT_PATH)),
        ("train_evaluation", EVALUATION_PATH),
    ):
        if artifact.is_file():
            artifacts[name] = ArtifactIdentity.from_artifact(artifact).to_dict()
    write_json_atomic(
        path,
        {
            "schema": "quantis.jepa_wm_physical_state_residual_failure.v1",
            "status": "failed",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "arguments": {
                name: _failure_argument(value)
                for name, value in sorted(arguments.items())
            },
            "error_type": type(error).__name__,
            "error": str(error),
            "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
            "artifacts": artifacts,
            "terminal_experiment_failure": terminal,
            "retry_same_command_authorized": not terminal,
            "retraining_authorized": False,
            "held_out_accessed": False,
            "canonical_accessed": False,
            "live_action_authorized": False,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _add_common_model_arguments(preflight_parser)
    preflight_parser.add_argument(
        "--recording", type=Path, action="append", required=True
    )
    preflight_parser.add_argument("--control-adapter", type=Path, required=True)
    preflight_parser.add_argument("--route-probe", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    _add_common_model_arguments(train_parser)
    train_parser.add_argument("--recording", type=Path, action="append", required=True)
    train_parser.add_argument("--control-adapter", type=Path, required=True)
    train_parser.add_argument("--route-probe", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate-train")
    _add_common_model_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--recording", type=Path, action="append", required=True
    )
    evaluate_parser.add_argument("--artifact", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "preflight":
            result = preflight(
                arguments.recording,
                arguments.source,
                arguments.checkpoint,
                arguments.control_adapter,
                arguments.route_probe,
                arguments.output,
                experiment_config=arguments.experiment_config,
            )
        elif arguments.command == "train":
            result = train_residual(
                arguments.source,
                arguments.checkpoint,
                arguments.control_adapter,
                arguments.recording,
                arguments.route_probe,
                arguments.output,
                experiment_config=arguments.experiment_config,
            )
        else:
            result = evaluate_train(
                arguments.source,
                arguments.checkpoint,
                arguments.recording,
                arguments.artifact,
                arguments.output,
                experiment_config=arguments.experiment_config,
            )
    except Exception as error:
        terminal = _command_entered_experiment(arguments.command)
        _write_failure_record(
            FAILURE_PATH if terminal else INVOCATION_FAILURE_PATH,
            arguments.command,
            vars(arguments),
            error,
            terminal=terminal,
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
