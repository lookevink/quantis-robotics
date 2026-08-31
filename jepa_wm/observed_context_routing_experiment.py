"""Frozen TRAIN-only experiment for candidate-independent context routing."""

from __future__ import annotations

import argparse
from collections import Counter
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
    BASE_COMMAND_ROUTE,
    COMMAND_ROUTE_NAMES,
    NEGATIVE_X_COMMAND_ROUTE,
    POSITIVE_X_COMMAND_ROUTE,
    ActionConditioningContract,
    ActionConditioningKind,
    ActionConditioningSpec,
    LoadedActionConditioning,
    ObservedContextResidualActionEncoder,
    ObservedContextRoutingSpec,
    action_conditioning_parameters,
    install_action_conditioning,
    observed_action_context,
    save_action_conditioning,
)
from jepa_wm.action_conditioning_experiment import (
    EXPERIMENT_WINDOW,
    TRAINING_BOUNDS,
    TRAINING_RECORDINGS,
    EvaluationEnergies,
    _metrics,
    _regime_for_context,
    _segment_for_context,
)
from jepa_wm.action_conditioning_training import (
    AlternatingCommandRouteSampler,
    signed_x_margin_loss,
    signed_x_negatives,
)
from jepa_wm.action_routing_experiment import (
    CONTROL_ARTIFACT_FINGERPRINT,
    ROUTING_SPEC as COMMAND_ROUTING_SPEC,
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
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_layout import ContactInsertionSegment
from jepa_wm.model import load_headless_model
from jepa_wm.persistence import write_json_atomic
from jepa_wm.readiness import ActionControlGate
from jepa_wm.rollout_scoring import rollout_action_tensor, score_actions
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    artifact_fingerprint,
    load_training_report,
    training_configuration_fingerprint,
    training_report_path,
)
from jepa_wm.trajectory import RecordedRollout, load_rollouts


EXPERIMENT_SCHEMA = "quantis.jepa_wm_observed_context_routing_experiment.v1"
FROZEN_EXPERIMENT_CONFIG_FINGERPRINT = (
    "2b57e748a1bf3e60af1e6ad0ec946a1ed502923240ea7b03765c7e808bb3abf6"
)
OUTPUT_ROOT = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/quantis_observed_context_routing_v1"
)
ARTIFACT_PATH = OUTPUT_ROOT / "observed_context_router.pth"
PREFLIGHT_PATH = OUTPUT_ROOT / "preflight.json"
EVALUATION_PATH = OUTPUT_ROOT / "train-evaluation.json"
ROUTING_SPEC = ObservedContextRoutingSpec(
    signed_x_deadband=0.0001,
    signed_x_transition_width=0.0001,
)
OBSERVED_CONTEXT_TREATMENT = "O"
OBSERVED_CONTEXT_TREATMENT_SPEC = ActionConditioningSpec(
    ActionConditioningKind.OBSERVED_CONTEXT_RESIDUAL,
    observed_context_routing=ROUTING_SPEC,
)
EXPECTED_OBSERVED_ROUTE_ROSTER = {
    "negative_x": 576,
    "positive_x": 1283,
    "base": 157,
    "total": 2016,
}
EXPECTED_FUTURE_ROUTE_MATCHES = 1895


def _load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT:
        raise ValueError("observed-context experiment configuration changed")
    payload = json.loads(encoded)
    corpus = payload.get("corpus", {})
    router = payload.get("router", {})
    evaluation = payload.get("evaluation", {})
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != EXPERIMENT_SCHEMA
        or tuple(corpus.get("training_recordings", ())) != TRAINING_RECORDINGS
        or corpus.get("window") != EXPERIMENT_WINDOW.to_dict()
        or payload.get("train_only_previous_action_roster")
        != {
            **EXPECTED_OBSERVED_ROUTE_ROSTER,
            "matches_recorded_future_route": EXPECTED_FUTURE_ROUTE_MATCHES,
        }
        or router.get("kind") != "observed_context_residual"
        or router.get("source") != "previous_realized_droid_action"
        or router.get("signed_x_deadband") != ROUTING_SPEC.signed_x_deadband
        or router.get("signed_x_transition_width")
        != ROUTING_SPEC.signed_x_transition_width
        or router.get("runtime_inputs") != ["previous_action"]
        or router.get("candidate_invariant") is not True
        or router.get("continuous_at_deadband") is not True
        or "candidate_action" not in router.get("forbidden_router_inputs", ())
        or "recorded_future_action"
        not in router.get("forbidden_router_inputs", ())
        or evaluation.get("residual_ratio_candidates")
        != ["recorded", "zero", "x_zero", "x_opposed"]
    ):
        raise ValueError("observed-context experiment contract is invalid")
    return payload


def previous_action_tensor(
    rollouts: Sequence[RecordedRollout],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not rollouts:
        raise ValueError("observed-context routing requires rollouts")
    return torch.tensor(
        [rollout.previous_action.values for rollout in rollouts],
        dtype=torch.float32,
        device=device,
    )


def observed_route_roster(previous_actions: torch.Tensor) -> dict[str, int]:
    routes = ROUTING_SPEC.classify(previous_actions)
    counts = Counter(COMMAND_ROUTE_NAMES[int(route)] for route in routes)
    return {
        "negative_x": counts["negative_x"],
        "positive_x": counts["positive_x"],
        "base": counts["base"],
        "total": int(routes.numel()),
    }


def _route_audit(rollouts: Sequence[RecordedRollout]) -> dict[str, Any]:
    previous = previous_action_tensor(rollouts)
    observed_routes = ROUTING_SPEC.classify(previous)
    future_routes, _ = COMMAND_ROUTING_SPEC.classify(
        rollout_action_tensor(rollouts).transpose(0, 1)
    )
    matches = int((observed_routes == future_routes).sum())
    roster = observed_route_roster(previous)
    if (
        roster != EXPECTED_OBSERVED_ROUTE_ROSTER
        or matches != EXPECTED_FUTURE_ROUTE_MATCHES
    ):
        raise ValueError("observed-context TRAIN route audit changed")
    return {
        "previous_action_routes": roster,
        "recorded_future_route_matches": matches,
        "recorded_future_route_match_fraction": matches / len(rollouts),
    }


def _validate_output(output: Path, expected: Path) -> None:
    if output.resolve() != expected:
        raise ValueError(f"observed-context output must be {expected}")
    if output.exists():
        raise ValueError(f"observed-context output already exists: {output}")


def preflight(
    recordings: Sequence[Path],
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
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
    selection = _validated_selection(recordings)
    audit = _route_audit(selection.rollouts)
    report = {
        "schema": "quantis.jepa_wm_observed_context_routing_preflight.v1",
        "status": "passed",
        "scope": "TRAIN-only offline routing; no held-out or live action",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "control_artifact": control_identity.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "router": ROUTING_SPEC.to_dict(),
        "route_audit": audit,
        "candidate_router_inputs": [],
        "held_out_accessed": False,
        "canonical_accessed": False,
        "live_action_authorized": False,
    }
    write_json_atomic(output, report)
    return report


def _score_observed(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
) -> torch.Tensor:
    with observed_action_context(model, previous_actions):
        return score_actions(model, context, target, actions)


def _mine_observed_candidates(
    model: Any,
    context: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    previous_actions: torch.Tensor,
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
        or previous_actions.shape != (batch, 7)
    ):
        raise ValueError("observed candidate batch is inconsistent")
    flattened = candidates.reshape(horizon, batch * candidate_count, 7)
    repeated_context = context.repeat_interleave(candidate_count, dim=0)
    repeated_target = target.repeat_interleave(candidate_count, dim=0)
    repeated_previous = previous_actions.repeat_interleave(candidate_count, dim=0)
    with torch.no_grad():
        chunks = []
        for start in range(0, batch * candidate_count, scoring_batch_size):
            stop = start + scoring_batch_size
            chunks.append(
                _score_observed(
                    model,
                    repeated_context[start:stop],
                    repeated_target[start:stop],
                    flattened[:, start:stop],
                    repeated_previous[start:stop],
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
        "treatment": OBSERVED_CONTEXT_TREATMENT,
        "spec": OBSERVED_CONTEXT_TREATMENT_SPEC.to_dict(),
        "control_artifact_fingerprint": CONTROL_ARTIFACT_FINGERPRINT,
        "training": experiment["training"],
    }


def train_router(
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    recordings: Sequence[Path],
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("observed-context training requires CUDA")
    _validate_output(output, ARTIFACT_PATH)
    if training_report_path(output).exists():
        raise ValueError("observed-context training report already exists")
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    selection = _validated_selection(recordings)
    rollouts = selection.rollouts
    audit = _route_audit(rollouts)
    training = experiment["training"]
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    load_started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=control_adapter,
    )
    encoder = install_action_conditioning(model, OBSERVED_CONTEXT_TREATMENT_SPEC)
    if not isinstance(encoder, ObservedContextResidualActionEncoder):
        raise ValueError("observed-context encoder installation failed")
    load_seconds = monotonic() - load_started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = action_conditioning_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)
    frozen_base = tuple(
        value.detach().clone() for value in encoder.base.state_dict().values()
    )

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
    previous_actions = previous_action_tensor(rollouts)
    routes = ROUTING_SPEC.classify(previous_actions)
    goal_actions = torch.tensor(
        [rollout.goal_action.values for rollout in rollouts],
        dtype=actions.dtype,
    )
    mismatched_candidates = mismatched_negative_candidates(rollouts)
    candidate_config = CandidateMiningConfig.from_dict(
        training["candidate_mining"]
    )
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
    sampler = AlternatingCommandRouteSampler(routes, seed=seed)
    mismatch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    candidate_generator = torch.Generator(device=device).manual_seed(seed)
    minimum_activity = float(
        training["objective"]["signed_x_activity_threshold"]
    )
    losses = []
    model.eval()
    training_started = monotonic()
    for _ in range(int(training["steps"])):
        indices = torch.tensor((sampler.next_index(),), dtype=torch.long)
        context = contexts[indices].to(device)
        target = targets[indices].to(device)
        action_batch = actions[:, indices].to(device)
        previous_batch = previous_actions[indices].to(device)
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
        mined = _mine_observed_candidates(
            model,
            context,
            target,
            local_candidates,
            previous_batch,
            scoring_batch_size=candidate_config.scoring_batch_size,
        )
        recorded_energy = _score_observed(
            model,
            context,
            target,
            action_batch,
            previous_batch,
        )
        zero_energy = _score_observed(
            model,
            context,
            target,
            torch.zeros_like(action_batch),
            previous_batch,
        )
        mismatched_energy = _score_observed(
            model,
            context,
            target,
            mismatched,
            previous_batch,
        )
        candidate_energy = _score_observed(
            model,
            context,
            target,
            mined,
            previous_batch,
        )
        x_zero, x_opposed = signed_x_negatives(action_batch)
        x_zero_energy = _score_observed(
            model,
            context,
            target,
            x_zero,
            previous_batch,
        )
        x_opposed_energy = _score_observed(
            model,
            context,
            target,
            x_opposed,
            previous_batch,
        )
        loss = (
            recorded_energy.mean()
            + terms["zero_negative"].loss(recorded_energy, zero_energy)
            + terms["mismatched_negative"].loss(
                recorded_energy,
                mismatched_energy,
            )
            + terms["candidate_negative"].loss(
                recorded_energy,
                candidate_energy,
            )
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
    if any(
        not torch.equal(before, after)
        for before, after in zip(frozen_base, encoder.base.state_dict().values())
    ):
        raise ValueError("frozen control action map changed during training")

    metadata = TrainingArtifactMetadata(
        MODEL_ID,
        os.environ.get("JEPA_WM_REVISION", "unknown"),
        "wrist",
        TRAINING_RECORDINGS,
        int(training["steps"]),
    )
    config_payload = _training_config(experiment)
    config_fingerprint = training_configuration_fingerprint(config_payload)
    contract = ActionConditioningContract(
        ACTION_CONDITIONING_SCHEMA,
        metadata,
        selection.fingerprint,
        config_fingerprint,
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        OBSERVED_CONTEXT_TREATMENT_SPEC,
    )
    save_action_conditioning(model, output, contract)
    report = {
        "schema": "quantis.jepa_wm_observed_context_routing_training.v1",
        "status": "trained",
        "treatment": OBSERVED_CONTEXT_TREATMENT,
        "scope": "TRAIN-only offline routing; no held-out or live action",
        "artifact": str(output.resolve()),
        "artifact_fingerprint": artifact_fingerprint(output),
        "metadata": metadata.to_dict(),
        "contract": contract.to_dict(),
        "config": config_payload,
        "training_config_fingerprint": config_fingerprint,
        **selection.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "route_audit": audit,
        "sampling": sampler.to_dict(),
        "control_artifact": control_identity.to_dict(),
        "base_model": base_identity,
        "base_map_unchanged": True,
        "candidate_invariant": True,
        "ordered_pairwise_objective": True,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": float(np.min(losses)),
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
    return report


def smoke(
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    recording: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    """Exercise both observed motion blends on real TRAIN latents without saving."""

    if not torch.cuda.is_available():
        raise RuntimeError("observed-context smoke requires CUDA")
    experiment = _load_experiment_config(experiment_config)
    _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    if recording.name != TRAINING_RECORDINGS[0]:
        raise ValueError("observed-context smoke requires TRAIN recording 00")
    by_context = {
        rollout.context[0].index: rollout
        for rollout in load_rollouts(
            recording,
            camera="wrist",
            bounds=TRAINING_BOUNDS,
        )
    }
    rollouts = (by_context[120], by_context[180])
    previous = previous_action_tensor(rollouts)
    if tuple(int(route) for route in ROUTING_SPEC.classify(previous)) != (
        NEGATIVE_X_COMMAND_ROUTE,
        POSITIVE_X_COMMAND_ROUTE,
    ):
        raise ValueError("observed-context smoke routes changed")
    actions = rollout_action_tensor(rollouts)
    device = torch.device("cuda", torch.cuda.current_device())
    seed = int(experiment["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=control_adapter,
    )
    encoder = install_action_conditioning(model, OBSERVED_CONTEXT_TREATMENT_SPEC)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = action_conditioning_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)
    frozen_base = tuple(
        value.detach().clone() for value in encoder.base.state_dict().values()
    )
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=2,
    ).to(device)
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=2,
    ).to(device)
    action_batch = actions.to(device)
    previous_batch = previous.to(device)
    recorded = _score_observed(
        model,
        contexts,
        targets,
        action_batch,
        previous_batch,
    )
    x_zero, x_opposed = signed_x_negatives(action_batch)
    x_zero_energy = _score_observed(
        model,
        contexts,
        targets,
        x_zero,
        previous_batch,
    )
    x_opposed_energy = _score_observed(
        model,
        contexts,
        targets,
        x_opposed,
        previous_batch,
    )
    loss = recorded.mean() + signed_x_margin_loss(
        x_zero_energy,
        x_opposed_energy,
        action_batch,
        weight=1.0,
        margin=0.001,
        minimum_activity=0.0001,
    )
    optimizer = torch.optim.AdamW(trainable, lr=0.001, weight_decay=0.01)
    before = tuple(parameter.detach().clone() for parameter in trainable)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in trainable
    ):
        raise ValueError("observed-context smoke gradient is invalid")
    optimizer.step()
    changed = tuple(
        not torch.equal(previous_value, parameter.detach())
        for previous_value, parameter in zip(before, trainable)
    )
    if changed != (True, True):
        raise ValueError("observed-context smoke did not update both residuals")
    if any(
        not torch.equal(previous_value, current)
        for previous_value, current in zip(
            frozen_base,
            encoder.base.state_dict().values(),
        )
    ):
        raise ValueError("observed-context smoke changed the frozen base map")
    return {
        "status": "smoke_passed",
        "recording": recording.name,
        "contexts": [rollout.context[0].index for rollout in rollouts],
        "routes": [
            COMMAND_ROUTE_NAMES[int(route)]
            for route in ROUTING_SPEC.classify(previous)
        ],
        "loss": float(loss.detach().cpu()),
        "residuals_updated": list(changed),
        "base_map_unchanged": True,
        "candidate_invariant": True,
        "artifacts_written": 0,
        "held_out_accessed": False,
    }


def _score_evaluation_batches(
    model: Any,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
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
            previous_batch = previous_actions[start:stop].to(device)
            for name, candidate in candidates.items():
                chunks[name].append(
                    _score_observed(
                        model,
                        contexts[start:stop].to(device),
                        targets[start:stop].to(device),
                        candidate[:, start:stop].to(device),
                        previous_batch,
                    ).cpu()
                )
    return EvaluationEnergies(
        **{name: torch.cat(values) for name, values in chunks.items()}
    )


def _residual_ratio_report(
    encoder: ObservedContextResidualActionEncoder,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
) -> dict[str, Any]:
    x_zero, x_opposed = signed_x_negatives(actions)
    candidates = {
        "recorded": actions,
        "zero": torch.zeros_like(actions),
        "x_zero": x_zero,
        "x_opposed": x_opposed,
    }
    routes = ROUTING_SPEC.classify(previous_actions).to(
        encoder.base.weight.device
    )
    report = {}
    maxima = []
    for name, route in (
        (
            ("negative_x", NEGATIVE_X_COMMAND_ROUTE),
            ("positive_x", POSITIVE_X_COMMAND_ROUTE),
        )
    ):
        selected_count = int((routes == route).sum())
        if selected_count == 0:
            raise ValueError(f"TRAIN selection has no observed {name} route")
        candidate_report = {}
        route_maxima = []
        for candidate_name, candidate in candidates.items():
            batch_actions = candidate.transpose(0, 1).to(
                encoder.base.weight.device
            )
            selected = batch_actions[routes == route]
            with torch.inference_mode():
                base = encoder.base(selected)
                residual = encoder.residual_for_route(route)(selected)
                ratios = torch.linalg.vector_norm(
                    residual,
                    dim=-1,
                ) / torch.clamp(
                    torch.linalg.vector_norm(base, dim=-1),
                    min=1e-12,
                )
            candidate_maximum = float(ratios.max())
            route_maxima.append(candidate_maximum)
            candidate_report[candidate_name] = {
                "mean_residual_to_base_embedding_ratio": float(ratios.mean()),
                "maximum_residual_to_base_embedding_ratio": candidate_maximum,
            }
        maximum = max(route_maxima)
        maxima.extend(route_maxima)
        report[name] = {
            "rollouts": selected_count,
            "candidates": candidate_report,
            "maximum_full_route_residual_to_base_embedding_ratio": maximum,
        }
    report["maximum"] = max(maxima)
    return report


def _gate_for_context_indices(
    energies: EvaluationEnergies,
    context_indices: Sequence[int],
    *,
    maximum_residual_ratio: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    aggregate = _metrics(list(range(len(context_indices))), energies)
    control_gate = ActionControlGate().evaluate(
        mean_improvement_over_zero=aggregate["mean_improvement_over_zero"],
        recorded_action_win_rate=aggregate["recorded_action_win_rate"],
    )
    by_segment = {}
    for segment in ContactInsertionSegment:
        indices = [
            index
            for index, context in enumerate(context_indices)
            if _segment_for_context(context) == segment.value
        ]
        if indices:
            by_segment[segment.value] = _metrics(indices, energies)
    retained_indices = [
        index
        for index, context in enumerate(context_indices)
        if _regime_for_context(context) == 0
    ]
    post_indices = [
        index
        for index, context in enumerate(context_indices)
        if _regime_for_context(context) == 1
    ]
    retained = _metrics(retained_indices, energies)
    post = _metrics(post_indices, energies)
    required_signed_segments = (
        ContactInsertionSegment.RETREAT.value,
        ContactInsertionSegment.ALIGN.value,
        ContactInsertionSegment.INSERT.value,
    )
    passed = (
        control_gate.passed
        and aggregate["recorded_action_win_rate"] >= 0.90
        and retained["recorded_action_win_rate"] >= 0.85
        and post["recorded_action_win_rate"] >= 0.95
        and all(
            segment["mean_improvement_over_zero"] > 0.0
            for segment in by_segment.values()
        )
        and all(
            by_segment[segment]["signed_order_fraction"] >= 0.75
            for segment in required_signed_segments
        )
        and maximum_residual_ratio <= 0.15
    )
    aggregate["control_gate"] = control_gate.to_dict()
    return aggregate, retained, post, by_segment, passed


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
        raise RuntimeError("observed-context evaluation requires CUDA")
    _validate_output(output, EVALUATION_PATH)
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    if artifact.resolve() != ARTIFACT_PATH or not artifact.is_file():
        raise ValueError("observed-context evaluation artifact path changed")
    identity = ArtifactIdentity.from_artifact(artifact)
    loaded = LoadedActionConditioning.load(
        artifact,
        expected_identity=identity,
    )
    if (
        loaded.contract.experiment_config_fingerprint
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
        or loaded.contract.spec != OBSERVED_CONTEXT_TREATMENT_SPEC
    ):
        raise ValueError("observed-context artifact contract changed")
    training_report = load_training_report(artifact)
    if (
        training_report.get("artifact_fingerprint") != identity.fingerprint
        or training_report.get("treatment") != OBSERVED_CONTEXT_TREATMENT
        or training_report.get("base_map_unchanged") is not True
        or training_report.get("candidate_invariant") is not True
        or training_report.get("ordered_pairwise_objective") is not True
    ):
        raise ValueError("observed-context training report is invalid")
    selection = _validated_selection(recordings)
    rollouts = selection.rollouts
    audit = _route_audit(rollouts)
    if loaded.contract.training_selection_fingerprint != selection.fingerprint:
        raise ValueError("observed-context artifact TRAIN selection changed")
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
    encoder = model.model.predictor.action_encoder
    if not isinstance(encoder, ObservedContextResidualActionEncoder):
        raise ValueError("observed-context artifact installed the wrong encoder")
    model.eval()
    load_seconds = monotonic() - load_started
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
    previous_actions = previous_action_tensor(rollouts)
    evaluation_started = monotonic()
    energies = _score_evaluation_batches(
        model,
        contexts,
        targets,
        actions,
        previous_actions,
        device=device,
    )
    torch.cuda.synchronize(device)
    evaluation_seconds = monotonic() - evaluation_started
    residual_ratios = _residual_ratio_report(
        encoder,
        actions,
        previous_actions,
    )
    aggregate, retained, post, by_segment, passed = _gate_for_context_indices(
        energies,
        tuple(rollout.context[0].index for rollout in rollouts),
        maximum_residual_ratio=float(residual_ratios["maximum"]),
    )
    report = {
        "schema": "quantis.jepa_wm_observed_context_routing_train_evaluation.v1",
        "status": "evaluated",
        "scope": "TRAIN optimization-contract gate; not generalization evidence",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "artifact": identity.to_dict(),
        "training_report": ArtifactIdentity.from_artifact(
            training_report_path(artifact)
        ).to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "route_audit": audit,
        "candidate_invariant": True,
        "router_inputs": ["previous_action"],
        "aggregate": aggregate,
        "retained": retained,
        "post": post,
        "by_segment": by_segment,
        "residual_ratios": residual_ratios,
        "experimental_gate": {
            "passed": passed,
            "minimum_overall_win_rate": 0.90,
            "minimum_retained_win_rate": 0.85,
            "minimum_post_win_rate": 0.95,
            "minimum_signed_order_fraction": {
                "retreat": 0.75,
                "align": 0.75,
                "insert": 0.75,
            },
            "requires_positive_mean_each_segment": True,
            "maximum_full_route_residual_to_base_embedding_ratio": 0.15,
        },
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "outcome": (
            "observed_context_router_train_candidate"
            if passed
            else "observed_context_router_train_failed"
        ),
        "fresh_canary_authorized": False,
        "held_out_accessed": False,
        "canonical_accessed": False,
        "live_action_authorized": False,
    }
    write_json_atomic(output, report)
    return report


def _add_common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    _add_common_model_arguments(preflight_parser)
    preflight_parser.add_argument(
        "--recording", type=Path, action="append", required=True
    )
    preflight_parser.add_argument("--control-adapter", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke")
    _add_common_model_arguments(smoke_parser)
    smoke_parser.add_argument("--recording", type=Path, required=True)
    smoke_parser.add_argument("--control-adapter", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    _add_common_model_arguments(train_parser)
    train_parser.add_argument(
        "--recording", type=Path, action="append", required=True
    )
    train_parser.add_argument("--control-adapter", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate-train")
    _add_common_model_arguments(evaluate_parser)
    evaluate_parser.add_argument(
        "--recording", type=Path, action="append", required=True
    )
    evaluate_parser.add_argument("--artifact", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = preflight(
            arguments.recording,
            arguments.source,
            arguments.checkpoint,
            arguments.control_adapter,
            arguments.output,
            experiment_config=arguments.experiment_config,
        )
    elif arguments.command == "smoke":
        result = smoke(
            arguments.source,
            arguments.checkpoint,
            arguments.control_adapter,
            arguments.recording,
            experiment_config=arguments.experiment_config,
        )
    elif arguments.command == "train":
        result = train_router(
            arguments.source,
            arguments.checkpoint,
            arguments.control_adapter,
            arguments.recording,
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
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
