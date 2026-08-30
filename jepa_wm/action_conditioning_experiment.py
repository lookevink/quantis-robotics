"""Frozen offline experiment for insertion action-conditioning capacity."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.action_conditioning import (
    ACTION_CONDITIONING_SCHEMA,
    POST_REGIME,
    RETAINED_REGIME,
    ActionConditioningContract,
    ActionConditioningKind,
    ActionConditioningSpec,
    LoadedActionConditioning,
    action_conditioning_parameters,
    install_action_conditioning,
    save_action_conditioning,
)
from jepa_wm.action_conditioning_training import (
    AlternatingStratumSampler,
    signed_x_margin_loss,
    signed_x_negatives,
)
from jepa_wm.adapt_recording import (
    ContrastiveTermConfig,
    _sample_mismatched_indices,
    mismatched_negative_candidates,
    validated_training_recordings,
)
from jepa_wm.candidate_negatives import (
    CandidateMiningConfig,
    mine_lowest_energy_candidates,
    sample_local_candidates,
)
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_layout import (
    CONTACT_INSERTION_LAYOUT,
    ContactInsertionSegment,
)
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.model import load_headless_model
from jepa_wm.persistence import write_json_atomic
from jepa_wm.readiness import ActionControlGate
from jepa_wm.rollout_scoring import (
    rollout_action_tensor,
    score_actions,
    score_recorded_against_mismatched,
)
from jepa_wm.rollout_training import RolloutTrainingSelection
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    artifact_fingerprint,
    load_training_report,
    training_configuration_fingerprint,
    training_report_path,
)
from jepa_wm.trajectory import RecordedRollout, RolloutWindow, load_rollouts


FROZEN_EXPERIMENT_CONFIG_FINGERPRINT = (
    "a8e111cbd197592091c93cf1d00adb751ede97a194c537559ffe10e7b5e7de14"
)
EXPERIMENT_SCHEMA = "quantis.jepa_wm_action_conditioning_experiment.v1"
TRAINING_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)
EXPERIMENT_WINDOW = RolloutWindow(113, 168, 1)
TRAINING_RECORDINGS = tuple(
    f"contact-insertion-v10-drive-slow-2600-train-{index:02d}"
    for index in range(12)
)
DEVELOPMENT_CANARY = "contact-insertion-v10-drive-slow-72600-held-00"
CANONICAL_HELD_OUT = (
    "contact-insertion-v10-drive-slow-2600-held-00",
    "contact-insertion-v10-drive-slow-2600-held-01",
)
TREATMENT_SPECS = {
    "B": ActionConditioningSpec(ActionConditioningKind.GLOBAL_LINEAR),
    "C": ActionConditioningSpec(
        ActionConditioningKind.NONLINEAR_RESIDUAL,
        hidden_dimension=32,
    ),
    "D": ActionConditioningSpec(ActionConditioningKind.ORACLE_REGIME_RESIDUAL),
}


def _load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT:
        raise ValueError("experiment configuration fingerprint changed")
    payload = json.loads(encoded)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != EXPERIMENT_SCHEMA
        or tuple(payload.get("corpus", {}).get("training_recordings", ()))
        != TRAINING_RECORDINGS
        or payload.get("corpus", {}).get("development_canary")
        != DEVELOPMENT_CANARY
        or tuple(payload.get("corpus", {}).get("canonical_held_out", ()))
        != CANONICAL_HELD_OUT
    ):
        raise ValueError("experiment configuration contract is invalid")
    return payload


def _regime_for_context(context_index: int) -> int:
    if 113 <= context_index <= 165:
        return RETAINED_REGIME
    if 166 <= context_index <= 280:
        return POST_REGIME
    raise ValueError("rollout context is outside the frozen insertion window")


def _rollout_regimes(rollouts: Sequence[RecordedRollout]) -> torch.Tensor:
    return torch.tensor(
        [_regime_for_context(rollout.context[0].index) for rollout in rollouts],
        dtype=torch.long,
    )


def _segment_for_context(context_index: int) -> str:
    start = 0
    for span in CONTACT_INSERTION_LAYOUT.spans:
        stop = start + span.frames
        if start <= context_index < stop:
            return span.segment.value
        start = stop
    raise ValueError("context index is outside the insertion layout")


def _validate_output_is_new(output: Path) -> None:
    if output.exists() or training_report_path(output).exists():
        raise ValueError(f"experiment output already exists: {output}")


def _treatment_training_config(
    experiment: dict[str, Any],
    treatment: str,
) -> dict[str, Any]:
    return {
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "treatment": treatment,
        "spec": TREATMENT_SPECS[treatment].to_dict(),
        "training": experiment["training"],
    }


def train_treatment(
    source: Path,
    checkpoint: Path,
    recordings: Sequence[Path],
    output: Path,
    *,
    treatment: str,
    experiment_config: Path,
) -> dict[str, Any]:
    if treatment not in TREATMENT_SPECS:
        raise ValueError("training treatment must be B, C, or D")
    if not torch.cuda.is_available():
        raise RuntimeError("action-conditioning training requires CUDA")
    _validate_output_is_new(output)
    experiment = _load_experiment_config(experiment_config)
    validated = validated_training_recordings(recordings)
    if tuple(recording.name for recording in validated) != TRAINING_RECORDINGS:
        raise ValueError("training inputs do not match the frozen TRAIN roster")
    selection = RolloutTrainingSelection.load(
        tuple(recording.path for recording in validated),
        camera="wrist",
        bounds=TRAINING_BOUNDS,
        window=EXPERIMENT_WINDOW,
    )
    rollouts = selection.rollouts
    if len(rollouts) != 12 * EXPERIMENT_WINDOW.count:
        raise ValueError("frozen action-conditioning selection is incomplete")

    training = experiment["training"]
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    load_started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    spec = TREATMENT_SPECS[treatment]
    install_action_conditioning(model, spec)
    load_seconds = monotonic() - load_started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = action_conditioning_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)

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
    goal_actions = torch.tensor(
        [rollout.goal_action.values for rollout in rollouts],
        dtype=actions.dtype,
    )
    regimes = _rollout_regimes(rollouts)
    mismatched_candidates = mismatched_negative_candidates(rollouts)
    candidate_config = CandidateMiningConfig.from_dict(training["candidate_mining"])
    terms = {
        name: ContrastiveTermConfig(**training["objective"][name])
        for name in (
            "zero_negative",
            "mismatched_negative",
            "candidate_negative",
            "signed_x_zero_negative",
            "signed_x_opposed_negative",
        )
    }

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sampler = AlternatingStratumSampler(regimes, seed=seed)
    mismatch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    candidate_generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    model.eval()
    training_started = monotonic()
    for _ in range(int(training["steps"])):
        indices = torch.tensor((sampler.next_index(),), dtype=torch.long)
        context = contexts[indices].to(device)
        target = targets[indices].to(device)
        action_batch = actions[:, indices].to(device)
        regime_batch = regimes[indices].to(device)
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
        mined = mine_lowest_energy_candidates(
            model,
            context,
            target,
            local_candidates,
            scoring_batch_size=candidate_config.scoring_batch_size,
            regimes=regime_batch,
        )
        energies = score_recorded_against_mismatched(
            model,
            context,
            target,
            action_batch,
            mismatched,
            regime_batch,
        )
        candidate_energy = score_actions(
            model,
            context,
            target,
            mined,
            regime_batch,
        )
        x_zero, x_opposed = signed_x_negatives(action_batch)
        x_zero_energy = score_actions(
            model,
            context,
            target,
            x_zero,
            regime_batch,
        )
        x_opposed_energy = score_actions(
            model,
            context,
            target,
            x_opposed,
            regime_batch,
        )
        loss = (
            energies.recorded.mean()
            + terms["zero_negative"].loss(energies.recorded, energies.zero)
            + terms["mismatched_negative"].loss(
                energies.recorded,
                energies.mismatched_negative,
            )
            + terms["candidate_negative"].loss(
                energies.recorded,
                candidate_energy,
            )
            + signed_x_margin_loss(
                energies.recorded,
                x_zero_energy,
                action_batch,
                weight=terms["signed_x_zero_negative"].weight,
                margin=terms["signed_x_zero_negative"].margin,
                minimum_activity=float(
                    training["objective"]["signed_x_activity_threshold"]
                ),
            )
            + signed_x_margin_loss(
                energies.recorded,
                x_opposed_energy,
                action_batch,
                weight=terms["signed_x_opposed_negative"].weight,
                margin=terms["signed_x_opposed_negative"].margin,
                minimum_activity=float(
                    training["objective"]["signed_x_activity_threshold"]
                ),
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    training_seconds = monotonic() - training_started

    metadata = TrainingArtifactMetadata(
        MODEL_ID,
        os.environ.get("JEPA_WM_REVISION", "unknown"),
        "wrist",
        TRAINING_RECORDINGS,
        int(training["steps"]),
    )
    config_payload = _treatment_training_config(experiment, treatment)
    config_fingerprint = training_configuration_fingerprint(config_payload)
    contract = ActionConditioningContract(
        ACTION_CONDITIONING_SCHEMA,
        metadata,
        selection.fingerprint,
        config_fingerprint,
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        spec,
    )
    save_action_conditioning(model, output, contract)
    report = {
        "schema": "quantis.jepa_wm_action_conditioning_training.v1",
        "status": "trained",
        "treatment": treatment,
        "artifact": str(output.resolve()),
        "artifact_fingerprint": artifact_fingerprint(output),
        "metadata": metadata.to_dict(),
        "contract": contract.to_dict(),
        "config": config_payload,
        "training_config_fingerprint": config_fingerprint,
        **selection.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "sampling": sampler.to_dict(),
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
    }
    report["report"] = str(training_report_path(output).resolve())
    write_json_atomic(training_report_path(output), report)
    return report


def smoke_treatments(
    source: Path,
    checkpoint: Path,
    recording: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    """Exercise one real TRAIN batch per family without saving an artifact."""

    if not torch.cuda.is_available():
        raise RuntimeError("action-conditioning smoke requires CUDA")
    experiment = _load_experiment_config(experiment_config)
    validated = validated_training_recordings((recording,))
    if validated[0].name != TRAINING_RECORDINGS[0] or validated[0].seed != 2600:
        raise ValueError("smoke requires frozen TRAIN recording 00")
    selection = RolloutTrainingSelection.load(
        (recording,),
        camera="wrist",
        bounds=TRAINING_BOUNDS,
        window=RolloutWindow(114, 1, 1),
    )
    rollout = selection.rollouts[0]
    actions = rollout_action_tensor((rollout,))
    routes = torch.tensor((RETAINED_REGIME,), dtype=torch.long)
    device = torch.device("cuda", torch.cuda.current_device())
    results = {}
    reference_energy: torch.Tensor | None = None
    for treatment, spec in TREATMENT_SPECS.items():
        torch.manual_seed(int(experiment["training"]["seed"]))
        torch.cuda.manual_seed_all(int(experiment["training"]["seed"]))
        model = load_headless_model(source, checkpoint, device=device)
        install_action_conditioning(model, spec)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        trainable = action_conditioning_parameters(model)
        for parameter in trainable:
            parameter.requires_grad_(True)
        contexts = encode_clips(model, [rollout.context_paths], batch_size=1)
        targets = encode_clips(model, [rollout.target_clip], batch_size=1)
        context = contexts.to(device)
        target = targets.to(device)
        action_batch = actions.to(device)
        route_batch = routes.to(device)
        initial = score_actions(
            model,
            context,
            target,
            action_batch,
            route_batch,
        )
        if reference_energy is None:
            reference_energy = initial.detach().cpu()
        elif not torch.equal(initial.detach().cpu(), reference_energy):
            raise ValueError("zero-residual treatment does not match linear baseline")
        x_zero, x_opposed = signed_x_negatives(action_batch)
        zero_energy = score_actions(
            model,
            context,
            target,
            x_zero,
            route_batch,
        )
        opposed_energy = score_actions(
            model,
            context,
            target,
            x_opposed,
            route_batch,
        )
        term = ContrastiveTermConfig(weight=1.0, margin=0.001)
        loss = initial.mean() + term.loss(initial, zero_energy) + term.loss(
            initial,
            opposed_energy,
        )
        if not torch.isfinite(loss):
            raise ValueError("action-conditioning smoke loss is not finite")
        before = tuple(parameter.detach().clone() for parameter in trainable)
        optimizer = torch.optim.AdamW(
            trainable,
            lr=float(experiment["training"]["learning_rate"]),
            weight_decay=float(experiment["training"]["weight_decay"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in trainable):
            raise ValueError("action-conditioning smoke gradient is invalid")
        optimizer.step()
        changed = any(
            not torch.equal(previous, parameter.detach())
            for previous, parameter in zip(before, trainable)
        )
        if not changed:
            raise ValueError("action-conditioning smoke optimizer made no update")
        results[treatment] = {
            "initial_energy": float(initial.detach().cpu()),
            "loss": float(loss.detach().cpu()),
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
            "updated": changed,
        }
        del model
        torch.cuda.empty_cache()
    return {
        "status": "smoke_passed",
        "recording": recording.name,
        "context_index": rollout.context[0].index,
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "treatments": results,
        "artifacts_written": 0,
        "held_out_accessed": False,
    }


@dataclass(frozen=True)
class EvaluationEnergies:
    recorded: torch.Tensor
    zero: torch.Tensor
    x_zero: torch.Tensor
    x_opposed: torch.Tensor


def _score_evaluation_batches(
    model: Any,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    regimes: torch.Tensor | None,
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
            route_batch = regimes[start:stop].to(device) if regimes is not None else None
            for name, candidate in candidates.items():
                chunks[name].append(
                    score_actions(
                        model,
                        contexts[start:stop].to(device),
                        targets[start:stop].to(device),
                        candidate[:, start:stop].to(device),
                        route_batch,
                    ).cpu()
                )
    return EvaluationEnergies(
        **{name: torch.cat(values) for name, values in chunks.items()}
    )


def _metrics(indices: Sequence[int], energies: EvaluationEnergies) -> dict[str, Any]:
    selected = torch.tensor(indices, dtype=torch.long)
    recorded = energies.recorded[selected]
    zero = energies.zero[selected]
    improvements = zero - recorded
    return {
        "rollouts": len(indices),
        "mean_improvement_over_zero": float(improvements.mean()),
        "recorded_action_win_rate": float((improvements > 0.0).float().mean()),
        "signed_order_fraction": float(
            (
                (recorded < energies.x_zero[selected])
                & (energies.x_zero[selected] < energies.x_opposed[selected])
            )
            .float()
            .mean()
        ),
    }


def evaluate_treatment(
    source: Path,
    checkpoint: Path,
    recording: Path,
    artifact: Path,
    output: Path,
    *,
    treatment: str,
    expected_seed: int,
    experiment_config: Path,
) -> dict[str, Any]:
    if treatment not in ("A", "B", "C", "D"):
        raise ValueError("evaluation treatment must be A, B, C, or D")
    if not torch.cuda.is_available():
        raise RuntimeError("action-conditioning evaluation requires CUDA")
    if output.exists():
        raise ValueError(f"evaluation output already exists: {output}")
    experiment = _load_experiment_config(experiment_config)
    allowed_recordings = {
        DEVELOPMENT_CANARY: 72600,
        CANONICAL_HELD_OUT[0]: 12600,
        CANONICAL_HELD_OUT[1]: 12601,
    }
    if allowed_recordings.get(recording.name) != expected_seed:
        raise ValueError("evaluation input is outside the frozen held-out roster")
    ContactInsertionEvidence.from_recording(
        recording,
        expected_split="held_out",
        expected_seed=expected_seed,
    )
    rollouts = EXPERIMENT_WINDOW.select(
        load_rollouts(recording, camera="wrist", bounds=TRAINING_BOUNDS)
    )
    if len(rollouts) != EXPERIMENT_WINDOW.count:
        raise ValueError("evaluation recording lacks the frozen insertion window")

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    load_started = monotonic()
    if treatment == "A":
        model = load_headless_model(
            source,
            checkpoint,
            device=device,
            adapter=artifact,
        )
        artifact_identity = ArtifactIdentity.from_artifact(artifact)
        if (
            artifact_identity.fingerprint
            != experiment["treatments"]["A"]["artifact_fingerprint"]
        ):
            raise ValueError("control artifact disagrees with the frozen experiment")
        artifact_spec = {"kind": "existing_v2_global_linear", "hidden_dimension": None}
        regimes = None
    else:
        model = load_headless_model(source, checkpoint, device=device)
        loaded = LoadedActionConditioning.load(artifact)
        if (
            loaded.contract.experiment_config_fingerprint
            != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
            or loaded.contract.spec != TREATMENT_SPECS[treatment]
        ):
            raise ValueError("evaluation artifact disagrees with its treatment")
        loaded.apply(
            model,
            expected_source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
        )
        artifact_identity = loaded.identity
        artifact_spec = loaded.contract.spec.to_dict()
        regimes = _rollout_regimes(rollouts)
        report = load_training_report(artifact)
        if (
            report.get("artifact_fingerprint") != artifact_identity.fingerprint
            or report.get("treatment") != treatment
        ):
            raise ValueError("training report disagrees with evaluation artifact")
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
    evaluation_started = monotonic()
    energies = _score_evaluation_batches(
        model,
        contexts,
        targets,
        actions,
        regimes,
        device=device,
    )
    torch.cuda.synchronize(device)
    evaluation_seconds = monotonic() - evaluation_started

    all_indices = list(range(len(rollouts)))
    aggregate = _metrics(all_indices, energies)
    control_gate = ActionControlGate().evaluate(
        mean_improvement_over_zero=aggregate["mean_improvement_over_zero"],
        recorded_action_win_rate=aggregate["recorded_action_win_rate"],
    )
    by_segment: dict[str, Any] = {}
    for segment in ContactInsertionSegment:
        indices = [
            index
            for index, rollout in enumerate(rollouts)
            if _segment_for_context(rollout.context[0].index) == segment.value
        ]
        if indices:
            by_segment[segment.value] = _metrics(indices, energies)
    retained_indices = [
        index
        for index, rollout in enumerate(rollouts)
        if _regime_for_context(rollout.context[0].index) == RETAINED_REGIME
    ]
    post_indices = [
        index
        for index, rollout in enumerate(rollouts)
        if _regime_for_context(rollout.context[0].index) == POST_REGIME
    ]
    retained = _metrics(retained_indices, energies)
    post = _metrics(post_indices, energies)
    retreat = by_segment[ContactInsertionSegment.RETREAT.value]
    align = by_segment[ContactInsertionSegment.ALIGN.value]
    experimental_passed = (
        control_gate.passed
        and aggregate["recorded_action_win_rate"] >= 0.90
        and retained["recorded_action_win_rate"] >= 0.85
        and post["recorded_action_win_rate"] >= 0.95
        and all(
            segment["mean_improvement_over_zero"] > 0.0
            for segment in by_segment.values()
        )
        and retreat["signed_order_fraction"] >= 0.75
        and align["signed_order_fraction"] >= 0.75
    )
    results = []
    for index, rollout in enumerate(rollouts):
        recorded = float(energies.recorded[index])
        zero = float(energies.zero[index])
        x_zero = float(energies.x_zero[index])
        x_opposed = float(energies.x_opposed[index])
        results.append(
            {
                "context_index": rollout.context[0].index,
                "target_index": rollout.target.index,
                "segment": _segment_for_context(rollout.context[0].index),
                "regime": (
                    "retained"
                    if _regime_for_context(rollout.context[0].index)
                    == RETAINED_REGIME
                    else "post"
                ),
                "recorded_energy": recorded,
                "zero_energy": zero,
                "x_zero_energy": x_zero,
                "x_opposed_energy": x_opposed,
                "improvement_over_zero": zero - recorded,
                "recorded_action_wins": recorded < zero,
                "signed_order_passed": recorded < x_zero < x_opposed,
            }
        )
    report = {
        "schema": "quantis.jepa_wm_action_conditioning_evaluation.v1",
        "status": "evaluated",
        "scope": "offline insertion energy only; no live action",
        "treatment": treatment,
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "artifact": str(artifact_identity.path),
        "artifact_fingerprint": artifact_identity.fingerprint,
        "artifact_spec": artifact_spec,
        "recording": recording.name,
        "expected_seed": expected_seed,
        "window": EXPERIMENT_WINDOW.to_dict(),
        "aggregate": {**aggregate, "control_gate": control_gate.to_dict()},
        "retained": retained,
        "post": post,
        "by_segment": by_segment,
        "experimental_gate": {
            "passed": experimental_passed,
            "minimum_overall_win_rate": 0.90,
            "minimum_retained_win_rate": 0.85,
            "minimum_post_win_rate": 0.95,
            "minimum_retreat_signed_order_fraction": 0.75,
            "minimum_alignment_signed_order_fraction": 0.75,
            "requires_positive_mean_each_segment": True,
        },
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "results": results,
    }
    write_json_atomic(output, report)
    return report


def summarize_canary(reports: Sequence[Path], output: Path) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"canary summary already exists: {output}")
    payloads = [json.loads(report.resolve().read_text()) for report in reports]
    if (
        len(payloads) != 4
        or {payload.get("treatment") for payload in payloads} != {"A", "B", "C", "D"}
        or any(payload.get("recording") != DEVELOPMENT_CANARY for payload in payloads)
        or any(
            payload.get("experiment_config_fingerprint")
            != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
            for payload in payloads
        )
    ):
        raise ValueError("canary reports do not match the frozen treatment matrix")
    by_treatment = {payload["treatment"]: payload for payload in payloads}
    selected = next(
        (
            treatment
            for treatment in ("B", "C")
            if by_treatment[treatment]["experimental_gate"]["passed"]
        ),
        None,
    )
    if selected == "B":
        outcome = "balanced_linear_candidate"
    elif selected == "C":
        outcome = "nonlinear_residual_candidate"
    elif by_treatment["D"]["experimental_gate"]["passed"]:
        outcome = "regime_conflict_confirmed"
    else:
        outcome = "frozen_dynamics_or_representation_blocker"
    summary = {
        "schema": "quantis.jepa_wm_action_conditioning_canary_summary.v1",
        "status": "complete",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "outcome": outcome,
        "selected_treatment": selected,
        "selected_artifact": (
            {
                "path": by_treatment[selected]["artifact"],
                "fingerprint": by_treatment[selected]["artifact_fingerprint"],
            }
            if selected is not None
            else None
        ),
        "treatments": {
            treatment: {
                "artifact_fingerprint": payload["artifact_fingerprint"],
                "aggregate": payload["aggregate"],
                "retained": payload["retained"],
                "post": payload["post"],
                "experimental_gate": payload["experimental_gate"],
            }
            for treatment, payload in sorted(by_treatment.items())
        },
    }
    write_json_atomic(output, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--source", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--recording", type=Path, action="append", required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--treatment", choices=("B", "C", "D"), required=True)
    train.add_argument("--experiment-config", type=Path, required=True)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--recording", type=Path, required=True)
    evaluate.add_argument("--artifact", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--treatment", choices=("A", "B", "C", "D"), required=True)
    evaluate.add_argument("--expected-seed", type=int, required=True)
    evaluate.add_argument("--experiment-config", type=Path, required=True)

    summarize = subparsers.add_parser("summarize-canary")
    summarize.add_argument("--report", type=Path, action="append", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--source", type=Path, required=True)
    smoke.add_argument("--checkpoint", type=Path, required=True)
    smoke.add_argument("--recording", type=Path, required=True)
    smoke.add_argument("--experiment-config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "train":
        result = train_treatment(
            arguments.source,
            arguments.checkpoint,
            arguments.recording,
            arguments.output,
            treatment=arguments.treatment,
            experiment_config=arguments.experiment_config,
        )
    elif arguments.command == "evaluate":
        result = evaluate_treatment(
            arguments.source,
            arguments.checkpoint,
            arguments.recording,
            arguments.artifact,
            arguments.output,
            treatment=arguments.treatment,
            expected_seed=arguments.expected_seed,
            experiment_config=arguments.experiment_config,
        )
    elif arguments.command == "summarize-canary":
        result = summarize_canary(arguments.report, arguments.output)
    else:
        result = smoke_treatments(
            arguments.source,
            arguments.checkpoint,
            arguments.recording,
            experiment_config=arguments.experiment_config,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
