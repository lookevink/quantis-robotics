"""Offline probes for the insertion world-model regime-boundary failure."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import torch
import torch.nn.functional as functional

from jepa_wm.frames import encode_clips
from jepa_wm.insertion_layout import (
    CONTACT_INSERTION_LAYOUT,
    ContactInsertionSegment,
)
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.insertion_wm_readiness import INSERTION_BOUNDS
from jepa_wm.model import load_headless_model
from jepa_wm.rollout_scoring import rollout_action_tensor
from jepa_wm.training_artifact import artifact_fingerprint
from jepa_wm.trajectory import RecordedRollout, load_rollouts


def x_axis_counterfactuals(
    actions: torch.Tensor,
) -> OrderedDict[str, torch.Tensor]:
    """Return matched actions that vary only base-frame translation X."""

    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise ValueError("actions must have shape [horizon, batch, 7]")
    magnitude = actions[..., 0].abs()
    negative = actions.clone()
    zero = actions.clone()
    positive = actions.clone()
    negative[..., 0] = -magnitude
    zero[..., 0] = 0.0
    positive[..., 0] = magnitude
    return OrderedDict(
        (
            ("recorded", actions.clone()),
            ("all_zero", torch.zeros_like(actions)),
            ("x_negative", negative),
            ("x_zero", zero),
            ("x_positive", positive),
        )
    )


def terminal_token_l2_energy(
    prediction: torch.Tensor | Sequence[torch.Tensor],
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute terminal latent L2 without averaging away the token axis."""

    terminal = prediction[-1]
    target_frame = target[:, -1]
    if terminal.shape != target_frame.shape or terminal.ndim < 3:
        raise ValueError(
            "terminal prediction and target must share batch, token, and feature axes"
        )
    squared_error = (target_frame - terminal).pow(2)
    return squared_error.flatten(start_dim=1, end_dim=-2).mean(dim=-1)


def phase_centroid_probe(
    embeddings: torch.Tensor,
    phases: Sequence[str],
    recordings: Sequence[str],
) -> dict[str, float | int]:
    """Measure phase separability while holding out each recording in turn."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [examples, features]")
    if len(phases) != embeddings.shape[0] or len(recordings) != embeddings.shape[0]:
        raise ValueError("phase probe metadata must match the examples")
    phase_roster = tuple(dict.fromkeys(phases))
    recording_roster = tuple(dict.fromkeys(recordings))
    if len(phase_roster) < 2 or len(recording_roster) < 2:
        raise ValueError("phase probe requires at least two phases and recordings")

    normalized = functional.normalize(embeddings.float(), dim=-1)
    correct = 0
    margins: list[float] = []
    evaluated = 0
    for held_out in recording_roster:
        training_indices = [
            index for index, recording in enumerate(recordings) if recording != held_out
        ]
        held_out_indices = [
            index for index, recording in enumerate(recordings) if recording == held_out
        ]
        centroids = []
        for phase in phase_roster:
            indices = [index for index in training_indices if phases[index] == phase]
            if not indices:
                raise ValueError("every training fold must contain every phase")
            centroids.append(
                functional.normalize(normalized[indices].mean(dim=0), dim=0)
            )
        centroid_tensor = torch.stack(centroids)
        for index in held_out_indices:
            scores = normalized[index] @ centroid_tensor.T
            expected = phase_roster.index(phases[index])
            competing = torch.cat((scores[:expected], scores[expected + 1 :])).max()
            margin = float(scores[expected] - competing)
            margins.append(margin)
            correct += int(int(scores.argmax()) == expected)
            evaluated += 1
    return {
        "examples": evaluated,
        "folds": len(recording_roster),
        "accuracy": correct / evaluated,
        "mean_cosine_margin": sum(margins) / len(margins),
        "minimum_cosine_margin": min(margins),
    }


def _sampled_context_indices(
    segment: ContactInsertionSegment,
    count: int,
) -> tuple[int, ...]:
    span = CONTACT_INSERTION_LAYOUT.span(segment)
    start = CONTACT_INSERTION_LAYOUT.start_index(segment)
    if count <= 0 or count > span.frames:
        raise ValueError("samples per phase must fit every selected phase")
    if count == 1:
        return (start + span.frames // 2,)
    return tuple(
        start + round(index * (span.frames - 1) / (count - 1))
        for index in range(count)
    )


def _token_concentration(delta: torch.Tensor) -> dict[str, float | int | list[int]]:
    mean_delta = delta.abs().mean(dim=0)
    token_count = mean_delta.numel()
    top_count = max(1, round(token_count * 0.1))
    values, indices = mean_delta.topk(top_count)
    total = float(mean_delta.sum())
    return {
        "tokens": token_count,
        "top_10_percent_tokens": top_count,
        "top_10_percent_absolute_delta_fraction": (
            float(values.sum()) / total if total > 0.0 else 0.0
        ),
        "most_sensitive_token_indices": [int(index) for index in indices[:10]],
    }


def _score_counterfactuals(
    model: Any,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    global_energies: dict[str, torch.Tensor] = {}
    token_energies: dict[str, torch.Tensor] = {}
    for name, candidate in x_axis_counterfactuals(actions).items():
        token_chunks = []
        with torch.inference_mode():
            for start in range(0, contexts.shape[0], batch_size):
                stop = start + batch_size
                prediction = model.unroll(
                    contexts[start:stop].to(device),
                    candidate[:, start:stop].to(device),
                )
                token_chunks.append(
                    terminal_token_l2_energy(
                        prediction,
                        targets[start:stop].to(device),
                    ).cpu()
                )
        per_token = torch.cat(token_chunks)
        token_energies[name] = per_token
        global_energies[name] = per_token.mean(dim=1)
    return global_energies, token_energies


def _segment_report(
    global_energies: dict[str, torch.Tensor],
    token_energies: dict[str, torch.Tensor],
    indices: Sequence[int],
) -> dict[str, Any]:
    selected = torch.tensor(indices, dtype=torch.long)
    means = {
        name: float(values[selected].mean()) for name, values in global_energies.items()
    }
    negative = token_energies["x_negative"][selected]
    positive = token_energies["x_positive"][selected]
    return {
        "examples": len(indices),
        "mean_terminal_l2": means,
        "negative_x_minus_positive_x": means["x_negative"] - means["x_positive"],
        "fraction_of_examples_preferring_negative_x": float(
            (negative.mean(dim=1) < positive.mean(dim=1)).float().mean()
        ),
        "fraction_of_tokens_preferring_negative_x": float(
            (negative < positive).float().mean()
        ),
        "signed_x_token_sensitivity": _token_concentration(negative - positive),
    }


def diagnose_insertion_latents(
    source: Path,
    checkpoint: Path,
    adapter: Path,
    held_out: Sequence[tuple[Path, int]],
    *,
    samples_per_phase: int = 8,
    encoding_batch_size: int = 4,
    scoring_batch_size: int = 2,
) -> dict[str, Any]:
    """Probe existing authenticated evidence without training or simulation."""

    if not torch.cuda.is_available():
        raise RuntimeError("insertion latent diagnostics require CUDA")
    if len(held_out) < 2:
        raise ValueError("at least two held-out recordings are required")
    if encoding_batch_size <= 0 or scoring_batch_size <= 0:
        raise ValueError("diagnostic batch sizes must be positive")

    selected_by_phase = OrderedDict(
        (
            ("retreat", _sampled_context_indices(ContactInsertionSegment.RETREAT, samples_per_phase)),
            ("align", _sampled_context_indices(ContactInsertionSegment.ALIGN, samples_per_phase)),
            ("insert", _sampled_context_indices(ContactInsertionSegment.INSERT, samples_per_phase)),
        )
    )
    rollouts: list[RecordedRollout] = []
    phases: list[str] = []
    recordings: list[str] = []
    evidence = []
    for recording, seed in held_out:
        validated = ContactInsertionEvidence.from_recording(
            recording,
            expected_split="held_out",
            expected_seed=seed,
        )
        evidence.append(validated.to_dict())
        by_context = {
            rollout.context[0].index: rollout
            for rollout in load_rollouts(
                recording,
                camera="wrist",
                bounds=INSERTION_BOUNDS,
            )
        }
        for phase, context_indices in selected_by_phase.items():
            for context_index in context_indices:
                try:
                    rollout = by_context[context_index]
                except KeyError as error:
                    raise ValueError(
                        f"{recording.name} lacks diagnostic context {context_index}"
                    ) from error
                rollouts.append(rollout)
                phases.append(phase)
                recordings.append(recording.name)

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=adapter,
    )
    load_seconds = monotonic() - started
    started = monotonic()
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=encoding_batch_size,
    )
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=encoding_batch_size,
    )
    encoding_seconds = monotonic() - started

    target_frames = targets[:, -1]
    token_targets = target_frames.flatten(start_dim=1, end_dim=-2)
    pooled_targets = token_targets.mean(dim=1)
    flattened_targets = targets[:, -1].flatten(start_dim=1)
    phase_probe = {
        "mean_pooled_tokens": phase_centroid_probe(
            pooled_targets,
            phases,
            recordings,
        ),
        "all_tokens": phase_centroid_probe(
            flattened_targets,
            phases,
            recordings,
        ),
    }

    action_phase_indices = [
        index for index, phase in enumerate(phases) if phase in ("retreat", "align")
    ]
    action_rollouts = [rollouts[index] for index in action_phase_indices]
    action_contexts = contexts[action_phase_indices]
    action_targets = targets[action_phase_indices]
    actions = rollout_action_tensor(action_rollouts)
    started = monotonic()
    global_energies, token_energies = _score_counterfactuals(
        model,
        action_contexts,
        action_targets,
        actions,
        batch_size=scoring_batch_size,
        device=device,
    )
    torch.cuda.synchronize(device)
    scoring_seconds = monotonic() - started
    action_phases = [phases[index] for index in action_phase_indices]
    signed_action_probe = {
        phase: _segment_report(
            global_energies,
            token_energies,
            [index for index, label in enumerate(action_phases) if label == phase],
        )
        for phase in ("retreat", "align")
    }
    return {
        "schema": "quantis.jepa_wm_insertion_latent_diagnostic.v1",
        "status": "diagnostic_only",
        "simulation_actions": 0,
        "training_steps": 0,
        "source_revision": os.environ.get("JEPA_WM_REVISION", "unknown"),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_fingerprint": artifact_fingerprint(checkpoint),
        "adapter": str(adapter.resolve()),
        "adapter_fingerprint": artifact_fingerprint(adapter),
        "recording_evidence": evidence,
        "samples_per_phase_per_recording": samples_per_phase,
        "context_indices": {
            phase: list(indices) for phase, indices in selected_by_phase.items()
        },
        "phase_representation_probe": phase_probe,
        "signed_action_probe": signed_action_probe,
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "scoring_seconds": round(scoring_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "limitations": [
            "Uses existing phase observations, not same-reset causal counterfactual clips.",
            "Centroid accuracy establishes decodability, not causal sufficiency.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe insertion phase latents and signed-X action sensitivity."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--held-out",
        nargs=2,
        action="append",
        metavar=("RECORDING", "SEED"),
        required=True,
    )
    parser.add_argument("--samples-per-phase", type=int, default=8)
    parser.add_argument("--encoding-batch-size", type=int, default=4)
    parser.add_argument("--scoring-batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    held_out = tuple(
        (Path(recording), int(seed)) for recording, seed in arguments.held_out
    )
    report = diagnose_insertion_latents(
        arguments.source,
        arguments.checkpoint,
        arguments.adapter,
        held_out,
        samples_per_phase=arguments.samples_per_phase,
        encoding_batch_size=arguments.encoding_batch_size,
        scoring_batch_size=arguments.scoring_batch_size,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
