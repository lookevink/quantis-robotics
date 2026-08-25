"""Fit the DROID predictor's small action encoder to Isaac trajectories."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.adapter import (
    ActionAdapterContract,
    LoadedActionAdapter,
    action_adapter_parameters,
    save_action_adapter,
)
from jepa_wm.contract import MODEL_ID
from jepa_wm.candidate_negatives import (
    CandidateMiningConfig,
    mine_lowest_energy_candidates,
    sample_local_candidates,
)
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_adapter_profile import InsertionAdapterProfile
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.model import load_headless_model
from jepa_wm.rollout_scoring import (
    rollout_action_tensor,
    score_actions,
    score_recorded_against_mismatched,
)
from jepa_wm.rollout_training import RolloutTrainingSelection
from jepa_wm.trajectory import RecordedRollout, RolloutWindow
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    artifact_fingerprint,
    training_report_path,
    training_configuration_fingerprint,
)
from sim.exploration import DatasetSplit


TRAINING_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)


@dataclass(frozen=True)
class ContrastiveTermConfig:
    weight: float = 1.0
    margin: float = 1e-3

    def __post_init__(self) -> None:
        if self.weight < 0.0 or self.margin < 0.0:
            raise ValueError("contrastive weight and margin must be non-negative")

    def loss(self, recorded: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        return self.weight * torch.relu(self.margin + recorded - negative).mean()

    def to_dict(self) -> dict[str, float]:
        return {"weight": self.weight, "margin": self.margin}


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-3
    zero_negative: ContrastiveTermConfig = ContrastiveTermConfig()
    mismatched_negative: ContrastiveTermConfig = ContrastiveTermConfig()
    candidate_negative: ContrastiveTermConfig = ContrastiveTermConfig()
    candidate_mining: CandidateMiningConfig = CandidateMiningConfig()
    encoding_batch_size: int = 4
    seed: int = 234
    initial_adapter: ArtifactIdentity | None = None

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.encoding_batch_size <= 0:
            raise ValueError("training steps and batch sizes must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "zero_negative": self.zero_negative.to_dict(),
            "mismatched_negative": self.mismatched_negative.to_dict(),
            "candidate_negative": self.candidate_negative.to_dict(),
            "candidate_mining": self.candidate_mining.to_dict(),
            "encoding_batch_size": self.encoding_batch_size,
            "seed": self.seed,
        }
        if self.initial_adapter is not None:
            payload["initial_adapter"] = self.initial_adapter.to_dict()
        return payload


class ShuffledEpochSampler:
    """Deterministic without-replacement rollout coverage across epochs."""

    def __init__(self, rollout_count: int, batch_size: int, seed: int) -> None:
        if rollout_count <= 0 or batch_size <= 0:
            raise ValueError("rollout count and sampling batch size must be positive")
        self.rollout_count = rollout_count
        self.batch_size = batch_size
        self.seed = seed
        self.samples_drawn = 0
        self._generator = torch.Generator(device="cpu").manual_seed(seed)
        self._order = torch.randperm(rollout_count, generator=self._generator)
        self._cursor = 0

    def next_indices(self) -> torch.Tensor:
        chunks = []
        remaining = self.batch_size
        while remaining:
            available = self.rollout_count - self._cursor
            take = min(remaining, available)
            chunks.append(self._order[self._cursor : self._cursor + take])
            self._cursor += take
            remaining -= take
            if self._cursor == self.rollout_count:
                self._order = torch.randperm(
                    self.rollout_count,
                    generator=self._generator,
                )
                self._cursor = 0
        self.samples_drawn += self.batch_size
        return torch.cat(chunks)

    def to_dict(self) -> dict[str, int | str]:
        return {
            "strategy": "seeded_shuffled_epochs",
            "rollouts_per_epoch": self.rollout_count,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "samples_drawn": self.samples_drawn,
            "complete_epochs": self.samples_drawn // self.rollout_count,
        }


def validated_training_recordings(
    recordings: Sequence[Path],
) -> tuple[DomainRecording, ...]:
    validated = tuple(
        DomainRecording.from_path(path, expected_split=DatasetSplit.TRAIN)
        for path in recordings
    )
    if not validated:
        raise ValueError("at least one training recording is required")
    names = {recording.name for recording in validated}
    seeds = {recording.seed for recording in validated}
    if len(names) != len(validated) or len(seeds) != len(validated):
        raise ValueError("adaptation recordings require unique identities and seeds")
    return validated


def mismatched_negative_candidates(
    rollouts: Sequence[RecordedRollout],
) -> tuple[tuple[int, ...], ...]:
    """Eligible negative indices with different task positions and actions."""

    action_sequences = tuple(
        tuple(action.values for action in rollout.actions) for rollout in rollouts
    )
    candidates = tuple(
        tuple(
            candidate_index
            for candidate_index, candidate in enumerate(rollouts)
            if candidate.task_context_index != rollout.task_context_index
            and action_sequences[candidate_index] != action_sequences[rollout_index]
        )
        for rollout_index, rollout in enumerate(rollouts)
    )
    if not candidates or any(not eligible for eligible in candidates):
        raise ValueError(
            "each adaptation rollout requires a different-context action negative"
        )
    return candidates


def _sample_mismatched_indices(
    candidates: tuple[tuple[int, ...], ...],
    positive_indices: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    selected = []
    for positive_index in positive_indices.tolist():
        eligible = candidates[positive_index]
        ordinal = int(torch.randint(len(eligible), (1,), generator=generator).item())
        selected.append(eligible[ordinal])
    return torch.tensor(selected, dtype=torch.long)


def adapt_recordings(
    source: Path,
    checkpoint: Path,
    recordings: Sequence[Path],
    output: Path,
    *,
    camera: str,
    config: AdaptationConfig,
    window: RolloutWindow | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM adaptation requires CUDA")
    training_recordings = validated_training_recordings(recordings)

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    training_selection = RolloutTrainingSelection.load(
        tuple(recording.path for recording in training_recordings),
        camera=camera,
        bounds=TRAINING_BOUNDS,
        window=window,
    )
    rollouts = training_selection.rollouts
    if len(rollouts) < 2:
        raise ValueError("adaptation requires at least two bounded rollouts")
    load_started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    if config.initial_adapter is not None:
        loaded_initial = LoadedActionAdapter.load(
            config.initial_adapter.path,
            expected_identity=config.initial_adapter,
        )
        initial_contract = loaded_initial.contract
        expected_recordings = tuple(
            recording.name for recording in training_recordings
        )
        if (
            initial_contract.metadata.camera != camera
            or initial_contract.metadata.training_recordings != expected_recordings
            or initial_contract.metadata.source_revision
            != os.environ.get("JEPA_WM_REVISION", "unknown")
            or initial_contract.training_selection_fingerprint
            != training_selection.fingerprint
        ):
            raise ValueError("initial adapter does not match the training corpus")
        loaded_initial.apply(
            model,
            expected_source_revision=initial_contract.metadata.source_revision,
        )
    load_seconds = monotonic() - load_started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter_parameters = action_adapter_parameters(model)
    for parameter in adapter_parameters:
        parameter.requires_grad_(True)

    encoding_started = monotonic()
    context_latents = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    )
    target_latents = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    )
    encoding_seconds = monotonic() - encoding_started
    actions = rollout_action_tensor(rollouts)
    goal_actions = torch.tensor(
        [rollout.goal_action.values for rollout in rollouts],
        dtype=actions.dtype,
    )
    negative_candidates = mismatched_negative_candidates(rollouts)

    optimizer = torch.optim.AdamW(adapter_parameters, lr=config.learning_rate)
    sampler = ShuffledEpochSampler(
        len(rollouts),
        config.batch_size,
        config.seed,
    )
    mismatch_generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    candidate_generator = torch.Generator(device=device).manual_seed(config.seed)
    losses = []
    training_started = monotonic()
    model.eval()
    for _ in range(config.steps):
        indices = sampler.next_indices()
        context = context_latents[indices].to(device)
        target = target_latents[indices].to(device)
        action_batch = actions[:, indices].to(device)
        negative_indices = _sample_mismatched_indices(
            negative_candidates,
            indices,
            mismatch_generator,
        )
        mismatched_negative_batch = actions[:, negative_indices].to(device)
        local_candidates = sample_local_candidates(
            action_batch,
            config=config.candidate_mining,
            generator=candidate_generator,
            goal_actions=goal_actions[indices].to(device),
        )
        candidate_negative_actions = mine_lowest_energy_candidates(
            model,
            context,
            target,
            local_candidates,
            scoring_batch_size=config.candidate_mining.scoring_batch_size,
        )
        energies = score_recorded_against_mismatched(
            model,
            context,
            target,
            action_batch,
            mismatched_negative_batch,
        )
        candidate_negative_energy = score_actions(
            model,
            context,
            target,
            candidate_negative_actions,
        )
        loss = (
            energies.recorded.mean()
            + config.zero_negative.loss(energies.recorded, energies.zero)
            + config.mismatched_negative.loss(
                energies.recorded, energies.mismatched_negative
            )
            + config.candidate_negative.loss(
                energies.recorded, candidate_negative_energy
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    training_seconds = monotonic() - training_started

    metadata = TrainingArtifactMetadata(
        base_model=MODEL_ID,
        source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
        camera=camera,
        training_recordings=tuple(recording.name for recording in training_recordings),
        training_steps=config.steps,
    )
    selection_payload = training_selection.to_dict()
    selection_fingerprint = training_selection.fingerprint
    config_payload = config.to_dict()
    config_fingerprint = training_configuration_fingerprint(config_payload)
    save_action_adapter(
        model,
        output,
        ActionAdapterContract.current(
            metadata,
            training_selection_fingerprint=selection_fingerprint,
            training_config_fingerprint=config_fingerprint,
        ),
    )
    report = {
        "status": "adapted",
        "adapter": str(output.resolve()),
        "adapter_fingerprint": artifact_fingerprint(output),
        "metadata": metadata.to_dict(),
        "config": config_payload,
        "training_config_fingerprint": config_fingerprint,
        "sampling": sampler.to_dict(),
        **selection_payload,
        "training_selection_fingerprint": selection_fingerprint,
        "trainable_parameters": sum(
            parameter.numel() for parameter in adapter_parameters
        ),
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
    report_path = training_report_path(output)
    report["report"] = str(report_path.resolve())
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recording", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--steps", type=int, default=AdaptationConfig.steps)
    parser.add_argument("--batch-size", type=int, default=AdaptationConfig.batch_size)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--candidate-minimum-goal-cosine", type=float)
    parser.add_argument(
        "--candidate-profile",
        choices=tuple(profile.value for profile in InsertionAdapterProfile),
    )
    args = parser.parse_args()
    window_values = (args.start_index, args.count, args.stride)
    if any(value is not None for value in window_values) and not all(
        value is not None for value in window_values
    ):
        parser.error("start-index, count, and stride must be supplied together")
    if (
        args.candidate_profile is not None
        and args.candidate_minimum_goal_cosine is not None
    ):
        parser.error("candidate profile and explicit goal cosine are mutually exclusive")
    profile = (
        InsertionAdapterProfile(args.candidate_profile)
        if args.candidate_profile is not None
        else None
    )
    if profile is not None and args.learning_rate is not None:
        parser.error("candidate profile and explicit learning rate are mutually exclusive")
    candidate_mining = (
        profile.descriptor.candidate_mining_config()
        if profile is not None
        else CandidateMiningConfig(
            minimum_goal_cosine=args.candidate_minimum_goal_cosine,
        )
    )
    initial_adapter = (
        ArtifactIdentity.from_artifact(
            profile.descriptor.initial_adapter_path(args.output, args.steps)
        )
        if profile is not None and profile.descriptor.initial_profile is not None
        else None
    )
    print(
        json.dumps(
            adapt_recordings(
                args.source,
                args.checkpoint,
                args.recording,
                args.output,
                camera=args.camera,
                config=AdaptationConfig(
                    steps=args.steps,
                    batch_size=args.batch_size,
                    learning_rate=(
                        profile.descriptor.learning_rate
                        if profile is not None
                        else (
                            args.learning_rate
                            if args.learning_rate is not None
                            else AdaptationConfig.learning_rate
                        )
                    ),
                    candidate_mining=candidate_mining,
                    initial_adapter=initial_adapter,
                ),
                window=(
                    RolloutWindow(args.start_index, args.count, args.stride)
                    if all(value is not None for value in window_values)
                    else None
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
