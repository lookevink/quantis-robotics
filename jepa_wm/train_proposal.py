"""Train a small inverse-action head on frozen JEPA context and goal features."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import encode_clips
from jepa_wm.model import load_headless_model
from jepa_wm.proposal import (
    ActionProposalNetwork,
    action_normalization,
    save_action_proposal,
)
from jepa_wm.proprioception import DroidValueNormalization
from jepa_wm.trajectory import RecordedRollout, RolloutWindow, load_rollouts
from jepa_wm.training_artifact import TrainingArtifactMetadata, training_report_path


TRAINING_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)


@dataclass(frozen=True)
class TrainingRecordingSelection:
    recording: str
    context_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording": self.recording,
            "context_indices": list(self.context_indices),
        }


@dataclass(frozen=True)
class ProposalTrainingConfig:
    steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dimension: int = 128
    encoding_batch_size: int = 4
    seed: int = 234

    def __post_init__(self) -> None:
        dimensions = (
            self.steps,
            self.batch_size,
            self.hidden_dimension,
            self.encoding_batch_size,
        )
        if any(value <= 0 for value in dimensions) or self.seed < 0:
            raise ValueError("proposal training dimensions must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("proposal optimizer values are invalid")


def _training_rollouts(
    recordings: Sequence[Path], camera: str, window: RolloutWindow | None
) -> tuple[tuple[RecordedRollout, ...], tuple[TrainingRecordingSelection, ...]]:
    def selected(recording: Path) -> tuple[RecordedRollout, ...]:
        recording_rollouts = load_rollouts(
            recording,
            camera=camera,
            bounds=TRAINING_BOUNDS,
        )
        return (
            window.select(recording_rollouts)
            if window is not None
            else recording_rollouts
        )

    rollouts = []
    selections = []
    for recording in recordings:
        recording_rollouts = selected(recording)
        rollouts.extend(recording_rollouts)
        selections.append(
            TrainingRecordingSelection(
                recording.name,
                tuple(rollout.context[0].index for rollout in recording_rollouts),
            )
        )
    if not rollouts:
        raise ValueError("proposal training recordings contain no rollouts")
    return tuple(rollouts), tuple(selections)


def train_action_proposal(
    source: Path,
    checkpoint: Path,
    recordings: Sequence[Path],
    output: Path,
    *,
    camera: str,
    config: ProposalTrainingConfig,
    window: RolloutWindow | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("action proposal training requires CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    rollouts, recording_selections = _training_rollouts(recordings, camera, window)
    model = load_headless_model(source, checkpoint, device=device)
    encoding_started = monotonic()
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    )
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    )
    encoding_seconds = monotonic() - encoding_started
    action_sequences = np.asarray(
        [[action.values for action in rollout.actions] for rollout in rollouts],
        dtype=np.float32,
    )
    action_mean, action_std = action_normalization(action_sequences)
    poses = np.asarray(
        [rollout.context_pose.values for rollout in rollouts],
        dtype=np.float32,
    )
    pose_normalization = DroidValueNormalization.from_samples(poses)
    pose_tensor = torch.from_numpy(poses)
    previous_actions = np.asarray(
        [rollout.previous_action.values for rollout in rollouts],
        dtype=np.float32,
    )
    previous_action_normalization = DroidValueNormalization.from_samples(
        previous_actions
    )
    previous_action_tensor = torch.from_numpy(previous_actions)
    actions = torch.from_numpy(action_sequences)
    standardized_actions = (actions - action_mean) / action_std
    feature_dimension = contexts.shape[-1]
    proposal = ActionProposalNetwork(
        feature_dimension,
        action_sequences.shape[1],
        config.hidden_dimension,
        action_mean,
        action_std,
        pose_normalization=pose_normalization,
        previous_action_normalization=previous_action_normalization,
    ).to(device)
    optimizer = torch.optim.AdamW(
        proposal.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses = []
    training_started = monotonic()
    proposal.train()
    for _ in range(config.steps):
        indices = torch.randint(
            len(rollouts),
            (config.batch_size,),
            generator=generator,
        )
        predicted = proposal.standardized_actions(
            contexts[indices].to(device),
            targets[indices].to(device),
            pose_tensor[indices].to(device),
            previous_action_tensor[indices].to(device),
        )
        loss = torch.nn.functional.mse_loss(
            predicted,
            standardized_actions[indices].to(device),
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
        training_recordings=tuple(recording.name for recording in recordings),
        training_steps=config.steps,
    )
    save_action_proposal(proposal, output, metadata)
    report = {
        "status": "trained",
        "proposal": str(output.resolve()),
        "metadata": metadata.to_dict(),
        "config": asdict(config),
        "rollouts": len(rollouts),
        "window": window.to_dict() if window is not None else None,
        "selection_bounds": TRAINING_BOUNDS.to_dict(),
        "recording_selections": [
            selection.to_dict() for selection in recording_selections
        ],
        "trainable_parameters": sum(
            parameter.numel() for parameter in proposal.parameters()
        ),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": float(np.min(losses)),
        "encoding_seconds": round(encoding_seconds, 3),
        "training_seconds": round(training_seconds, 3),
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
    parser.add_argument("--steps", type=int, default=ProposalTrainingConfig.steps)
    parser.add_argument(
        "--hidden-dimension",
        type=int,
        default=ProposalTrainingConfig.hidden_dimension,
    )
    parser.add_argument(
        "--learning-rate", type=float, default=ProposalTrainingConfig.learning_rate
    )
    parser.add_argument(
        "--weight-decay", type=float, default=ProposalTrainingConfig.weight_decay
    )
    parser.add_argument("--seed", type=int, default=ProposalTrainingConfig.seed)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--count", type=int)
    parser.add_argument("--stride", type=int)
    args = parser.parse_args()
    window_values = (args.start_index, args.count, args.stride)
    if any(value is not None for value in window_values) and not all(
        value is not None for value in window_values
    ):
        parser.error("start-index, count, and stride must be provided together")
    window = (
        RolloutWindow(*window_values)
        if all(value is not None for value in window_values)
        else None
    )
    print(
        json.dumps(
            train_action_proposal(
                args.source,
                args.checkpoint,
                args.recording,
                args.output,
                camera=args.camera,
                config=ProposalTrainingConfig(
                    steps=args.steps,
                    hidden_dimension=args.hidden_dimension,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                ),
                window=window,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
