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
    ActionAdapterMetadata,
    action_adapter_parameters,
    save_action_adapter,
)
from jepa_wm.adapter_metadata import adapter_report_path
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import video_batch
from jepa_wm.model import load_headless_model
from jepa_wm.rollout_scoring import (
    rollout_action_tensor,
    score_recorded_against_zero,
)
from jepa_wm.trajectory import RecordedRollout, load_rollouts


TRAINING_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)


@dataclass(frozen=True)
class AdaptationConfig:
    steps: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-3
    contrastive_weight: float = 1.0
    contrastive_margin: float = 1e-3
    encoding_batch_size: int = 4
    seed: int = 234

    def __post_init__(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.encoding_batch_size <= 0:
            raise ValueError("training steps and batch sizes must be positive")
        if self.learning_rate <= 0 or self.contrastive_margin < 0:
            raise ValueError("learning rate must be positive and margin non-negative")
        if self.contrastive_weight < 0:
            raise ValueError("contrastive weight must be non-negative")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "contrastive_weight": self.contrastive_weight,
            "contrastive_margin": self.contrastive_margin,
            "encoding_batch_size": self.encoding_batch_size,
            "seed": self.seed,
        }


def _encode_clips(
    model: Any,
    clips: Sequence[Sequence[Path]],
    *,
    batch_size: int,
) -> torch.Tensor:
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(clips), batch_size):
            frames = video_batch(clips[start : start + batch_size])
            encoded.append(model.encode(frames).cpu())
    return torch.cat(encoded)


def _load_training_rollouts(
    recordings: Sequence[Path],
    camera: str,
) -> tuple[RecordedRollout, ...]:
    rollouts = tuple(
        rollout
        for recording in recordings
        for rollout in load_rollouts(
            recording,
            camera=camera,
            bounds=TRAINING_BOUNDS,
        )
    )
    if not rollouts:
        raise ValueError("training recordings contain no bounded rollouts")
    return rollouts


def adapt_recordings(
    source: Path,
    checkpoint: Path,
    recordings: Sequence[Path],
    output: Path,
    *,
    camera: str,
    config: AdaptationConfig,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM adaptation requires CUDA")
    if not recordings:
        raise ValueError("at least one training recording is required")

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    rollouts = _load_training_rollouts(recordings, camera)
    load_started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    load_seconds = monotonic() - load_started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter_parameters = action_adapter_parameters(model)
    for parameter in adapter_parameters:
        parameter.requires_grad_(True)

    encoding_started = monotonic()
    context_latents = _encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    )
    target_latents = _encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    )
    encoding_seconds = monotonic() - encoding_started
    actions = rollout_action_tensor(rollouts)

    optimizer = torch.optim.AdamW(adapter_parameters, lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses = []
    training_started = monotonic()
    model.eval()
    for _ in range(config.steps):
        indices = torch.randint(
            len(rollouts),
            (config.batch_size,),
            generator=generator,
        )
        context = context_latents[indices].to(device)
        target = target_latents[indices].to(device)
        action_batch = actions[:, indices].to(device)

        energies = score_recorded_against_zero(
            model,
            context,
            target,
            action_batch,
        )
        contrastive = torch.relu(
            config.contrastive_margin + energies.recorded - energies.zero
        )
        loss = energies.recorded.mean() + config.contrastive_weight * contrastive.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    training_seconds = monotonic() - training_started

    metadata = ActionAdapterMetadata(
        base_model=MODEL_ID,
        source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
        camera=camera,
        training_recordings=tuple(recording.name for recording in recordings),
        training_steps=config.steps,
    )
    save_action_adapter(model, output, metadata)
    report = {
        "status": "adapted",
        "adapter": str(output.resolve()),
        "metadata": metadata.to_dict(),
        "config": config.to_dict(),
        "rollouts": len(rollouts),
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
    report_path = adapter_report_path(output)
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
    parser.add_argument("--steps", type=int, default=AdaptationConfig.steps)
    parser.add_argument("--batch-size", type=int, default=AdaptationConfig.batch_size)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=AdaptationConfig.learning_rate,
    )
    args = parser.parse_args()
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
                    learning_rate=args.learning_rate,
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
