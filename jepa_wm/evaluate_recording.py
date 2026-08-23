"""Compare recorded DROID actions with a zero-action JEPA-WM baseline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    DEFAULT_ACTION_SELECTION_BOUNDS,
    ActionSelectionBounds,
)
from jepa_wm.model import load_headless_model
from jepa_wm.trajectory import RecordedTransition, TransitionWindow, load_transitions


MODEL_NAME = "jepa_wm_droid"


def _frame_batch(paths: Sequence[Path]) -> torch.Tensor:
    frames = []
    expected_size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if expected_size is None:
                expected_size = rgb.size
            elif rgb.size != expected_size:
                raise ValueError("recording frames must share one resolution")
            frames.append(np.asarray(rgb, dtype=np.uint8).copy())
    batch = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
    return batch.unsqueeze(1)


def _terminal_energy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    terminal = prediction[-1]
    target_frame = target[:, -1]
    reduction_dimensions = tuple(range(1, terminal.ndim))
    return (target_frame - terminal).pow(2).mean(dim=reduction_dimensions)


@dataclass(frozen=True)
class TransitionEvaluation:
    transition: RecordedTransition
    recorded_energy: float
    zero_energy: float

    @property
    def improvement_over_zero(self) -> float:
        return self.zero_energy - self.recorded_energy

    @property
    def recorded_action_wins(self) -> bool:
        return self.improvement_over_zero > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_index": self.transition.current_index,
            "next_index": self.transition.next_index,
            "action": list(self.transition.action.values),
            "recorded_action_energy": self.recorded_energy,
            "zero_action_energy": self.zero_energy,
            "improvement_over_zero": self.improvement_over_zero,
            "recorded_action_wins": self.recorded_action_wins,
        }


def _transition_evaluations(
    transitions: Sequence[RecordedTransition],
    recorded_energy: torch.Tensor,
    zero_energy: torch.Tensor,
) -> tuple[TransitionEvaluation, ...]:
    return tuple(
        TransitionEvaluation(transition, float(recorded), float(zero))
        for transition, recorded, zero in zip(
            transitions,
            recorded_energy.tolist(),
            zero_energy.tolist(),
        )
    )


def evaluate_recording(
    source: Path,
    checkpoint: Path,
    recording: Path,
    *,
    camera: str,
    window: TransitionWindow,
    bounds: ActionSelectionBounds,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM recording evaluation requires CUDA")
    transitions = load_transitions(
        recording,
        camera=camera,
        window=window,
        bounds=bounds,
    )
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    load_started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    load_seconds = monotonic() - load_started
    current_frames = _frame_batch(
        [transition.current_frame for transition in transitions]
    )
    next_frames = _frame_batch([transition.next_frame for transition in transitions])
    actions = torch.tensor(
        [transition.action.values for transition in transitions],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    evaluation_started = monotonic()
    with torch.inference_mode():
        current_latents = model.encode(current_frames)
        target_latents = model.encode(next_frames)
        recorded_prediction = model.unroll(current_latents, actions)
        zero_prediction = model.unroll(current_latents, torch.zeros_like(actions))
        recorded_energy = _terminal_energy(recorded_prediction, target_latents)
        zero_energy = _terminal_energy(zero_prediction, target_latents)
    torch.cuda.synchronize(device)
    evaluation_seconds = monotonic() - evaluation_started

    evaluations = _transition_evaluations(
        transitions,
        recorded_energy,
        zero_energy,
    )
    improvements = [evaluation.improvement_over_zero for evaluation in evaluations]
    wins = sum(evaluation.recorded_action_wins for evaluation in evaluations)
    report = {
        "status": "evaluated",
        "model": MODEL_NAME,
        "source_revision": os.environ.get("JEPA_WM_REVISION", "unknown"),
        "recording": str(recording.resolve()),
        "camera": camera,
        "transitions": len(transitions),
        "transition_window": window.to_dict(),
        "action_selection": bounds.to_dict(),
        "action_format": ACTION_RECORDING_CONTRACT.format,
        "objective": "terminal_latent_l2",
        "mean_recorded_action_energy": float(recorded_energy.mean().item()),
        "mean_zero_action_energy": float(zero_energy.mean().item()),
        "mean_improvement_over_zero": float(np.mean(improvements)),
        "recorded_action_win_rate": wins / len(transitions),
        "load_seconds": round(load_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "results": [evaluation.to_dict() for evaluation in evaluations],
    }
    output_dir = recording.resolve() / "jepa_wm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / window.report_name(camera)
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--minimum-action-norm",
        type=float,
        default=DEFAULT_ACTION_SELECTION_BOUNDS.minimum_action_norm,
    )
    parser.add_argument(
        "--maximum-pose-action-norm",
        type=float,
        default=DEFAULT_ACTION_SELECTION_BOUNDS.maximum_pose_action_norm,
    )
    parser.add_argument(
        "--maximum-gripper-action",
        type=float,
        default=DEFAULT_ACTION_SELECTION_BOUNDS.maximum_gripper_action,
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_recording(
                args.source,
                args.checkpoint,
                args.recording,
                camera=args.camera,
                window=TransitionWindow(
                    start_index=args.start_index,
                    count=args.count,
                    stride=args.stride,
                ),
                bounds=ActionSelectionBounds(
                    minimum_action_norm=args.minimum_action_norm,
                    maximum_pose_action_norm=args.maximum_pose_action_norm,
                    maximum_gripper_action=args.maximum_gripper_action,
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
