"""Compare recorded DROID rollouts with a zero-action JEPA-WM baseline."""

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

from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    DEFAULT_ACTION_SELECTION_BOUNDS,
    ActionSelectionBounds,
)
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import video_batch
from jepa_wm.model import load_headless_model
from jepa_wm.readiness import ActionControlGate
from jepa_wm.rollout_scoring import (
    rollout_action_tensor,
    score_recorded_against_zero,
)
from jepa_wm.trajectory import (
    DROID_ROLLOUT_PROTOCOL,
    RecordedRollout,
    RolloutProtocol,
    RolloutWindow,
    load_rollouts,
)


@dataclass(frozen=True)
class RolloutEvaluation:
    rollout: RecordedRollout
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
            "context_indices": [frame.index for frame in self.rollout.context],
            "target_index": self.rollout.target.index,
            "actions": [list(action.values) for action in self.rollout.actions],
            "recorded_action_energy": self.recorded_energy,
            "zero_action_energy": self.zero_energy,
            "improvement_over_zero": self.improvement_over_zero,
            "recorded_action_wins": self.recorded_action_wins,
        }


def _rollout_evaluations(
    rollouts: Sequence[RecordedRollout],
    recorded_energy: torch.Tensor,
    zero_energy: torch.Tensor,
) -> tuple[RolloutEvaluation, ...]:
    return tuple(
        RolloutEvaluation(rollout, float(recorded), float(zero))
        for rollout, recorded, zero in zip(
            rollouts,
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
    window: RolloutWindow,
    bounds: ActionSelectionBounds,
    protocol: RolloutProtocol = DROID_ROLLOUT_PROTOCOL,
    adapter: Path | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM recording evaluation requires CUDA")
    rollouts = window.select(
        load_rollouts(
            recording,
            camera=camera,
            bounds=bounds,
            protocol=protocol,
        )
    )
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    load_started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=adapter,
    )
    load_seconds = monotonic() - load_started
    context_frames = video_batch([rollout.context_paths for rollout in rollouts])
    target_frames = video_batch([rollout.target_clip for rollout in rollouts])
    actions = rollout_action_tensor(rollouts, device=device)

    evaluation_started = monotonic()
    with torch.inference_mode():
        context_latents = model.encode(context_frames)
        target_latents = model.encode(target_frames)
        energies = score_recorded_against_zero(
            model,
            context_latents,
            target_latents,
            actions,
        )
    torch.cuda.synchronize(device)
    evaluation_seconds = monotonic() - evaluation_started

    evaluations = _rollout_evaluations(
        rollouts,
        energies.recorded,
        energies.zero,
    )
    improvements = [evaluation.improvement_over_zero for evaluation in evaluations]
    wins = sum(evaluation.recorded_action_wins for evaluation in evaluations)
    mean_improvement = float(np.mean(improvements))
    win_rate = wins / len(rollouts)
    control_gate = ActionControlGate().evaluate(
        mean_improvement_over_zero=mean_improvement,
        recorded_action_win_rate=win_rate,
    )
    report = {
        "status": "evaluated",
        "model": MODEL_ID,
        "source_revision": os.environ.get("JEPA_WM_REVISION", "unknown"),
        "adapter": str(adapter.resolve()) if adapter is not None else None,
        "recording": str(recording.resolve()),
        "camera": camera,
        "rollouts": len(rollouts),
        "rollout_protocol": protocol.to_dict(),
        "rollout_window": window.to_dict(),
        "action_selection": bounds.to_dict(),
        "action_format": ACTION_RECORDING_CONTRACT.format,
        "objective": "terminal_latent_l2",
        "mean_recorded_action_energy": float(energies.recorded.mean().item()),
        "mean_zero_action_energy": float(energies.zero.mean().item()),
        "mean_improvement_over_zero": mean_improvement,
        "recorded_action_win_rate": win_rate,
        "control_gate": control_gate.to_dict(),
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
    report_camera = f"{camera}_adapted" if adapter is not None else camera
    output_path = output_dir / window.report_name(report_camera)
    report["output_path"] = str(output_path)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--adapter", type=Path)
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
                window=RolloutWindow(
                    start_index=args.start_index,
                    count=args.count,
                    stride=args.stride,
                ),
                bounds=ActionSelectionBounds(
                    minimum_action_norm=args.minimum_action_norm,
                    maximum_pose_action_norm=args.maximum_pose_action_norm,
                    maximum_gripper_action=args.maximum_gripper_action,
                ),
                adapter=args.adapter,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
