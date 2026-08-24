"""Evaluate a frozen-JEPA inverse-action proposal on held-out recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch

from jepa_wm.action import (
    DEFAULT_ACTION_SELECTION_BOUNDS,
    ActionSelectionBounds,
    DroidAction,
)
from jepa_wm.frames import encode_clips
from jepa_wm.model import load_headless_model
from jepa_wm.planner import PlannerActionBounds
from jepa_wm.planner_readiness import evaluate_first_actions
from jepa_wm.proposal import ProposalInputs, load_action_proposal
from jepa_wm.proposal_evidence import ProposalGoalEvidence
from jepa_wm.trajectory import RolloutWindow, load_rollouts
from jepa_wm.training_artifact import artifact_fingerprint


REPORT_SCHEMA = "quantis.jepa_wm_action_proposal_evaluation.v1"


def evaluate_action_proposal(
    source: Path,
    checkpoint: Path,
    proposal_path: Path,
    recording: Path,
    *,
    camera: str,
    window: RolloutWindow,
    selection_bounds: ActionSelectionBounds = DEFAULT_ACTION_SELECTION_BOUNDS,
    planner_bounds: PlannerActionBounds = PlannerActionBounds(),
    encoding_batch_size: int = 4,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("action proposal evaluation requires CUDA")
    rollouts = window.select(
        load_rollouts(recording, camera=camera, bounds=selection_bounds)
    )
    device = torch.device("cuda", torch.cuda.current_device())
    model = load_headless_model(source, checkpoint, device=device)
    proposal, metadata = load_action_proposal(proposal_path, device=device)
    if metadata.camera != camera:
        raise ValueError("proposal camera does not match evaluation camera")
    if recording.name in metadata.training_recordings:
        raise ValueError("proposal evaluation recording was used for training")
    encoding_started = monotonic()
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
    encoding_seconds = monotonic() - encoding_started
    with torch.inference_mode():
        inputs = ProposalInputs.from_rollouts(
            rollouts,
            conditioning=proposal.conditioning,
            device=device,
            dtype=contexts.dtype,
        )
        predicted = (
            proposal(contexts.to(device), targets.to(device), inputs)
            .cpu()
            .numpy()
        )
    predicted = planner_bounds.clip(predicted)
    recorded = np.asarray(
        [[action.values for action in rollout.actions] for rollout in rollouts],
        dtype=np.float64,
    )
    summary = evaluate_first_actions(
        [DroidAction(tuple(actions[0])) for actions in recorded],
        [DroidAction(tuple(actions[0])) for actions in predicted],
    )
    results = [
        {
            **ProposalGoalEvidence.from_rollout(rollout).to_dict(),
            "recorded_actions": recorded_action.tolist(),
            "proposed_actions": predicted_action.tolist(),
            "sequence_mse": float(np.square(recorded_action - predicted_action).mean()),
            "first_action_cosine": decision.cosine,
            "first_action_gate": decision.to_dict(),
        }
        for rollout, recorded_action, predicted_action, decision in zip(
            rollouts, recorded, predicted, summary.decisions
        )
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "evaluated",
        "proposal": str(proposal_path.resolve()),
        "proposal_fingerprint": artifact_fingerprint(proposal_path),
        "recording": str(recording.resolve()),
        "camera": camera,
        "rollouts": len(rollouts),
        "window": window.to_dict(),
        "selection_bounds": selection_bounds.to_dict(),
        "conditioning": proposal.conditioning.to_dict(),
        "mean_sequence_mse": float(
            np.mean([result["sequence_mse"] for result in results])
        ),
        "mean_first_action_cosine": float(
            summary.mean_cosine
        ),
        **summary.to_dict(),
        "encoding_seconds": round(encoding_seconds, 3),
        "results": results,
    }
    output_dir = recording.resolve() / "jepa_wm"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / (
        f"{camera}_{proposal_path.stem}_proposal_eval_"
        f"{window.start_index:06d}_{window.count:03d}_{window.stride:03d}.json"
    )
    report["output_path"] = str(output)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--include-stationary",
        action="store_true",
        help="include zero-action rollouts in the requested evaluation window",
    )
    args = parser.parse_args()
    selection_bounds = (
        ActionSelectionBounds(minimum_action_norm=0.0)
        if args.include_stationary
        else DEFAULT_ACTION_SELECTION_BOUNDS
    )
    print(
        json.dumps(
            evaluate_action_proposal(
                args.source,
                args.checkpoint,
                args.proposal,
                args.recording,
                camera=args.camera,
                window=RolloutWindow(args.start_index, args.count, args.stride),
                selection_bounds=selection_bounds,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
