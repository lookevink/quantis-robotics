"""TRAIN-only alignment diagnosis for a frozen insertion proposal."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import fsum
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch

from jepa_wm.action import ActionSelectionBounds, DroidAction
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_planner import INSERTION_PLANNER_PROFILE
from jepa_wm.insertion_proposal_readiness import validate_insertion_proposal
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.model import load_headless_model
from jepa_wm.persistence import write_json_atomic
from jepa_wm.planner import PlannerActionBounds
from jepa_wm.planner_readiness import FirstActionGate
from jepa_wm.proposal import ProposalInputs, load_action_proposal
from jepa_wm.task_windows import INSERTION_PROPOSAL_WINDOW
from jepa_wm.trajectory import load_rollouts
from sim.exploration import DatasetSplit


REPORT_SCHEMA = "quantis.jepa_wm_insertion_proposal_training_diagnostic.v1"


@dataclass(frozen=True)
class ContextAlignmentDiagnostic:
    context_index: int
    samples: int
    active_goal_samples: int
    mean_goal_cosine: float | None
    minimum_goal_cosine: float | None
    goal_alignment_pass_rate: float | None
    mean_recorded_cosine: float
    recorded_direction_pass_rate: float
    mean_sequence_mse: float

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "context_index": self.context_index,
            "samples": self.samples,
            "active_goal_samples": self.active_goal_samples,
            "mean_goal_cosine": self.mean_goal_cosine,
            "minimum_goal_cosine": self.minimum_goal_cosine,
            "goal_alignment_pass_rate": self.goal_alignment_pass_rate,
            "mean_recorded_cosine": self.mean_recorded_cosine,
            "recorded_direction_pass_rate": self.recorded_direction_pass_rate,
            "mean_sequence_mse": self.mean_sequence_mse,
        }


def diagnose_insertion_proposal_training(
    source: Path,
    checkpoint: Path,
    proposal_path: Path,
    recording_root: Path,
    output: Path,
    *,
    encoding_batch_size: int = 4,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("insertion proposal diagnosis requires CUDA")
    if encoding_batch_size <= 0:
        raise ValueError("encoding batch size must be positive")
    proposal_evidence = validate_insertion_proposal(proposal_path.resolve())
    metadata = proposal_evidence.metadata
    recordings = tuple(
        (recording_root.resolve() / name).resolve()
        for name in metadata.training_recordings
    )
    for recording in recordings:
        ContactInsertionEvidence.from_recording(
            recording,
            expected_split=DatasetSplit.TRAIN.value,
        )
    rollouts = tuple(
        rollout
        for recording in recordings
        for rollout in INSERTION_PROPOSAL_WINDOW.select(
            load_rollouts(
                recording,
                camera="wrist",
                bounds=ActionSelectionBounds(minimum_action_norm=0.0),
            )
        )
    )
    device = torch.device("cuda", torch.cuda.current_device())
    model = load_headless_model(source, checkpoint, device=device)
    proposal, loaded_metadata = load_action_proposal(
        proposal_evidence.identity.path,
        device=device,
    )
    if loaded_metadata != metadata:
        raise ValueError("proposal checkpoint metadata changed during diagnosis")
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
    inputs = ProposalInputs.from_rollouts(
        rollouts,
        conditioning=proposal.conditioning,
        device=device,
        dtype=contexts.dtype,
    )
    with torch.inference_mode():
        predicted = proposal(
            contexts.to(device),
            targets.to(device),
            inputs,
        ).cpu().numpy()
    predicted = PlannerActionBounds().clip(predicted)
    goal_policy = INSERTION_PLANNER_PROFILE.task_policy.goal_action_alignment
    if goal_policy is None:
        raise ValueError("insertion proposal diagnosis requires goal alignment")
    first_action_gate = FirstActionGate(
        INSERTION_PLANNER_PROFILE.task_policy.first_action_thresholds
    )
    raw = []
    for rollout, proposed in zip(rollouts, predicted):
        recorded = np.asarray(
            [action.values for action in rollout.actions],
            dtype=np.float64,
        )
        recorded_direction = first_action_gate.evaluate(
            rollout.actions[0],
            DroidAction(tuple(proposed[0])),
        )
        goal_is_active = np.linalg.norm(rollout.goal_action.values) > 1e-12
        goal = (
            goal_policy.evaluate(
                DroidAction(tuple(proposed[0])),
                rollout.goal_action,
            )
            if goal_is_active
            else None
        )
        raw.append(
            {
                "context_index": rollout.context[0].index,
                "goal_cosine": goal.cosine if goal is not None else None,
                "goal_passed": goal.passed if goal is not None else None,
                "recorded_cosine": recorded_direction.cosine,
                "recorded_passed": recorded_direction.passed,
                "sequence_mse": float(np.square(recorded - proposed).mean()),
            }
        )
    contexts_evaluated = []
    for context_index in INSERTION_PROPOSAL_WINDOW.context_indices:
        items = tuple(item for item in raw if item["context_index"] == context_index)
        if len(items) != len(recordings):
            raise ValueError("proposal diagnostic context coverage is incomplete")
        active_goal_items = tuple(
            item for item in items if item["goal_cosine"] is not None
        )
        contexts_evaluated.append(
            ContextAlignmentDiagnostic(
                context_index=context_index,
                samples=len(items),
                active_goal_samples=len(active_goal_items),
                mean_goal_cosine=(
                    fsum(item["goal_cosine"] for item in active_goal_items)
                    / len(active_goal_items)
                    if active_goal_items
                    else None
                ),
                minimum_goal_cosine=(
                    min(item["goal_cosine"] for item in active_goal_items)
                    if active_goal_items
                    else None
                ),
                goal_alignment_pass_rate=(
                    sum(item["goal_passed"] for item in active_goal_items)
                    / len(active_goal_items)
                    if active_goal_items
                    else None
                ),
                mean_recorded_cosine=fsum(
                    item["recorded_cosine"] for item in items
                )
                / len(items),
                recorded_direction_pass_rate=sum(
                    item["recorded_passed"] for item in items
                )
                / len(items),
                mean_sequence_mse=fsum(item["sequence_mse"] for item in items)
                / len(items),
            )
        )
    active_goal_results = tuple(
        item for item in raw if item["goal_cosine"] is not None
    )
    payload = {
        "schema": REPORT_SCHEMA,
        "scope": "TRAIN-only insertion proposal alignment; no control authority",
        "proposal": proposal_evidence.identity.to_dict(),
        "training_recordings": list(metadata.training_recordings),
        "window": INSERTION_PROPOSAL_WINDOW.to_dict(),
        "rollouts": len(rollouts),
        "goal_alignment_threshold": goal_policy.minimum_cosine,
        "active_goal_rollouts": len(active_goal_results),
        "goal_alignment_pass_rate": sum(
            item["goal_passed"] for item in active_goal_results
        )
        / len(active_goal_results),
        "recorded_direction_pass_rate": sum(
            item["recorded_passed"] for item in raw
        )
        / len(raw),
        "mean_sequence_mse": fsum(item["sequence_mse"] for item in raw)
        / len(raw),
        "encoding_seconds": round(encoding_seconds, 3),
        "contexts": [item.to_dict() for item in contexts_evaluated],
        "live_insertion_authority_granted": False,
    }
    write_json_atomic(output.resolve(), payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoding-batch-size", type=int, default=4)
    arguments = parser.parse_args(argv)
    report = diagnose_insertion_proposal_training(
        arguments.source,
        arguments.checkpoint,
        arguments.proposal,
        arguments.recording_root,
        arguments.output,
        encoding_batch_size=arguments.encoding_batch_size,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
