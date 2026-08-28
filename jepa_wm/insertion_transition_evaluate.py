"""Evaluate a transition-fine-tuned proposal on a disjoint live grasp endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from jepa_wm.action import DroidAction
from jepa_wm.control_safety import INSERTION_TARGET_PROGRESS
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_transition import (
    INSERTION_TRANSITION_FINETUNE_SCHEMA,
    INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
    InsertionTransitionTrainingSelection,
)
from jepa_wm.insertion_transition_evidence import transition_example_from_session
from jepa_wm.insertion_transition_finetune import _first_action_goal_cosine
from jepa_wm.model import load_headless_model
from jepa_wm.proposal import ProposalInputs, load_action_proposal
from jepa_wm.training_artifact import training_report_path


INSERTION_TRANSITION_EVALUATION_SCHEMA = (
    "quantis.jepa_wm_insertion_transition_evaluation.v1"
)


def evaluate_insertion_transition(
    source: Path,
    checkpoint: Path,
    data_root: Path,
    proposal_path: Path,
    source_session_id: str,
    output: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("insertion transition evaluation requires CUDA")
    proposal_path = proposal_path.resolve()
    training = json.loads(training_report_path(proposal_path).read_text())
    if (
        not isinstance(training, dict)
        or training.get("schema") != INSERTION_TRANSITION_FINETUNE_SCHEMA
        or training.get("output_constraint")
        != INSERTION_TRANSITION_OUTPUT_CONSTRAINT
    ):
        raise ValueError("proposal is not a transition fine-tune artifact")
    selection = InsertionTransitionTrainingSelection.from_dict(
        training.get("training_selection")
    )
    example = transition_example_from_session(data_root.resolve(), source_session_id)
    if (
        example.reference_recording in selection.evaluation_exclusions
        or example.source_proposal.fingerprint != selection.parent.fingerprint
    ):
        raise ValueError("transition evaluation overlaps or mismatches training evidence")

    device = torch.device("cuda", torch.cuda.current_device())
    proposal, _ = load_action_proposal(proposal_path, device=device)
    rollout = example.rollout(data_root.resolve())
    encoder = load_headless_model(source, checkpoint, device=device)
    contexts = encode_clips(encoder, [rollout.context_paths], batch_size=1).to(device)
    targets = encode_clips(encoder, [rollout.target_clip], batch_size=1).to(device)
    inputs = ProposalInputs.from_rollouts(
        (rollout,), conditioning=proposal.conditioning, device=device
    )
    if inputs.goal_delta is None:
        raise ValueError("transition evaluation requires goal-delta conditioning")
    with torch.no_grad():
        predicted = proposal(contexts, targets, inputs)
    actions = predicted[0].detach().cpu().tolist()
    first = tuple(float(value) for value in actions[0])
    projected = example.context_pose.applied(DroidAction(first))
    progress_reason = INSERTION_TARGET_PROGRESS.failure_reason(
        example.context_pose,
        example.target_pose,
        projected,
    )
    cosine = _first_action_goal_cosine(predicted, inputs.goal_delta)
    maximum_nontranslation_magnitude = max(
        abs(float(value)) for action in actions for value in action[3:]
    )
    passed = (
        cosine >= 0.95
        and progress_reason is None
        and maximum_nontranslation_magnitude <= 1e-8
    )
    report = {
        "schema": INSERTION_TRANSITION_EVALUATION_SCHEMA,
        "status": "passed" if passed else "failed",
        "proposal": ArtifactIdentity.from_artifact(proposal_path).to_dict(),
        "source_session_id": source_session_id,
        "reference_recording": example.reference_recording,
        "seed": example.seed,
        "training_exclusions": exclusions,
        "source_parent": parent.to_dict(),
        "example_fingerprint": example.fingerprint,
        "first_action_goal_cosine": cosine,
        "projected_progress_reason": (
            progress_reason.value if progress_reason is not None else None
        ),
        "maximum_nontranslation_magnitude": maximum_nontranslation_magnitude,
        "predicted_actions": actions,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    report["report"] = str(output)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = evaluate_insertion_transition(
        args.source,
        args.checkpoint,
        args.data_root,
        args.proposal,
        args.source_session,
        args.output,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
