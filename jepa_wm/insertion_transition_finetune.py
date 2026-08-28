"""Fine-tune the insertion proposal on authenticated grasp-endpoint negatives."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch

from jepa_wm.action import ActionSelectionBounds, DroidAction
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_proposal_readiness import validate_insertion_proposal
from jepa_wm.insertion_transition import (
    INSERTION_TRANSITION_FINETUNE_SCHEMA,
    INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
    InsertionTransitionCandidateRank,
    InsertionTransitionHardExampleEvaluation,
    InsertionTransitionExample,
    InsertionTransitionTrainingSelection,
    bounded_insertion_transition_cosine,
    transition_hard_evaluations_fingerprint,
    transition_training_examples,
    validate_insertion_transition_proposal,
)
from jepa_wm.insertion_transition_evidence import transition_example_from_session
from jepa_wm.model import load_headless_model
from jepa_wm.proposal import (
    ActionProposalNetwork,
    ProposalInputs,
    load_action_proposal,
    save_action_proposal,
)
from jepa_wm.task_windows import INSERTION_PROPOSAL_WINDOW
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    artifact_fingerprint,
    training_report_path,
)
from jepa_wm.trajectory import load_rollout_at


def constrain_insertion_transition_output(
    proposal: ActionProposalNetwork,
) -> None:
    """Make the specialized bridge learn XYZ while holding every other axis."""

    shared_dimensions = 6 if proposal.conditioned_gripper_head else 7
    standardized_zero = -proposal.action_mean / proposal.action_standard_deviation

    def constrain_shared(layer: torch.nn.Linear, *, residual: bool) -> None:
        weights = layer.weight.reshape(
            proposal.horizon,
            shared_dimensions,
            layer.in_features,
        )
        bias = layer.bias.reshape(proposal.horizon, shared_dimensions)
        weights[:, 3:, :].zero_()
        if residual:
            bias[:, 3:].zero_()
        else:
            bias[:, 3:].copy_(standardized_zero[:, 3:shared_dimensions])

    with torch.no_grad():
        constrain_shared(proposal.network[-1], residual=False)
        if proposal.conditioning_network is not None:
            constrain_shared(proposal.conditioning_network[-1], residual=True)
        if proposal.gripper_network is not None:
            gripper = proposal.gripper_network[-1]
            gripper.weight.zero_()
            gripper.bias.copy_(standardized_zero[:, 6])


@dataclass(frozen=True)
class InsertionTransitionFinetuneConfig:
    steps: int = 500
    rehearsal_batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    distillation_weight: float = 1.0
    goal_direction_weight: float = 1.0
    encoding_batch_size: int = 4
    seed: int = 52600

    def __post_init__(self) -> None:
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in (
                    self.steps,
                    self.rehearsal_batch_size,
                    self.encoding_batch_size,
                )
            )
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or any(
                not np.isfinite(value) or value < 0.0
                for value in (
                    self.learning_rate,
                    self.weight_decay,
                    self.distillation_weight,
                    self.goal_direction_weight,
                )
            )
            or self.learning_rate <= 0.0
        ):
            raise ValueError("insertion transition fine-tune config is invalid")


def _rehearsal_indices() -> tuple[int, ...]:
    indices = INSERTION_PROPOSAL_WINDOW.context_indices
    selected = indices[::8]
    return selected if selected[-1] == indices[-1] else (*selected, indices[-1])


def _first_action_goal_cosine(actions: torch.Tensor, goal: torch.Tensor) -> float:
    first_translation = actions[0, 0, :3]
    goal_translation = goal[0, :3]
    return bounded_insertion_transition_cosine(
        float(
            torch.nn.functional.cosine_similarity(
                first_translation[None], goal_translation[None], dim=1, eps=1e-12
            ).item()
        )
    )


@dataclass(frozen=True)
class InsertionTransitionHardObjective:
    """Differentiable objective shared by candidate training and retention."""

    hard_loss: torch.Tensor
    direction_loss: torch.Tensor
    total: torch.Tensor

    @classmethod
    def from_predictions(
        cls,
        standardized_actions: torch.Tensor,
        expected_standardized_actions: torch.Tensor,
        action_mean: torch.Tensor,
        action_standard_deviation: torch.Tensor,
        goal: torch.Tensor,
        goal_direction_weight: float,
    ) -> InsertionTransitionHardObjective:
        hard_loss = torch.nn.functional.mse_loss(
            standardized_actions,
            expected_standardized_actions,
        )
        first_actions = (
            standardized_actions * action_standard_deviation + action_mean
        )[:, 0, :3]
        direction_loss = (
            1.0
            - torch.nn.functional.cosine_similarity(
                first_actions,
                goal,
                dim=1,
                eps=1e-12,
            )
        ).mean()
        return cls(
            hard_loss=hard_loss,
            direction_loss=direction_loss,
            total=hard_loss + goal_direction_weight * direction_loss,
        )


def _hard_evaluations_from_tensor(
    examples: tuple[InsertionTransitionExample, ...],
    actions: torch.Tensor,
) -> tuple[InsertionTransitionHardExampleEvaluation, ...]:
    action_rows = actions.detach().cpu().tolist()
    if len(action_rows) != len(examples):
        raise ValueError("insertion transition hard prediction roster is invalid")
    return tuple(
        InsertionTransitionHardExampleEvaluation.from_prediction(
            example,
            tuple(DroidAction(tuple(values)) for values in predicted_actions),
        )
        for example, predicted_actions in zip(examples, action_rows)
    )


def finetune_insertion_transition(
    source: Path,
    checkpoint: Path,
    data_root: Path,
    parent_path: Path,
    source_session_id: str,
    output: Path,
    *,
    config: InsertionTransitionFinetuneConfig = InsertionTransitionFinetuneConfig(),
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("insertion transition fine-tuning requires CUDA")
    root = data_root.resolve()
    parent_path = parent_path.resolve()
    try:
        parent_identity = validate_insertion_proposal(parent_path).identity
        prior_transition_examples: tuple[InsertionTransitionExample, ...] = ()
    except ValueError:
        parent_identity = validate_insertion_transition_proposal(parent_path)
        prior_transition_examples = transition_training_examples(parent_path)
    example = transition_example_from_session(root, source_session_id)
    if example.source_proposal != parent_identity:
        raise ValueError("transition example was not produced by the parent proposal")
    if not example.target_progress_admissible:
        raise ValueError(
            "transition example requires a target-selection correction, not proposal training"
        )

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    proposal, parent_metadata = load_action_proposal(parent_path, device=device)
    teacher = deepcopy(proposal).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    rehearsal_indices = _rehearsal_indices()
    training_selection = InsertionTransitionTrainingSelection(
        parent=parent_identity,
        transition_example=example,
        evaluation_exclusions=tuple(
            dict.fromkeys(
                hard_example.reference_recording
                for hard_example in (*prior_transition_examples, example)
            )
        ),
        rehearsal_recordings=parent_metadata.training_recordings,
        rehearsal_context_indices=rehearsal_indices,
        rehearsal_transition_examples=prior_transition_examples,
    )
    admissible_prior_examples = (
        training_selection.actionable_rehearsal_transition_examples
    )
    rehearsal = tuple(
        load_rollout_at(
            root / "recordings" / recording,
            camera="wrist",
            context_index=context_index,
            bounds=ActionSelectionBounds(minimum_action_norm=0.0),
        )
        for recording in parent_metadata.training_recordings
        for context_index in rehearsal_indices
    )
    hard = example.rollout(root)
    prior_transition_rollouts = tuple(
        prior.rollout(root) for prior in admissible_prior_examples
    )
    rollouts = (hard, *prior_transition_rollouts, *rehearsal)
    encoder = load_headless_model(source, checkpoint, device=device)
    encoding_started = monotonic()
    contexts = encode_clips(
        encoder,
        [rollout.context_paths for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    ).to(device)
    targets = encode_clips(
        encoder,
        [rollout.target_clip for rollout in rollouts],
        batch_size=config.encoding_batch_size,
    ).to(device)
    inputs = ProposalInputs.from_rollouts(
        rollouts,
        conditioning=proposal.conditioning,
        device=device,
    )
    if inputs.goal_delta is None:
        raise ValueError("transition fine-tuning requires goal-delta conditioning")
    with torch.no_grad():
        teacher_standardized = teacher.standardized_actions(contexts, targets, inputs)
    hard_examples = (example, *admissible_prior_examples)
    hard_actions = torch.tensor(
        [
            [action.values for action in hard_example.actions]
            for hard_example in hard_examples
        ],
        device=device,
        dtype=torch.float32,
    )
    hard_standardized = (
        hard_actions - proposal.action_mean
    ) / proposal.action_standard_deviation
    constrain_insertion_transition_output(proposal)
    initial_actions = proposal(
        contexts[:1],
        targets[:1],
        inputs.indexed(torch.tensor([0], device=device)),
    )
    initial_cosine = _first_action_goal_cosine(initial_actions, inputs.goal_delta[:1])

    optimizer = torch.optim.AdamW(
        proposal.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses: list[float] = []
    proposal.train()
    hard_count = len(hard_examples)
    rehearsal_count = len(rehearsal)
    best_rank: InsertionTransitionCandidateRank | None = None
    best_state = {
        name: value.detach().clone() for name, value in proposal.state_dict().items()
    }

    def retain_best_state(
        standardized_actions: torch.Tensor,
        objective: torch.Tensor,
    ) -> None:
        nonlocal best_rank
        objective_value = float(objective.detach().cpu())
        predicted_actions = (
            standardized_actions * proposal.action_standard_deviation
            + proposal.action_mean
        )
        evaluations = _hard_evaluations_from_tensor(
            hard_examples,
            predicted_actions,
        )
        rank = InsertionTransitionCandidateRank.from_evaluations(
            evaluations,
            hard_objective=objective_value,
        )
        if best_rank is not None and rank >= best_rank:
            return
        best_rank = rank
        for name, value in proposal.state_dict().items():
            best_state[name].copy_(value.detach())

    for _ in range(config.steps):
        sampled = (
            torch.randint(
                rehearsal_count,
                (config.rehearsal_batch_size,),
                generator=generator,
            )
            + hard_count
        )
        batch_indices = torch.cat(
            (torch.arange(hard_count, dtype=torch.long), sampled)
        ).to(device)
        predicted = proposal.standardized_actions(
            contexts[batch_indices],
            targets[batch_indices],
            inputs.indexed(batch_indices),
        )
        hard_objective = InsertionTransitionHardObjective.from_predictions(
            predicted[:hard_count],
            hard_standardized,
            proposal.action_mean,
            proposal.action_standard_deviation,
            inputs.goal_delta[:hard_count, :3],
            config.goal_direction_weight,
        )
        distillation_loss = torch.nn.functional.mse_loss(
            predicted[hard_count:], teacher_standardized[batch_indices[hard_count:]]
        )
        retain_best_state(predicted[:hard_count], hard_objective.total)
        loss = hard_objective.total + config.distillation_weight * distillation_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        constrain_insertion_transition_output(proposal)
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    proposal.eval()
    with torch.no_grad():
        hard_indices = torch.arange(hard_count, device=device)
        optimized_standardized = proposal.standardized_actions(
            contexts[:hard_count],
            targets[:hard_count],
            inputs.indexed(hard_indices),
        )
        optimized_hard_objective = InsertionTransitionHardObjective.from_predictions(
            optimized_standardized,
            hard_standardized,
            proposal.action_mean,
            proposal.action_standard_deviation,
            inputs.goal_delta[:hard_count, :3],
            config.goal_direction_weight,
        )
        retain_best_state(optimized_standardized, optimized_hard_objective.total)
        proposal.load_state_dict(best_state)
        final_hard_actions = proposal(
            contexts[:hard_count],
            targets[:hard_count],
            inputs.indexed(hard_indices),
        )
    hard_evaluations = _hard_evaluations_from_tensor(
        hard_examples,
        final_hard_actions,
    )
    if best_rank is None:
        raise RuntimeError("insertion transition candidate selection did not run")
    final_actions = final_hard_actions[:1]
    final_cosine = hard_evaluations[0].first_action_goal_cosine
    hard_evaluations_fingerprint = transition_hard_evaluations_fingerprint(
        hard_evaluations
    )
    metadata = TrainingArtifactMetadata(
        base_model=parent_metadata.base_model,
        source_revision=parent_metadata.source_revision,
        camera=parent_metadata.camera,
        training_recordings=parent_metadata.training_recordings,
        training_steps=config.steps,
    )
    save_action_proposal(
        proposal,
        output.resolve(),
        metadata,
        training_selection_fingerprint=training_selection.fingerprint,
        training_evaluation_fingerprint=hard_evaluations_fingerprint,
    )
    report = {
        "schema": INSERTION_TRANSITION_FINETUNE_SCHEMA,
        "status": (
            "trained" if all(item.passed for item in hard_evaluations) else "failed"
        ),
        "proposal": str(output.resolve()),
        "proposal_fingerprint": artifact_fingerprint(output.resolve()),
        "metadata": metadata.to_dict(),
        "config": asdict(config),
        "output_constraint": INSERTION_TRANSITION_OUTPUT_CONSTRAINT,
        "training_selection": training_selection.to_dict(),
        "training_selection_fingerprint": training_selection.fingerprint,
        "initial_first_action_goal_cosine": initial_cosine,
        "final_first_action_goal_cosine": final_cosine,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "selected_hard_failure_count": best_rank.failed_hard_examples,
        "selected_hard_objective": best_rank.hard_objective,
        "hard_example_evaluations": [item.to_dict() for item in hard_evaluations],
        "hard_example_evaluations_fingerprint": hard_evaluations_fingerprint,
        "encoding_seconds": round(monotonic() - encoding_started, 3),
        "predicted_actions": final_actions[0].detach().cpu().tolist(),
    }
    report_path = training_report_path(output.resolve())
    report["report"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-session", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args(argv)
    report = finetune_insertion_transition(
        args.source,
        args.checkpoint,
        args.data_root,
        args.parent,
        args.source_session,
        args.output,
        config=InsertionTransitionFinetuneConfig(
            steps=args.steps,
            learning_rate=args.learning_rate,
        ),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "trained" else 2


if __name__ == "__main__":
    raise SystemExit(main())
