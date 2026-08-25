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
from jepa_wm.planner_readiness import FirstActionThresholds
from jepa_wm.proposal import (
    ActionProposalNetwork,
    ProposalConditioning,
    ProposalInputs,
    action_normalization,
    save_action_proposal,
)
from jepa_wm.proprioception import DroidValueNormalization, ScalarNormalization
from jepa_wm.rollout_training import RolloutTrainingSelection
from jepa_wm.trajectory import RecordedRollout, RolloutWindow
from jepa_wm.training_artifact import (
    TrainingArtifactMetadata,
    artifact_fingerprint,
    training_report_path,
)


TRAINING_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)


@dataclass(frozen=True)
class ProposalLossWeights:
    action_mse: float = 1.0
    goal_consistency: float = 1.0
    first_action_mse: float = 1.0
    active_direction: float = 0.1
    inactive_gripper: float = 0.01
    first_gripper_mse: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not np.isfinite(value) or value < 0.0
            for value in (
                self.action_mse,
                self.goal_consistency,
                self.first_action_mse,
                self.active_direction,
                self.inactive_gripper,
                self.first_gripper_mse,
            )
        ):
            raise ValueError("proposal loss weights must be finite and non-negative")


@dataclass(frozen=True)
class ProposalTrainingConfig:
    steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dimension: int = 128
    encoding_batch_size: int = 4
    seed: int = 234
    loss_weights: ProposalLossWeights = ProposalLossWeights()
    conditioning_residual: bool = True
    conditioned_gripper_head: bool = True

    def __post_init__(self) -> None:
        dimensions = (
            self.steps,
            self.batch_size,
            self.hidden_dimension,
            self.encoding_batch_size,
        )
        if any(value <= 0 for value in dimensions) or self.seed < 0:
            raise ValueError("proposal training dimensions must be positive")
        if (
            self.learning_rate <= 0
            or self.weight_decay < 0
        ):
            raise ValueError("proposal optimizer values are invalid")


@dataclass(frozen=True)
class ProposalLoss:
    action_mse: torch.Tensor
    goal_consistency_mse: torch.Tensor
    first_action_mse: torch.Tensor
    active_direction_loss: torch.Tensor
    inactive_gripper_loss: torch.Tensor
    first_gripper_mse: torch.Tensor

    def total(self, weights: ProposalLossWeights) -> torch.Tensor:
        return (
            weights.action_mse * self.action_mse
            + weights.goal_consistency * self.goal_consistency_mse
            + weights.first_action_mse * self.first_action_mse
            + weights.active_direction * self.active_direction_loss
            + weights.inactive_gripper * self.inactive_gripper_loss
            + weights.first_gripper_mse * self.first_gripper_mse
        )


@dataclass(frozen=True)
class ProposalLossSnapshot:
    action_mse: float
    goal_consistency_mse: float
    first_action_mse: float
    active_direction_loss: float
    inactive_gripper_loss: float
    first_gripper_mse: float

    @classmethod
    def from_loss(cls, loss: ProposalLoss) -> ProposalLossSnapshot:
        return cls(
            *(float(value.detach().cpu()) for value in loss.__dict__.values())
        )

    @staticmethod
    def report(
        initial: ProposalLossSnapshot,
        final: ProposalLossSnapshot,
    ) -> dict[str, float]:
        return {
            **{
                f"initial_{name}": value
                for name, value in initial.__dict__.items()
            },
            **{
                f"final_{name}": value
                for name, value in final.__dict__.items()
            },
        }


def proposal_loss(
    predicted_standardized_actions: torch.Tensor,
    target_standardized_actions: torch.Tensor,
    goal_delta: torch.Tensor,
    *,
    action_mean: torch.Tensor,
    action_standard_deviation: torch.Tensor,
    goal_standard_deviation: torch.Tensor,
    thresholds: FirstActionThresholds = FirstActionThresholds(),
) -> ProposalLoss:
    if (
        predicted_standardized_actions.shape != target_standardized_actions.shape
        or predicted_standardized_actions.ndim != 3
        or predicted_standardized_actions.shape[-1] != 7
        or goal_delta.shape != (predicted_standardized_actions.shape[0], 7)
        or action_mean.shape != predicted_standardized_actions.shape[1:]
        or action_standard_deviation.shape != action_mean.shape
        or goal_standard_deviation.shape != (7,)
    ):
        raise ValueError("proposal loss tensors do not match the action horizon")
    predicted_actions = (
        predicted_standardized_actions * action_standard_deviation + action_mean
    )
    target_actions = (
        target_standardized_actions * action_standard_deviation + action_mean
    )
    predicted_additive_goal = torch.cat(
        (predicted_actions[:, :, :3].sum(dim=1), predicted_actions[:, :, 6:].sum(dim=1)),
        dim=1,
    )
    additive_goal_delta = torch.cat((goal_delta[:, :3], goal_delta[:, 6:]), dim=1)
    additive_goal_standard_deviation = torch.cat(
        (goal_standard_deviation[:3], goal_standard_deviation[6:])
    )
    standardized_goal_error = (
        predicted_additive_goal - additive_goal_delta
    ) / additive_goal_standard_deviation
    recorded_first = target_actions[:, 0]
    planned_first = predicted_actions[:, 0]
    active = thresholds.recorded_activity.active_tensor(recorded_first)
    cosine = torch.nn.functional.cosine_similarity(
        recorded_first, planned_first, dim=1, eps=1e-12
    )
    active_weights = active.to(dtype=cosine.dtype)
    direction_loss = (
        ((1.0 - cosine) * active_weights).sum()
        / torch.clamp_min(active_weights.sum(), 1.0)
    )
    inactive_gripper = (
        torch.abs(recorded_first[:, 6])
        <= thresholds.recorded_activity.gripper_delta
    ).to(dtype=planned_first.dtype)
    inactive_gripper_loss = (
        torch.square(
            planned_first[:, 6] / thresholds.maximum_stationary_gripper
        )
        * inactive_gripper
    ).sum() / torch.clamp_min(inactive_gripper.sum(), 1.0)
    return ProposalLoss(
        torch.nn.functional.mse_loss(
            predicted_standardized_actions, target_standardized_actions
        ),
        torch.mean(torch.square(standardized_goal_error)),
        torch.nn.functional.mse_loss(
            predicted_standardized_actions[:, 0],
            target_standardized_actions[:, 0],
        ),
        direction_loss,
        inactive_gripper_loss,
        torch.nn.functional.mse_loss(
            predicted_standardized_actions[:, 0, 6],
            target_standardized_actions[:, 0, 6],
        ),
    )


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
    training_selection = RolloutTrainingSelection.load(
        recordings,
        camera=camera,
        bounds=TRAINING_BOUNDS,
        window=window,
    )
    rollouts = training_selection.rollouts
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
    proposal_inputs = ProposalInputs.from_rollouts(rollouts)
    if any(value is None for value in proposal_inputs.values):
        raise ValueError("training rollouts did not produce complete proposal inputs")
    pose_normalization = DroidValueNormalization.from_samples(
        proposal_inputs.pose.numpy()
    )
    previous_action_normalization = DroidValueNormalization.from_samples(
        proposal_inputs.previous_action.numpy()
    )
    goal_delta_normalization = DroidValueNormalization.from_samples(
        proposal_inputs.goal_delta.numpy()
    )
    task_progress_normalization = ScalarNormalization.from_samples(
        proposal_inputs.task_progress.numpy()
    )
    actions = torch.from_numpy(action_sequences)
    standardized_actions = (actions - action_mean) / action_std
    feature_dimension = contexts.shape[-1]
    proposal = ActionProposalNetwork(
        feature_dimension,
        action_sequences.shape[1],
        config.hidden_dimension,
        action_mean,
        action_std,
        conditioning=ProposalConditioning(
            pose=pose_normalization,
            previous_action=previous_action_normalization,
            goal_delta=goal_delta_normalization,
            task_progress=task_progress_normalization,
        ),
        conditioning_residual=config.conditioning_residual,
        conditioned_gripper_head=config.conditioned_gripper_head,
    ).to(device)
    optimizer = torch.optim.AdamW(
        proposal.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    losses = []
    component_history = []
    training_started = monotonic()
    proposal.train()
    for _ in range(config.steps):
        indices = torch.randint(
            len(rollouts),
            (config.batch_size,),
            generator=generator,
        )
        inputs = proposal_inputs.indexed(indices).to(device)
        predicted = proposal.standardized_actions(
            contexts[indices].to(device), targets[indices].to(device), inputs
        )
        components = proposal_loss(
            predicted,
            standardized_actions[indices].to(device),
            inputs.goal_delta,
            action_mean=proposal.action_mean,
            action_standard_deviation=proposal.action_standard_deviation,
            goal_standard_deviation=proposal.goal_delta_standard_deviation,
        )
        loss = components.total(config.loss_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        component_history.append(ProposalLossSnapshot.from_loss(components))
    torch.cuda.synchronize(device)
    training_seconds = monotonic() - training_started
    metadata = TrainingArtifactMetadata(
        base_model=MODEL_ID,
        source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
        camera=camera,
        training_recordings=tuple(recording.name for recording in recordings),
        training_steps=config.steps,
    )
    selection_payload = training_selection.to_dict()
    selection_fingerprint = training_selection.fingerprint
    save_action_proposal(
        proposal,
        output,
        metadata,
        training_selection_fingerprint=selection_fingerprint,
    )
    report = {
        "status": "trained",
        "proposal": str(output.resolve()),
        "proposal_fingerprint": artifact_fingerprint(output),
        "metadata": metadata.to_dict(),
        "config": asdict(config),
        **selection_payload,
        "training_selection_fingerprint": selection_fingerprint,
        "conditioning": proposal.conditioning.to_dict(),
        "trainable_parameters": sum(
            parameter.numel() for parameter in proposal.parameters()
        ),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": float(np.min(losses)),
        **ProposalLossSnapshot.report(
            component_history[0], component_history[-1]
        ),
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
    parser.add_argument(
        "--goal-consistency-weight",
        type=float,
        default=ProposalLossWeights.goal_consistency,
    )
    parser.add_argument(
        "--first-action-weight",
        type=float,
        default=ProposalLossWeights.first_action_mse,
    )
    parser.add_argument(
        "--active-direction-weight",
        type=float,
        default=ProposalLossWeights.active_direction,
    )
    parser.add_argument(
        "--inactive-gripper-weight",
        type=float,
        default=ProposalLossWeights.inactive_gripper,
    )
    parser.add_argument(
        "--first-gripper-weight",
        type=float,
        default=ProposalLossWeights.first_gripper_mse,
    )
    parser.add_argument(
        "--conditioning-residual",
        action=argparse.BooleanOptionalAction,
        default=ProposalTrainingConfig.conditioning_residual,
    )
    parser.add_argument(
        "--conditioned-gripper-head",
        action=argparse.BooleanOptionalAction,
        default=ProposalTrainingConfig.conditioned_gripper_head,
    )
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
                    loss_weights=ProposalLossWeights(
                        goal_consistency=args.goal_consistency_weight,
                        first_action_mse=args.first_action_weight,
                        active_direction=args.active_direction_weight,
                        inactive_gripper=args.inactive_gripper_weight,
                        first_gripper_mse=args.first_gripper_weight,
                    ),
                    conditioning_residual=args.conditioning_residual,
                    conditioned_gripper_head=args.conditioned_gripper_head,
                ),
                window=window,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
