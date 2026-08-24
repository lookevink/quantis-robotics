"""Small inverse-action proposal head over frozen JEPA context and goal latents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isqrt
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.action_prior import ActionPriorConfig
from jepa_wm.proprioception import DroidValueNormalization
from jepa_wm.training_artifact import (
    ProposalConditioningCapabilities,
    TrainingArtifactMetadata,
)

if TYPE_CHECKING:
    from jepa_wm.control_protocol import ControlObservation
    from jepa_wm.trajectory import RecordedRollout


PROPOSAL_SCHEMA = "quantis.jepa_wm_action_proposal.v1"


@dataclass(frozen=True)
class ProposalConditioning:
    pose: DroidValueNormalization | None = None
    previous_action: DroidValueNormalization | None = None
    goal_delta: DroidValueNormalization | None = None

    def __post_init__(self) -> None:
        if self.previous_action is not None and self.pose is None:
            raise ValueError("action-history proposals also require the current pose")

    @property
    def input_dimension(self) -> int:
        return ACTION_DIMENSIONS * sum(
            value is not None
            for value in (self.pose, self.previous_action, self.goal_delta)
        )

    @property
    def uses_proprioception(self) -> bool:
        return self.pose is not None

    @property
    def uses_action_history(self) -> bool:
        return self.previous_action is not None

    @property
    def uses_goal_delta(self) -> bool:
        return self.goal_delta is not None

    def to_dict(self) -> dict[str, bool]:
        return self.capabilities.to_dict()

    @property
    def capabilities(self) -> ProposalConditioningCapabilities:
        return ProposalConditioningCapabilities(
            self.uses_proprioception,
            self.uses_action_history,
            self.uses_goal_delta,
        )


@dataclass(frozen=True)
class ProposalInputs:
    pose: torch.Tensor | None = None
    previous_action: torch.Tensor | None = None
    goal_delta: torch.Tensor | None = None

    @classmethod
    def from_rollouts(
        cls,
        rollouts: Sequence[RecordedRollout],
        *,
        conditioning: ProposalConditioning | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> ProposalInputs:
        include_pose = conditioning is None or conditioning.uses_proprioception
        include_history = conditioning is None or conditioning.uses_action_history
        include_goal_delta = conditioning is None or conditioning.uses_goal_delta
        return cls(
            pose=torch.tensor(
                [rollout.context_pose.values for rollout in rollouts],
                device=device,
                dtype=dtype,
            ) if include_pose else None,
            previous_action=torch.tensor(
                [rollout.previous_action.values for rollout in rollouts],
                device=device,
                dtype=dtype,
            ) if include_history else None,
            goal_delta=torch.tensor(
                [rollout.goal_action.values for rollout in rollouts],
                device=device,
                dtype=dtype,
            ) if include_goal_delta else None,
        )

    @classmethod
    def from_observation(
        cls,
        observation: ControlObservation,
        *,
        conditioning: ProposalConditioning,
        device: torch.device,
        dtype: torch.dtype,
    ) -> ProposalInputs:
        return cls(
            pose=torch.tensor(
                (observation.pose.values,), device=device, dtype=dtype
            ) if conditioning.uses_proprioception else None,
            previous_action=torch.tensor(
                (observation.previous_action.values,), device=device, dtype=dtype
            ) if conditioning.uses_action_history else None,
            goal_delta=torch.tensor(
                (observation.goal_action.values,), device=device, dtype=dtype
            ) if conditioning.uses_goal_delta else None,
        )

    def indexed(self, indices: torch.Tensor) -> ProposalInputs:
        return ProposalInputs(
            *(value[indices] if value is not None else None for value in self.values)
        )

    def to(
        self,
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> ProposalInputs:
        return ProposalInputs(
            *(
                value.to(device=device, dtype=dtype)
                if value is not None
                else None
                for value in self.values
            )
        )

    @property
    def values(self) -> tuple[torch.Tensor | None, ...]:
        return self.pose, self.previous_action, self.goal_delta


class ProposalFeatureMode(str, Enum):
    GLOBAL = "global_pool"
    SPATIAL_MOMENTS = "spatial_moments"

    @property
    def multiplier(self) -> int:
        return 3 if self is ProposalFeatureMode.GLOBAL else 5

    @classmethod
    def parse(cls, value: object) -> ProposalFeatureMode:
        try:
            return cls(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"unsupported proposal feature mode: {value}") from error


def pool_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim < 2:
        raise ValueError("JEPA latents must have a batch and feature dimension")
    reduction_dimensions = tuple(range(1, latents.ndim - 1))
    return latents.mean(dim=reduction_dimensions) if reduction_dimensions else latents


def _token_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim < 3:
        raise ValueError("spatial proposal features require tokenized JEPA latents")
    temporal_dimensions = tuple(range(1, latents.ndim - 2))
    return latents.mean(dim=temporal_dimensions) if temporal_dimensions else latents


def proposal_features(
    context: torch.Tensor,
    target: torch.Tensor,
    mode: ProposalFeatureMode,
) -> torch.Tensor:
    if mode is ProposalFeatureMode.GLOBAL:
        context_features = pool_latents(context)
        target_features = pool_latents(target)
        return torch.cat(
            (context_features, target_features, target_features - context_features),
            dim=-1,
        )
    context_tokens = _token_latents(context)
    target_tokens = _token_latents(target)
    if context_tokens.shape != target_tokens.shape:
        raise ValueError("context and target token shapes do not match")
    token_count = context_tokens.shape[-2]
    grid_size = isqrt(token_count)
    if grid_size * grid_size != token_count:
        raise ValueError("spatial proposal requires a square JEPA token grid")
    coordinates = torch.linspace(
        -1.0,
        1.0,
        grid_size,
        device=context_tokens.device,
        dtype=context_tokens.dtype,
    )
    y_coordinates, x_coordinates = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )
    difference = target_tokens - context_tokens
    x_moment = (difference * x_coordinates.reshape(1, -1, 1)).mean(dim=-2)
    y_moment = (difference * y_coordinates.reshape(1, -1, 1)).mean(dim=-2)
    return torch.cat(
        (
            context_tokens.mean(dim=-2),
            target_tokens.mean(dim=-2),
            difference.mean(dim=-2),
            x_moment,
            y_moment,
        ),
        dim=-1,
    )


class ActionProposalNetwork(torch.nn.Module):
    def __init__(
        self,
        feature_dimension: int,
        horizon: int,
        hidden_dimension: int,
        action_mean: torch.Tensor,
        action_standard_deviation: torch.Tensor,
        feature_mode: ProposalFeatureMode = ProposalFeatureMode.SPATIAL_MOMENTS,
        conditioning: ProposalConditioning = ProposalConditioning(),
    ) -> None:
        super().__init__()
        if feature_dimension <= 0 or horizon <= 0 or hidden_dimension <= 0:
            raise ValueError("proposal network dimensions must be positive")
        expected_shape = (horizon, ACTION_DIMENSIONS)
        if (
            tuple(action_mean.shape) != expected_shape
            or tuple(action_standard_deviation.shape) != expected_shape
            or torch.any(action_standard_deviation <= 0)
        ):
            raise ValueError("proposal action normalization has an invalid shape")
        self.feature_dimension = feature_dimension
        self.horizon = horizon
        self.hidden_dimension = hidden_dimension
        self.feature_mode = ProposalFeatureMode.parse(feature_mode)
        self.conditioning = conditioning
        input_dimension = (
            feature_dimension * self.feature_mode.multiplier
            + conditioning.input_dimension
        )
        if conditioning.pose is not None:
            pose_mean = torch.from_numpy(conditioning.pose.mean)
            pose_standard_deviation = torch.from_numpy(
                conditioning.pose.standard_deviation
            )
        else:
            pose_mean = torch.zeros(ACTION_DIMENSIONS)
            pose_standard_deviation = torch.ones(ACTION_DIMENSIONS)
        if conditioning.previous_action is not None:
            previous_action_mean = torch.from_numpy(conditioning.previous_action.mean)
            previous_action_standard_deviation = torch.from_numpy(
                conditioning.previous_action.standard_deviation
            )
        else:
            previous_action_mean = torch.zeros(ACTION_DIMENSIONS)
            previous_action_standard_deviation = torch.ones(ACTION_DIMENSIONS)
        if conditioning.goal_delta is not None:
            goal_delta_mean = torch.from_numpy(conditioning.goal_delta.mean)
            goal_delta_standard_deviation = torch.from_numpy(
                conditioning.goal_delta.standard_deviation
            )
        else:
            goal_delta_mean = torch.zeros(ACTION_DIMENSIONS)
            goal_delta_standard_deviation = torch.ones(ACTION_DIMENSIONS)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dimension, hidden_dimension),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dimension, horizon * ACTION_DIMENSIONS),
        )
        self.register_buffer("action_mean", action_mean.clone().float())
        self.register_buffer(
            "action_standard_deviation",
            action_standard_deviation.clone().float(),
        )
        self.register_buffer("pose_mean", pose_mean.clone().float())
        self.register_buffer(
            "pose_standard_deviation",
            pose_standard_deviation.clone().float(),
        )
        self.register_buffer(
            "previous_action_mean", previous_action_mean.clone().float()
        )
        self.register_buffer(
            "previous_action_standard_deviation",
            previous_action_standard_deviation.clone().float(),
        )
        self.register_buffer("goal_delta_mean", goal_delta_mean.clone().float())
        self.register_buffer(
            "goal_delta_standard_deviation",
            goal_delta_standard_deviation.clone().float(),
        )

    @property
    def uses_proprioception(self) -> bool:
        return self.conditioning.uses_proprioception

    @property
    def uses_action_history(self) -> bool:
        return self.conditioning.uses_action_history

    @property
    def uses_goal_delta(self) -> bool:
        return self.conditioning.uses_goal_delta

    def standardized_actions(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        inputs: ProposalInputs = ProposalInputs(),
    ) -> torch.Tensor:
        features = proposal_features(context, target, self.feature_mode)
        expected_features = self.feature_dimension * self.feature_mode.multiplier
        if features.shape[-1] != expected_features:
            raise ValueError(
                "context and target latent features do not match the proposal"
            )
        if self.uses_proprioception:
            pose = inputs.pose
            if pose is None or pose.ndim != 2 or pose.shape[-1] != ACTION_DIMENSIONS:
                raise ValueError("proposal requires one seven-value pose per sample")
            standardized_pose = (
                pose.to(device=features.device, dtype=features.dtype) - self.pose_mean
            ) / self.pose_standard_deviation
            features = torch.cat((features, standardized_pose), dim=-1)
        elif inputs.pose is not None:
            raise ValueError("proposal checkpoint does not accept proprioception")
        if self.uses_action_history:
            previous_action = inputs.previous_action
            if (
                previous_action is None
                or previous_action.ndim != 2
                or previous_action.shape[-1] != ACTION_DIMENSIONS
            ):
                raise ValueError(
                    "proposal requires one seven-value previous action per sample"
                )
            standardized_previous_action = (
                previous_action.to(device=features.device, dtype=features.dtype)
                - self.previous_action_mean
            ) / self.previous_action_standard_deviation
            features = torch.cat((features, standardized_previous_action), dim=-1)
        elif inputs.previous_action is not None:
            raise ValueError("proposal checkpoint does not accept action history")
        if self.uses_goal_delta:
            goal_delta = inputs.goal_delta
            if (
                goal_delta is None
                or goal_delta.ndim != 2
                or goal_delta.shape[-1] != ACTION_DIMENSIONS
            ):
                raise ValueError(
                    "goal-conditioned proposal requires one seven-value goal delta "
                    "per sample"
                )
            standardized_goal_delta = (
                goal_delta.to(device=features.device, dtype=features.dtype)
                - self.goal_delta_mean
            ) / self.goal_delta_standard_deviation
            features = torch.cat((features, standardized_goal_delta), dim=-1)
        elif inputs.goal_delta is not None:
            raise ValueError("proposal checkpoint does not accept goal conditioning")
        return self.network(features).reshape(-1, self.horizon, ACTION_DIMENSIONS)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        inputs: ProposalInputs = ProposalInputs(),
    ) -> torch.Tensor:
        return (
            self.standardized_actions(context, target, inputs)
            * self.action_standard_deviation
            + self.action_mean
        )


def action_normalization(
    action_sequences: np.ndarray,
    config: ActionPriorConfig = ActionPriorConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    actions = np.asarray(action_sequences, dtype=np.float32)
    if actions.ndim != 3 or actions.shape[2] != ACTION_DIMENSIONS:
        raise ValueError("proposal actions must have shape [samples, horizon, 7]")
    distribution = config.distribution_for(actions)
    return (
        torch.from_numpy(distribution.mean.astype(np.float32)),
        torch.from_numpy(distribution.standard_deviation.astype(np.float32)),
    )


def save_action_proposal(
    proposal: ActionProposalNetwork,
    path: Path,
    metadata: TrainingArtifactMetadata,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": PROPOSAL_SCHEMA,
            "metadata": metadata.to_dict(),
            "feature_dimension": proposal.feature_dimension,
            "horizon": proposal.horizon,
            "hidden_dimension": proposal.hidden_dimension,
            "feature_mode": proposal.feature_mode.value,
            "conditioning": proposal.conditioning.to_dict(),
            "state_dict": proposal.state_dict(),
        },
        path,
    )


def load_action_proposal(
    path: Path,
    *,
    device: torch.device,
) -> tuple[ActionProposalNetwork, TrainingArtifactMetadata]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("schema") != PROPOSAL_SCHEMA:
        raise ValueError("action proposal schema is unsupported")
    metadata_payload = payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError("action proposal metadata is missing")
    metadata = TrainingArtifactMetadata.from_dict(metadata_payload)
    raw_state = payload.get("state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError("action proposal state is missing")
    state = dict(raw_state)
    if "goal_mean" in state or "goal_standard_deviation" in state:
        if "goal_delta_mean" in state or "goal_delta_standard_deviation" in state:
            raise ValueError("action proposal has conflicting goal-delta state")
        try:
            state["goal_delta_mean"] = state.pop("goal_mean")
            state["goal_delta_standard_deviation"] = state.pop(
                "goal_standard_deviation"
            )
        except KeyError as error:
            raise ValueError("action proposal goal-delta state is incomplete") from error
    conditioning_payload = payload.get("conditioning")
    if conditioning_payload is None:
        capabilities = ProposalConditioningCapabilities(
            bool(payload.get("uses_proprioception", False)),
            bool(payload.get("uses_action_history", False)),
            bool(payload.get("uses_goal_conditioning", False)),
        )
    else:
        capabilities = ProposalConditioningCapabilities.from_dict(
            conditioning_payload
        )
    goal_delta_state_keys = {
        "goal_delta_mean",
        "goal_delta_standard_deviation",
    }
    present_goal_delta_state = goal_delta_state_keys.intersection(state)
    if present_goal_delta_state and present_goal_delta_state != goal_delta_state_keys:
        raise ValueError("action proposal goal-delta state is incomplete")
    if not present_goal_delta_state:
        if conditioning_payload is not None or capabilities.goal_delta:
            raise ValueError("action proposal goal-delta state is incomplete")
        state["goal_delta_mean"] = torch.zeros(ACTION_DIMENSIONS)
        state["goal_delta_standard_deviation"] = torch.ones(ACTION_DIMENSIONS)
    pose_normalization = (
        DroidValueNormalization(
            state["pose_mean"].cpu().numpy(),
            state["pose_standard_deviation"].cpu().numpy(),
        )
        if capabilities.proprioception
        else None
    )
    previous_action_normalization = (
        DroidValueNormalization(
            state["previous_action_mean"].cpu().numpy(),
            state["previous_action_standard_deviation"].cpu().numpy(),
        )
        if capabilities.action_history
        else None
    )
    goal_delta_normalization = (
        DroidValueNormalization(
            state["goal_delta_mean"].cpu().numpy(),
            state["goal_delta_standard_deviation"].cpu().numpy(),
        )
        if capabilities.goal_delta
        else None
    )
    conditioning = ProposalConditioning(
        pose=pose_normalization,
        previous_action=previous_action_normalization,
        goal_delta=goal_delta_normalization,
    )
    proposal = ActionProposalNetwork(
        int(payload["feature_dimension"]),
        int(payload["horizon"]),
        int(payload["hidden_dimension"]),
        state["action_mean"],
        state["action_standard_deviation"],
        feature_mode=ProposalFeatureMode.parse(
            payload.get("feature_mode", ProposalFeatureMode.GLOBAL.value)
        ),
        conditioning=conditioning,
    ).to(device)
    proposal.load_state_dict(state, strict=True)
    proposal.eval()
    return proposal, metadata
