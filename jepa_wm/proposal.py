"""Small inverse-action proposal head over frozen JEPA context and goal latents."""

from __future__ import annotations

from enum import Enum
from math import isqrt
from pathlib import Path

import numpy as np
import torch

from jepa_wm.action import ACTION_DIMENSIONS
from jepa_wm.action_prior import ActionPriorConfig
from jepa_wm.proprioception import DroidValueNormalization
from jepa_wm.training_artifact import TrainingArtifactMetadata


PROPOSAL_SCHEMA = "quantis.jepa_wm_action_proposal.v1"


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
        pose_normalization: DroidValueNormalization | None = None,
        previous_action_normalization: DroidValueNormalization | None = None,
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
        self.uses_proprioception = pose_normalization is not None
        self.uses_action_history = previous_action_normalization is not None
        if self.uses_action_history and not self.uses_proprioception:
            raise ValueError("action-history proposals also require the current pose")
        input_dimension = feature_dimension * self.feature_mode.multiplier
        if pose_normalization is not None:
            input_dimension += ACTION_DIMENSIONS
            pose_mean = torch.from_numpy(pose_normalization.mean)
            pose_standard_deviation = torch.from_numpy(
                pose_normalization.standard_deviation
            )
        else:
            pose_mean = torch.zeros(ACTION_DIMENSIONS)
            pose_standard_deviation = torch.ones(ACTION_DIMENSIONS)
        if previous_action_normalization is not None:
            input_dimension += ACTION_DIMENSIONS
            previous_action_mean = torch.from_numpy(previous_action_normalization.mean)
            previous_action_standard_deviation = torch.from_numpy(
                previous_action_normalization.standard_deviation
            )
        else:
            previous_action_mean = torch.zeros(ACTION_DIMENSIONS)
            previous_action_standard_deviation = torch.ones(ACTION_DIMENSIONS)
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

    def standardized_actions(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        pose: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = proposal_features(context, target, self.feature_mode)
        expected_features = self.feature_dimension * self.feature_mode.multiplier
        if features.shape[-1] != expected_features:
            raise ValueError(
                "context and target latent features do not match the proposal"
            )
        if self.uses_proprioception:
            if pose is None or pose.ndim != 2 or pose.shape[-1] != ACTION_DIMENSIONS:
                raise ValueError("proposal requires one seven-value pose per sample")
            standardized_pose = (
                pose.to(device=features.device, dtype=features.dtype) - self.pose_mean
            ) / self.pose_standard_deviation
            features = torch.cat((features, standardized_pose), dim=-1)
        elif pose is not None:
            raise ValueError("proposal checkpoint does not accept proprioception")
        if self.uses_action_history:
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
        elif previous_action is not None:
            raise ValueError("proposal checkpoint does not accept action history")
        return self.network(features).reshape(-1, self.horizon, ACTION_DIMENSIONS)

    def forward(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        pose: torch.Tensor | None = None,
        previous_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return (
            self.standardized_actions(context, target, pose, previous_action)
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
            "uses_proprioception": proposal.uses_proprioception,
            "uses_action_history": proposal.uses_action_history,
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
    state = payload["state_dict"]
    uses_proprioception = bool(payload.get("uses_proprioception", False))
    uses_action_history = bool(payload.get("uses_action_history", False))
    pose_normalization = (
        DroidValueNormalization(
            state["pose_mean"].cpu().numpy(),
            state["pose_standard_deviation"].cpu().numpy(),
        )
        if uses_proprioception
        else None
    )
    previous_action_normalization = (
        DroidValueNormalization(
            state["previous_action_mean"].cpu().numpy(),
            state["previous_action_standard_deviation"].cpu().numpy(),
        )
        if uses_action_history
        else None
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
        pose_normalization=pose_normalization,
        previous_action_normalization=previous_action_normalization,
    ).to(device)
    proposal.load_state_dict(
        state,
        strict=uses_proprioception and uses_action_history,
    )
    proposal.eval()
    return proposal, metadata
