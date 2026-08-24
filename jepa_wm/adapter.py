"""Small persistent action adapter for the pinned DROID JEPA-WM predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from jepa_wm.training_artifact import TrainingArtifactMetadata
from jepa_wm.training_artifact import validate_artifact_fingerprint
from jepa_wm.contract import MODEL_ID


ADAPTER_SCHEMA = "quantis.jepa_wm_action_adapter.v1"


def _action_encoder(model: Any) -> torch.nn.Module:
    try:
        encoder = model.model.predictor.action_encoder
    except AttributeError as error:
        raise ValueError("model has no in-predictor action encoder") from error
    if not isinstance(encoder, torch.nn.Module):
        raise ValueError("model action encoder is not a PyTorch module")
    return encoder


def action_adapter_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    encoder = _action_encoder(model)
    weight = getattr(encoder, "weight", None)
    if not isinstance(weight, torch.nn.Parameter):
        raise ValueError("model action encoder has no trainable weight")
    return (weight,)


def save_action_adapter(
    model: Any,
    path: Path,
    metadata: TrainingArtifactMetadata,
    *,
    training_selection_fingerprint: str | None = None,
) -> None:
    if training_selection_fingerprint is not None:
        validate_artifact_fingerprint(training_selection_fingerprint)
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {
            "schema": ADAPTER_SCHEMA,
            "metadata": metadata.to_dict(),
            "training_selection_fingerprint": training_selection_fingerprint,
            "action_encoder": _action_encoder(model).state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def _parse_action_adapter(
    payload: Any,
) -> tuple[TrainingArtifactMetadata, str | None, dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("unsupported JEPA-WM action adapter")
    metadata = TrainingArtifactMetadata.from_dict(payload.get("metadata"))
    fingerprint = payload.get("training_selection_fingerprint")
    if fingerprint is not None:
        if not isinstance(fingerprint, str):
            raise ValueError("adapter training selection fingerprint is invalid")
        validate_artifact_fingerprint(fingerprint)
    state = payload.get("action_encoder")
    if not isinstance(state, dict):
        raise ValueError("adapter has no action encoder state")
    return metadata, fingerprint, state


def load_action_adapter_contract(
    path: Path,
) -> tuple[TrainingArtifactMetadata, str | None]:
    metadata, fingerprint, _ = _parse_action_adapter(
        torch.load(path, map_location="cpu", weights_only=True)
    )
    return metadata, fingerprint


def apply_action_adapter(
    model: Any,
    path: Path,
    *,
    expected_base_model: str = MODEL_ID,
    expected_source_revision: str | None = None,
) -> TrainingArtifactMetadata:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata, _, state = _parse_action_adapter(payload)
    if metadata.base_model != expected_base_model:
        raise ValueError(
            f"adapter base model is {metadata.base_model}, expected {expected_base_model}"
        )
    if (
        expected_source_revision is not None
        and metadata.source_revision != expected_source_revision
    ):
        raise ValueError("adapter source revision does not match the installed model")
    _action_encoder(model).load_state_dict(state, strict=True)
    return metadata
