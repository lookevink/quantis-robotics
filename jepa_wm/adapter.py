"""Small persistent action adapter for the pinned DROID JEPA-WM predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from jepa_wm.contract import MODEL_ID


ADAPTER_SCHEMA = "quantis.jepa_wm_action_adapter.v1"


@dataclass(frozen=True)
class ActionAdapterMetadata:
    base_model: str
    source_revision: str
    camera: str
    training_recordings: tuple[str, ...]
    training_steps: int

    def __post_init__(self) -> None:
        if not self.base_model or not self.source_revision or not self.camera:
            raise ValueError("adapter metadata fields must be non-empty")
        if not self.training_recordings or self.training_steps <= 0:
            raise ValueError("adapter training data and steps must be non-empty")

    @classmethod
    def from_dict(cls, payload: Any) -> ActionAdapterMetadata:
        if not isinstance(payload, dict):
            raise ValueError("adapter metadata must be an object")
        try:
            return cls(
                base_model=str(payload["base_model"]),
                source_revision=str(payload["source_revision"]),
                camera=str(payload["camera"]),
                training_recordings=tuple(payload["training_recordings"]),
                training_steps=int(payload["training_steps"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("adapter metadata is incomplete") from error

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    metadata: ActionAdapterMetadata,
) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {
            "schema": ADAPTER_SCHEMA,
            "metadata": metadata.to_dict(),
            "action_encoder": _action_encoder(model).state_dict(),
        },
        temporary,
    )
    temporary.replace(path)


def apply_action_adapter(
    model: Any,
    path: Path,
    *,
    expected_base_model: str = MODEL_ID,
    expected_source_revision: str | None = None,
) -> ActionAdapterMetadata:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("unsupported JEPA-WM action adapter")
    metadata = ActionAdapterMetadata.from_dict(payload.get("metadata"))
    if metadata.base_model != expected_base_model:
        raise ValueError(
            f"adapter base model is {metadata.base_model}, expected {expected_base_model}"
        )
    if (
        expected_source_revision is not None
        and metadata.source_revision != expected_source_revision
    ):
        raise ValueError("adapter source revision does not match the installed model")
    state = payload.get("action_encoder")
    if not isinstance(state, dict):
        raise ValueError("adapter has no action encoder state")
    _action_encoder(model).load_state_dict(state, strict=True)
    return metadata
