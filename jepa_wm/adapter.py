"""Small persistent action adapter for the pinned DROID JEPA-WM predictor."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from jepa_wm.training_artifact import ArtifactIdentity, TrainingArtifactMetadata
from jepa_wm.training_artifact import validate_artifact_fingerprint
from jepa_wm.contract import MODEL_ID


ADAPTER_SCHEMA = "quantis.jepa_wm_action_adapter.v2"
LEGACY_ADAPTER_SCHEMA = "quantis.jepa_wm_action_adapter.v1"


@dataclass(frozen=True)
class ActionAdapterContract:
    schema: str
    metadata: TrainingArtifactMetadata
    training_selection_fingerprint: str | None
    training_config_fingerprint: str | None

    def __post_init__(self) -> None:
        if self.schema not in (ADAPTER_SCHEMA, LEGACY_ADAPTER_SCHEMA):
            raise ValueError("unsupported JEPA-WM action adapter")
        for fingerprint in (
            self.training_selection_fingerprint,
            self.training_config_fingerprint,
        ):
            if fingerprint is not None:
                validate_artifact_fingerprint(fingerprint)
        if (
            self.schema == LEGACY_ADAPTER_SCHEMA
            and self.training_config_fingerprint is not None
        ):
            raise ValueError(
                "legacy adapter has an unexpected training config fingerprint"
            )
        if (
            self.schema == ADAPTER_SCHEMA
            and self.training_config_fingerprint is None
        ):
            raise ValueError("current adapter requires a training config fingerprint")

    @classmethod
    def current(
        cls,
        metadata: TrainingArtifactMetadata,
        *,
        training_selection_fingerprint: str | None,
        training_config_fingerprint: str,
    ) -> ActionAdapterContract:
        return cls(
            ADAPTER_SCHEMA,
            metadata,
            training_selection_fingerprint,
            training_config_fingerprint,
        )

    @classmethod
    def from_dict(cls, payload: Any) -> ActionAdapterContract:
        if not isinstance(payload, dict):
            raise ValueError("unsupported JEPA-WM action adapter")
        selection_fingerprint = payload.get("training_selection_fingerprint")
        config_fingerprint = payload.get("training_config_fingerprint")
        if selection_fingerprint is not None and not isinstance(
            selection_fingerprint, str
        ):
            raise ValueError("adapter training selection fingerprint is invalid")
        if config_fingerprint is not None and not isinstance(config_fingerprint, str):
            raise ValueError("adapter training config fingerprint is invalid")
        return cls(
            str(payload.get("schema")),
            TrainingArtifactMetadata.from_dict(payload.get("metadata")),
            selection_fingerprint,
            config_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "metadata": self.metadata.to_dict(),
            "training_selection_fingerprint": self.training_selection_fingerprint,
            "training_config_fingerprint": self.training_config_fingerprint,
        }


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
    contract: ActionAdapterContract,
) -> None:
    if contract.schema != ADAPTER_SCHEMA:
        raise ValueError("new adapters must use the current adapter schema")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {**contract.to_dict(), "action_encoder": _action_encoder(model).state_dict()},
        temporary,
    )
    temporary.replace(path)


def _parse_action_adapter(
    payload: Any,
) -> tuple[ActionAdapterContract, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("unsupported JEPA-WM action adapter")
    contract = ActionAdapterContract.from_dict(payload)
    state = payload.get("action_encoder")
    if not isinstance(state, dict):
        raise ValueError("adapter has no action encoder state")
    return contract, state


@dataclass(frozen=True)
class LoadedActionAdapter:
    """One immutable adapter read shared by identity checks and application."""

    identity: ArtifactIdentity
    contract: ActionAdapterContract
    state: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_identity: ArtifactIdentity | None = None,
    ) -> LoadedActionAdapter:
        resolved = path.resolve()
        encoded = resolved.read_bytes()
        identity = ArtifactIdentity(resolved, sha256(encoded).hexdigest())
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("action adapter identity changed before loading")
        contract, state = _parse_action_adapter(
            torch.load(BytesIO(encoded), map_location="cpu", weights_only=True)
        )
        return cls(identity, contract, state)

    def apply(
        self,
        model: Any,
        *,
        expected_base_model: str = MODEL_ID,
        expected_source_revision: str | None = None,
    ) -> TrainingArtifactMetadata:
        metadata = self.contract.metadata
        if metadata.base_model != expected_base_model:
            raise ValueError(
                f"adapter base model is {metadata.base_model}, "
                f"expected {expected_base_model}"
            )
        if (
            expected_source_revision is not None
            and metadata.source_revision != expected_source_revision
        ):
            raise ValueError("adapter source revision does not match the installed model")
        _action_encoder(model).load_state_dict(self.state, strict=True)
        return metadata


def load_action_adapter_contract(
    path: Path,
) -> ActionAdapterContract:
    return LoadedActionAdapter.load(path).contract


def apply_action_adapter(
    model: Any,
    path: Path,
    *,
    expected_base_model: str = MODEL_ID,
    expected_source_revision: str | None = None,
) -> TrainingArtifactMetadata:
    return LoadedActionAdapter.load(path).apply(
        model,
        expected_base_model=expected_base_model,
        expected_source_revision=expected_source_revision,
    )
