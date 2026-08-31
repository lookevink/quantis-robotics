"""Fail-closed dispatch for versioned JEPA action-model artifacts."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import torch

from jepa_wm.action_conditioning import (
    ACTION_CONDITIONING_SCHEMA,
    apply_action_conditioning,
)
from jepa_wm.adapter import (
    ADAPTER_SCHEMA,
    LEGACY_ADAPTER_SCHEMA,
    apply_action_adapter,
)


def apply_action_model_artifact(
    model: Any,
    path: Path,
    *,
    expected_source_revision: str | None = None,
) -> Any:
    """Apply exactly one authenticated adapter or conditioning family."""

    resolved = path.resolve()
    payload = torch.load(
        BytesIO(resolved.read_bytes()),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("unsupported JEPA action-model artifact")
    schema = payload.get("schema")
    if schema == ACTION_CONDITIONING_SCHEMA:
        return apply_action_conditioning(
            model,
            resolved,
            expected_source_revision=expected_source_revision,
        )
    if schema in (ADAPTER_SCHEMA, LEGACY_ADAPTER_SCHEMA):
        return apply_action_adapter(
            model,
            resolved,
            expected_source_revision=expected_source_revision,
        )
    raise ValueError("unsupported JEPA action-model artifact schema")
