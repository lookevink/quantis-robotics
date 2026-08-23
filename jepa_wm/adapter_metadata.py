"""Dependency-free metadata and report contract for JEPA-WM action adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


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


def adapter_report_path(adapter: Path) -> Path:
    return adapter.with_suffix(adapter.suffix + ".json")


def load_adapter_report_metadata(adapter: Path) -> ActionAdapterMetadata:
    report = adapter_report_path(adapter)
    if not report.is_file():
        raise ValueError(f"adapter training report does not exist: {report}")
    payload = json.loads(report.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"adapter training report must be an object: {report}")
    return ActionAdapterMetadata.from_dict(payload.get("metadata"))
