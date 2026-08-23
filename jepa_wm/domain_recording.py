"""Validated whole-seed recording provenance for the JEPA-WM domain."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from sim.exploration import DOMAIN_DATASET_ID, DatasetSplit


@dataclass(frozen=True)
class DomainRecording:
    path: Path
    name: str
    split: DatasetSplit
    seed: int

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        expected_split: DatasetSplit,
    ) -> DomainRecording:
        path = path.resolve()
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"recording manifest does not exist: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
        if (
            not isinstance(metadata, dict)
            or metadata.get("dataset") != DOMAIN_DATASET_ID
        ):
            raise ValueError(f"recording is not a domain dataset: {path}")
        try:
            split = DatasetSplit(metadata.get("split"))
        except ValueError as error:
            raise ValueError(f"recording split is invalid: {path}") from error
        seed = metadata.get("seed")
        name = manifest.get("recording_id")
        if split != expected_split:
            raise ValueError(
                f"recording {path.name} has split {split.value!r}, "
                f"expected {expected_split.value!r}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"recording seed is invalid: {path}")
        if not isinstance(name, str) or name != path.name:
            raise ValueError(f"recording identity does not match its directory: {path}")
        return cls(path, name, split, seed)
