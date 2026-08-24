"""Validated whole-seed recording provenance for the JEPA-WM domain."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping

from sim.exploration import DOMAIN_DATASET_ID, DatasetSplit, validate_sample_times


@dataclass(frozen=True)
class DomainRecording:
    path: Path
    name: str
    split: DatasetSplit
    seed: int
    manifest: Mapping[str, Any]

    def load_steps(self) -> tuple[dict[str, Any], ...]:
        """Load frames and validate the manifest-owned count and cadence contract."""

        steps = tuple(
            json.loads(line)
            for line in (self.path / "steps.jsonl").read_text().splitlines()
            if line
        )
        if not all(isinstance(step, dict) for step in steps):
            raise ValueError(f"recording steps are invalid: {self.path}")
        if len(steps) != self.manifest.get("frames"):
            raise ValueError(f"recording frame count is inconsistent: {self.path}")
        fps = self.manifest.get("fps")
        if (
            isinstance(fps, bool)
            or not isinstance(fps, (int, float))
            or not isfinite(float(fps))
            or fps <= 0
        ):
            raise ValueError(f"recording FPS is invalid: {self.path}")
        sample_times = tuple(
            float(step["simulation_time_seconds"])
            for step in steps
            if step.get("simulation_time_seconds") is not None
        )
        if len(sample_times) != len(steps):
            raise ValueError(f"recording simulation times are incomplete: {self.path}")
        validate_sample_times(sample_times, 1.0 / float(fps))
        return steps

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
        if not isinstance(manifest, dict):
            raise ValueError(f"recording manifest is invalid: {manifest_path}")
        metadata = manifest.get("metadata")
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
        return cls(path, name, split, seed, manifest)
