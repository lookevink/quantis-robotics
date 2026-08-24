"""Validated reach-and-grasp demonstration recording evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import numpy as np

from jepa_wm.domain_recording import DomainRecording
from jepa_wm.grasp_contract import GRASP_TASK_ID
from sim.exploration import DatasetSplit


@dataclass(frozen=True)
class GraspDemonstrationEvidence:
    recording: str
    acquisition_index: int
    attached_observations: int
    retained_displacement_meters: float

    @classmethod
    def from_recording(
        cls,
        path: Path,
        *,
        expected_split: str,
        minimum_retained_displacement_meters: float = 0.02,
    ) -> GraspDemonstrationEvidence:
        """Validate task identity, cadence, attachment transition, and retention."""

        if (
            not isfinite(minimum_retained_displacement_meters)
            or minimum_retained_displacement_meters <= 0.0
        ):
            raise ValueError("minimum retained displacement must be positive")
        recording = DomainRecording.from_path(
            path,
            expected_split=DatasetSplit(expected_split),
        )
        manifest = recording.manifest
        metadata = manifest.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("task") != GRASP_TASK_ID:
            raise ValueError("recording is not a reach-and-grasp demonstration")
        steps = recording.load_steps()
        acquisition_index = next(
            (
                index
                for index in range(1, len(steps))
                if not steps[index - 1].get("plug_attached")
                and steps[index].get("plug_attached") is True
            ),
            None,
        )
        if acquisition_index is None:
            raise ValueError("grasp demonstration has no attachment transition")
        attached_steps = steps[acquisition_index:]
        if any(step.get("plug_attached") is not True for step in attached_steps):
            raise ValueError("grasp demonstration loses the connector after acquisition")
        origin = np.asarray(steps[acquisition_index]["plug_position"], dtype=np.float64)
        positions = tuple(
            np.asarray(step["plug_position"], dtype=np.float64)
            for step in attached_steps
        )
        if origin.shape != (3,) or any(
            position.shape != (3,) or not np.all(np.isfinite(position))
            for position in positions
        ):
            raise ValueError("grasp demonstration plug positions are invalid")
        displacement = max(
            float(np.linalg.norm(position - origin)) for position in positions
        )
        if displacement < minimum_retained_displacement_meters:
            raise ValueError("grasp demonstration does not retain and move the connector")
        return cls(
            recording.name,
            acquisition_index,
            len(attached_steps),
            displacement,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "recording": self.recording,
            "acquisition_index": self.acquisition_index,
            "attached_observations": self.attached_observations,
            "retained_displacement_meters": self.retained_displacement_meters,
        }
