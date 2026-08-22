"""Validated frame layouts for demo recordings and legacy capture episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from jepa.contract import ObservationStage


@dataclass(frozen=True)
class ObservationSource:
    path: Path
    cameras: tuple[str, ...] | None

    @classmethod
    def open(cls, path: Path) -> ObservationSource:
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            return cls(path, None)

        manifest = json.loads(manifest_path.read_text())
        cameras = manifest.get("cameras", [])
        if not isinstance(cameras, list) or not all(
            isinstance(camera, str) for camera in cameras
        ):
            raise ValueError("recording manifest cameras must be a list of names")
        return cls(path, tuple(cameras))

    def frame_paths(self, camera: str = "wrist") -> list[Path]:
        if self.cameras is not None:
            self._validate_camera(camera)
            return sorted((self.path / camera).glob("frame_*.png"))
        return sorted((self.path / "rgb").rglob("*.png"))

    def staged_frame_paths(
        self, camera: str
    ) -> dict[ObservationStage, list[Path]]:
        if self.cameras is None:
            raise ValueError("legacy capture episodes do not contain stage labels")
        self._validate_camera(camera)
        recording_root = self.path.resolve()
        grouped = {stage: [] for stage in ObservationStage}
        with (self.path / "steps.jsonl").open(encoding="utf-8") as steps_file:
            for line in steps_file:
                step = json.loads(line)
                try:
                    stage = ObservationStage(step["stage"])
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        "every recording step must have a known stage"
                    ) from error
                try:
                    frame = (self.path / step["frames"][camera]).resolve()
                    frame.relative_to(recording_root)
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        f"recording step has no safe {camera!r} frame path"
                    ) from error
                if not frame.is_file():
                    raise ValueError(f"recording frame does not exist: {frame}")
                grouped[stage].append(frame)
        return grouped

    def default_embedding_path(self, camera: str) -> Path:
        if self.cameras is not None:
            return self.path / f"{camera}_vjepa2_embedding.npy"
        return self.path / "vjepa2_embedding.npy"

    def _validate_camera(self, camera: str) -> None:
        if self.cameras is None or camera not in self.cameras:
            raise ValueError(
                f"recording has no {camera!r} camera; "
                f"available cameras: {list(self.cameras or ())}"
            )
