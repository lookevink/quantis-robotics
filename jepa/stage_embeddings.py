"""Build and cache frozen V-JEPA embeddings for labeled demo stages."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Callable, Protocol

import numpy as np

from jepa.contract import DEFAULT_FRAMES, ObservationStage
from jepa.observation_source import ObservationSource


CACHE_SCHEMA = "quantis.jepa_stage_embeddings.v1"


class PathEncoder(Protocol):
    def embed_paths(self, paths: list[Path]) -> np.ndarray:
        ...


@dataclass(frozen=True)
class StageWindow:
    stage: ObservationStage
    frames: list[Path]
    fingerprint: str


@dataclass(frozen=True)
class StageEmbedding:
    stage: ObservationStage
    path: Path
    cached: bool


def _load_windows(
    recording: Path,
    *,
    camera: str,
    frame_count: int,
) -> list[StageWindow]:
    recording_root = recording.resolve()
    grouped = ObservationSource.open(recording).staged_frame_paths(camera)

    windows = []
    for stage in ObservationStage:
        frames = grouped[stage]
        if len(frames) < frame_count:
            raise ValueError(
                f"{stage.value} has {len(frames)} frames but "
                f"{frame_count} are required"
            )
        selected = frames[-frame_count:]
        digest = sha256()
        for frame in selected:
            stat = frame.stat()
            digest.update(frame.relative_to(recording_root).as_posix().encode())
            digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode())
        windows.append(StageWindow(stage, selected, digest.hexdigest()))
    return windows


def _read_cache(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def embed_recording_stages(
    recording: Path,
    *,
    camera: str,
    model_id: str,
    encoder_factory: Callable[[], PathEncoder],
    frame_count: int = DEFAULT_FRAMES,
) -> list[StageEmbedding]:
    """Embed one window per stage and reuse artifacts while inputs are unchanged."""

    windows = _load_windows(recording, camera=camera, frame_count=frame_count)
    recording_root = recording.resolve()
    output_dir = recording / "jepa" / camera
    cache_path = output_dir / "manifest.json"
    cache = _read_cache(cache_path)
    cache_matches = (
        cache.get("schema") == CACHE_SCHEMA
        and cache.get("model") == model_id
        and cache.get("camera") == camera
        and cache.get("frame_count") == frame_count
    )
    cached_stages = cache.get("stages", {}) if cache_matches else {}
    if not isinstance(cached_stages, dict):
        cached_stages = {}

    encoder: PathEncoder | None = None
    results = []
    stage_entries = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for window in windows:
        output = output_dir / f"{window.stage.value}.npy"
        cached_entry = cached_stages.get(window.stage.value, {})
        cached = (
            isinstance(cached_entry, dict)
            and cached_entry.get("fingerprint") == window.fingerprint
            and output.is_file()
        )
        if not cached:
            if encoder is None:
                encoder = encoder_factory()
            np.save(output, encoder.embed_paths(window.frames))
        stage_entries[window.stage.value] = {
            "embedding": output.name,
            "fingerprint": window.fingerprint,
                "frames": [
                frame.relative_to(recording_root).as_posix()
                for frame in window.frames
            ],
        }
        results.append(StageEmbedding(window.stage, output, cached))

    cache_path.write_text(
        json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "model": model_id,
                "camera": camera,
                "frame_count": frame_count,
                "stages": stage_entries,
            },
            indent=2,
        )
        + "\n"
    )
    return results
