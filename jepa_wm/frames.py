"""RGB frame batching shared by JEPA-WM evaluation and adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch


def video_batch(clips: Sequence[Sequence[Path]]) -> torch.Tensor:
    if not clips or not clips[0]:
        raise ValueError("video batch must contain at least one frame")
    expected_frames = len(clips[0])
    videos = []
    expected_size: tuple[int, int] | None = None
    for paths in clips:
        if len(paths) != expected_frames:
            raise ValueError("video clips must share one frame count")
        frames = []
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if expected_size is None:
                    expected_size = rgb.size
                elif rgb.size != expected_size:
                    raise ValueError("recording frames must share one resolution")
                frames.append(np.asarray(rgb, dtype=np.uint8).copy())
        videos.append(np.stack(frames))
    return torch.from_numpy(np.stack(videos)).permute(0, 1, 4, 2, 3)


def encode_clips(
    model: Any,
    clips: Sequence[Sequence[Path]],
    *,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("encoding batch size must be positive")
    encoded = []
    with torch.inference_mode():
        for start in range(0, len(clips), batch_size):
            encoded.append(
                model.encode(video_batch(clips[start : start + batch_size])).cpu()
            )
    if not encoded:
        raise ValueError("at least one clip is required for encoding")
    return torch.cat(encoded)
