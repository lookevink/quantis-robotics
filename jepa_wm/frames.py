"""RGB frame batching shared by JEPA-WM evaluation and adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

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
