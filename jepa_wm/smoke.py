"""Load the pinned DROID JEPA-WM and execute one latent action rollout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any

import torch

from jepa_wm.model import load_headless_model


MODEL_NAME = "jepa_wm_droid"


def _shape(value: Any) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def run_smoke(source: Path, checkpoint: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM smoke test requires CUDA")

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    load_seconds = monotonic() - started

    frames = torch.zeros((1, 1, 3, 256, 256), dtype=torch.uint8)
    actions = torch.zeros((1, 1, model.action_dim), device=device)
    inference_started = monotonic()
    with torch.inference_mode():
        context = model.encode(frames)
        prediction = model.unroll(context, actions)
    torch.cuda.synchronize(device)
    inference_seconds = monotonic() - inference_started

    return {
        "status": "ready",
        "model": MODEL_NAME,
        "source_revision": os.environ.get("JEPA_WM_REVISION", "unknown"),
        "device": torch.cuda.get_device_name(device),
        "action_dimensions": int(model.action_dim),
        "context_shape": _shape(context),
        "prediction_shape": _shape(prediction),
        "load_seconds": round(load_seconds, 3),
        "inference_seconds": round(inference_seconds, 3),
        "allocated_gib": round(torch.cuda.memory_allocated(device_index) / 2**30, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.source, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
