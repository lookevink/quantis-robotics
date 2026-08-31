"""Load the pinned DROID JEPA-WM and execute one latent action rollout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any

import torch

from jepa_wm.contract import MODEL_ID
from jepa_wm.model import load_headless_model
from jepa_wm.persistence import write_json_atomic
from jepa_wm.runtime_environment import (
    claim_model_load_preflight,
    runtime_artifact_fingerprint,
    validate_headless_runtime,
)


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
        "model": MODEL_ID,
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


def run_model_load_preflight(source: Path, checkpoint: Path) -> dict[str, Any]:
    """Load the full frozen model without opening any recording or frame."""

    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM model-load preflight requires CUDA")
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    runtime = validate_headless_runtime(source, checkpoint)
    started = monotonic()
    model = load_headless_model(source, checkpoint, device=device)
    torch.cuda.synchronize(device)
    return {
        "schema": "quantis.jepa_wm_model_load_preflight.v1",
        "status": "passed",
        "model": MODEL_ID,
        "source_revision": os.environ.get("JEPA_WM_REVISION", "unknown"),
        "device": torch.cuda.get_device_name(device),
        "action_dimensions": int(model.action_dim),
        "runtime": runtime,
        "load_seconds": round(monotonic() - started, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "recordings_loaded": False,
        "canonical_accessed": False,
        "trained": False,
        "live_action_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    claim = None
    claim_payload = None
    if args.output is not None:
        if not args.load_only:
            raise ValueError("only model-load preflight accepts an output path")
        claim, claim_payload = claim_model_load_preflight(args.output)
    result = (
        run_model_load_preflight(args.source, args.checkpoint)
        if args.load_only
        else run_smoke(args.source, args.checkpoint)
    )
    if args.output is not None:
        assert claim is not None and claim_payload is not None
        result["claim"] = {
            "path": str(claim),
            "fingerprint": runtime_artifact_fingerprint(claim),
            "payload": claim_payload,
        }
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
