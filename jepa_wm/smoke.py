"""Load the pinned DROID JEPA-WM and execute one latent action rollout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from time import monotonic
from typing import Any

import torch
import yaml


MODEL_NAME = "jepa_wm_droid"
CONFIG_RELATIVE_PATH = Path(
    "configs/evals/simu_env_planning/droid/jepa-wm/"
    "droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_"
    "r256_alpha0_ep64_decode.yaml"
)


def _shape(value: Any) -> list[int]:
    return [int(dimension) for dimension in value.shape]


def load_headless_model(
    source: Path,
    checkpoint: Path,
    *,
    device: torch.device,
) -> Any:
    """Load the official model without its optional pixel-decoder head."""

    sys.path.insert(0, str(source))
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import (
        make_inverse_transforms,
        make_transforms,
    )
    from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import (
        init_module,
    )

    config = yaml.safe_load((source / CONFIG_RELATIVE_PATH).read_text())
    model_config = config["model_kwargs"]
    pretrain_config = model_config["pretrain_kwargs"]
    data_config = model_config["data"]
    augmentation = model_config["data_aug"]
    pretrain_config["heads_cfg"] = {}

    stats = get_data_stats("droid")
    transform = make_transforms(
        img_size=data_config["img_size"],
        normalize=augmentation["normalize"],
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    preprocessor = Preprocessor(
        action_mean=torch.tensor(stats["action_mean"]),
        action_std=torch.tensor(stats["action_std"]),
        state_mean=torch.tensor(stats["state_mean"]),
        state_std=torch.tensor(stats["state_std"]),
        proprio_mean=torch.tensor(stats["proprio_mean"]),
        proprio_std=torch.tensor(stats["proprio_std"]),
        transform=transform,
        inverse_transform=make_inverse_transforms(
            img_size=data_config["img_size"],
            normalize=augmentation["normalize"],
        ),
    )
    model = init_module(
        folder=checkpoint.parent,
        checkpoint=checkpoint.name,
        model_kwargs=pretrain_config,
        device=device,
        action_dim=stats["action_dim"],
        proprio_dim=stats["proprio_dim"],
        preprocessor=preprocessor,
        cfgs_data=data_config,
        wrapper_kwargs=model_config["wrapper_kwargs"],
    )
    return model


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
