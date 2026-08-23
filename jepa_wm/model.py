"""Official headless DROID JEPA-WM model initialization."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import torch
import yaml

from jepa_wm.adapter import apply_action_adapter


CONFIG_RELATIVE_PATH = Path(
    "configs/evals/simu_env_planning/droid/jepa-wm/"
    "droid_L2_cem_sourcedset_H3_nas3_maxnorm01_ctxt2_gH3_"
    "r256_alpha0_ep64_decode.yaml"
)


def load_headless_model(
    source: Path,
    checkpoint: Path,
    *,
    device: torch.device,
    adapter: Path | None = None,
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
    if adapter is not None:
        apply_action_adapter(
            model,
            adapter,
            expected_source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
        )
    return model
