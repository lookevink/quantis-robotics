"""Fail-closed filesystem and environment contract for headless JEPA-WM."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping


DINOV3_CHECKPOINT_NAME = (
    "dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
)
DINOV3_APPROVED_CHECKPOINT_NAME = (
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
DINOV3_APPROVED_CHECKPOINT_FINGERPRINT = (
    "8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035"
)
JEPA_WM_CHECKPOINT_FINGERPRINT = (
    "daa69198aef764932f1cb809239a4e19c71da20a93c6a0b9f3869cb30a13f4aa"
)


def _required_path(environment: Mapping[str, str], name: str) -> Path:
    value = environment.get(name)
    if not value:
        raise ValueError(f"headless JEPA-WM runtime requires {name}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"headless JEPA-WM runtime requires absolute {name}")
    return path.resolve()


def runtime_artifact_fingerprint(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_model_load_preflight(output: Path) -> tuple[Path, dict[str, object]]:
    """Exclusively reserve one model-load preflight before touching CUDA."""

    output = output.resolve()
    claim = output.with_name(f"{output.stem}-claim{output.suffix}")
    if output.exists():
        raise ValueError(f"model-load preflight already exists: {output}")
    payload: dict[str, object] = {
        "schema": "quantis.jepa_wm_model_load_preflight_claim.v1",
        "status": "claimed",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output),
        "recordings_loaded": False,
        "canonical_accessed": False,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("model-load preflight was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(claim.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return claim, payload


def validate_headless_runtime(
    source: Path,
    checkpoint: Path,
    *,
    environment: Mapping[str, str] = os.environ,
    expected_dinov3_fingerprint: str = DINOV3_APPROVED_CHECKPOINT_FINGERPRINT,
    expected_checkpoint_fingerprint: str = JEPA_WM_CHECKPOINT_FINGERPRINT,
) -> dict[str, str]:
    """Authenticate the paths upstream uses before constructing the model."""

    source = source.resolve()
    checkpoint = checkpoint.resolve()
    source_root = _required_path(environment, "JEPAWM_HOME")
    checkpoint_root = _required_path(environment, "JEPAWM_OSSCKPT")
    torch_home = _required_path(environment, "TORCH_HOME")
    runtime_root = source_root.parent
    if checkpoint_root != runtime_root / "checkpoints":
        raise ValueError("JEPAWM_OSSCKPT is outside the authenticated runtime root")
    if torch_home != runtime_root / "cache" / "torch":
        raise ValueError("TORCH_HOME is outside the authenticated runtime root")
    expected_source = source_root / "jepa-wms"
    if source != expected_source:
        raise ValueError(
            f"JEPA-WM source must be {expected_source}, received {source}"
        )
    if checkpoint.parent != checkpoint_root:
        raise ValueError(
            "JEPA-WM checkpoint is outside the authenticated checkpoint root"
        )
    dinov3_source = source_root / "dinov3"
    hubconf = dinov3_source / "hubconf.py"
    dinov3_checkpoint = checkpoint_root / "dinov3" / DINOV3_CHECKPOINT_NAME
    approved_checkpoint = (
        checkpoint_root / "dinov3" / DINOV3_APPROVED_CHECKPOINT_NAME
    )
    cached_checkpoint = (
        torch_home / "hub" / "checkpoints" / DINOV3_APPROVED_CHECKPOINT_NAME
    )
    required_files = {
        "JEPA-WM checkpoint": checkpoint,
        "DINOv3 hubconf": hubconf,
        "DINOv3 checkpoint": dinov3_checkpoint,
        "approved DINOv3 checkpoint": approved_checkpoint,
        "cached DINOv3 checkpoint": cached_checkpoint,
    }
    missing = [name for name, path in required_files.items() if not path.is_file()]
    if missing:
        raise ValueError(
            "headless JEPA-WM runtime is incomplete: " + ", ".join(missing)
        )
    if not (
        os.path.samefile(dinov3_checkpoint, approved_checkpoint)
        and os.path.samefile(cached_checkpoint, approved_checkpoint)
    ):
        raise ValueError("DINOv3 checkpoint aliases do not share one artifact")
    dinov3_fingerprint = runtime_artifact_fingerprint(approved_checkpoint)
    if dinov3_fingerprint != expected_dinov3_fingerprint:
        raise ValueError("approved DINOv3 checkpoint fingerprint changed")
    checkpoint_fingerprint = runtime_artifact_fingerprint(checkpoint)
    if checkpoint_fingerprint != expected_checkpoint_fingerprint:
        raise ValueError("JEPA-WM checkpoint fingerprint changed")
    revisions = {}
    for name, path, variable in (
        ("JEPA-WM", source, "JEPA_WM_REVISION"),
        ("DINOv3", dinov3_source, "DINOV3_REVISION"),
    ):
        expected_revision = environment.get(variable)
        if not expected_revision:
            raise ValueError(f"headless JEPA-WM runtime requires {variable}")
        try:
            actual_revision = subprocess.run(
                ("git", "-C", str(path), "rev-parse", "HEAD"),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            tracked = subprocess.run(
                (
                    "git",
                    "-C",
                    str(path),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ),
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"{name} source revision cannot be authenticated"
            ) from error
        if actual_revision != expected_revision or tracked:
            raise ValueError(f"{name} source revision changed")
        revisions[variable] = actual_revision
    return {
        "source": str(source),
        "checkpoint": str(checkpoint),
        "dinov3_source": str(dinov3_source),
        "dinov3_checkpoint": str(dinov3_checkpoint),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "dinov3_checkpoint_fingerprint": dinov3_fingerprint,
        "torch_home": str(torch_home),
        "source_revision": revisions["JEPA_WM_REVISION"],
        "dinov3_revision": revisions["DINOV3_REVISION"],
    }
