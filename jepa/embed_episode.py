"""Embed a captured episode with the official Hugging Face V-JEPA 2 model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.contract import DEFAULT_FRAMES

if TYPE_CHECKING:
    import torch


DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"


@dataclass(frozen=True)
class ObservationSource:
    """A demo recording or legacy capture episode and its frame layout."""

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
            if camera not in self.cameras:
                raise ValueError(
                    f"recording has no {camera!r} camera; "
                    f"available cameras: {list(self.cameras)}"
                )
            return sorted((self.path / camera).glob("frame_*.png"))
        return sorted((self.path / "rgb").rglob("*.png"))

    def default_embedding_path(self, camera: str) -> Path:
        if self.cameras is not None:
            return self.path / f"{camera}_vjepa2_embedding.npy"
        return self.path / "vjepa2_embedding.npy"


def sample_paths(paths: list[Path], frames: int) -> list[Path]:
    """Pick `frames` evenly spaced frames, refusing to invent any."""
    if not paths:
        raise ValueError("episode contains no PNG frames")
    if len(paths) < frames:
        raise ValueError(
            f"episode has {len(paths)} frames but {frames} were requested; "
            "upsampling would repeat frames and flatten the motion the encoder "
            f"reads. Capture at least {frames} frames, or pass "
            f"--frames {len(paths)} to embed the shorter clip as captured."
        )
    indices = np.linspace(0, len(paths) - 1, num=frames).round().astype(int)
    return [paths[index] for index in indices]


def pool_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise RuntimeError(f"unexpected V-JEPA feature shape: {tuple(features.shape)}")
    features = features.float()
    if features.ndim == 2:
        return features
    return features.mean(dim=tuple(range(1, features.ndim - 1)))


def resolve_device() -> torch.device:
    """Prefer CUDA on the GPU host, Metal on an Apple workstation."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Encoder:
    """A loaded V-JEPA 2 encoder. Construct once, embed many episodes."""

    def __init__(self, model_id: str = DEFAULT_MODEL) -> None:
        from transformers import AutoModel, AutoVideoProcessor

        self.model_id = model_id
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = resolve_device()
        self.model.to(self.device).eval()

    def embed(
        self,
        source: ObservationSource,
        *,
        camera: str = "wrist",
        frame_count: int = DEFAULT_FRAMES,
    ) -> np.ndarray:
        import torch
        from PIL import Image

        frame_paths = sample_paths(source.frame_paths(camera), frame_count)
        images = [Image.open(path).convert("RGB") for path in frame_paths]
        video = torch.stack(
            [
                torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
                for image in images
            ]
        )

        inputs = self.processor(video, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self.model.get_vision_features(**inputs)
            embedding = pool_features(features)
            embedding = torch.nn.functional.normalize(embedding, dim=-1)
        return embedding[0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--goal", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    source = ObservationSource.open(args.source)
    encoder = Encoder(args.model)
    embedding = encoder.embed(
        source,
        camera=args.camera,
        frame_count=args.frames,
    )
    output = args.output or source.default_embedding_path(args.camera)
    np.save(output, embedding)

    result: dict[str, object] = {
        "source": str(args.source),
        "camera": args.camera,
        "model": args.model,
        "device": str(encoder.device),
        "frames": args.frames,
        "embedding": str(output),
        "dimensions": int(embedding.shape[0]),
    }
    if args.goal:
        goal_source = ObservationSource.open(args.goal)
        goal = encoder.embed(
            goal_source,
            camera=args.camera,
            frame_count=args.frames,
        )
        result["goal"] = str(args.goal)
        result["cosine_similarity"] = float(np.dot(embedding, goal))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
