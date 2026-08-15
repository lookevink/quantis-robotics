"""Embed a captured episode with the official Hugging Face V-JEPA 2 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor


DEFAULT_MODEL = "facebook/vjepa2-vitl-fpc64-256"


def sample_paths(paths: list[Path], frames: int) -> list[Path]:
    if not paths:
        raise ValueError("episode contains no PNG frames")
    indices = np.linspace(0, len(paths) - 1, num=frames).round().astype(int)
    return [paths[index] for index in indices]


def pool_features(features: torch.Tensor) -> torch.Tensor:
    if features.ndim < 2:
        raise RuntimeError(f"unexpected V-JEPA feature shape: {tuple(features.shape)}")
    features = features.float()
    if features.ndim == 2:
        return features
    return features.mean(dim=tuple(range(1, features.ndim - 1)))


def embed_episode(
    episode_dir: Path,
    *,
    model_id: str,
    frame_count: int,
) -> np.ndarray:
    frame_paths = sample_paths(sorted((episode_dir / "rgb").rglob("*.png")), frame_count)
    images = [Image.open(path).convert("RGB") for path in frame_paths]
    video = torch.stack(
        [torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1) for image in images]
    )
    processor = AutoVideoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    inputs = processor(video, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_vision_features(**inputs)
        embedding = pool_features(features)
        embedding = torch.nn.functional.normalize(embedding, dim=-1)
    return embedding[0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--goal", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    embedding = embed_episode(args.episode, model_id=args.model, frame_count=args.frames)
    output = args.output or args.episode / "vjepa2_embedding.npy"
    np.save(output, embedding)

    result: dict[str, object] = {
        "episode": str(args.episode),
        "model": args.model,
        "frames": args.frames,
        "embedding": str(output),
        "dimensions": int(embedding.shape[0]),
    }
    if args.goal:
        goal = embed_episode(args.goal, model_id=args.model, frame_count=args.frames)
        result["goal"] = str(args.goal)
        result["cosine_similarity"] = float(np.dot(embedding, goal))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
