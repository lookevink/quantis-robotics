"""CLI for cached V-JEPA embeddings of labeled recording stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa.embed_episode import DEFAULT_MODEL, Encoder
from jepa.stage_embeddings import embed_recording_stages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    results = embed_recording_stages(
        args.recording,
        camera=args.camera,
        model_id=args.model,
        encoder_factory=lambda: Encoder(args.model),
    )
    print(
        json.dumps(
            {
                "recording": str(args.recording),
                "camera": args.camera,
                "model": args.model,
                "embeddings": {
                    result.stage.value: {
                        "path": str(result.path),
                        "cached": result.cached,
                    }
                    for result in results
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
