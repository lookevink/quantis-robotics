"""JSONL worker process for online stage predictions; not wired to Isaac yet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jepa.contract import ONLINE_CONFIDENCE_THRESHOLDS, ConfidenceThresholds
from jepa.embed_episode import DEFAULT_MODEL, Encoder
from jepa.online_predictor import OnlineStagePredictor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=ONLINE_CONFIDENCE_THRESHOLDS.min_similarity,
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=ONLINE_CONFIDENCE_THRESHOLDS.min_margin,
    )
    args = parser.parse_args()

    predictor = OnlineStagePredictor(
        args.reference,
        camera=args.camera,
        encoder=Encoder(args.model),
        model_id=args.model,
        thresholds=ConfidenceThresholds(args.min_similarity, args.min_margin),
    )
    for line in sys.stdin:
        request = json.loads(line)
        frames = [Path(frame) for frame in request.get("frames", [])]
        missing = [str(frame) for frame in frames if not frame.is_file()]
        if missing:
            raise ValueError(f"online request contains missing frames: {missing}")
        prediction = predictor.predict(frames)
        print(json.dumps(prediction.to_dict()), flush=True)


if __name__ == "__main__":
    main()
