"""CLI for held-out JEPA stage-similarity evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa.contract import OFFLINE_CONFIDENCE_THRESHOLDS, ConfidenceThresholds
from jepa.embed_episode import DEFAULT_MODEL, Encoder
from jepa.stage_embeddings import embed_recording_stages
from jepa.stage_scoring import score_stage_recording


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=OFFLINE_CONFIDENCE_THRESHOLDS.min_similarity,
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=OFFLINE_CONFIDENCE_THRESHOLDS.min_margin,
    )
    args = parser.parse_args()

    encoder: Encoder | None = None

    def encoder_factory() -> Encoder:
        nonlocal encoder
        if encoder is None:
            encoder = Encoder(args.model)
        return encoder

    for recording in [*args.reference, args.query]:
        embed_recording_stages(
            recording,
            camera=args.camera,
            model_id=args.model,
            encoder_factory=encoder_factory,
        )
    report = score_stage_recording(
        args.reference,
        args.query,
        camera=args.camera,
        thresholds=ConfidenceThresholds(args.min_similarity, args.min_margin),
    )
    report_path = args.query / "jepa" / args.camera / "stage_report.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
