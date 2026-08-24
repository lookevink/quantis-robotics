"""Task-specific offline gate for the reach-and-grasp inverse proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from jepa_wm.grasp_recording import GraspDemonstrationEvidence
from jepa_wm.proposal_readiness import (
    ProposalReadinessThresholds,
    summarize_proposal_readiness,
)
from jepa_wm.training_artifact import load_training_report_metadata


GRASP_READINESS_SCHEMA = "quantis.jepa_wm_grasp_proposal_readiness.v1"
GRASP_WINDOW_START = 69
GRASP_WINDOW_COUNT = 30
GRASP_WINDOW_STRIDE = 1


def validate_grasp_evaluation_window(report: Path) -> Path:
    payload = json.loads(report.read_text())
    window = payload.get("window") if isinstance(payload, dict) else None
    if window != {
        "start_index": GRASP_WINDOW_START,
        "count": GRASP_WINDOW_COUNT,
        "stride": GRASP_WINDOW_STRIDE,
    }:
        raise ValueError("grasp proposal evaluation must cover the complete task window")
    recording = Path(str(payload.get("recording"))).resolve()
    GraspDemonstrationEvidence.from_recording(
        recording,
        expected_split="held_out",
    )
    return recording


def summarize_grasp_proposal_readiness(
    proposal: Path,
    evaluation_reports: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    if not evaluation_reports:
        raise ValueError("grasp readiness requires held-out evaluation reports")
    held_out_recordings = tuple(
        validate_grasp_evaluation_window(report.resolve())
        for report in evaluation_reports
    )
    metadata = load_training_report_metadata(proposal.resolve())
    recording_root = held_out_recordings[0].parent
    for name in metadata.training_recordings:
        GraspDemonstrationEvidence.from_recording(
            recording_root / name,
            expected_split="train",
        )
    summary = summarize_proposal_readiness(
        proposal,
        evaluation_reports,
        output,
        thresholds=ProposalReadinessThresholds(
            minimum_rollouts_per_seed=GRASP_WINDOW_COUNT,
        ),
    )
    summary["schema"] = GRASP_READINESS_SCHEMA
    summary["scope"] = (
        "offline reach-and-grasp inverse proposal; no live task completion"
    )
    output.resolve().write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = summarize_grasp_proposal_readiness(
        arguments.proposal,
        arguments.evaluation_report,
        arguments.output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
