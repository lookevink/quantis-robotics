"""Task-specific offline gate for the reach-and-grasp inverse proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.grasp_recording import GraspDemonstrationEvidence
from jepa_wm.proposal_readiness import (
    ProposalReadinessThresholds,
    summarize_proposal_readiness,
)
from jepa_wm.trajectory import RolloutWindow
from jepa_wm.training_artifact import (
    TrainingArtifactMetadata,
    load_training_report_metadata,
    training_report_path,
)


GRASP_READINESS_SCHEMA = "quantis.jepa_wm_grasp_proposal_readiness.v1"
GRASP_WINDOW = RolloutWindow(69, 30, 1)
GRASP_EVALUATION_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)


def _grasp_training_report(proposal: Path) -> dict[str, object]:
    report = training_report_path(proposal.resolve())
    payload = json.loads(report.read_text())
    if not isinstance(payload, dict):
        raise ValueError("grasp proposal training report must be an object")
    return payload


def _validate_grasp_training_window(payload: dict[str, object]) -> None:
    if payload.get("window") != GRASP_WINDOW.to_dict():
        raise ValueError("grasp proposal training must select the complete task window")


def validate_grasp_training_window(proposal: Path) -> None:
    _validate_grasp_training_window(_grasp_training_report(proposal))


def validate_grasp_training_selection(
    proposal: Path,
    metadata: TrainingArtifactMetadata,
) -> None:
    payload = _grasp_training_report(proposal)
    _validate_grasp_training_window(payload)
    if payload.get("selection_bounds") != GRASP_EVALUATION_BOUNDS.to_dict():
        raise ValueError("grasp proposal training must include stationary actions")
    expected_indices = list(GRASP_WINDOW.context_indices)
    expected_selections = [
        {"recording": name, "context_indices": expected_indices}
        for name in metadata.training_recordings
    ]
    if (
        payload.get("rollouts")
        != len(metadata.training_recordings) * GRASP_WINDOW.count
        or payload.get("recording_selections") != expected_selections
    ):
        raise ValueError("grasp proposal training selection evidence is invalid")


def validate_grasp_evaluation_window(report: Path) -> Path:
    payload = json.loads(report.read_text())
    window = payload.get("window") if isinstance(payload, dict) else None
    if window != GRASP_WINDOW.to_dict():
        raise ValueError("grasp proposal evaluation must cover the complete task window")
    if payload.get("selection_bounds") != GRASP_EVALUATION_BOUNDS.to_dict():
        raise ValueError(
            "grasp proposal evaluation must include stationary hold windows"
        )
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
    validate_grasp_training_selection(proposal, metadata)
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
            minimum_rollouts_per_seed=GRASP_WINDOW.count,
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
