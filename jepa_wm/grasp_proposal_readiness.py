"""Task-specific offline gate for the reach-and-grasp inverse proposal."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.grasp_recording import GraspDemonstrationEvidence
from jepa_wm.task_proposal_readiness import (
    TaskProposalReadinessPolicy,
    run_task_proposal_readiness_cli,
)
from jepa_wm.task_windows import GRASP_PROPOSAL_WINDOW
from jepa_wm.training_artifact import (
    ProposalArtifactIdentity,
    ProposalConditioningCapabilities,
    TrainingArtifactMetadata,
)


GRASP_READINESS_SCHEMA = "quantis.jepa_wm_grasp_proposal_readiness.v1"
GRASP_WINDOW = GRASP_PROPOSAL_WINDOW
GRASP_EVALUATION_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)
GRASP_CONDITIONING = ProposalConditioningCapabilities(True, True, True, True)


def _validate_recording(recording: Path, split: str) -> None:
    GraspDemonstrationEvidence.from_recording(recording, expected_split=split)


GRASP_READINESS = TaskProposalReadinessPolicy(
    task_name="grasp",
    window=GRASP_WINDOW,
    bounds=GRASP_EVALUATION_BOUNDS,
    conditioning=GRASP_CONDITIONING,
    schema=GRASP_READINESS_SCHEMA,
    scope="offline reach-and-grasp inverse proposal; no live task completion",
    window_description="the complete task window",
    stationary_description="stationary hold windows while attached",
    validate_recording=_validate_recording,
)


def validate_grasp_proposal_identity(proposal: Path) -> ProposalArtifactIdentity:
    return GRASP_READINESS.validate_proposal_identity(proposal)


def validate_grasp_training_window(proposal: Path) -> None:
    GRASP_READINESS.validate_training_window(proposal)


def validate_grasp_training_selection(
    proposal: Path,
    metadata: TrainingArtifactMetadata,
) -> None:
    GRASP_READINESS.validate_training_selection(proposal, metadata)


def validate_grasp_evaluation_window(
    report: Path,
    *,
    proposal_identity: ProposalArtifactIdentity | None = None,
) -> Path:
    return GRASP_READINESS.validate_evaluation(
        report,
        proposal_identity=proposal_identity,
    )


def validate_grasp_goal_deltas(recording: Path, results: object) -> None:
    GRASP_READINESS.validate_goal_deltas(recording, results)


def summarize_grasp_proposal_readiness(
    proposal: Path,
    evaluation_reports: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    return GRASP_READINESS.summarize(proposal, evaluation_reports, output)


def main(argv: Sequence[str] | None = None) -> int:
    return run_task_proposal_readiness_cli(GRASP_READINESS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
