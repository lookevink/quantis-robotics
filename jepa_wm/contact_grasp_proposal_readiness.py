"""Strict offline gate for grasping from the contact-insertion domain."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.insertion_recording import ContactGraspEvidence
from jepa_wm.task_proposal_readiness import (
    TaskProposalReadinessPolicy,
    run_task_proposal_readiness_cli,
)
from jepa_wm.task_windows import CONTACT_GRASP_PROPOSAL_WINDOW
from jepa_wm.training_artifact import (
    ProposalArtifactIdentity,
    ProposalConditioningCapabilities,
    TrainingArtifactMetadata,
)


CONTACT_GRASP_READINESS_SCHEMA = (
    "quantis.jepa_wm_contact_grasp_proposal_readiness.v1"
)
CONTACT_GRASP_WINDOW = CONTACT_GRASP_PROPOSAL_WINDOW
CONTACT_GRASP_EVALUATION_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)
CONTACT_GRASP_CONDITIONING = ProposalConditioningCapabilities(True, True, True, True)


def _validate_recording(recording: Path, split: str) -> None:
    ContactGraspEvidence.from_recording(recording, expected_split=split)


CONTACT_GRASP_READINESS = TaskProposalReadinessPolicy(
    task_name="contact-grasp",
    window=CONTACT_GRASP_WINDOW,
    bounds=CONTACT_GRASP_EVALUATION_BOUNDS,
    conditioning=CONTACT_GRASP_CONDITIONING,
    schema=CONTACT_GRASP_READINESS_SCHEMA,
    scope="offline contact-domain grasp inverse proposal; no live task completion",
    window_description="the grasp-close and retained-attachment window",
    stationary_description="stationary retained-grasp windows",
    validate_recording=_validate_recording,
    require_training_selection_fingerprint=True,
)


def validate_contact_grasp_proposal_identity(
    proposal: Path,
) -> ProposalArtifactIdentity:
    return CONTACT_GRASP_READINESS.validate_proposal_identity(proposal)


def validate_contact_grasp_training_selection(
    proposal: Path,
    metadata: TrainingArtifactMetadata,
) -> None:
    CONTACT_GRASP_READINESS.validate_training_selection(proposal, metadata)


def validate_contact_grasp_evaluation_window(
    report: Path,
    *,
    proposal_identity: ProposalArtifactIdentity | None = None,
) -> Path:
    return CONTACT_GRASP_READINESS.validate_evaluation(
        report,
        proposal_identity=proposal_identity,
    )


def validate_contact_grasp_goal_deltas(recording: Path, results: object) -> None:
    CONTACT_GRASP_READINESS.validate_goal_deltas(recording, results)


def summarize_contact_grasp_proposal_readiness(
    proposal: Path,
    evaluation_reports: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    return CONTACT_GRASP_READINESS.summarize(
        proposal,
        evaluation_reports,
        output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_task_proposal_readiness_cli(CONTACT_GRASP_READINESS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
