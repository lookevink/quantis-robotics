"""Task-specific offline gate for the post-attachment insertion proposal."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Sequence

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.insertion_corpus import InsertionCorpusRoster
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.task_proposal_readiness import (
    TaskCorpusExpectation,
    TaskProposalArtifactEvidence,
    TaskProposalReadinessPolicy,
)
from jepa_wm.task_windows import INSERTION_PROPOSAL_WINDOW
from jepa_wm.training_artifact import (
    ProposalArtifactIdentity,
    ProposalConditioningCapabilities,
    TrainingArtifactMetadata,
)


INSERTION_WINDOW = INSERTION_PROPOSAL_WINDOW
INSERTION_EVALUATION_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)
INSERTION_CONDITIONING = ProposalConditioningCapabilities(True, True, True, True)
INSERTION_READINESS_SCHEMA = "quantis.jepa_wm_insertion_proposal_readiness.v1"


def _validate_recording(recording: Path, split: str) -> None:
    ContactInsertionEvidence.from_recording(recording, expected_split=split)


INSERTION_READINESS = TaskProposalReadinessPolicy(
    task_name="insertion",
    window=INSERTION_WINDOW,
    bounds=INSERTION_EVALUATION_BOUNDS,
    conditioning=INSERTION_CONDITIONING,
    schema=INSERTION_READINESS_SCHEMA,
    scope="offline post-attachment insertion inverse proposal; no live insertion",
    window_description="the complete insertion window",
    stationary_description="stationary seated windows",
    validate_recording=_validate_recording,
    require_training_selection_fingerprint=True,
)


def validate_insertion_proposal_identity(
    proposal: Path,
) -> ProposalArtifactIdentity:
    return INSERTION_READINESS.validate_proposal_identity(proposal)


def validate_insertion_proposal(
    proposal: Path,
) -> TaskProposalArtifactEvidence:
    return INSERTION_READINESS.validate_proposal(proposal)


def validate_insertion_training_selection(
    proposal: Path,
    metadata: TrainingArtifactMetadata,
) -> None:
    INSERTION_READINESS.validate_training_selection(proposal, metadata)


def validate_insertion_evaluation_window(
    report: Path,
    *,
    proposal_identity: ProposalArtifactIdentity | None = None,
) -> Path:
    return INSERTION_READINESS.validate_evaluation(
        report,
        proposal_identity=proposal_identity,
    )


def validate_insertion_goal_deltas(recording: Path, results: object) -> None:
    INSERTION_READINESS.validate_goal_deltas(recording, results)


def summarize_insertion_proposal_readiness(
    proposal: Path,
    evaluation_reports: Sequence[Path],
    output: Path,
    roster: Path,
) -> dict[str, object]:
    corpus = InsertionCorpusRoster.load(roster.resolve())
    return INSERTION_READINESS.summarize(
        proposal,
        evaluation_reports,
        output,
        corpus=tuple(
            TaskCorpusExpectation(
                recording.recording_id,
                recording.split,
                recording.seed,
            )
            for recording in corpus.recordings
        ),
        summary_evidence={"corpus_roster": corpus.to_dict()},
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = summarize_insertion_proposal_readiness(
        arguments.proposal,
        arguments.evaluation_report,
        arguments.output,
        arguments.roster,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
