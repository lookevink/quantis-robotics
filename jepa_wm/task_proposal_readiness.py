"""Shared strict readiness boundary for task-conditioned inverse proposals."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.action import DroidAction
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.proposal_evidence import ProposalGoalEvidence
from jepa_wm.proposal_readiness import (
    ProposalReadinessThresholds,
    build_proposal_readiness,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.trajectory import RolloutWindow, load_rollout_at
from jepa_wm.training_artifact import (
    ProposalArtifactIdentity,
    ProposalConditioningCapabilities,
    TrainingArtifactMetadata,
    load_training_report_metadata,
    rollout_training_selection_fingerprint,
    training_report_path,
)
from sim.exploration import DatasetSplit


RecordingValidator = Callable[[Path, str], None]


@dataclass(frozen=True)
class TaskProposalArtifactEvidence:
    identity: ProposalArtifactIdentity
    metadata: TrainingArtifactMetadata


@dataclass(frozen=True)
class TaskCorpusExpectation:
    recording: str
    split: str
    seed: int

    def __post_init__(self) -> None:
        DatasetSplit(self.split)
        if not self.recording or self.seed < 0:
            raise ValueError("task corpus expectation is invalid")


@dataclass(frozen=True)
class TaskProposalReadinessPolicy:
    task_name: str
    window: RolloutWindow
    bounds: ActionSelectionBounds
    conditioning: ProposalConditioningCapabilities
    schema: str
    scope: str
    window_description: str
    stationary_description: str
    validate_recording: RecordingValidator
    require_training_selection_fingerprint: bool = False
    minimum_warmup_frames: int = 4

    def training_report(self, proposal: Path) -> dict[str, object]:
        payload = json.loads(training_report_path(proposal.resolve()).read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                f"{self.task_name} proposal training report must be an object"
            )
        return payload

    def validate_conditioning(self, payload: dict[str, object]) -> None:
        try:
            conditioning = ProposalConditioningCapabilities.from_dict(
                payload.get("conditioning")
            )
        except ValueError as error:
            raise ValueError(
                f"{self.task_name} proposal conditioning evidence is invalid"
            ) from error
        if conditioning != self.conditioning:
            raise ValueError(
                f"{self.task_name} proposal training requires pose, action-history, "
                "goal-delta, and task-progress conditioning"
            )

    def _validate_proposal_identity(
        self,
        proposal: Path,
        payload: dict[str, object],
    ) -> ProposalArtifactIdentity:
        proposal = proposal.resolve()
        self.validate_conditioning(payload)
        identity = ProposalArtifactIdentity.from_artifact(proposal)
        if payload.get("proposal_fingerprint") != identity.fingerprint:
            raise ValueError(
                f"{self.task_name} proposal fingerprint does not match its "
                "training report"
            )
        from jepa_wm.proposal import load_action_proposal_with_training_selection
        import torch

        checkpoint, checkpoint_metadata, checkpoint_selection_fingerprint = (
            load_action_proposal_with_training_selection(
            proposal,
            device=torch.device("cpu"),
            )
        )
        sidecar_metadata = TrainingArtifactMetadata.from_dict(payload.get("metadata"))
        checkpoint_conditioning = ProposalConditioningCapabilities(
            checkpoint.uses_proprioception,
            checkpoint.uses_action_history,
            checkpoint.uses_goal_delta,
            checkpoint.uses_task_progress,
        )
        selection_fingerprint = rollout_training_selection_fingerprint(
            {
                field: payload.get(field)
                for field in (
                    "window",
                    "selection_bounds",
                    "recording_selections",
                    "rollouts",
                )
            }
        )
        if (
            checkpoint_metadata != sidecar_metadata
            or checkpoint_conditioning != self.conditioning
            or (
                self.require_training_selection_fingerprint
                and (
                    checkpoint_selection_fingerprint != selection_fingerprint
                    or payload.get("training_selection_fingerprint")
                    != selection_fingerprint
                )
            )
        ):
            raise ValueError(
                f"{self.task_name} proposal checkpoint and training report disagree"
            )
        return identity

    def validate_proposal_identity(self, proposal: Path) -> ProposalArtifactIdentity:
        proposal = proposal.resolve()
        return self._validate_proposal_identity(
            proposal,
            self.training_report(proposal),
        )

    def validate_training_window(self, proposal: Path) -> None:
        if self.training_report(proposal).get("window") != self.window.to_dict():
            raise ValueError(
                f"{self.task_name} proposal training must select "
                f"{self.window_description}"
            )

    def _validate_training_selection(
        self,
        payload: dict[str, object],
        metadata: TrainingArtifactMetadata,
    ) -> None:
        if payload.get("window") != self.window.to_dict():
            raise ValueError(
                f"{self.task_name} proposal training must select "
                f"{self.window_description}"
            )
        self.validate_conditioning(payload)
        if payload.get("selection_bounds") != self.bounds.to_dict():
            raise ValueError(
                f"{self.task_name} proposal training must include "
                f"{self.stationary_description}"
            )
        expected_indices = list(self.window.context_indices)
        expected_selections = [
            {"recording": name, "context_indices": expected_indices}
            for name in metadata.training_recordings
        ]
        if (
            payload.get("rollouts")
            != len(metadata.training_recordings) * self.window.count
            or payload.get("recording_selections") != expected_selections
        ):
            raise ValueError(
                f"{self.task_name} proposal training selection evidence is invalid"
            )

    def validate_training_selection(
        self,
        proposal: Path,
        metadata: TrainingArtifactMetadata,
    ) -> None:
        self._validate_training_selection(self.training_report(proposal), metadata)

    def validate_proposal(self, proposal: Path) -> TaskProposalArtifactEvidence:
        proposal = proposal.resolve()
        payload = self.training_report(proposal)
        identity = self._validate_proposal_identity(proposal, payload)
        metadata = TrainingArtifactMetadata.from_dict(payload.get("metadata"))
        self._validate_training_selection(payload, metadata)
        return TaskProposalArtifactEvidence(identity, metadata)

    def validate_goal_deltas(self, recording: Path, results: object) -> None:
        if not isinstance(results, list) or len(results) != self.window.count:
            raise ValueError(
                f"{self.task_name} evaluation goal-delta evidence is incomplete"
            )
        for context_index, result in zip(self.window.context_indices, results):
            rollout = load_rollout_at(
                recording,
                camera="wrist",
                context_index=context_index,
                bounds=self.bounds,
            )
            try:
                evidence = ProposalGoalEvidence.from_dict(result)
            except ValueError as error:
                raise ValueError(
                    f"{self.task_name} evaluation goal delta is invalid"
                ) from error
            if not evidence.validates(rollout):
                raise ValueError(
                    f"{self.task_name} evaluation goal delta does not match telemetry"
                )
            raw_actions = result.get("recorded_actions") if isinstance(result, dict) else None
            try:
                recorded_actions = tuple(
                    DroidAction(tuple(action)) for action in raw_actions
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{self.task_name} evaluation recorded actions are invalid"
                ) from error
            if recorded_actions != rollout.actions:
                raise ValueError(
                    f"{self.task_name} evaluation recorded actions do not match telemetry"
                )

    def validate_evaluation(
        self,
        report: Path,
        *,
        proposal_identity: ProposalArtifactIdentity | None = None,
    ) -> Path:
        payload = json.loads(report.read_text())
        window = payload.get("window") if isinstance(payload, dict) else None
        if window != self.window.to_dict():
            raise ValueError(
                f"{self.task_name} proposal evaluation must cover "
                f"{self.window_description}"
            )
        if payload.get("selection_bounds") != self.bounds.to_dict():
            raise ValueError(
                f"{self.task_name} proposal evaluation must include "
                f"{self.stationary_description}"
            )
        self.validate_conditioning(payload)
        if proposal_identity is not None and (
            Path(str(payload.get("proposal"))).resolve() != proposal_identity.path
            or payload.get("proposal_fingerprint") != proposal_identity.fingerprint
        ):
            raise ValueError(
                f"{self.task_name} evaluation proposal identity is invalid"
            )
        recording = Path(str(payload.get("recording"))).resolve()
        self.validate_recording(recording, "held_out")
        self.validate_goal_deltas(recording, payload.get("results"))
        return recording

    def summarize(
        self,
        proposal: Path,
        evaluation_reports: Sequence[Path],
        output: Path,
        *,
        corpus: Sequence[TaskCorpusExpectation] | None = None,
        summary_evidence: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        if not evaluation_reports:
            raise ValueError(
                f"{self.task_name} readiness requires held-out evaluation reports"
            )
        proposal = proposal.resolve()
        identity = self.validate_proposal_identity(proposal)
        held_out_recordings = tuple(
            self.validate_evaluation(report.resolve(), proposal_identity=identity)
            for report in evaluation_reports
        )
        metadata = load_training_report_metadata(proposal)
        self.validate_training_selection(proposal, metadata)
        recording_root = held_out_recordings[0].parent
        if corpus is not None:
            training_expectations = tuple(
                item for item in corpus if item.split == DatasetSplit.TRAIN.value
            )
            held_out_expectations = tuple(
                item for item in corpus if item.split == DatasetSplit.HELD_OUT.value
            )
            if (
                metadata.training_recordings
                != tuple(item.recording for item in training_expectations)
                or tuple(path.name for path in held_out_recordings)
                != tuple(item.recording for item in held_out_expectations)
            ):
                raise ValueError(
                    f"{self.task_name} proposal corpus does not match its roster"
                )
            for item in corpus:
                recording = DomainRecording.from_path(
                    recording_root / item.recording,
                    expected_split=DatasetSplit(item.split),
                )
                if recording.seed != item.seed:
                    raise ValueError(
                        f"{self.task_name} proposal corpus seed does not match its roster"
                    )
        for name in metadata.training_recordings:
            self.validate_recording(recording_root / name, "train")
        summary = build_proposal_readiness(
            proposal,
            evaluation_reports,
            ProposalReadinessThresholds(
                minimum_rollouts_per_seed=self.window.count,
                minimum_warmup_frames=self.minimum_warmup_frames,
            ),
        )
        summary["schema"] = self.schema
        summary["scope"] = self.scope
        summary["proposal_fingerprint"] = identity.fingerprint
        summary["training_selection_fingerprint"] = (
            self.training_report(proposal).get("training_selection_fingerprint")
        )
        if summary_evidence is not None:
            summary.update(summary_evidence)
        write_json_atomic(output.resolve(), summary)
        return summary


def run_task_proposal_readiness_cli(
    policy: TaskProposalReadinessPolicy,
    argv: Sequence[str] | None = None,
) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = policy.summarize(
        arguments.proposal,
        arguments.evaluation_report,
        arguments.output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2
