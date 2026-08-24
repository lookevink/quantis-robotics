"""Strict offline readiness for the insertion-window JEPA-WM adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import fsum, isclose, isfinite
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    ActionSelectionBounds,
    DroidAction,
)
from jepa_wm.adapter import load_action_adapter_contract
from jepa_wm.contract import MODEL_ID
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.experiment import HeldOutEvaluation, build_experiment_from_evidence
from jepa_wm.insertion_corpus import InsertionCorpusRoster
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.persistence import write_json_atomic
from jepa_wm.readiness import ActionControlGate
from jepa_wm.task_windows import INSERTION_PROPOSAL_WINDOW
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    artifact_fingerprint,
    load_training_report,
    rollout_training_selection_fingerprint,
)
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL, load_rollout_at
from sim.exploration import DatasetSplit


INSERTION_WM_SCHEMA = "quantis.jepa_wm_insertion_world_model_readiness.v1"
INSERTION_WINDOW = INSERTION_PROPOSAL_WINDOW
INSERTION_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)


@dataclass(frozen=True)
class InsertionAdapterEvidence:
    identity: ArtifactIdentity
    metadata: TrainingArtifactMetadata
    training_selection_fingerprint: str


def _training_selection(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        field: payload.get(field)
        for field in (
            "window",
            "selection_bounds",
            "recording_selections",
            "rollouts",
        )
    }


def validate_insertion_adapter(adapter: Path) -> InsertionAdapterEvidence:
    adapter = adapter.resolve()
    payload = load_training_report(adapter)
    identity = ArtifactIdentity.from_artifact(adapter)
    checkpoint_metadata, checkpoint_selection = load_action_adapter_contract(adapter)
    sidecar_metadata = TrainingArtifactMetadata.from_dict(payload.get("metadata"))
    selection = _training_selection(payload)
    selection_fingerprint = rollout_training_selection_fingerprint(selection)
    expected_indices = list(INSERTION_WINDOW.context_indices)
    expected_selections = [
        {"recording": recording, "context_indices": expected_indices}
        for recording in sidecar_metadata.training_recordings
    ]
    if (
        payload.get("adapter_fingerprint") != identity.fingerprint
        or checkpoint_metadata != sidecar_metadata
        or payload.get("window") != INSERTION_WINDOW.to_dict()
        or payload.get("selection_bounds") != INSERTION_BOUNDS.to_dict()
        or payload.get("recording_selections") != expected_selections
        or payload.get("rollouts")
        != len(sidecar_metadata.training_recordings) * INSERTION_WINDOW.count
        or payload.get("training_selection_fingerprint") != selection_fingerprint
        or checkpoint_selection != selection_fingerprint
    ):
        raise ValueError("insertion adapter checkpoint and training evidence disagree")
    return InsertionAdapterEvidence(
        identity,
        sidecar_metadata,
        selection_fingerprint,
    )


def _validated_evaluation_metrics(
    recording: Path,
    results: Any,
) -> tuple[float, float]:
    if not isinstance(results, list) or len(results) != INSERTION_WINDOW.count:
        raise ValueError("insertion adapter evaluation results are incomplete")
    improvements = []
    wins = 0
    for context_index, result in zip(INSERTION_WINDOW.context_indices, results):
        if not isinstance(result, dict):
            raise ValueError("insertion adapter rollout result is invalid")
        rollout = load_rollout_at(
            recording,
            camera="wrist",
            context_index=context_index,
            bounds=INSERTION_BOUNDS,
        )
        try:
            actions = tuple(
                DroidAction(tuple(action)) for action in result.get("actions", ())
            )
            recorded_value = result["recorded_action_energy"]
            zero_value = result["zero_action_energy"]
            improvement_value = result["improvement_over_zero"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion adapter rollout evidence is invalid") from error
        if (
            type(recorded_value) not in (int, float)
            or type(zero_value) not in (int, float)
            or type(improvement_value) not in (int, float)
        ):
            raise ValueError("insertion adapter rollout energies must be native numbers")
        recorded = float(recorded_value)
        zero = float(zero_value)
        improvement = float(improvement_value)
        win = result.get("recorded_action_wins")
        if (
            actions != rollout.actions
            or result.get("context_indices")
            != [frame.index for frame in rollout.context]
            or result.get("target_index") != rollout.target.index
            or not all(isfinite(value) for value in (recorded, zero, improvement))
            or recorded < 0.0
            or zero < 0.0
            or not isclose(improvement, zero - recorded, rel_tol=1e-9, abs_tol=1e-12)
            or not isinstance(win, bool)
            or win != (improvement > 0.0)
        ):
            raise ValueError("insertion adapter rollout evidence is inconsistent")
        improvements.append(improvement)
        wins += int(win)
    return fsum(improvements) / len(improvements), wins / len(improvements)


def validate_insertion_adapter_evaluation(
    report: Path,
    adapter: InsertionAdapterEvidence,
    *,
    expected_recording: str,
    expected_seed: int,
) -> HeldOutEvaluation:
    payload = json.loads(report.resolve().read_text())
    if not isinstance(payload, dict):
        raise ValueError("insertion adapter evaluation must be an object")
    recording = Path(str(payload.get("recording"))).resolve()
    if (
        recording.name != expected_recording
        or payload.get("camera") != "wrist"
        or Path(str(payload.get("adapter"))).resolve() != adapter.identity.path
        or payload.get("adapter_fingerprint") != adapter.identity.fingerprint
        or payload.get("rollout_window") != INSERTION_WINDOW.to_dict()
        or payload.get("action_selection") != INSERTION_BOUNDS.to_dict()
        or payload.get("rollouts") != INSERTION_WINDOW.count
        or payload.get("model") != MODEL_ID
        or payload.get("source_revision") != adapter.metadata.source_revision
        or payload.get("rollout_protocol") != DROID_ROLLOUT_PROTOCOL.to_dict()
        or payload.get("action_format") != ACTION_RECORDING_CONTRACT.format
        or payload.get("objective") != "terminal_latent_l2"
    ):
        raise ValueError("insertion adapter evaluation identity is invalid")
    ContactInsertionEvidence.from_recording(
        recording,
        expected_split="held_out",
        expected_seed=expected_seed,
    )
    mean_improvement, win_rate = _validated_evaluation_metrics(
        recording,
        payload.get("results"),
    )
    decision = ActionControlGate().evaluate(
        mean_improvement_over_zero=mean_improvement,
        recorded_action_win_rate=win_rate,
    )
    try:
        reported_improvement = float(payload["mean_improvement_over_zero"])
        reported_win_rate = float(payload["recorded_action_win_rate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("insertion adapter evaluation aggregates are invalid") from error
    if (
        not isclose(
            reported_improvement,
            mean_improvement,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or not isclose(
            reported_win_rate,
            win_rate,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
        or payload.get("control_gate") != decision.to_dict()
    ):
        raise ValueError("insertion adapter evaluation aggregates are inconsistent")
    return HeldOutEvaluation.from_payload(report.resolve(), payload)


def summarize_insertion_world_model_readiness(
    experiment_id: str,
    adapter_path: Path,
    evaluation_reports: Sequence[Path],
    roster_path: Path,
    output: Path,
) -> dict[str, Any]:
    roster = InsertionCorpusRoster.load(roster_path.resolve())
    adapter = validate_insertion_adapter(adapter_path)
    training = roster.for_split("train")
    held_out = roster.for_split("held_out")
    if adapter.metadata.training_recordings != tuple(
        recording.recording_id for recording in training
    ) or len(evaluation_reports) != len(held_out):
        raise ValueError("insertion adapter corpus does not match its roster")
    held_out_evidence = tuple(
        validate_insertion_adapter_evaluation(
            report,
            adapter,
            expected_recording=recording.recording_id,
            expected_seed=recording.seed,
        )
        for report, recording in zip(evaluation_reports, held_out)
    )
    recording_root = held_out_evidence[0].recording.path.parent
    training_evidence = []
    for recording in training:
        ContactInsertionEvidence.from_recording(
            recording_root / recording.recording_id,
            expected_split="train",
            expected_seed=recording.seed,
        )
        training_evidence.append(
            DomainRecording.from_path(
                recording_root / recording.recording_id,
                expected_split=DatasetSplit.TRAIN,
            )
        )
    summary = build_experiment_from_evidence(
        experiment_id,
        tuple(training_evidence),
        held_out_evidence,
    )
    summary.update(
        {
            "schema": INSERTION_WM_SCHEMA,
            "scope": "offline insertion world-model energy; no live insertion",
            "adapter_fingerprint": adapter.identity.fingerprint,
            "training_selection_fingerprint": (
                adapter.training_selection_fingerprint
            ),
            "corpus_roster": roster.to_dict(),
        }
    )
    write_json_atomic(output.resolve(), summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, action="append", required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = summarize_insertion_world_model_readiness(
        args.experiment_id,
        args.adapter,
        args.evaluation_report,
        args.roster,
        args.output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
