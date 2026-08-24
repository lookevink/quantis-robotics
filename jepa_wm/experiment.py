"""Aggregate whole-seed JEPA-WM evaluations into one readiness decision."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from math import fsum, isfinite
import json
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.domain_recording import DomainRecording
from jepa_wm.persistence import write_json_atomic
from jepa_wm.readiness import ActionControlDecision, ActionControlGate
from jepa_wm.training_artifact import load_training_report_metadata
from sim.exploration import DatasetSplit


EXPERIMENT_SCHEMA = "quantis.jepa_wm_domain_experiment.v1"


@dataclass(frozen=True)
class EvaluationMetrics:
    rollouts: int
    improvement_sum: float
    recorded_action_wins: int

    def __post_init__(self) -> None:
        if (
            self.rollouts <= 0
            or not isfinite(self.improvement_sum)
            or not 0 <= self.recorded_action_wins <= self.rollouts
        ):
            raise ValueError("evaluation evidence is inconsistent")

    @property
    def mean_improvement_over_zero(self) -> float:
        return self.improvement_sum / self.rollouts

    @property
    def recorded_action_win_rate(self) -> float:
        return self.recorded_action_wins / self.rollouts

    @property
    def control_gate(self) -> ActionControlDecision:
        return ActionControlGate().evaluate(
            mean_improvement_over_zero=self.mean_improvement_over_zero,
            recorded_action_win_rate=self.recorded_action_win_rate,
        )

    @classmethod
    def from_results(cls, results: Any) -> EvaluationMetrics:
        if not isinstance(results, list) or not results:
            raise ValueError("held-out report must contain rollout results")
        improvements = []
        wins = 0
        for result in results:
            if not isinstance(result, dict):
                raise ValueError("held-out rollout result must be an object")
            improvement = result.get("improvement_over_zero")
            recorded_action_wins = result.get("recorded_action_wins")
            if (
                not isinstance(improvement, (int, float))
                or not isfinite(improvement)
                or not isinstance(recorded_action_wins, bool)
                or recorded_action_wins != (improvement > 0.0)
            ):
                raise ValueError("held-out rollout metrics are inconsistent")
            improvements.append(float(improvement))
            wins += int(recorded_action_wins)
        return cls(
            rollouts=len(improvements),
            improvement_sum=fsum(improvements),
            recorded_action_wins=wins,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rollouts": self.rollouts,
            "mean_improvement_over_zero": self.mean_improvement_over_zero,
            "recorded_action_win_rate": self.recorded_action_win_rate,
            "control_gate": self.control_gate.to_dict(),
        }

    @classmethod
    def aggregate(
        cls,
        metrics: Sequence[EvaluationMetrics],
    ) -> EvaluationMetrics:
        if not metrics:
            raise ValueError("at least one held-out metric is required")
        return cls(
            rollouts=sum(metric.rollouts for metric in metrics),
            improvement_sum=fsum(metric.improvement_sum for metric in metrics),
            recorded_action_wins=sum(metric.recorded_action_wins for metric in metrics),
        )


@dataclass(frozen=True)
class HeldOutEvaluation:
    recording: DomainRecording
    report: Path
    camera: str
    adapter: Path
    metrics: EvaluationMetrics

    @classmethod
    def from_path(cls, path: Path) -> HeldOutEvaluation:
        payload = json.loads(path.read_text())
        return cls.from_payload(path, payload)

    @classmethod
    def from_payload(cls, path: Path, payload: Any) -> HeldOutEvaluation:
        if not isinstance(payload, dict):
            raise ValueError(f"held-out report must be an object: {path}")
        recording_path = payload.get("recording")
        camera = payload.get("camera")
        adapter = payload.get("adapter")
        if not all(
            isinstance(value, str) and value
            for value in (recording_path, camera, adapter)
        ):
            raise ValueError(f"held-out report provenance is incomplete: {path}")
        recording = DomainRecording.from_path(
            Path(recording_path),
            expected_split=DatasetSplit.HELD_OUT,
        )
        if path.resolve().parent.parent != recording.path:
            raise ValueError(f"held-out report is outside its recording: {path}")
        return cls(
            recording=recording,
            report=path.resolve(),
            camera=camera,
            adapter=Path(adapter).resolve(),
            metrics=EvaluationMetrics.from_results(payload.get("results")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording": self.recording.name,
            "seed": self.recording.seed,
            "report": str(self.report),
            **self.metrics.to_dict(),
        }


def _require_unique_disjoint_recordings(
    training: Sequence[DomainRecording],
    held_out: Sequence[HeldOutEvaluation],
) -> None:
    training_names = [recording.name for recording in training]
    held_out_names = [evaluation.recording.name for evaluation in held_out]
    all_seeds = [recording.seed for recording in training] + [
        evaluation.recording.seed for evaluation in held_out
    ]
    if len(set(training_names)) != len(training_names):
        raise ValueError("training recordings must be unique")
    if len(set(held_out_names)) != len(held_out_names):
        raise ValueError("held-out recordings must be unique")
    if set(training_names) & set(held_out_names):
        raise ValueError("training and held-out recordings must be disjoint")
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError("training and held-out seeds must be unique")


def _validate_adapter_training_set(
    adapter: Path,
    camera: str,
    training: Sequence[DomainRecording],
) -> None:
    metadata = load_training_report_metadata(adapter)
    expected_names = tuple(recording.name for recording in training)
    if metadata.camera != camera:
        raise ValueError("adapter camera does not match held-out reports")
    if metadata.training_recordings != expected_names:
        raise ValueError("adapter training recordings do not match the experiment")


def build_experiment_from_evidence(
    experiment_id: str,
    training: Sequence[DomainRecording],
    evaluations: Sequence[HeldOutEvaluation],
) -> dict[str, Any]:
    if not experiment_id or not training or not evaluations:
        raise ValueError(
            "experiment, training recordings, and held-out reports are required"
        )
    training = tuple(training)
    evaluations = tuple(evaluations)
    _require_unique_disjoint_recordings(training, evaluations)
    cameras = {evaluation.camera for evaluation in evaluations}
    adapters = {evaluation.adapter for evaluation in evaluations}
    if len(cameras) != 1 or len(adapters) != 1:
        raise ValueError("held-out reports must use one camera and adapter")
    _validate_adapter_training_set(
        evaluations[0].adapter,
        evaluations[0].camera,
        training,
    )
    aggregate = EvaluationMetrics.aggregate(
        tuple(evaluation.metrics for evaluation in evaluations)
    )
    passed = aggregate.control_gate.passed and all(
        evaluation.metrics.control_gate.passed for evaluation in evaluations
    )
    return {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": experiment_id,
        "camera": evaluations[0].camera,
        "adapter": str(evaluations[0].adapter),
        "training_recordings": [recording.name for recording in training],
        "held_out_evaluations": [
            evaluation.to_dict() for evaluation in evaluations
        ],
        "aggregate": aggregate.to_dict(),
        "passed": passed,
    }


def build_experiment(
    experiment_id: str,
    training_recordings: Sequence[Path],
    held_out_reports: Sequence[Path],
) -> dict[str, Any]:
    if not experiment_id or not training_recordings or not held_out_reports:
        raise ValueError(
            "experiment, training recordings, and held-out reports are required"
        )
    training = tuple(
        DomainRecording.from_path(path, expected_split=DatasetSplit.TRAIN)
        for path in training_recordings
    )
    evaluations = tuple(HeldOutEvaluation.from_path(path) for path in held_out_reports)
    return build_experiment_from_evidence(experiment_id, training, evaluations)


def summarize_experiment(
    experiment_id: str,
    training_recordings: Sequence[Path],
    held_out_reports: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    summary = build_experiment(
        experiment_id,
        training_recordings,
        held_out_reports,
    )
    write_json_atomic(output.resolve(), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--training-recording",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--held-out-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize_experiment(
                args.experiment_id,
                args.training_recording,
                args.held_out_report,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
