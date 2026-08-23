"""Strict whole-seed readiness gate for inverse-action proposal checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from math import fsum, isclose, isfinite
import json
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.action import DroidAction
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.planner import PlannerActionBounds
from jepa_wm.planner_readiness import FirstActionGate, FirstActionSummary
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL, RolloutWindow
from jepa_wm.training_artifact import load_training_report_metadata
from sim.exploration import DatasetSplit


READINESS_SCHEMA = "quantis.jepa_wm_action_proposal_readiness.v1"


@dataclass(frozen=True)
class ProposalReadinessThresholds:
    minimum_training_seeds: int = 12
    minimum_held_out_seeds: int = 2
    minimum_rollouts_per_seed: int = 50
    minimum_gate_pass_rate: float = 0.95
    minimum_active_direction_pass_rate: float = 0.98
    minimum_mean_cosine: float = 0.9
    maximum_mean_sequence_mse: float = 0.001
    minimum_warmup_frames: int = 4

    def __post_init__(self) -> None:
        rates = (
            self.minimum_gate_pass_rate,
            self.minimum_active_direction_pass_rate,
            self.minimum_mean_cosine,
        )
        if (
            self.minimum_training_seeds <= 0
            or self.minimum_held_out_seeds <= 0
            or self.minimum_rollouts_per_seed <= 0
        ):
            raise ValueError("readiness sample thresholds must be positive")
        if self.minimum_warmup_frames < 0:
            raise ValueError("readiness warm-up must be non-negative")
        if not all(isfinite(value) and 0.0 <= value <= 1.0 for value in rates):
            raise ValueError("readiness rate thresholds must be between zero and one")
        if (
            not isfinite(self.maximum_mean_sequence_mse)
            or self.maximum_mean_sequence_mse < 0
        ):
            raise ValueError("readiness sequence error threshold must be non-negative")


@dataclass(frozen=True)
class ProposalMetrics:
    rollouts: int
    sequence_error_sum: float
    first_actions: FirstActionSummary
    warmup_frames: int

    def __post_init__(self) -> None:
        if (
            self.rollouts <= 0
            or self.rollouts != self.first_actions.count
            or not isfinite(self.sequence_error_sum)
            or self.sequence_error_sum < 0.0
            or self.warmup_frames < 0
        ):
            raise ValueError("proposal metrics are inconsistent")

    @classmethod
    def aggregate(cls, metrics: Sequence[ProposalMetrics]) -> ProposalMetrics:
        if not metrics:
            raise ValueError("at least one proposal metric is required")
        return cls(
            rollouts=sum(metric.rollouts for metric in metrics),
            sequence_error_sum=fsum(metric.sequence_error_sum for metric in metrics),
            first_actions=FirstActionSummary.aggregate(
                tuple(metric.first_actions for metric in metrics)
            ),
            warmup_frames=min(metric.warmup_frames for metric in metrics),
        )

    @property
    def mean_sequence_mse(self) -> float:
        return self.sequence_error_sum / self.rollouts

    @property
    def mean_first_action_cosine(self) -> float:
        return self.first_actions.mean_cosine

    @property
    def gate_pass_rate(self) -> float:
        return self.first_actions.pass_rate

    @property
    def active_direction_pass_rate(self) -> float:
        rate = self.first_actions.active_direction_pass_rate
        if rate is None:
            raise ValueError("proposal readiness requires active actions")
        return rate

    def passes(self, thresholds: ProposalReadinessThresholds) -> bool:
        return (
            self.rollouts >= thresholds.minimum_rollouts_per_seed
            and self.warmup_frames >= thresholds.minimum_warmup_frames
            and self.gate_pass_rate >= thresholds.minimum_gate_pass_rate
            and self.active_direction_pass_rate
            >= thresholds.minimum_active_direction_pass_rate
            and self.mean_first_action_cosine >= thresholds.minimum_mean_cosine
            and self.mean_sequence_mse <= thresholds.maximum_mean_sequence_mse
        )

    def metrics_dict(self, thresholds: ProposalReadinessThresholds) -> dict[str, Any]:
        return {
            "rollouts": self.rollouts,
            "warmup_frames": self.warmup_frames,
            "mean_sequence_mse": self.mean_sequence_mse,
            "mean_first_action_cosine": self.mean_first_action_cosine,
            "first_action_gate_pass_rate": self.gate_pass_rate,
            "active_first_action_direction_pass_rate": (
                self.active_direction_pass_rate
            ),
            "passed": self.passes(thresholds),
        }


@dataclass(frozen=True)
class ProposalEvaluationEvidence:
    recording: DomainRecording
    report: Path
    metrics: ProposalMetrics

    def to_dict(self, thresholds: ProposalReadinessThresholds) -> dict[str, Any]:
        return {
            "recording": self.recording.name,
            "seed": self.recording.seed,
            "report": str(self.report),
            **self.metrics.metrics_dict(thresholds),
        }

    @classmethod
    def from_report(
        cls,
        report: Path,
        *,
        proposal: Path,
        camera: str,
    ) -> ProposalEvaluationEvidence:
        payload = json.loads(report.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "quantis.jepa_wm_action_proposal_evaluation.v1"
            or payload.get("status") != "evaluated"
            or Path(str(payload.get("proposal"))).resolve() != proposal
            or payload.get("camera") != camera
        ):
            raise ValueError(f"proposal evaluation provenance is invalid: {report}")
        recording = DomainRecording.from_path(
            Path(str(payload.get("recording"))),
            expected_split=DatasetSplit.HELD_OUT,
        )
        if report.resolve().parent.parent != recording.path:
            raise ValueError("proposal evaluation is outside its recording")
        results = payload.get("results")
        window = payload.get("window")
        if not isinstance(results, list) or not results or not isinstance(window, dict):
            raise ValueError("proposal evaluation evidence is incomplete")
        window_values = (
            window.get("start_index"),
            window.get("count"),
            window.get("stride"),
        )
        if any(type(value) is not int for value in window_values):
            raise ValueError("proposal evaluation window is invalid")
        rollout_window = RolloutWindow(*window_values)
        errors: list[float] = []
        decisions = []
        planner_bounds = PlannerActionBounds()
        for ordinal, result in enumerate(results):
            gate = result.get("first_action_gate") if isinstance(result, dict) else None
            error = result.get("sequence_mse") if isinstance(result, dict) else None
            cosine = (
                result.get("first_action_cosine") if isinstance(result, dict) else None
            )
            recorded_payload = (
                result.get("recorded_actions") if isinstance(result, dict) else None
            )
            proposed_payload = (
                result.get("proposed_actions") if isinstance(result, dict) else None
            )
            expected_context_index = (
                rollout_window.start_index + ordinal * rollout_window.stride
            )
            if (
                not isinstance(gate, dict)
                or not isinstance(gate.get("passed"), bool)
                or not isinstance(gate.get("recorded_action_is_active"), bool)
                or not isinstance(error, (int, float))
                or not isinstance(cosine, (int, float))
                or not isfinite(error)
                or not isfinite(cosine)
                or not isinstance(recorded_payload, list)
                or not isinstance(proposed_payload, list)
                or len(recorded_payload) != len(proposed_payload)
                or len(recorded_payload) != DROID_ROLLOUT_PROTOCOL.action_horizon
                or result.get("context_index") != expected_context_index
                or result.get("target_index")
                != expected_context_index + DROID_ROLLOUT_PROTOCOL.action_horizon
            ):
                raise ValueError("proposal rollout evidence is inconsistent")
            try:
                recorded_actions = tuple(
                    DroidAction(tuple(action)) for action in recorded_payload
                )
                proposed_actions = tuple(
                    DroidAction(tuple(action)) for action in proposed_payload
                )
            except (TypeError, ValueError) as validation_error:
                raise ValueError(
                    "proposal rollout actions are invalid"
                ) from validation_error
            if not planner_bounds.accepts(proposed_actions):
                raise ValueError("proposed rollout actions exceed planner bounds")
            squared_error = fsum(
                (left - right) ** 2
                for recorded_action, proposed_action in zip(
                    recorded_actions, proposed_actions
                )
                for left, right in zip(recorded_action.values, proposed_action.values)
            ) / (len(recorded_actions) * len(recorded_actions[0].values))
            decision = FirstActionGate().evaluate(
                recorded_actions[0], proposed_actions[0]
            )
            if (
                not isclose(float(error), squared_error, rel_tol=1e-9, abs_tol=1e-12)
                or not isclose(
                    float(cosine), decision.cosine, rel_tol=1e-9, abs_tol=1e-12
                )
                or gate != decision.to_dict()
            ):
                raise ValueError("proposal rollout metrics do not match raw actions")
            errors.append(squared_error)
            decisions.append(decision)
        rollouts = len(results)
        first_actions = FirstActionSummary(tuple(decisions))
        if (
            not first_actions.active_count
            or payload.get("rollouts") != rollouts
            or rollout_window.count != rollouts
            or rollout_window.stride != 1
        ):
            raise ValueError("proposal evaluation rollout counts are inconsistent")
        metrics = ProposalMetrics(
            rollouts=rollouts,
            sequence_error_sum=fsum(errors),
            first_actions=first_actions,
            warmup_frames=rollout_window.start_index,
        )
        reported_values = (
            (payload.get("mean_sequence_mse"), metrics.mean_sequence_mse),
            (
                payload.get("mean_first_action_cosine"),
                metrics.mean_first_action_cosine,
            ),
            (payload.get("first_action_gate_pass_rate"), metrics.gate_pass_rate),
            (
                payload.get("active_first_action_direction_pass_rate"),
                metrics.active_direction_pass_rate,
            ),
        )
        if any(
            not isinstance(reported, (int, float))
            or not isclose(float(reported), computed, rel_tol=1e-9, abs_tol=1e-12)
            for reported, computed in reported_values
        ):
            raise ValueError("proposal evaluation aggregates are inconsistent")
        return cls(recording, report.resolve(), metrics)


def summarize_proposal_readiness(
    proposal: Path,
    evaluation_reports: Sequence[Path],
    output: Path,
    thresholds: ProposalReadinessThresholds = ProposalReadinessThresholds(),
) -> dict[str, Any]:
    proposal = proposal.resolve()
    if not evaluation_reports:
        raise ValueError("at least one proposal evaluation report is required")
    metadata = load_training_report_metadata(proposal)
    evidence = tuple(
        ProposalEvaluationEvidence.from_report(
            report.resolve(),
            proposal=proposal,
            camera=metadata.camera,
        )
        for report in evaluation_reports
    )
    held_names = [item.recording.name for item in evidence]
    held_seeds = [item.recording.seed for item in evidence]
    if len(set(held_names)) != len(held_names) or len(set(held_seeds)) != len(
        held_seeds
    ):
        raise ValueError("held-out proposal evaluations must be unique")
    if set(held_names) & set(metadata.training_recordings):
        raise ValueError("proposal training and held-out recordings overlap")
    training = tuple(
        DomainRecording.from_path(
            evidence[0].recording.path.parent / training_name,
            expected_split=DatasetSplit.TRAIN,
        )
        for training_name in metadata.training_recordings
    )
    training_seeds = [recording.seed for recording in training]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("proposal training seeds must be unique")
    if set(training_seeds) & set(held_seeds):
        raise ValueError("proposal training and held-out seeds overlap")
    aggregate = ProposalMetrics.aggregate(tuple(item.metrics for item in evidence))
    passed = (
        len(training) >= thresholds.minimum_training_seeds
        and len(evidence) >= thresholds.minimum_held_out_seeds
        and aggregate.passes(thresholds)
        and all(item.metrics.passes(thresholds) for item in evidence)
    )
    summary = {
        "schema": READINESS_SCHEMA,
        "proposal": str(proposal),
        "metadata": metadata.to_dict(),
        "thresholds": asdict(thresholds),
        "held_out_evaluations": [item.to_dict(thresholds) for item in evidence],
        "aggregate": aggregate.metrics_dict(thresholds),
        "passed": passed,
        "scope": "simulator-only inverse-action proposal; no live execution",
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize_proposal_readiness(
                args.proposal,
                args.evaluation_report,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
