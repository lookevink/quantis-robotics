"""Strict whole-seed readiness for the frozen insertion planner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import isclose, isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from jepa_wm.action import ActionSelectionBounds, DroidAction
from jepa_wm.contract import MODEL_ID
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.insertion_corpus import InsertionFreshEvaluationRoster
from jepa_wm.insertion_planner import INSERTION_PLANNER_PROFILE
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.persistence import write_json_atomic
from jepa_wm.planner import PlannerActionBounds
from jepa_wm.planner_objective import CandidateObjective
from jepa_wm.planner_report import (
    REPORT_SCHEMA,
    BlockedRefinement,
    CandidateEvaluation,
    PlannerBenchmarkProvenance,
    PlannerBenchmarkReport,
    PlannerInitialization,
    PlannerRolloutEvaluation,
    PlannerRunSummary,
    PlannerTimings,
    SelectedRefinement,
)
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactIdentity,
)
from jepa_wm.trajectory import load_rollouts
from sim.exploration import DatasetSplit

if TYPE_CHECKING:
    from jepa_wm.insertion_wm_readiness import InsertionAdapterEvidence
    from jepa_wm.task_proposal_readiness import TaskProposalArtifactEvidence


INSERTION_PLANNER_READINESS_SCHEMA = (
    "quantis.jepa_wm_insertion_planner_readiness.v1"
)
MINIMUM_SELECTED_WIN_RATE = 0.75
INSERTION_PLANNER_SELECTION_BOUNDS = ActionSelectionBounds(
    minimum_action_norm=0.0
)


def _number(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _action_matrix(value: Any, name: str) -> np.ndarray:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(row, list) or len(row) != 7 for row in value)
    ):
        raise ValueError(f"{name} must contain three seven-value actions")
    matrix = np.asarray(
        [
            [_number(component, f"{name} component") for component in row]
            for row in value
        ],
        dtype=np.float64,
    )
    return matrix


def _action(value: Any, name: str) -> DroidAction:
    if not isinstance(value, list) or len(value) != 7:
        raise ValueError(f"{name} must contain seven values")
    return DroidAction(tuple(_number(component, name) for component in value))


def _candidate(payload: dict[str, Any], prefix: str) -> CandidateEvaluation:
    action_field = (
        "searched_actions" if prefix == "searched" else f"{prefix}_action"
    )
    return CandidateEvaluation(
        _action_matrix(payload.get(action_field), f"{prefix} action"),
        CandidateObjective(
            _number(payload.get(f"{prefix}_energy"), f"{prefix} energy"),
            _number(
                payload.get(f"{prefix}_prior_penalty"),
                f"{prefix} prior penalty",
            ),
            _number(
                payload.get(f"{prefix}_task_penalty"),
                f"{prefix} task penalty",
            ),
        ),
    )


def _assert_same(actual: Any, expected: Any, path: str = "report") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path} fields do not match reconstructed evidence")
        for key, value in expected.items():
            _assert_same(actual[key], value, f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path} does not match reconstructed evidence")
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_same(actual_item, expected_item, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not isfinite(float(actual))
            or not isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(f"{path} does not match reconstructed evidence")
        return
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"{path} does not match reconstructed evidence")


@dataclass(frozen=True)
class InsertionPlannerSeedEvidence:
    report: Path
    recording: str
    seed: int
    base_checkpoint: ArtifactIdentity
    training_action_library: int
    selection_rate: float
    selected_goal_alignment_pass_rate: float | None
    selected_first_action_pass_rate: float | None
    mean_selected_improvement_over_zero: float | None
    selected_win_rate_over_zero: float | None
    mean_selected_improvement_over_recorded: float | None
    selected_win_rate_over_recorded: float | None

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons = []
        if self.selection_rate != 1.0:
            reasons.append("blocked_context")
        if self.selected_goal_alignment_pass_rate != 1.0:
            reasons.append("goal_alignment")
        if self.selected_first_action_pass_rate != 1.0:
            reasons.append("first_action")
        if (
            self.mean_selected_improvement_over_zero is None
            or self.mean_selected_improvement_over_zero <= 0.0
            or self.selected_win_rate_over_zero is None
            or self.selected_win_rate_over_zero < MINIMUM_SELECTED_WIN_RATE
        ):
            reasons.append("zero_action_energy")
        if (
            self.mean_selected_improvement_over_recorded is None
            or self.mean_selected_improvement_over_recorded <= 0.0
            or self.selected_win_rate_over_recorded is None
            or self.selected_win_rate_over_recorded < MINIMUM_SELECTED_WIN_RATE
        ):
            reasons.append("recorded_action_energy")
        return tuple(reasons)

    @property
    def passed(self) -> bool:
        return not self.reasons

    @classmethod
    def from_report(
        cls,
        report: Path,
        recording: Path,
        adapter: InsertionAdapterEvidence,
        proposal: TaskProposalArtifactEvidence,
        expected_base_checkpoint: ArtifactIdentity,
        expected_training_action_library: int,
    ) -> InsertionPlannerSeedEvidence:
        report = report.resolve()
        recording = recording.resolve()
        payload = json.loads(report.read_text())
        if not isinstance(payload, dict):
            raise ValueError("insertion planner report must be an object")
        domain_recording = DomainRecording.from_path(
            recording,
            expected_split=DatasetSplit.HELD_OUT,
        )
        ContactInsertionEvidence.from_recording(
            recording,
            expected_split=DatasetSplit.HELD_OUT.value,
        )
        if (
            adapter.contract.metadata.corpus_identity
            != proposal.metadata.corpus_identity
            or adapter.contract.metadata.camera != "wrist"
        ):
            raise ValueError("planner artifacts do not share one insertion corpus")
        base_path = payload.get("base_checkpoint")
        if not isinstance(base_path, str):
            raise ValueError("planner base checkpoint path is invalid")
        base_checkpoint = ArtifactIdentity.from_artifact(Path(base_path))
        if base_checkpoint != expected_base_checkpoint:
            raise ValueError("planner report uses the wrong base checkpoint")
        adapter_identity = TrainingArtifactIdentity(
            adapter.identity.path,
            adapter.identity.fingerprint,
            adapter.contract.metadata,
        )
        proposal_identity = TrainingArtifactIdentity(
            proposal.identity.path,
            proposal.identity.fingerprint,
            proposal.metadata,
        )
        selected_rollouts = INSERTION_PLANNER_PROFILE.window.select(
            load_rollouts(
                recording,
                camera="wrist",
                bounds=INSERTION_PLANNER_SELECTION_BOUNDS,
            )
        )
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or len(raw_results) != len(
            selected_rollouts
        ):
            raise ValueError("planner report has the wrong rollout roster")
        evaluations = []
        bounds = PlannerActionBounds()
        for raw, rollout in zip(raw_results, selected_rollouts):
            if not isinstance(raw, dict):
                raise ValueError("planner rollout evidence must be an object")
            evaluation = PlannerRolloutEvaluation(
                context_index=rollout.context[0].index,
                target_index=rollout.target.index,
                recorded_actions=_action_matrix(
                    raw.get("recorded_actions"), "recorded actions"
                ),
                recorded_energy=_number(
                    raw.get("recorded_energy"), "recorded energy"
                ),
                zero_energy=_number(raw.get("zero_energy"), "zero energy"),
                initialization=PlannerInitialization.PROPOSAL,
                initial_candidate=_candidate(raw, "proposal"),
                searched_candidate=_candidate(raw, "searched"),
                goal_action=_action(raw.get("goal_action"), "goal action"),
            )
            if not np.allclose(
                evaluation.recorded_actions,
                np.asarray([action.values for action in rollout.actions]),
                rtol=0.0,
                atol=1e-12,
            ) or any(
                not isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    evaluation.goal_action.values,
                    rollout.goal_action.values,
                )
            ):
                raise ValueError("planner report does not match recording telemetry")
            initial_actions = tuple(
                DroidAction(tuple(action))
                for action in evaluation.initial_candidate.actions
            )
            searched_actions = tuple(
                DroidAction(tuple(action))
                for action in evaluation.searched_candidate.actions
            )
            if not bounds.accepts(initial_actions) or not bounds.accepts(
                searched_actions
            ):
                raise ValueError("planner report contains out-of-bounds actions")
            _assert_same(
                raw,
                evaluation.to_dict(INSERTION_PLANNER_PROFILE.task_policy),
                f"result[{evaluation.context_index}]",
            )
            evaluations.append(evaluation)
        training_action_library = _positive_integer(
            payload.get("planner", {}).get("training_action_library")
            if isinstance(payload.get("planner"), dict)
            else None,
            "training action library",
        )
        if training_action_library != expected_training_action_library:
            raise ValueError(
                "planner action library does not match the training corpus"
            )
        run = PlannerBenchmarkReport(
            provenance=PlannerBenchmarkProvenance(
                model=MODEL_ID,
                source_revision=adapter.contract.metadata.source_revision,
                adapter=adapter_identity,
                proposal=proposal_identity,
                base_checkpoint=base_checkpoint,
                recording=domain_recording,
                camera="wrist",
                window=INSERTION_PLANNER_PROFILE.window,
                selection_bounds=INSERTION_PLANNER_SELECTION_BOUNDS,
                scoring_batch_size=INSERTION_PLANNER_PROFILE.scoring_batch_size,
            ),
            planner=PlannerRunSummary(
                config=INSERTION_PLANNER_PROFILE.planner,
                training_action_library=training_action_library,
                prior_penalty_weight=INSERTION_PLANNER_PROFILE.prior.penalty_weight,
                initialization=PlannerInitialization.PROPOSAL,
                task_policy=INSERTION_PLANNER_PROFILE.task_policy,
            ),
            bounds=bounds,
            timings=PlannerTimings(
                _number(payload.get("load_seconds"), "load seconds"),
                _number(payload.get("encoding_seconds"), "encoding seconds"),
                _number(payload.get("planning_seconds"), "planning seconds"),
                _number(payload.get("peak_allocated_gib"), "peak allocated GiB"),
            ),
            evaluations=tuple(evaluations),
        )
        expected = run.to_dict()
        expected["output_path"] = str(report)
        _assert_same(payload, expected)
        selected_first_action = expected.get("selected_first_action")
        return cls(
            report=report,
            recording=domain_recording.name,
            seed=domain_recording.seed,
            base_checkpoint=base_checkpoint,
            training_action_library=training_action_library,
            selection_rate=float(expected["selection_rate"]),
            selected_goal_alignment_pass_rate=expected[
                "selected_goal_action_alignment_pass_rate"
            ],
            selected_first_action_pass_rate=(
                selected_first_action["first_action_gate_pass_rate"]
                if isinstance(selected_first_action, dict)
                else None
            ),
            mean_selected_improvement_over_zero=expected[
                "mean_selected_improvement_over_zero"
            ],
            selected_win_rate_over_zero=expected[
                "selected_action_win_rate_over_zero"
            ],
            mean_selected_improvement_over_recorded=expected[
                "mean_selected_improvement_over_recorded"
            ],
            selected_win_rate_over_recorded=expected[
                "selected_action_win_rate_over_recorded"
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "report": str(self.report),
            "recording": self.recording,
            "seed": self.seed,
            "base_checkpoint": self.base_checkpoint.to_dict(),
            "training_action_library": self.training_action_library,
            "selection_rate": self.selection_rate,
            "selected_goal_action_alignment_pass_rate": (
                self.selected_goal_alignment_pass_rate
            ),
            "selected_first_action_pass_rate": self.selected_first_action_pass_rate,
            "mean_selected_improvement_over_zero": (
                self.mean_selected_improvement_over_zero
            ),
            "selected_action_win_rate_over_zero": self.selected_win_rate_over_zero,
            "mean_selected_improvement_over_recorded": (
                self.mean_selected_improvement_over_recorded
            ),
            "selected_action_win_rate_over_recorded": (
                self.selected_win_rate_over_recorded
            ),
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def _select_current_policy_evidence(
    candidate_reports: Sequence[Path],
    recording: Path,
    expected_recording_id: str,
    expected_seed: int,
    adapter: InsertionAdapterEvidence,
    proposal: TaskProposalArtifactEvidence,
    base_checkpoint: ArtifactIdentity,
    expected_training_action_library: int,
) -> InsertionPlannerSeedEvidence:
    matching_evidence = []
    for candidate_report in candidate_reports:
        try:
            item = InsertionPlannerSeedEvidence.from_report(
                candidate_report,
                recording,
                adapter,
                proposal,
                base_checkpoint,
                expected_training_action_library,
            )
        except (KeyError, OSError, TypeError, ValueError):
            continue
        if item.recording == expected_recording_id and item.seed == expected_seed:
            matching_evidence.append(item)
    if len(matching_evidence) != 1:
        raise ValueError(
            "expected exactly one current-policy insertion planner report "
            f"for {expected_recording_id}; found {len(matching_evidence)}"
        )
    return matching_evidence[0]


def summarize_insertion_planner_readiness(
    roster_path: Path,
    recording_root: Path,
    adapter_path: Path,
    proposal_path: Path,
    base_checkpoint_path: Path,
    reports: Sequence[Path] | None,
    output: Path,
) -> dict[str, Any]:
    from jepa_wm.insertion_proposal_readiness import validate_insertion_proposal
    from jepa_wm.insertion_wm_readiness import validate_insertion_adapter

    roster = InsertionFreshEvaluationRoster.load(roster_path.resolve())
    adapter = validate_insertion_adapter(adapter_path.resolve())
    proposal = validate_insertion_proposal(proposal_path.resolve())
    base_checkpoint = ArtifactIdentity.from_artifact(base_checkpoint_path.resolve())
    if (
        adapter.identity.path.stem != roster.adapter.name
        or adapter.identity.fingerprint != roster.adapter.fingerprint
    ):
        raise ValueError("planner adapter does not match the frozen fresh roster")
    expected_training_action_library = sum(
        len(
            load_rollouts(
                recording_root.resolve() / training_name,
                camera="wrist",
                bounds=INSERTION_PLANNER_SELECTION_BOUNDS,
            )
        )
        for training_name in adapter.contract.metadata.training_recordings
    )
    supplied_reports = tuple(path.resolve() for path in reports or ())
    evidence_items = []
    for expected in roster.recordings:
        recording = recording_root.resolve() / expected.recording_id
        candidate_reports = supplied_reports or tuple(
            sorted(
                (recording / "jepa_wm").glob(
                    "wrist_cem_benchmark_"
                    f"{INSERTION_PLANNER_PROFILE.window.start_index:06d}_"
                    f"{INSERTION_PLANNER_PROFILE.window.count:03d}_"
                    "held_out_proposal_prior_*.json"
                )
            )
        )
        evidence_items.append(
            _select_current_policy_evidence(
                candidate_reports,
                recording,
                expected.recording_id,
                expected.seed,
                adapter,
                proposal,
                base_checkpoint,
                expected_training_action_library,
            )
        )
    evidence = tuple(evidence_items)
    if any(
        item.recording != expected.recording_id or item.seed != expected.seed
        for item, expected in zip(evidence, roster.recordings)
    ):
        raise ValueError("planner reports do not match the fresh roster")
    if (
        len({item.base_checkpoint for item in evidence}) != 1
        or len({item.training_action_library for item in evidence}) != 1
    ):
        raise ValueError("planner reports do not share one complete search identity")
    passed = all(item.passed for item in evidence)
    payload = {
        "schema": INSERTION_PLANNER_READINESS_SCHEMA,
        "scope": "offline frozen insertion planner; no live insertion",
        "fresh_evaluation_roster": roster.to_dict(),
        "adapter": adapter.identity.to_dict(),
        "proposal": proposal.identity.to_dict(),
        "base_checkpoint": evidence[0].base_checkpoint.to_dict(),
        "planner": {
            **INSERTION_PLANNER_PROFILE.planner.to_dict(),
            "scoring_batch_size": INSERTION_PLANNER_PROFILE.scoring_batch_size,
            "training_action_library": evidence[0].training_action_library,
            "prior_penalty_weight": INSERTION_PLANNER_PROFILE.prior.penalty_weight,
            "task_policy": INSERTION_PLANNER_PROFILE.task_policy.to_dict(),
        },
        "window": INSERTION_PLANNER_PROFILE.window.to_dict(),
        "selection_bounds": INSERTION_PLANNER_SELECTION_BOUNDS.to_dict(),
        "planner_bounds": PlannerActionBounds().to_dict(),
        "minimum_selected_win_rate": MINIMUM_SELECTED_WIN_RATE,
        "whole_seed_evaluations": [item.to_dict() for item in evidence],
        "whole_seeds_passed": sum(item.passed for item in evidence),
        "whole_seeds_required": len(roster.recordings),
        "passed": passed,
        "live_insertion_authority_granted": False,
        "production_authority_granted": False,
    }
    write_json_atomic(output.resolve(), payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-roster", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        action="append",
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = summarize_insertion_planner_readiness(
        arguments.fresh_roster,
        arguments.recording_root,
        arguments.adapter,
        arguments.proposal,
        arguments.base_checkpoint,
        arguments.evaluation_report,
        arguments.output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
