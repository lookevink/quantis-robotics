"""Strict full-acquisition gate for the contact-aware grasp proposal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.insertion_layout import CONTACT_INSERTION_LAYOUT, ContactInsertionSegment
from jepa_wm.insertion_recording import ContactGraspEvidence
from jepa_wm.persistence import write_json_atomic
from jepa_wm.task_proposal_readiness import TaskProposalReadinessPolicy
from jepa_wm.task_windows import CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW
from jepa_wm.training_artifact import ProposalConditioningCapabilities


CONTACT_GRASP_ACQUISITION_READINESS_SCHEMA = (
    "quantis.jepa_wm_contact_grasp_acquisition_readiness.v1"
)
CONTACT_GRASP_ACQUISITION_BOUNDS = ActionSelectionBounds(minimum_action_norm=0.0)
CONTACT_GRASP_ACQUISITION_CONDITIONING = ProposalConditioningCapabilities(
    True, True, True, True
)
MAXIMUM_HOLD_GRIPPER_DELTA = 0.02


def _validate_recording(recording: Path, split: str) -> None:
    ContactGraspEvidence.from_recording(recording, expected_split=split)


CONTACT_GRASP_ACQUISITION_READINESS = TaskProposalReadinessPolicy(
    task_name="contact-grasp-acquisition",
    window=CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW,
    bounds=CONTACT_GRASP_ACQUISITION_BOUNDS,
    conditioning=CONTACT_GRASP_ACQUISITION_CONDITIONING,
    schema=CONTACT_GRASP_ACQUISITION_READINESS_SCHEMA,
    scope=(
        "offline full contact-domain acquisition and retained-grasp inverse "
        "proposal; no live task completion"
    ),
    window_description="the full approach, open, close, attach, and retained window",
    stationary_description="stationary gripper holds throughout acquisition",
    validate_recording=_validate_recording,
    require_training_selection_fingerprint=True,
    minimum_warmup_frames=0,
)


def _gripper_phase_evidence(report: Path) -> dict[str, Any]:
    payload = json.loads(report.read_text())
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list):
        raise ValueError("contact-grasp acquisition gripper evidence is incomplete")
    close_start = CONTACT_INSERTION_LAYOUT.start_index(
        ContactInsertionSegment.GRASP_CLOSE
    ) - 1
    attach = CONTACT_INSERTION_LAYOUT.start_index(
        ContactInsertionSegment.GRASP_ATTACH
    )
    holds = 0
    closes = 0
    retained = 0
    for result in results:
        if not isinstance(result, dict):
            raise ValueError("contact-grasp acquisition gripper evidence is invalid")
        context_index = result.get("context_index")
        proposed = result.get("proposed_actions")
        if (
            type(context_index) is not int
            or not isinstance(proposed, list)
            or not proposed
            or not isinstance(proposed[0], list)
            or len(proposed[0]) != 7
        ):
            raise ValueError("contact-grasp acquisition gripper evidence is invalid")
        gripper = float(proposed[0][6])
        if context_index < close_start:
            holds += abs(gripper) <= MAXIMUM_HOLD_GRIPPER_DELTA
        elif context_index < attach:
            closes += gripper > 0.0
        else:
            retained += gripper >= -MAXIMUM_HOLD_GRIPPER_DELTA
    expected_holds = close_start
    expected_closes = attach - close_start
    expected_retained = (
        CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW.count
        - attach
    )
    passed = (
        holds == expected_holds
        and closes == expected_closes
        and retained == expected_retained
    )
    return {
        "report": str(report.resolve()),
        "maximum_hold_gripper_delta": MAXIMUM_HOLD_GRIPPER_DELTA,
        "hold_passes": holds,
        "expected_holds": expected_holds,
        "close_direction_passes": closes,
        "expected_closes": expected_closes,
        "retained_hold_passes": retained,
        "expected_retained_holds": expected_retained,
        "passed": passed,
    }


def summarize_contact_grasp_acquisition_readiness(
    proposal: Path,
    evaluation_reports: Sequence[Path],
    output: Path,
) -> dict[str, object]:
    training = CONTACT_GRASP_ACQUISITION_READINESS.training_report(
        proposal.resolve()
    )
    config = training.get("config")
    metadata = training.get("metadata")
    training_recordings = (
        metadata.get("training_recordings")
        if isinstance(metadata, dict)
        else None
    )
    training_rollouts = training.get("rollouts")
    if (
        not isinstance(training_recordings, list)
        or type(training_rollouts) is not int
        or not isinstance(config, dict)
    ):
        raise ValueError(
            "contact-grasp acquisition proposal lacks open-gripper recovery evidence"
        )
    expected_counterfactuals = (
        len(training_recordings)
        * (
            CONTACT_INSERTION_LAYOUT.start_index(
                ContactInsertionSegment.GRASP_CLOSE
            )
            - 1
        )
    )
    if (
        config.get("open_gripper_counterfactuals") is not True
        or training.get("open_gripper_counterfactual_examples")
        != expected_counterfactuals
        or training.get("training_examples")
        != training_rollouts + expected_counterfactuals
    ):
        raise ValueError(
            "contact-grasp acquisition proposal lacks open-gripper recovery evidence"
        )
    phase_evidence = tuple(
        _gripper_phase_evidence(report.resolve()) for report in evaluation_reports
    )
    summary = CONTACT_GRASP_ACQUISITION_READINESS.summarize(
        proposal,
        evaluation_reports,
        output,
        summary_evidence={"gripper_phase_evidence": list(phase_evidence)},
    )
    summary["passed"] = bool(summary["passed"]) and all(
        bool(item["passed"]) for item in phase_evidence
    )
    write_json_atomic(output.resolve(), summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument(
        "--evaluation-report", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = summarize_contact_grasp_acquisition_readiness(
        arguments.proposal,
        arguments.evaluation_report,
        arguments.output,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
