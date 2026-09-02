"""Typed terminal evidence for one bounded grasp-to-insertion milestone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jepa_wm.control_rollout import ControlRolloutReport
from jepa_wm.grasp_task import MAXIMUM_CONTACT_GRASP_ACTIONS
from jepa_wm.insertion_rollout import DEMO_INSERTION_ROLLOUT
from jepa_wm.insertion_task import InsertionDecision, evaluate_insertion
from sim.recording import validate_recording_id


GRASP_TO_INSERTION_SCHEMA = "quantis.jepa_wm_grasp_to_insertion.v2"
GRASP_ACTIONS = MAXIMUM_CONTACT_GRASP_ACTIONS


@dataclass(frozen=True)
class GraspToInsertionReport:
    """One task-terminal bounded grasp followed by four insertion actions."""

    run_id: str
    grasp: ControlRolloutReport
    insertion: ControlRolloutReport

    def __post_init__(self) -> None:
        validate_recording_id(self.run_id)
        grasp_decision = self.grasp.reach_and_grasp
        acquisition_evidence = tuple(
            step.result.post_action
            for step in self.grasp.complete_steps
            if step.result.post_action is not None
            and step.result.post_action.plug_attached
            and step.result.post_action.grasp_acquisition is not None
            and step.result.post_action.grasp_acquisition.decision.passed
            and step.result.post_action.attachment_mechanism is not None
        )
        attachment_mechanisms = {
            step.result.post_action.attachment_mechanism
            for report in (self.grasp, self.insertion)
            for step in report.complete_steps
            if step.result.post_action is not None
            and step.result.post_action.plug_attached
        }
        insertion_steps = self.insertion.complete_steps
        if (
            self.grasp.requested_steps != GRASP_ACTIONS
            or len(self.grasp.applied_steps) != len(self.grasp.complete_steps)
            or self.grasp.orchestration_failure is not None
            or grasp_decision is None
            or not grasp_decision.passed
            or not acquisition_evidence
            or None in attachment_mechanisms
            or len(attachment_mechanisms) != 1
            or self.insertion.requested_steps
            != DEMO_INSERTION_ROLLOUT.maximum_steps
            or not self.insertion.all_steps_applied
            or self.grasp.reference_recording
            != self.insertion.reference_recording
            or self.grasp.seed != self.insertion.seed
            or not self.grasp.complete_steps
            or not insertion_steps
            or self.insertion.predecessor_session_id
            != self.grasp.complete_steps[-1].session_id
            or tuple(
                step.state.resolved_insertion_rollout_position().step_index
                for step in insertion_steps
            )
            != tuple(range(1, DEMO_INSERTION_ROLLOUT.maximum_steps + 1))
            or self.insertion_decision is None
            or not self.insertion_decision.passed
            or len(self.insertion_decision.seated_indices) < 4
        ):
            raise ValueError("grasp-to-insertion report is invalid")

    @property
    def insertion_decision(self) -> InsertionDecision | None:
        target = self.insertion.insertion_target
        task_steps = []
        for report in (self.grasp, self.insertion):
            for step in report.complete_steps:
                post_action = step.result.post_action
                if (
                    post_action is None
                    or post_action.insertion_task_step is None
                    or post_action.command_realization is None
                    or not post_action.command_realization.passed
                ):
                    return None
                task_steps.append(post_action.insertion_task_step)
        if target is None or not task_steps:
            return None
        terminal_insertion_indices = frozenset(
            range(
                len(task_steps) - DEMO_INSERTION_ROLLOUT.maximum_steps,
                len(task_steps),
            )
        )
        return evaluate_insertion(
            tuple(task_steps),
            target,
            eligible_seating_indices=terminal_insertion_indices,
            require_terminal_attachment=True,
        )

    @property
    def production_authority_granted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        insertion_decision = self.insertion_decision
        if insertion_decision is None:
            raise AssertionError("validated report has no insertion decision")
        return {
            "schema": GRASP_TO_INSERTION_SCHEMA,
            "run_id": self.run_id,
            "reference_recording": self.grasp.reference_recording,
            "seed": self.grasp.seed,
            "grasp_actions_applied": len(self.grasp.applied_steps),
            "insertion_actions_applied": len(self.insertion.applied_steps),
            "grasp_passed": True,
            "attachment_mechanisms": sorted(
                {
                    step.result.post_action.attachment_mechanism.value
                    for report in (self.grasp, self.insertion)
                    for step in report.complete_steps
                    if step.result.post_action is not None
                    and step.result.post_action.attachment_mechanism is not None
                }
            ),
            "insertion_rollout_passed": True,
            "insertion": insertion_decision.evidence_dict(),
            "production_authority_granted": self.production_authority_granted,
            "grasp": self.grasp.to_dict(),
            "insertion_rollout": self.insertion.to_dict(),
        }
