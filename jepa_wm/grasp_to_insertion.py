"""Typed terminal evidence for one bounded grasp-to-insertion milestone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jepa_wm.control_rollout import ControlRolloutReport
from jepa_wm.grasp_task import MAXIMUM_CONTACT_GRASP_ACTIONS
from jepa_wm.insertion_rollout import DEMO_INSERTION_ROLLOUT
from sim.recording import validate_recording_id


GRASP_TO_INSERTION_SCHEMA = "quantis.jepa_wm_grasp_to_insertion.v1"
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
        insertion_steps = self.insertion.complete_steps
        if (
            self.grasp.requested_steps != GRASP_ACTIONS
            or len(self.grasp.applied_steps) != len(self.grasp.complete_steps)
            or self.grasp.orchestration_failure is not None
            or grasp_decision is None
            or not grasp_decision.passed
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
        ):
            raise ValueError("grasp-to-insertion report is invalid")

    @property
    def production_authority_granted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRASP_TO_INSERTION_SCHEMA,
            "run_id": self.run_id,
            "reference_recording": self.grasp.reference_recording,
            "seed": self.grasp.seed,
            "grasp_actions_applied": len(self.grasp.applied_steps),
            "insertion_actions_applied": len(self.insertion.applied_steps),
            "grasp_passed": True,
            "insertion_rollout_passed": True,
            "production_authority_granted": self.production_authority_granted,
            "grasp": self.grasp.to_dict(),
            "insertion": self.insertion.to_dict(),
        }
