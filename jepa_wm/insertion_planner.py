"""Exact offline planner contract for the contact-insertion stroke."""

from __future__ import annotations

from dataclasses import dataclass

from jepa_wm.action_prior import ActionPriorConfig
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
)
from jepa_wm.planner import CandidateTrustRegion, CEMConfig
from jepa_wm.planner_readiness import FirstActionThresholds
from jepa_wm.planner_policy import GoalActionAlignment, PlannerTaskPolicy
from jepa_wm.trajectory import RolloutWindow


@dataclass(frozen=True)
class InsertionPlannerProfile:
    """Pinned search and task semantics for one auditable insertion slice."""

    window: RolloutWindow = RolloutWindow(
        CONTACT_INSERTION_RECORDING.start_index(ContactInsertionSegment.INSERT),
        8,
        8,
    )
    planner: CEMConfig = CEMConfig(iterations=4, samples=64, elites=8, seed=234)
    prior: ActionPriorConfig = ActionPriorConfig(penalty_weight=1e-5)
    task_policy: PlannerTaskPolicy = PlannerTaskPolicy(
        proposal_trust_region=CandidateTrustRegion(
            maximum_translation_deviation=0.001,
            maximum_rotation_deviation=0.004,
            maximum_gripper_deviation=0.02,
        ),
        first_action_thresholds=FirstActionThresholds(
            recorded_translation_activity=1e-5,
            recorded_rotation_activity=1e-5,
            recorded_gripper_activity=0.005,
            maximum_stationary_translation=5e-5,
            maximum_stationary_rotation=5e-4,
            maximum_stationary_gripper=0.005,
            minimum_active_cosine=0.9,
        ),
        goal_action_alignment=GoalActionAlignment(
            minimum_cosine=0.95,
            failure_penalty=0.01,
        ),
    )


INSERTION_PLANNER_PROFILE = InsertionPlannerProfile()
