"""Exact offline planner contract for the contact-insertion stroke."""

from __future__ import annotations

from dataclasses import dataclass
from jepa_wm.action_prior import ActionPriorConfig
from jepa_wm.insertion_adapter_profile import InsertionAdapterProfile
from jepa_wm.insertion_planner_profile import InsertionPlannerProfileName
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
)
from jepa_wm.planner import CandidateTrustRegion, CEMConfig
from jepa_wm.planner_readiness import FirstActionThresholds
from jepa_wm.planner_policy import (
    ContextMatchedCandidatePolicy,
    GoalActionAlignment,
    PlannerTaskPolicy,
    RefinementAcceptancePolicy,
)
from jepa_wm.trajectory import RolloutWindow


@dataclass(frozen=True)
class InsertionPlannerProfile:
    """Pinned search and task semantics for one auditable insertion slice."""

    name: InsertionPlannerProfileName
    window: RolloutWindow
    planner: CEMConfig = CEMConfig(iterations=4, samples=64, elites=8, seed=234)
    scoring_batch_size: int = 64
    prior: ActionPriorConfig = ActionPriorConfig(penalty_weight=1e-5)
    task_policy: PlannerTaskPolicy = PlannerTaskPolicy(
        proposal_trust_region=CandidateTrustRegion(
            maximum_translation_deviation=0.001,
            maximum_rotation_deviation=0.004,
            maximum_gripper_deviation=0.02,
        ),
        first_action_thresholds=FirstActionThresholds(
            recorded_activity=(
                InsertionAdapterProfile.GOAL_ALIGNED.descriptor.first_action_activity
            ),
            maximum_stationary_translation=5e-5,
            maximum_stationary_rotation=5e-4,
            maximum_stationary_gripper=0.005,
            minimum_active_cosine=0.9,
        ),
        goal_action_alignment=GoalActionAlignment(
            minimum_cosine=(
                InsertionAdapterProfile.GOAL_ALIGNED.descriptor.minimum_goal_cosine
            ),
            failure_penalty=0.01,
        ),
        refinement_acceptance=RefinementAcceptancePolicy(
            minimum_latent_improvement=1e-6,
        ),
        context_matched_candidates=ContextMatchedCandidatePolicy(
            candidates_per_context=12,
        ),
    )

    def __post_init__(self) -> None:
        if self.scoring_batch_size <= 0:
            raise ValueError("insertion planner scoring batch size must be positive")

_INSERTION_START = CONTACT_INSERTION_RECORDING.start_index(
    ContactInsertionSegment.INSERT
)
INSERTION_SAMPLED_READINESS_PLANNER_PROFILE = InsertionPlannerProfile(
    InsertionPlannerProfileName.SAMPLED_READINESS,
    RolloutWindow(_INSERTION_START, 8, 8),
)
INSERTION_DENSE_PLANNER_PROFILE = InsertionPlannerProfile(
    InsertionPlannerProfileName.DENSE_EXECUTION,
    RolloutWindow(
        _INSERTION_START,
        CONTACT_INSERTION_RECORDING.span(ContactInsertionSegment.INSERT).frames,
        1,
    ),
)
_INSERTION_PLANNER_PROFILES = {
    profile.name: profile
    for profile in (
        INSERTION_SAMPLED_READINESS_PLANNER_PROFILE,
        INSERTION_DENSE_PLANNER_PROFILE,
    )
}


def insertion_planner_profile(name: str) -> InsertionPlannerProfile:
    return _INSERTION_PLANNER_PROFILES[InsertionPlannerProfileName(name)]
