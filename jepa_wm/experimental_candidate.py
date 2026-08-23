"""Explicitly isolated rebinding contract for one realized shadow-candidate trial."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from pathlib import Path
from time import time
from typing import Any, Mapping

from jepa_wm.action import DroidAction
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.shadow_planning import ShadowSearchEvidence
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from jepa_wm.trial_equivalence import TrialResetState, validate_reset_equivalence
from sim.recording import validate_recording_id


EXPERIMENTAL_CANDIDATE_SCHEMA = "quantis.jepa_wm_experimental_candidate.v1"


class ExperimentalCandidateAuthority(str, Enum):
    RESET_TRIAL_ONLY = "reset_trial_only"


@dataclass(frozen=True)
class ExperimentalCandidateBinding:
    execution_session_id: str
    source_session_id: str
    source_observation_id: int
    execution_observation_id: int
    source_proposal: Path
    adapter: Path
    actions: tuple[DroidAction, ...]
    energy_improvement: float
    objective_improvement: float
    authority: ExperimentalCandidateAuthority = (
        ExperimentalCandidateAuthority.RESET_TRIAL_ONLY
    )

    def __post_init__(self) -> None:
        validate_recording_id(self.execution_session_id)
        validate_recording_id(self.source_session_id)
        if (
            self.execution_session_id == self.source_session_id
            or self.source_observation_id <= 0
            or self.execution_observation_id <= 0
            or not self.source_proposal.is_absolute()
            or not self.adapter.is_absolute()
            or len(self.actions) != 3
            or not isfinite(self.energy_improvement)
            or self.energy_improvement <= 0.0
            or not isfinite(self.objective_improvement)
            or self.objective_improvement <= 0.0
            or self.authority
            is not ExperimentalCandidateAuthority.RESET_TRIAL_ONLY
        ):
            raise ValueError("experimental candidate binding is invalid")

    @property
    def production_authority_granted(self) -> bool:
        return False

    def validate_execution(
        self,
        source: CandidateSourceEvidence,
        execution: CandidateExecutionEvidence,
    ) -> None:
        source_context = source.context
        execution_context = execution.context
        shadow = source.shadow
        safety = source.safety
        observation = execution_context.observation
        response = execution.response
        if (
            self.execution_observation_id != observation.observation_id
            or self.source_observation_id != source_context.observation.observation_id
            or self.source_observation_id != shadow.observation_id
            or self.source_proposal != shadow.proposal
            or self.adapter != shadow.adapter
            or self.actions != shadow.planned.actions
            or not isclose(
                self.energy_improvement,
                shadow.energy_improvement,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isclose(
                self.objective_improvement,
                shadow.objective_improvement,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not shadow.passes_shadow_gate
            or not safety.passed
            or safety.planned_actions != self.actions
            or execution_context.policy
            is not ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
            or source_context.policy is not ControlExecutionPolicy.DIRECT
            or execution_context.reference_recording
            != source_context.reference_recording
            or execution_context.seed != source_context.seed
            or execution_context.previous_session_id
            != source_context.previous_session_id
            or observation.target_frame != source_context.observation.target_frame
            or observation.warmup_frames != source_context.observation.warmup_frames
            or observation.previous_action != source_context.observation.previous_action
        ):
            raise ValueError("experimental candidate is not bound to its source trial")
        validate_reset_equivalence(source_context.reset, execution_context.reset)
        if response is not None and (
            response.observation_id != self.execution_observation_id
            or response.actions != self.actions
            or response.proposal != observation.expected_proposal
        ):
            raise ValueError("experimental candidate response does not match its binding")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXPERIMENTAL_CANDIDATE_SCHEMA,
            "execution_session_id": self.execution_session_id,
            "source_session_id": self.source_session_id,
            "source_observation_id": self.source_observation_id,
            "execution_observation_id": self.execution_observation_id,
            "source_proposal": str(self.source_proposal),
            "adapter": str(self.adapter),
            "actions": [list(action.values) for action in self.actions],
            "energy_improvement": self.energy_improvement,
            "objective_improvement": self.objective_improvement,
            "authority": self.authority.value,
            "production_authority_granted": self.production_authority_granted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExperimentalCandidateBinding:
        if payload.get("schema") != EXPERIMENTAL_CANDIDATE_SCHEMA:
            raise ValueError("experimental candidate schema is invalid")
        if payload.get("production_authority_granted") is not False:
            raise ValueError("experimental candidate cannot have production authority")
        try:
            return cls(
                execution_session_id=str(payload["execution_session_id"]),
                source_session_id=str(payload["source_session_id"]),
                source_observation_id=int(payload["source_observation_id"]),
                execution_observation_id=int(payload["execution_observation_id"]),
                source_proposal=Path(payload["source_proposal"]),
                adapter=Path(payload["adapter"]),
                actions=tuple(
                    DroidAction(tuple(values)) for values in payload["actions"]
                ),
                energy_improvement=float(payload["energy_improvement"]),
                objective_improvement=float(payload["objective_improvement"]),
                authority=ExperimentalCandidateAuthority(payload["authority"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("experimental candidate binding is incomplete") from error


@dataclass(frozen=True)
class CandidateTrialContext:
    observation: ControlObservation
    reset: TrialResetState
    policy: ControlExecutionPolicy
    reference_recording: str
    seed: int
    previous_session_id: str | None


@dataclass(frozen=True)
class CandidateSourceEvidence:
    context: CandidateTrialContext
    shadow: ShadowSearchEvidence
    safety: ShadowSafetyEvidence


@dataclass(frozen=True)
class CandidateExecutionEvidence:
    context: CandidateTrialContext
    response: ProposedControl | None


def build_experimental_candidate_response(
    *,
    execution_session_id: str,
    source_session_id: str,
    observation: ControlObservation,
    shadow: ShadowSearchEvidence,
    safety: ShadowSafetyEvidence,
    created_at_unix_seconds: float | None = None,
) -> tuple[ExperimentalCandidateBinding, ProposedControl]:
    """Rebind a proven shadow winner to a fresh reset-trial observation."""

    if (
        not shadow.passes_shadow_gate
        or not safety.passed
        or safety.observation_id != shadow.observation_id
        or safety.planned_actions != shadow.planned.actions
    ):
        raise ValueError("candidate source did not pass shadow search and safety")
    binding = ExperimentalCandidateBinding(
        execution_session_id=execution_session_id,
        source_session_id=source_session_id,
        source_observation_id=shadow.observation_id,
        execution_observation_id=observation.observation_id,
        source_proposal=shadow.proposal,
        adapter=shadow.adapter,
        actions=shadow.planned.actions,
        energy_improvement=shadow.energy_improvement,
        objective_improvement=shadow.objective_improvement,
    )
    response = ProposedControl(
        observation_id=observation.observation_id,
        created_at_unix_seconds=(
            time() if created_at_unix_seconds is None else created_at_unix_seconds
        ),
        actions=binding.actions,
        proposal=observation.expected_proposal,
    )
    return binding, response
