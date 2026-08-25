"""Reset-bound authority for one realized insertion proposal action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import time
from typing import Any, Mapping

from jepa_wm.action import DroidAction, DroidActionScale
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ProposedControl
from jepa_wm.control_safety import insertion_projection_policy_for_scale
from jepa_wm.direct_safety import DirectInsertionSafetyEvidence
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.trial_equivalence import ControlTrialContext, validate_reset_equivalence
from sim.recording import validate_recording_id


INSERTION_TRIAL_SCHEMA = "quantis.jepa_wm_insertion_trial.v1"


class InsertionTrialAuthority(str, Enum):
    RESET_TRIAL_ONLY = "reset_trial_only"


@dataclass(frozen=True)
class InsertionTrialSourceEvidence:
    context: ControlTrialContext
    response: ProposedControl
    safety: DirectInsertionSafetyEvidence


@dataclass(frozen=True)
class InsertionTrialExecutionEvidence:
    context: ControlTrialContext
    response: ProposedControl | None


@dataclass(frozen=True)
class InsertionTrialBinding:
    execution_session_id: str
    source_session_id: str
    source_observation_id: int
    execution_observation_id: int
    proposal: ArtifactIdentity
    actions: tuple[DroidAction, ...]
    source_selected_action_scale: DroidActionScale
    authority: InsertionTrialAuthority = InsertionTrialAuthority.RESET_TRIAL_ONLY

    def __post_init__(self) -> None:
        validate_recording_id(self.execution_session_id)
        validate_recording_id(self.source_session_id)
        try:
            insertion_projection_policy_for_scale(self.source_selected_action_scale)
        except ValueError as error:
            raise ValueError("insertion trial binding is invalid") from error
        if (
            self.execution_session_id == self.source_session_id
            or self.source_observation_id <= 0
            or self.execution_observation_id <= 0
            or len(self.actions) != 3
            or self.authority is not InsertionTrialAuthority.RESET_TRIAL_ONLY
        ):
            raise ValueError("insertion trial binding is invalid")

    @property
    def production_authority_granted(self) -> bool:
        return False

    @property
    def allowed_projection_scales(self) -> tuple[DroidActionScale, ...]:
        policy = insertion_projection_policy_for_scale(
            self.source_selected_action_scale
        )
        source_index = policy.index(self.source_selected_action_scale)
        return policy[source_index:]

    def validate_attempted_projection_scales(
        self,
        attempted: tuple[DroidActionScale, ...],
    ) -> None:
        if (
            not attempted
            or attempted != self.allowed_projection_scales[: len(attempted)]
        ):
            raise ValueError("insertion trial exceeded its source projection")

    def validate_execution(
        self,
        source: InsertionTrialSourceEvidence,
        execution: InsertionTrialExecutionEvidence,
    ) -> None:
        source_context = source.context
        execution_context = execution.context
        source_response = source.response
        safety = source.safety
        response = execution.response
        source_identity = (
            ArtifactIdentity(source_response.proposal, source_response.proposal_fingerprint)
            if source_response.proposal_fingerprint is not None
            else None
        )
        if (
            self.source_observation_id
            != source_context.observation.observation_id
            or self.source_observation_id != source_response.observation_id
            or self.source_observation_id != safety.observation_id
            or self.execution_observation_id
            != execution_context.observation.observation_id
            or source_identity != self.proposal
            or safety.proposal != self.proposal
            or source_response.actions != self.actions
            or safety.proposed_actions != self.actions
            or safety.selected_action_scale != self.source_selected_action_scale
            or not safety.passed
            or not safety.live_state.plug_attached
            or source_context.policy
            is not ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION
            or execution_context.policy
            is not ControlExecutionPolicy.INSERTION_RESET_TRIAL
            or execution_context.reference_recording
            != source_context.reference_recording
            or execution_context.seed != source_context.seed
            or execution_context.previous_session_id
            != source_context.previous_session_id
            or execution_context.observation.target
            != source_context.observation.target
            or execution_context.observation.warmup_frames
            != source_context.observation.warmup_frames
            or execution_context.observation.previous_action
            != source_context.observation.previous_action
        ):
            raise ValueError("insertion trial is not bound to its safety source")
        validate_reset_equivalence(source_context.reset, execution_context.reset)
        if response is not None and (
            response.observation_id != self.execution_observation_id
            or response.actions != self.actions
            or response.proposal != self.proposal.path
            or response.proposal_fingerprint != self.proposal.fingerprint
        ):
            raise ValueError("insertion trial response does not match its binding")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INSERTION_TRIAL_SCHEMA,
            "execution_session_id": self.execution_session_id,
            "source_session_id": self.source_session_id,
            "source_observation_id": self.source_observation_id,
            "execution_observation_id": self.execution_observation_id,
            "proposal": self.proposal.to_dict(),
            "actions": [list(action.values) for action in self.actions],
            "source_selected_action_scale": self.source_selected_action_scale.to_dict(),
            "authority": self.authority.value,
            "production_authority_granted": self.production_authority_granted,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> InsertionTrialBinding:
        if payload.get("schema") != INSERTION_TRIAL_SCHEMA:
            raise ValueError("insertion trial schema is invalid")
        if payload.get("production_authority_granted") is not False:
            raise ValueError("insertion trial cannot have production authority")
        try:
            return cls(
                execution_session_id=str(payload["execution_session_id"]),
                source_session_id=str(payload["source_session_id"]),
                source_observation_id=int(payload["source_observation_id"]),
                execution_observation_id=int(payload["execution_observation_id"]),
                proposal=ArtifactIdentity.from_dict(payload["proposal"]),
                actions=tuple(
                    DroidAction(tuple(values)) for values in payload["actions"]
                ),
                source_selected_action_scale=DroidActionScale.from_payload(
                    payload["source_selected_action_scale"]
                ),
                authority=InsertionTrialAuthority(payload["authority"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion trial binding is incomplete") from error


def build_insertion_trial_response(
    *,
    execution_session_id: str,
    source_session_id: str,
    execution: ControlTrialContext,
    source: InsertionTrialSourceEvidence,
    created_at_unix_seconds: float | None = None,
) -> tuple[InsertionTrialBinding, ProposedControl]:
    """Rebind a passing no-actuation proposal to an equivalent fresh reset."""

    source_response = source.response
    safety = source.safety
    if source_response.proposal_fingerprint is None or safety.selected_action_scale is None:
        raise ValueError("insertion trial source has no selected exact proposal")
    binding = InsertionTrialBinding(
        execution_session_id=execution_session_id,
        source_session_id=source_session_id,
        source_observation_id=source_response.observation_id,
        execution_observation_id=execution.observation.observation_id,
        proposal=ArtifactIdentity(
            source_response.proposal, source_response.proposal_fingerprint
        ),
        actions=source_response.actions,
        source_selected_action_scale=safety.selected_action_scale,
    )
    response = ProposedControl(
        observation_id=execution.observation.observation_id,
        created_at_unix_seconds=(
            time() if created_at_unix_seconds is None else created_at_unix_seconds
        ),
        actions=binding.actions,
        proposal=binding.proposal.path,
        proposal_fingerprint=binding.proposal.fingerprint,
    )
    binding.validate_execution(
        source,
        InsertionTrialExecutionEvidence(execution, response),
    )
    return binding, response
