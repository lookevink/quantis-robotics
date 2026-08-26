"""Typed no-actuation safety evidence for one direct insertion proposal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_safety import (
    SafetyProjectionAttempt,
    insertion_projection_policy_for_attempts,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.insertion_refresh import (
    ControlSafetySnapshot,
    InsertionEvaluationRefresh,
)
from jepa_wm.insertion_task import InsertionTaskLimits
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.training_artifact import ArtifactIdentity


DIRECT_SAFETY_SCHEMA_V1 = "quantis.jepa_wm_direct_insertion_safety.v1"
DIRECT_SAFETY_SCHEMA_V2 = "quantis.jepa_wm_direct_insertion_safety.v2"
DIRECT_SAFETY_SCHEMA = "quantis.jepa_wm_direct_insertion_safety.v3"


class DirectSafetyAuthority(str, Enum):
    NO_ACTUATION = "no_actuation"


def _strict_number_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"direct insertion safety {field} must be numeric")
    return float(value)


def _strict_number(payload: dict[str, Any], field: str) -> float:
    return _strict_number_value(payload[field], field)


def _strict_positive_int(payload: dict[str, Any], field: str) -> int:
    value = payload[field]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"direct insertion safety {field} must be a positive integer")
    return value


@dataclass(frozen=True)
class _DirectSafetySchema:
    name: str
    requires_active_drive_target: bool
    nested_evaluation: bool

    @classmethod
    def for_evidence(
        cls,
        evidence: DirectInsertionSafetyEvidence,
    ) -> _DirectSafetySchema:
        if evidence.live_pose is not None:
            return _DIRECT_SAFETY_SCHEMAS[DIRECT_SAFETY_SCHEMA]
        if evidence.active_drive_target is not None:
            return _DIRECT_SAFETY_SCHEMAS[DIRECT_SAFETY_SCHEMA_V2]
        return _DIRECT_SAFETY_SCHEMAS[DIRECT_SAFETY_SCHEMA_V1]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> _DirectSafetySchema:
        try:
            schema = _DIRECT_SAFETY_SCHEMAS[payload["schema"]]
        except (KeyError, TypeError) as error:
            raise ValueError("direct insertion safety schema is invalid") from error
        present_target = "active_drive_target" in payload
        present_evaluation = "evaluation" in payload
        present_legacy_evaluation = any(
            field in payload
            for field in ("evaluated_at_unix_seconds", "live_state", "live_pose")
        )
        complete_legacy_evaluation = all(
            field in payload for field in ("evaluated_at_unix_seconds", "live_state")
        ) and "live_pose" not in payload
        if (
            present_target is not schema.requires_active_drive_target
            or present_evaluation is not schema.nested_evaluation
            or (
                schema.nested_evaluation
                and present_legacy_evaluation
            )
            or (
                not schema.nested_evaluation
                and not complete_legacy_evaluation
            )
        ):
            raise ValueError("direct insertion safety evidence is incomplete")
        return schema

    def evidence_fields(
        self,
        evidence: DirectInsertionSafetyEvidence,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if self.requires_active_drive_target:
            assert evidence.active_drive_target is not None
            fields["active_drive_target"] = evidence.active_drive_target.to_dict()
        if self.nested_evaluation:
            fields["evaluation"] = evidence.evaluation.to_dict()
        else:
            fields["evaluated_at_unix_seconds"] = evidence.evaluated_at_unix_seconds
            fields["live_state"] = evidence.live_state.to_dict()
        return fields

    def parse_evaluation(
        self,
        payload: dict[str, Any],
    ) -> InsertionEvaluationRefresh:
        if self.nested_evaluation:
            evaluation = InsertionEvaluationRefresh.from_dict(payload["evaluation"])
            if evaluation.live_pose is None:
                raise ValueError("direct insertion safety live pose is missing")
            return evaluation
        return InsertionEvaluationRefresh(
            _strict_number(payload, "evaluated_at_unix_seconds"),
            ControlSafetySnapshot.from_dict(payload["live_state"]),
        )

    def parse_active_drive_target(
        self,
        payload: dict[str, Any],
    ) -> JointDriveTarget | None:
        if not self.requires_active_drive_target:
            return None
        return JointDriveTarget.from_dict(payload["active_drive_target"])


_DIRECT_SAFETY_SCHEMAS = {
    DIRECT_SAFETY_SCHEMA_V1: _DirectSafetySchema(
        DIRECT_SAFETY_SCHEMA_V1,
        requires_active_drive_target=False,
        nested_evaluation=False,
    ),
    DIRECT_SAFETY_SCHEMA_V2: _DirectSafetySchema(
        DIRECT_SAFETY_SCHEMA_V2,
        requires_active_drive_target=True,
        nested_evaluation=False,
    ),
    DIRECT_SAFETY_SCHEMA: _DirectSafetySchema(
        DIRECT_SAFETY_SCHEMA,
        requires_active_drive_target=True,
        nested_evaluation=True,
    ),
}


@dataclass(frozen=True)
class DirectInsertionSafetyEvidence:
    observation_id: int
    evaluation: InsertionEvaluationRefresh
    proposed_actions: tuple[DroidAction, ...]
    proposal: ArtifactIdentity
    attempts: tuple[SafetyProjectionAttempt, ...]
    selected_action_scale: DroidActionScale | None
    active_drive_target: JointDriveTarget | None = None
    authority: DirectSafetyAuthority = DirectSafetyAuthority.NO_ACTUATION

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_id, bool)
            or self.observation_id <= 0
            or len(self.proposed_actions) != 3
            or not self.attempts
            or not isinstance(self.evaluation, InsertionEvaluationRefresh)
            or (
                self.evaluation.live_pose is not None
                and self.active_drive_target is None
            )
            or self.authority is not DirectSafetyAuthority.NO_ACTUATION
        ):
            raise ValueError("direct insertion safety evidence is invalid")
        selected = tuple(
            attempt
            for attempt in self.attempts
            if attempt.scale == self.selected_action_scale and attempt.gate.passed
        )
        if (self.selected_action_scale is None) == bool(selected):
            raise ValueError("direct insertion safety selection is inconsistent")
        projection_policy = insertion_projection_policy_for_attempts(
            attempt.scale for attempt in self.attempts
        )
        if any(
            attempt.gate.observation_id != self.observation_id
            for attempt in self.attempts
        ) or any(attempt.gate.passed for attempt in self.attempts[:-1]):
            raise ValueError("direct insertion safety projection is inconsistent")
        if self.attempts[-1].gate.passed:
            if self.selected_action_scale != self.attempts[-1].scale:
                raise ValueError("direct insertion safety selected the wrong projection")
        elif self.selected_action_scale is not None or len(self.attempts) != len(
            projection_policy
        ):
            raise ValueError("direct insertion safety stopped before exhaustion")

    @property
    def evaluated_at_unix_seconds(self) -> float:
        return self.evaluation.refreshed_at_unix_seconds

    @property
    def live_state(self) -> ControlSafetySnapshot:
        return self.evaluation.live_state

    @property
    def live_pose(self) -> DroidPose | None:
        return self.evaluation.live_pose

    @property
    def passed(self) -> bool:
        return (
            self.selected_action_scale is not None
            and self.live_state.plug_attached
            and not self.live_state.collision_detected
            and self.live_state.contact_force_newtons
            <= InsertionTaskLimits().maximum_contact_force_newtons
        )

    def to_dict(self) -> dict[str, Any]:
        schema = _DirectSafetySchema.for_evidence(self)
        return {
            "schema": schema.name,
            "task": INSERTION_TASK_ID,
            "observation_id": self.observation_id,
            "proposed_actions": [list(action.values) for action in self.proposed_actions],
            "proposal": self.proposal.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_action_scale": (
                self.selected_action_scale.to_dict()
                if self.selected_action_scale is not None
                else None
            ),
            **schema.evidence_fields(self),
            "passed": self.passed,
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DirectInsertionSafetyEvidence:
        if not isinstance(payload, dict) or payload.get("task") != INSERTION_TASK_ID:
            raise ValueError("direct insertion safety schema is invalid")
        try:
            schema = _DirectSafetySchema.from_payload(payload)
            selected = payload["selected_action_scale"]
            evidence = cls(
                observation_id=_strict_positive_int(payload, "observation_id"),
                evaluation=schema.parse_evaluation(payload),
                proposed_actions=tuple(
                    DroidAction(tuple(values)) for values in payload["proposed_actions"]
                ),
                proposal=ArtifactIdentity.from_dict(payload["proposal"]),
                attempts=tuple(
                    SafetyProjectionAttempt.from_dict(attempt)
                    for attempt in payload["attempts"]
                ),
                selected_action_scale=(
                    DroidActionScale.from_payload(selected)
                    if selected is not None
                    else None
                ),
                active_drive_target=schema.parse_active_drive_target(payload),
                authority=DirectSafetyAuthority(payload["authority"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("direct insertion safety evidence is incomplete") from error
        if payload.get("passed") is not evidence.passed:
            raise ValueError("direct insertion safety pass claim is inconsistent")
        return evidence
