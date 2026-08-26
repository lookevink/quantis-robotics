"""Typed no-actuation safety evidence for one direct insertion proposal."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist, isclose, isfinite
from pathlib import Path
from typing import Any

from jepa_wm.action import DroidAction, DroidActionScale
from jepa_wm.control_safety import (
    ControlInterlockEvidence,
    SafetyProjectionAttempt,
    SimulatorSafetyLimits,
    insertion_projection_policy_for_attempts,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.insertion_task import InsertionTaskLimits
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.training_artifact import ArtifactIdentity


DIRECT_SAFETY_SCHEMA_V1 = "quantis.jepa_wm_direct_insertion_safety.v1"
DIRECT_SAFETY_SCHEMA = "quantis.jepa_wm_direct_insertion_safety.v2"
MAXIMUM_CAPTURED_GRIPPER_DRIFT_METERS = 1e-6
MAXIMUM_CAPTURED_CONTACT_DRIFT_NEWTONS = 1e-9


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
class ControlSafetySnapshot:
    joint_positions: tuple[float, ...]
    gripper_width_m: float
    plug_position: tuple[float, ...]
    contact_force_newtons: float
    collision_detected: bool
    plug_attached: bool

    def __post_init__(self) -> None:
        if (
            len(self.joint_positions) != 7
            or not all(
                not isinstance(value, bool) and isfinite(value)
                for value in self.joint_positions
            )
            or isinstance(self.gripper_width_m, bool)
            or not isfinite(self.gripper_width_m)
            or not 0.0 <= self.gripper_width_m <= 0.08
            or len(self.plug_position) != 3
            or not all(
                not isinstance(value, bool) and isfinite(value)
                for value in self.plug_position
            )
            or isinstance(self.contact_force_newtons, bool)
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
            or not isinstance(self.collision_detected, bool)
            or not isinstance(self.plug_attached, bool)
        ):
            raise ValueError("control safety snapshot is invalid")

    def validate_continuity(
        self,
        captured: ControlSafetySnapshot,
        limits: SimulatorSafetyLimits = SimulatorSafetyLimits(),
    ) -> None:
        captured.validate_contact_continuity(
            ControlInterlockEvidence(
                self.contact_force_newtons,
                self.collision_detected,
            )
        )
        if (
            max(
                abs(live - expected)
                for live, expected in zip(
                    self.joint_positions, captured.joint_positions
                )
            )
            > limits.maximum_observation_joint_drift_radians
            or not isclose(
                self.gripper_width_m,
                captured.gripper_width_m,
                rel_tol=0.0,
                abs_tol=MAXIMUM_CAPTURED_GRIPPER_DRIFT_METERS,
            )
            or dist(self.plug_position, captured.plug_position)
            > limits.maximum_observation_plug_drift_meters
            or self.plug_attached is not captured.plug_attached
        ):
            raise ValueError("live control safety state changed after capture")

    def validate_contact_continuity(
        self,
        evidence: ControlInterlockEvidence,
    ) -> None:
        if (
            evidence.collision_detected is not self.collision_detected
            or not isclose(
                evidence.maximum_contact_force_newtons,
                self.contact_force_newtons,
                rel_tol=0.0,
                abs_tol=MAXIMUM_CAPTURED_CONTACT_DRIFT_NEWTONS,
            )
        ):
            raise ValueError("live control contact state changed after capture")

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_positions": list(self.joint_positions),
            "gripper_width_m": self.gripper_width_m,
            "plug_position": list(self.plug_position),
            "contact_force_newtons": self.contact_force_newtons,
            "collision_detected": self.collision_detected,
            "plug_attached": self.plug_attached,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlSafetySnapshot:
        if not isinstance(payload, dict):
            raise ValueError("control safety snapshot must be an object")
        try:
            return cls(
                joint_positions=tuple(
                    _strict_number_value(value, "joint_positions")
                    for value in payload["joint_positions"]
                ),
                gripper_width_m=_strict_number(payload, "gripper_width_m"),
                plug_position=tuple(
                    _strict_number_value(value, "plug_position")
                    for value in payload["plug_position"]
                ),
                contact_force_newtons=_strict_number(
                    payload, "contact_force_newtons"
                ),
                collision_detected=payload["collision_detected"],
                plug_attached=payload["plug_attached"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control safety snapshot is incomplete") from error


@dataclass(frozen=True)
class DirectInsertionSafetyEvidence:
    observation_id: int
    evaluated_at_unix_seconds: float
    proposed_actions: tuple[DroidAction, ...]
    proposal: ArtifactIdentity
    attempts: tuple[SafetyProjectionAttempt, ...]
    selected_action_scale: DroidActionScale | None
    live_state: ControlSafetySnapshot
    active_drive_target: JointDriveTarget | None = None
    authority: DirectSafetyAuthority = DirectSafetyAuthority.NO_ACTUATION

    def __post_init__(self) -> None:
        if (
            isinstance(self.observation_id, bool)
            or self.observation_id <= 0
            or isinstance(self.evaluated_at_unix_seconds, bool)
            or not isfinite(self.evaluated_at_unix_seconds)
            or len(self.proposed_actions) != 3
            or not self.attempts
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
    def passed(self) -> bool:
        return (
            self.selected_action_scale is not None
            and self.live_state.plug_attached
            and not self.live_state.collision_detected
            and self.live_state.contact_force_newtons
            <= InsertionTaskLimits().maximum_contact_force_newtons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                DIRECT_SAFETY_SCHEMA
                if self.active_drive_target is not None
                else DIRECT_SAFETY_SCHEMA_V1
            ),
            "task": INSERTION_TASK_ID,
            "observation_id": self.observation_id,
            "evaluated_at_unix_seconds": self.evaluated_at_unix_seconds,
            "proposed_actions": [list(action.values) for action in self.proposed_actions],
            "proposal": self.proposal.to_dict(),
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "selected_action_scale": (
                self.selected_action_scale.to_dict()
                if self.selected_action_scale is not None
                else None
            ),
            "live_state": self.live_state.to_dict(),
            **(
                {"active_drive_target": self.active_drive_target.to_dict()}
                if self.active_drive_target is not None
                else {}
            ),
            "passed": self.passed,
            "authority": self.authority.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> DirectInsertionSafetyEvidence:
        if (
            not isinstance(payload, dict)
            or payload.get("schema")
            not in (DIRECT_SAFETY_SCHEMA_V1, DIRECT_SAFETY_SCHEMA)
            or payload.get("task") != INSERTION_TASK_ID
        ):
            raise ValueError("direct insertion safety schema is invalid")
        try:
            selected = payload["selected_action_scale"]
            evidence = cls(
                observation_id=_strict_positive_int(payload, "observation_id"),
                evaluated_at_unix_seconds=_strict_number(
                    payload, "evaluated_at_unix_seconds"
                ),
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
                live_state=ControlSafetySnapshot.from_dict(payload["live_state"]),
                active_drive_target=(
                    JointDriveTarget.from_dict(payload["active_drive_target"])
                    if payload.get("schema") == DIRECT_SAFETY_SCHEMA
                    else None
                ),
                authority=DirectSafetyAuthority(payload["authority"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("direct insertion safety evidence is incomplete") from error
        if payload.get("passed") is not evidence.passed:
            raise ValueError("direct insertion safety pass claim is inconsistent")
        if (
            payload.get("schema") == DIRECT_SAFETY_SCHEMA
            and "active_drive_target" not in payload
        ):
            raise ValueError("direct insertion safety evidence is incomplete")
        return evidence
