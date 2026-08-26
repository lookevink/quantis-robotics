"""Exact synchronized insertion state used to reauthorize bound control inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import dist, isclose, isfinite
from typing import Any, Mapping

from jepa_wm.action import DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import ControlInterlockEvidence, SimulatorSafetyLimits


INSERTION_EVALUATION_REFRESH_SCHEMA = "quantis.jepa_wm_insertion_trial_refresh.v1"
MAXIMUM_CAPTURED_GRIPPER_DRIFT_METERS = 1e-6
MAXIMUM_CAPTURED_CONTACT_DRIFT_NEWTONS = 1e-9


def _strict_number_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"insertion refresh {field} must be numeric")
    return float(value)


def _strict_number(payload: Mapping[str, Any], field: str) -> float:
    return _strict_number_value(payload[field], field)


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
class InsertionEvaluationRefresh:
    """Reauthorize exact bound inputs after synchronized live continuity passes."""

    refreshed_at_unix_seconds: float
    live_state: ControlSafetySnapshot
    live_pose: DroidPose | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.refreshed_at_unix_seconds, bool)
            or not isfinite(self.refreshed_at_unix_seconds)
        ):
            raise ValueError("insertion evaluation refresh time is invalid")

    def authorize(
        self,
        captured: ControlObservation,
        response: ProposedControl,
        captured_state: ControlSafetySnapshot,
    ) -> tuple[ControlObservation, ProposedControl]:
        self.live_state.validate_continuity(captured_state)
        if self.refreshed_at_unix_seconds < max(
            captured.captured_at_unix_seconds,
            response.created_at_unix_seconds,
        ):
            raise ValueError("insertion evaluation refresh precedes its source")
        return (
            replace(
                captured,
                captured_at_unix_seconds=self.refreshed_at_unix_seconds,
                pose=(self.live_pose if self.live_pose is not None else captured.pose),
            ),
            replace(
                response,
                created_at_unix_seconds=self.refreshed_at_unix_seconds,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": INSERTION_EVALUATION_REFRESH_SCHEMA,
            "refreshed_at_unix_seconds": self.refreshed_at_unix_seconds,
            "live_state": self.live_state.to_dict(),
        }
        if self.live_pose is not None:
            payload["live_pose"] = list(self.live_pose.values)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionEvaluationRefresh:
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema") != INSERTION_EVALUATION_REFRESH_SCHEMA
        ):
            raise ValueError("insertion evaluation refresh schema is invalid")
        try:
            return cls(
                _strict_number(payload, "refreshed_at_unix_seconds"),
                ControlSafetySnapshot.from_dict(payload["live_state"]),
                (
                    DroidPose(tuple(payload["live_pose"]))
                    if "live_pose" in payload
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion evaluation refresh is incomplete") from error
