"""Fail-closed simulator safety policy for one proposed control action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from time import time
from typing import Any, Sequence

from jepa_wm.action import DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.planner import PlannerActionBounds


FRANKA_LOWER_JOINT_LIMITS = (
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
)
FRANKA_UPPER_JOINT_LIMITS = (
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
)

# Ordered fail-closed projections shared by execution and evidence validation.
ACTION_SCALES = (
    DroidActionScale(1.0, 0.25, 0.25),
    DroidActionScale(0.5, 0.125, 0.125),
    DroidActionScale.uniform(0.25),
    DroidActionScale.uniform(0.125),
)


def _finite_tuple(name: str, values: Sequence[float], count: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != count or not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {count} finite values")
    return result


@dataclass(frozen=True)
class SimulatorSafetyState:
    observed_joint_positions: tuple[float, ...]
    current_joint_positions: tuple[float, ...]
    proposed_joint_positions: tuple[float, ...]
    control_period_seconds: float
    contact_force_newtons: float
    collision_detected: bool

    def __post_init__(self) -> None:
        for field in (
            "observed_joint_positions", "current_joint_positions", "proposed_joint_positions"
        ):
            object.__setattr__(self, field, _finite_tuple(field, getattr(self, field), 7))
        if (
            not isfinite(self.control_period_seconds)
            or self.control_period_seconds <= 0.0
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
        ):
            raise ValueError("simulator safety telemetry is invalid")


@dataclass(frozen=True)
class SimulatorSafetyLimits:
    minimum_workspace_xyz: tuple[float, ...] = (-0.10, -0.60, 0.00)
    maximum_workspace_xyz: tuple[float, ...] = (0.85, 0.60, 1.20)
    lower_joint_limits: tuple[float, ...] = FRANKA_LOWER_JOINT_LIMITS
    upper_joint_limits: tuple[float, ...] = FRANKA_UPPER_JOINT_LIMITS
    maximum_joint_velocity_radians_per_second: float = 0.5
    maximum_observation_joint_drift_radians: float = 0.002
    maximum_observation_plug_drift_meters: float = 0.002
    maximum_contact_force_newtons: float = 2.0
    maximum_observation_age_seconds: float = 3.0
    maximum_command_age_seconds: float = 2.5
    minimum_warmup_frames: int = 4
    action_bounds: PlannerActionBounds = PlannerActionBounds()

    def __post_init__(self) -> None:
        minimum = _finite_tuple("workspace minimum", self.minimum_workspace_xyz, 3)
        maximum = _finite_tuple("workspace maximum", self.maximum_workspace_xyz, 3)
        lower = _finite_tuple("joint lower limits", self.lower_joint_limits, 7)
        upper = _finite_tuple("joint upper limits", self.upper_joint_limits, 7)
        if any(left >= right for left, right in zip(minimum, maximum)):
            raise ValueError("workspace minimum must be below its maximum")
        if any(left >= right for left, right in zip(lower, upper)):
            raise ValueError("joint minimum must be below its maximum")
        scalars = (
            self.maximum_joint_velocity_radians_per_second,
            self.maximum_observation_joint_drift_radians,
            self.maximum_observation_plug_drift_meters,
            self.maximum_contact_force_newtons,
            self.maximum_observation_age_seconds,
            self.maximum_command_age_seconds,
        )
        if not all(isfinite(value) and value > 0.0 for value in scalars):
            raise ValueError("simulator safety limits must be finite and positive")
        if self.minimum_warmup_frames < 0:
            raise ValueError("minimum warm-up frames must be non-negative")


class ControlGateReason(str, Enum):
    STALE_OBSERVATION = "stale_observation"
    OBSERVATION_MISMATCH = "observation_mismatch"
    PROPOSAL_MISMATCH = "proposal_mismatch"
    COMMAND_TIME_INVALID = "command_time_invalid"
    WARMUP_INCOMPLETE = "warmup_incomplete"
    ACTION_OUT_OF_BOUNDS = "action_out_of_bounds"
    WORKSPACE_VIOLATION = "workspace_violation"
    GRIPPER_VIOLATION = "gripper_violation"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    OBSERVATION_STATE_DRIFT = "observation_state_drift"
    JOINT_VELOCITY_VIOLATION = "joint_velocity_violation"
    IK_SOLUTION_FAILED = "ik_solution_failed"
    COLLISION_DETECTED = "collision_detected"
    FORCE_LIMIT_EXCEEDED = "force_limit_exceeded"


@dataclass(frozen=True)
class ControlGateDecision:
    observation_id: int
    next_pose: DroidPose
    reasons: tuple[ControlGateReason, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons

    @classmethod
    def from_dict(cls, payload: Any) -> ControlGateDecision:
        if not isinstance(payload, dict):
            raise ValueError("control gate decision must be an object")
        try:
            decision = cls(
                observation_id=int(payload["observation_id"]),
                next_pose=DroidPose(tuple(payload["next_pose"])),
                reasons=tuple(ControlGateReason(reason) for reason in payload["reasons"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control gate decision is incomplete") from error
        if (
            isinstance(decision.observation_id, bool)
            or decision.observation_id <= 0
            or payload.get("passed") is not decision.passed
        ):
            raise ValueError("control gate decision is inconsistent")
        return decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "passed": self.passed,
            "next_pose": list(self.next_pose.values),
            "reasons": [reason.value for reason in self.reasons],
        }


@dataclass(frozen=True)
class SafetyProjectionAttempt:
    scale: DroidActionScale
    gate: ControlGateDecision
    maximum_joint_delta_rad: float
    proposed_joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isfinite(self.maximum_joint_delta_rad)
            or self.maximum_joint_delta_rad < 0.0
        ):
            raise ValueError("safety projection evidence is invalid")
        object.__setattr__(
            self,
            "proposed_joint_positions",
            _finite_tuple(
                "proposed joint positions", self.proposed_joint_positions, 7
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.to_dict(),
            "gate": self.gate.to_dict(),
            "maximum_joint_delta_rad": self.maximum_joint_delta_rad,
            "proposed_joint_positions": list(self.proposed_joint_positions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> SafetyProjectionAttempt:
        if not isinstance(payload, dict):
            raise ValueError("safety projection attempt must be an object")
        try:
            return cls(
                DroidActionScale.from_payload(payload["scale"]),
                ControlGateDecision.from_dict(payload["gate"]),
                float(payload["maximum_joint_delta_rad"]),
                tuple(payload["proposed_joint_positions"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("safety projection attempt is incomplete") from error


class SimulatorControlGate:
    def __init__(self, limits: SimulatorSafetyLimits = SimulatorSafetyLimits()):
        self.limits = limits

    def evaluate(
        self,
        observation: ControlObservation,
        proposal: ProposedControl,
        state: SimulatorSafetyState,
        *,
        now_unix_seconds: float | None = None,
    ) -> ControlGateDecision:
        now = time() if now_unix_seconds is None else now_unix_seconds
        reasons = []
        if (
            now < observation.captured_at_unix_seconds
            or now - observation.captured_at_unix_seconds
            > self.limits.maximum_observation_age_seconds
        ):
            reasons.append(ControlGateReason.STALE_OBSERVATION)
        if proposal.observation_id != observation.observation_id:
            reasons.append(ControlGateReason.OBSERVATION_MISMATCH)
        if proposal.proposal != observation.expected_proposal:
            reasons.append(ControlGateReason.PROPOSAL_MISMATCH)
        if (
            proposal.created_at_unix_seconds < observation.captured_at_unix_seconds
            or proposal.created_at_unix_seconds > now
            or now - proposal.created_at_unix_seconds
            > self.limits.maximum_command_age_seconds
        ):
            reasons.append(ControlGateReason.COMMAND_TIME_INVALID)
        if observation.warmup_frames < self.limits.minimum_warmup_frames:
            reasons.append(ControlGateReason.WARMUP_INCOMPLETE)
        if not self.limits.action_bounds.accepts(proposal.actions):
            reasons.append(ControlGateReason.ACTION_OUT_OF_BOUNDS)
        try:
            next_pose = observation.pose.applied(proposal.first_action)
        except ValueError:
            next_pose = observation.pose
            reasons.append(ControlGateReason.GRIPPER_VIOLATION)
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                next_pose.values[:3],
                self.limits.minimum_workspace_xyz,
                self.limits.maximum_workspace_xyz,
            )
        ):
            reasons.append(ControlGateReason.WORKSPACE_VIOLATION)
        if not 0.0 <= next_pose.values[6] <= 1.0:
            reasons.append(ControlGateReason.GRIPPER_VIOLATION)
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                state.proposed_joint_positions,
                self.limits.lower_joint_limits,
                self.limits.upper_joint_limits,
            )
        ):
            reasons.append(ControlGateReason.JOINT_LIMIT_VIOLATION)
        maximum_joint_delta = (
            self.limits.maximum_joint_velocity_radians_per_second
            * state.control_period_seconds
        )
        if any(
            abs(current - observed)
            > self.limits.maximum_observation_joint_drift_radians
            for observed, current in zip(
                state.observed_joint_positions, state.current_joint_positions
            )
        ):
            reasons.append(ControlGateReason.OBSERVATION_STATE_DRIFT)
        if any(
            abs(proposed - current) > maximum_joint_delta
            for current, proposed in zip(
                state.current_joint_positions, state.proposed_joint_positions
            )
        ):
            reasons.append(ControlGateReason.JOINT_VELOCITY_VIOLATION)
        if state.collision_detected:
            reasons.append(ControlGateReason.COLLISION_DETECTED)
        if state.contact_force_newtons > self.limits.maximum_contact_force_newtons:
            reasons.append(ControlGateReason.FORCE_LIMIT_EXCEEDED)
        return ControlGateDecision(observation.observation_id, next_pose, tuple(reasons))
