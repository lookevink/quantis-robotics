"""Stable pre-probe baseline contract for control-resolution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any, Mapping

from jepa_wm.control_resolution_profile import ControlResolutionLoad
from jepa_wm.control_safety import ControlInterlockEvidence, SimulatorSafetyLimits
from jepa_wm.trial_equivalence import (
    ResetEquivalenceMeasurement,
    ResetEquivalenceTolerances,
    TrialResetState,
)


CONTROL_RESOLUTION_BASELINE_TOLERANCES = ResetEquivalenceTolerances(
    maximum_translation_difference_meters=1.25e-4,
    maximum_rotation_difference_radians=5e-4,
    maximum_gripper_difference=5e-4,
    maximum_joint_difference_radians=2.5e-4,
    maximum_reset_contact_force_newtons=0.01,
    maximum_plug_position_difference_meters=1.25e-4,
)


@dataclass(frozen=True)
class ControlResolutionDriveTarget:
    joint_positions: tuple[float, ...]
    gripper_width_m: float

    def __post_init__(self) -> None:
        if (
            len(self.joint_positions) != 7
            or not all(isfinite(value) for value in self.joint_positions)
            or not isfinite(self.gripper_width_m)
            or not 0.0 <= self.gripper_width_m <= 0.08
        ):
            raise ValueError("control resolution drive target is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_positions": list(self.joint_positions),
            "gripper_width_m": self.gripper_width_m,
        }

    def validate_active(
        self,
        joint_positions: tuple[float, ...],
        gripper_width_m: float,
    ) -> None:
        if (
            len(joint_positions) != 7
            or not all(
                isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    joint_positions,
                    self.joint_positions,
                )
            )
            or not isclose(
                gripper_width_m,
                self.gripper_width_m,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("control resolution active drive target changed")

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionDriveTarget:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution drive target must be an object")
        try:
            return cls(
                tuple(float(value) for value in payload["joint_positions"]),
                float(payload["gripper_width_m"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution drive target is incomplete"
            ) from error


@dataclass(frozen=True)
class ControlResolutionBaselinePolicy:
    observation_period_seconds: float = 0.25
    maximum_interval_overrun_seconds: float = 0.05
    required_consecutive_intervals: int = 8
    maximum_intervals: int = 80
    tolerances: ResetEquivalenceTolerances = (
        CONTROL_RESOLUTION_BASELINE_TOLERANCES
    )

    def __post_init__(self) -> None:
        if (
            not isfinite(self.observation_period_seconds)
            or self.observation_period_seconds <= 0.0
            or not isfinite(self.maximum_interval_overrun_seconds)
            or self.maximum_interval_overrun_seconds < 0.0
            or isinstance(self.required_consecutive_intervals, bool)
            or not isinstance(self.required_consecutive_intervals, int)
            or self.required_consecutive_intervals <= 0
            or isinstance(self.maximum_intervals, bool)
            or not isinstance(self.maximum_intervals, int)
            or self.maximum_intervals < self.required_consecutive_intervals
        ):
            raise ValueError("control resolution baseline policy is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_period_seconds": self.observation_period_seconds,
            "maximum_interval_overrun_seconds": (
                self.maximum_interval_overrun_seconds
            ),
            "required_consecutive_intervals": self.required_consecutive_intervals,
            "maximum_intervals": self.maximum_intervals,
            "tolerances": self.tolerances.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionBaselinePolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution baseline policy must be an object")
        required = payload.get("required_consecutive_intervals")
        maximum = payload.get("maximum_intervals")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (required, maximum)
        ):
            raise ValueError("baseline interval counts must be integers")
        try:
            return cls(
                float(payload["observation_period_seconds"]),
                float(payload["maximum_interval_overrun_seconds"]),
                required,
                maximum,
                ResetEquivalenceTolerances.from_dict(payload["tolerances"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution baseline policy is incomplete"
            ) from error


@dataclass(frozen=True)
class ControlResolutionBaselineTrace:
    states: tuple[TrialResetState, ...]
    interval_seconds: tuple[float, ...]
    interlock: ControlInterlockEvidence
    drive_target: ControlResolutionDriveTarget | None = None

    @property
    def initial_reset(self) -> TrialResetState:
        if not self.states:
            raise ValueError("control resolution baseline has no states")
        return self.states[0]

    def interval_passes(
        self,
        policy: ControlResolutionBaselinePolicy,
        load: ControlResolutionLoad,
        safety_limits: SimulatorSafetyLimits,
    ) -> tuple[bool, ...]:
        if (
            not self.states
            or len(self.states) > policy.maximum_intervals + 1
            or len(self.interval_seconds) != len(self.states) - 1
            or any(
                not isfinite(interval)
                or interval < policy.observation_period_seconds
                or interval
                > policy.observation_period_seconds
                + policy.maximum_interval_overrun_seconds
                for interval in self.interval_seconds
            )
            or self.interlock.collision_detected
            or self.interlock.maximum_contact_force_newtons
            > safety_limits.maximum_contact_force_newtons
            or any(
                state.plug_attached is not load.plug_attached
                or state.collision_detected
                or state.contact_force_newtons
                > safety_limits.maximum_contact_force_newtons
                for state in self.states
            )
            or (
                self.drive_target is not None
                and any(
                    value < lower or value > upper
                    for value, lower, upper in zip(
                        self.drive_target.joint_positions,
                        safety_limits.lower_joint_limits,
                        safety_limits.upper_joint_limits,
                    )
                )
            )
        ):
            raise ValueError("control resolution baseline trace is invalid")
        return tuple(
            ResetEquivalenceMeasurement.between(left, right).passes(
                policy.tolerances
            )
            for left, right in zip(self.states, self.states[1:])
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "states": [state.to_dict() for state in self.states],
            "interval_seconds": list(self.interval_seconds),
            "interlock": self.interlock.to_dict(),
        }
        if self.drive_target is not None:
            payload["drive_target"] = self.drive_target.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionBaselineTrace:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution baseline trace must be an object")
        try:
            return cls(
                tuple(
                    TrialResetState.from_dict(state)
                    for state in payload["states"]
                ),
                tuple(float(value) for value in payload["interval_seconds"]),
                ControlInterlockEvidence.from_dict(payload["interlock"]),
                (
                    ControlResolutionDriveTarget.from_dict(
                        payload["drive_target"]
                    )
                    if "drive_target" in payload
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution baseline trace is incomplete"
            ) from error

    def first_qualifying_end(
        self,
        policy: ControlResolutionBaselinePolicy,
        load: ControlResolutionLoad,
        safety_limits: SimulatorSafetyLimits,
    ) -> int | None:
        interval_passes = self.interval_passes(policy, load, safety_limits)
        return next(
            (
                end
                for end in range(
                    policy.required_consecutive_intervals - 1,
                    len(interval_passes),
                )
                if self._window_passes(policy, interval_passes, end)
            ),
            None,
        )

    def _window_passes(
        self,
        policy: ControlResolutionBaselinePolicy,
        interval_passes: tuple[bool, ...],
        end: int,
    ) -> bool:
        start = end - policy.required_consecutive_intervals + 1
        window = self.states[start : end + 2]
        return all(interval_passes[start : end + 1]) and all(
            ResetEquivalenceMeasurement.between(left, right).passes(
                policy.tolerances
            )
            for left_index, left in enumerate(window)
            for right in window[left_index + 1 :]
        )


@dataclass(frozen=True)
class ControlResolutionBaselineAttempt:
    trace: ControlResolutionBaselineTrace

    @property
    def initial_reset(self) -> TrialResetState:
        return self.trace.initial_reset

    @property
    def drive_target(self) -> ControlResolutionDriveTarget | None:
        return self.trace.drive_target

    def validate(
        self,
        policy: ControlResolutionBaselinePolicy,
        load: ControlResolutionLoad,
        safety_limits: SimulatorSafetyLimits = SimulatorSafetyLimits(),
    ) -> None:
        first_qualifying_end = self.trace.first_qualifying_end(
            policy,
            load,
            safety_limits,
        )
        if (
            len(self.trace.interval_seconds) != policy.maximum_intervals
            or first_qualifying_end is not None
        ):
            raise ValueError("control resolution baseline attempt is not a failure")

    def to_dict(self) -> dict[str, Any]:
        return self.trace.to_dict()

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionBaselineAttempt:
        return cls(ControlResolutionBaselineTrace.from_dict(payload))


@dataclass(frozen=True)
class ControlResolutionBaselineEvidence:
    trace: ControlResolutionBaselineTrace

    @property
    def initial_reset(self) -> TrialResetState:
        return self.trace.initial_reset

    @property
    def drive_target(self) -> ControlResolutionDriveTarget | None:
        return self.trace.drive_target

    @property
    def reference_reset(self) -> TrialResetState:
        if not self.trace.states:
            raise ValueError("control resolution baseline has no states")
        return self.trace.states[-1]

    def validate(
        self,
        policy: ControlResolutionBaselinePolicy,
        load: ControlResolutionLoad,
        safety_limits: SimulatorSafetyLimits = SimulatorSafetyLimits(),
    ) -> None:
        first_qualifying_end = self.trace.first_qualifying_end(
            policy,
            load,
            safety_limits,
        )
        if len(self.trace.states) < policy.required_consecutive_intervals + 1:
            raise ValueError("control resolution baseline is not stable")
        if first_qualifying_end != len(self.trace.interval_seconds) - 1:
            raise ValueError("control resolution baseline is not stable")

    def to_dict(self) -> dict[str, Any]:
        return self.trace.to_dict()

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionBaselineEvidence:
        return cls(ControlResolutionBaselineTrace.from_dict(payload))
