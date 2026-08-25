"""Stable pre-probe baseline contract for control-resolution experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
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
class ControlResolutionBaselinePolicy:
    observation_period_seconds: float = 0.25
    maximum_interval_overrun_seconds: float = 0.05
    required_consecutive_intervals: int = 2
    maximum_intervals: int = 8
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
class ControlResolutionBaselineEvidence:
    states: tuple[TrialResetState, ...]
    interval_seconds: tuple[float, ...]
    interlock: ControlInterlockEvidence

    @property
    def reference_reset(self) -> TrialResetState:
        if not self.states:
            raise ValueError("control resolution baseline has no states")
        return self.states[-1]

    def validate(
        self,
        policy: ControlResolutionBaselinePolicy,
        load: ControlResolutionLoad,
        safety_limits: SimulatorSafetyLimits = SimulatorSafetyLimits(),
    ) -> None:
        expected_state_count = policy.maximum_intervals + 1
        if (
            len(self.states) > expected_state_count
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
        ):
            raise ValueError("control resolution baseline is invalid")
        if len(self.states) < policy.required_consecutive_intervals + 1:
            raise ValueError("control resolution baseline is not stable")
        interval_passes = tuple(
            ResetEquivalenceMeasurement.between(left, right)
            .passes(policy.tolerances)
            for left, right in zip(self.states, self.states[1:])
        )
        first_qualifying_end = next(
            (
                end
                for end in range(
                    policy.required_consecutive_intervals - 1,
                    len(interval_passes),
                )
                if all(
                    interval_passes[
                        end - policy.required_consecutive_intervals + 1 : end + 1
                    ]
                )
            ),
            None,
        )
        if first_qualifying_end != len(interval_passes) - 1:
            raise ValueError("control resolution baseline is not stable")

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [state.to_dict() for state in self.states],
            "interval_seconds": list(self.interval_seconds),
            "interlock": self.interlock.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionBaselineEvidence:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution baseline must be an object")
        try:
            return cls(
                tuple(
                    TrialResetState.from_dict(state)
                    for state in payload["states"]
                ),
                tuple(float(value) for value in payload["interval_seconds"]),
                ControlInterlockEvidence.from_dict(payload["interlock"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution baseline is incomplete") from error
