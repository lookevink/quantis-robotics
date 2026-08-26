"""Typed evidence for a reset-repeatable insertion control-resolution experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Mapping, Union

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import (
    DROID_FPS,
    DroidAction,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.control_safety import (
    ControlGateReason,
    ControlInterlockEvidence,
    SafetyProjectionAttempt,
    SimulatorSafetyLimits,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.control_resolution_baseline import (
    CONTROL_RESOLUTION_BASELINE_TOLERANCES,
    ControlResolutionBaselineAttempt,
    ControlResolutionBaselineEvidence,
    ControlResolutionBaselinePolicy,
    ControlResolutionBaselineTrace,
    ControlResolutionCaptureSourceIdentity,
    ControlResolutionDriveTarget,
)
from jepa_wm.control_resolution_profile import ControlResolutionLoad
from jepa_wm.direct_safety import ControlSafetySnapshot
from jepa_wm.trial_equivalence import (
    ResetEquivalenceMeasurement,
    ResetEquivalenceTolerances,
    TrialResetState,
    validate_reset_equivalence,
)
from sim.recording import validate_recording_id


CONTROL_RESOLUTION_SCHEMA_V1 = "quantis.jepa_wm_control_resolution.v1"
CONTROL_RESOLUTION_SCHEMA_V2 = "quantis.jepa_wm_control_resolution.v2"
CONTROL_RESOLUTION_SCHEMA = "quantis.jepa_wm_control_resolution.v3"
CONTROL_RESOLUTION_FAILURE_SCHEMA_V1 = (
    "quantis.jepa_wm_control_resolution_failure.v1"
)
CONTROL_RESOLUTION_FAILURE_SCHEMA_V2 = (
    "quantis.jepa_wm_control_resolution_failure.v2"
)
CONTROL_RESOLUTION_FAILURE_SCHEMA_V3 = (
    "quantis.jepa_wm_control_resolution_failure.v3"
)
CONTROL_RESOLUTION_FAILURE_SCHEMA_V4 = (
    "quantis.jepa_wm_control_resolution_failure.v4"
)
CONTROL_RESOLUTION_FAILURE_SCHEMA = "quantis.jepa_wm_control_resolution_failure.v5"
CONTROL_RESOLUTION_RESET_TOLERANCES = ResetEquivalenceTolerances(
    maximum_translation_difference_meters=5e-4,
    maximum_rotation_difference_radians=1e-3,
    maximum_gripper_difference=1e-3,
    maximum_joint_difference_radians=5e-4,
    maximum_reset_contact_force_newtons=0.01,
    maximum_plug_position_difference_meters=5e-4,
)


def maximum_joint_position_delta(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> float:
    """Return the maximum absolute delta between two finite Franka arm states."""

    if (
        len(left) != 7
        or len(right) != 7
        or not all(isfinite(value) for value in (*left, *right))
    ):
        raise ValueError("joint positions must be finite seven-axis values")
    return max(abs(left_value - right_value) for left_value, right_value in zip(left, right))


def _reconstructed_metric_payload_matches(expected: Any, actual: Any) -> bool:
    """Compare reconstructed metrics across Python/NumPy runtimes."""

    if isinstance(expected, float):
        return (
            not isinstance(actual, bool)
            and isinstance(actual, (int, float))
            and isfinite(actual)
            and isclose(expected, float(actual), rel_tol=1e-12, abs_tol=1e-15)
        )
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(
                _reconstructed_metric_payload_matches(value, actual[key])
                for key, value in expected.items()
            )
        )
    if isinstance(expected, (list, tuple)):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(
                _reconstructed_metric_payload_matches(left, right)
                for left, right in zip(expected, actual)
            )
        )
    return type(expected) is type(actual) and expected == actual


@dataclass(frozen=True)
class ControlResolutionMotionPeriod:
    translation_meters: float
    seconds: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.translation_meters)
            or self.translation_meters <= 0.0
            or not isfinite(self.seconds)
            or self.seconds <= 0.0
        ):
            raise ValueError("control resolution motion period is invalid")

    def to_dict(self) -> dict[str, float]:
        return {
            "translation_meters": self.translation_meters,
            "seconds": self.seconds,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionMotionPeriod:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution motion period must be an object")
        try:
            return cls(
                float(payload["translation_meters"]),
                float(payload["seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution motion period is incomplete") from error


def retreat_direction(
    captured_pose: DroidPose,
    recorded_target_pose: DroidPose,
) -> tuple[float, float, float]:
    """Return the unit direction away from the recorded insertion target."""

    delta = np.asarray(recorded_target_pose.values[:3]) - np.asarray(
        captured_pose.values[:3]
    )
    norm = float(np.linalg.norm(delta))
    if not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("control resolution target has no translation direction")
    return tuple(float(value) for value in (-delta / norm))


@dataclass(frozen=True)
class TrackedErrorSettlement:
    absolute_tracking_floor_radians: float = 5e-4
    tracking_error_fraction_of_requested_motion: float = 0.25
    rollback_tracking_error_cap_radians: float | None = (
        CONTROL_RESOLUTION_RESET_TOLERANCES.maximum_joint_difference_radians
    )
    required_consecutive_updates: int = 2
    maximum_updates: int = 48

    def __post_init__(self) -> None:
        if (
            not isfinite(self.absolute_tracking_floor_radians)
            or self.absolute_tracking_floor_radians <= 0.0
            or not isfinite(self.tracking_error_fraction_of_requested_motion)
            or not 0.0 < self.tracking_error_fraction_of_requested_motion < 1.0
            or (
                self.rollback_tracking_error_cap_radians is not None
                and (
                    not isfinite(self.rollback_tracking_error_cap_radians)
                    or self.rollback_tracking_error_cap_radians <= 0.0
                )
            )
            or isinstance(self.required_consecutive_updates, bool)
            or not isinstance(self.required_consecutive_updates, int)
            or self.required_consecutive_updates <= 0
            or isinstance(self.maximum_updates, bool)
            or not isinstance(self.maximum_updates, int)
            or self.maximum_updates < self.required_consecutive_updates
        ):
            raise ValueError("control resolution settlement policy is invalid")

    def maximum_tracking_error(
        self,
        requested_joint_motion_radians: float,
        cap_radians: float | None = None,
    ) -> float:
        if (
            not isfinite(requested_joint_motion_radians)
            or requested_joint_motion_radians < 0.0
        ):
            raise ValueError("requested joint motion is invalid")
        maximum_error = max(
            self.absolute_tracking_floor_radians,
            requested_joint_motion_radians
            * self.tracking_error_fraction_of_requested_motion,
        )
        if cap_radians is None:
            return maximum_error
        if not isfinite(cap_radians) or cap_radians <= 0.0:
            raise ValueError("tracking error cap is invalid")
        return min(maximum_error, cap_radians)

    def to_dict(self) -> dict[str, Any]:
        return {
            "absolute_tracking_floor_radians": (
                self.absolute_tracking_floor_radians
            ),
            "tracking_error_fraction_of_requested_motion": (
                self.tracking_error_fraction_of_requested_motion
            ),
            **(
                {
                    "rollback_tracking_error_cap_radians": (
                        self.rollback_tracking_error_cap_radians
                    )
                }
                if self.rollback_tracking_error_cap_radians is not None
                else {}
            ),
            "required_consecutive_updates": self.required_consecutive_updates,
            "maximum_updates": self.maximum_updates,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TrackedErrorSettlement:
        if not isinstance(payload, dict):
            raise ValueError("control resolution settlement policy must be an object")
        required_updates = payload.get("required_consecutive_updates")
        maximum_updates = payload.get("maximum_updates")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (required_updates, maximum_updates)
        ):
            raise ValueError("settlement update counts must be integers")
        try:
            return cls(
                absolute_tracking_floor_radians=float(
                    payload["absolute_tracking_floor_radians"]
                ),
                tracking_error_fraction_of_requested_motion=float(
                    payload["tracking_error_fraction_of_requested_motion"]
                ),
                rollback_tracking_error_cap_radians=(
                    float(payload["rollback_tracking_error_cap_radians"])
                    if "rollback_tracking_error_cap_radians" in payload
                    else None
                ),
                required_consecutive_updates=required_updates,
                maximum_updates=maximum_updates,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution settlement policy is incomplete"
            ) from error

    def protocol_fields(self) -> dict[str, Any]:
        return {"tracked_error_settlement": self.to_dict()}

    def validate_evidence(
        self,
        evidence: TrackedSettlementEvidence | None,
        motion_requested_joint_motion_radians: float,
        motion_final_tracking_error_radians: float,
        rollback_requested_joint_motion_radians: float,
        rollback_final_tracking_error_radians: float,
    ) -> None:
        if evidence is None:
            raise ValueError("control resolution settlement is missing")
        evidence.validate(
            self,
            motion_requested_joint_motion_radians,
            motion_final_tracking_error_radians,
            rollback_requested_joint_motion_radians,
            rollback_final_tracking_error_radians,
        )

    def complete_evidence(
        self,
        motion: ControlResolutionSettlementEvidence | None,
        rollback: ControlResolutionSettlementEvidence | None,
        rollback_interlock: ControlInterlockEvidence,
    ) -> TrackedSettlementEvidence:
        if motion is None or rollback is None:
            raise ValueError("tracked settlement evidence is incomplete")
        return TrackedSettlementEvidence(motion, rollback, rollback_interlock)


@dataclass(frozen=True)
class FixedUpdateSettlement:
    updates: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.updates, bool)
            or not isinstance(self.updates, int)
            or self.updates <= 0
        ):
            raise ValueError("fixed settlement update count is invalid")

    def protocol_fields(self) -> dict[str, Any]:
        return {"settling_updates": self.updates}

    def validate_evidence(
        self,
        evidence: TrackedSettlementEvidence | None,
        motion_requested_joint_motion_radians: float,
        motion_final_tracking_error_radians: float,
        rollback_requested_joint_motion_radians: float,
        rollback_final_tracking_error_radians: float,
    ) -> None:
        del (
            motion_requested_joint_motion_radians,
            motion_final_tracking_error_radians,
            rollback_requested_joint_motion_radians,
            rollback_final_tracking_error_radians,
        )
        if evidence is not None:
            raise ValueError("fixed-update control resolution sample has settlement")

    def complete_evidence(
        self,
        motion: ControlResolutionSettlementEvidence | None,
        rollback: ControlResolutionSettlementEvidence | None,
        rollback_interlock: ControlInterlockEvidence,
    ) -> None:
        del rollback_interlock
        if motion is not None or rollback is not None:
            raise ValueError("fixed-update settlement produced tracked evidence")
        return None


ControlResolutionSettlement = Union[
    FixedUpdateSettlement,
    TrackedErrorSettlement,
]


class ControlResolutionProbeKind(str, Enum):
    HOLD = "hold"
    TRANSLATION = "translation"


@dataclass(frozen=True)
class DriveCommandSkipped:
    def to_dict(self) -> dict[str, str]:
        return {"status": "skipped"}


@dataclass(frozen=True)
class DriveCommandApplied:
    period_seconds: float

    def __post_init__(self) -> None:
        if not isfinite(self.period_seconds) or self.period_seconds <= 0.0:
            raise ValueError("control resolution drive command period is invalid")

    def to_dict(self) -> dict[str, float | str]:
        return {"status": "applied", "period_seconds": self.period_seconds}


ControlResolutionDriveCommand = Union[DriveCommandSkipped, DriveCommandApplied]


def _drive_command_from_dict(payload: Any) -> ControlResolutionDriveCommand:
    if not isinstance(payload, Mapping):
        raise ValueError("control resolution drive command must be an object")
    status = payload.get("status")
    if status == "skipped":
        if "period_seconds" in payload:
            raise ValueError("skipped drive command carries a period")
        return DriveCommandSkipped()
    if status == "applied":
        try:
            return DriveCommandApplied(float(payload["period_seconds"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("applied drive command is incomplete") from error
    raise ValueError("control resolution drive command status is invalid")


@dataclass(frozen=True)
class ControlResolutionProbePlan:
    sample_index: int
    requested_translation_meters: float
    kind: ControlResolutionProbeKind

    def __post_init__(self) -> None:
        expected_kind = (
            ControlResolutionProbeKind.HOLD
            if self.requested_translation_meters == 0.0
            else ControlResolutionProbeKind.TRANSLATION
        )
        if (
            isinstance(self.sample_index, bool)
            or not isinstance(self.sample_index, int)
            or self.sample_index < 0
            or not isfinite(self.requested_translation_meters)
            or self.requested_translation_meters < 0.0
            or self.kind is not expected_kind
        ):
            raise ValueError("control resolution probe plan is invalid")

    @classmethod
    def for_request(
        cls,
        sample_index: int,
        requested_translation_meters: float,
    ) -> ControlResolutionProbePlan:
        return cls(
            sample_index,
            requested_translation_meters,
            (
                ControlResolutionProbeKind.HOLD
                if requested_translation_meters == 0.0
                else ControlResolutionProbeKind.TRANSLATION
            ),
        )

    @property
    def applies_drive_command(self) -> bool:
        return self.kind is ControlResolutionProbeKind.TRANSLATION

    def drive_command(
        self,
        period_seconds: float | None,
    ) -> ControlResolutionDriveCommand:
        if not self.applies_drive_command:
            if period_seconds is not None:
                raise ValueError("hold probe cannot carry a drive command period")
            return DriveCommandSkipped()
        if period_seconds is None:
            raise ValueError("translation probe requires a drive command period")
        return DriveCommandApplied(period_seconds)

    def settlement_joint_target(
        self,
        live_start_joint_positions: tuple[float, ...],
        projected_joint_positions: tuple[float, ...] | None,
    ) -> tuple[float, ...]:
        if self.kind is ControlResolutionProbeKind.HOLD:
            if projected_joint_positions is not None:
                raise ValueError("hold probe cannot replace the active drive target")
            return live_start_joint_positions
        if projected_joint_positions is None:
            raise ValueError("translation probe has no projected joint target")
        return projected_joint_positions

    def controller_tracking_joint_target(
        self,
        drive_target: ControlResolutionDriveTarget,
        projected_joint_positions: tuple[float, ...],
    ) -> tuple[float, ...]:
        return (
            projected_joint_positions
            if self.applies_drive_command
            else drive_target.joint_positions
        )

    def rollback_joint_target(
        self,
        _drive_target: ControlResolutionDriveTarget,
        reference_reset: TrialResetState,
    ) -> tuple[float, ...]:
        return reference_reset.joint_positions

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_index": self.sample_index,
            "requested_translation_meters": self.requested_translation_meters,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionProbePlan:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution probe plan must be an object")
        sample_index = payload.get("sample_index")
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise ValueError("control resolution probe index must be an integer")
        try:
            return cls(
                sample_index,
                float(payload["requested_translation_meters"]),
                ControlResolutionProbeKind(payload["kind"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution probe plan is incomplete") from error


@dataclass(frozen=True)
class ControlResolutionProbeExecution:
    probe: ControlResolutionProbePlan
    recorded_target_pose: DroidPose
    start_reset: TrialResetState
    commanded_action: DroidAction
    target_pose: DroidPose
    projection: SafetyProjectionAttempt

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        observation_id: int,
    ) -> None:
        expected_action = protocol.probe_action(
            self.start_reset.pose,
            self.recorded_target_pose,
            self.probe.requested_translation_meters,
        )
        expected_target = self.start_reset.pose.applied(expected_action)
        expected_joint_delta = maximum_joint_position_delta(
            self.projection.proposed_joint_positions,
            self.start_reset.joint_positions,
        )
        projected_positions = (
            None
            if self.probe.kind is ControlResolutionProbeKind.HOLD
            else self.projection.proposed_joint_positions
        )
        expected_joint_target = self.probe.settlement_joint_target(
            self.start_reset.joint_positions,
            projected_positions,
        )
        limits = protocol.safety_limits
        start_target_distance = float(
            np.linalg.norm(
                np.asarray(self.start_reset.pose.values[:3])
                - np.asarray(self.recorded_target_pose.values[:3])
            )
        )
        probe_target_distance = float(
            np.linalg.norm(
                np.asarray(self.target_pose.values[:3])
                - np.asarray(self.recorded_target_pose.values[:3])
            )
        )
        if (
            not _reconstructed_metric_payload_matches(
                list(expected_action.values),
                list(self.commanded_action.values),
            )
            or not _reconstructed_metric_payload_matches(
                list(expected_target.values),
                list(self.target_pose.values),
            )
            or self.projection.proposed_joint_positions != expected_joint_target
            or not self.projection.gate.passed
            or self.projection.gate.observation_id != observation_id
            or self.projection.gate.next_pose != self.target_pose
            or not isclose(
                self.projection.maximum_joint_delta_rad,
                expected_joint_delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or (
                self.probe.kind is ControlResolutionProbeKind.TRANSLATION
                and probe_target_distance <= start_target_distance
            )
            or not limits.action_bounds.accepts((self.commanded_action,))
            or any(
                value < lower or value > upper
                for value, lower, upper in zip(
                    self.target_pose.values[:3],
                    limits.minimum_workspace_xyz,
                    limits.maximum_workspace_xyz,
                )
            )
            or not 0.0 <= self.target_pose.values[6] <= 1.0
            or any(
                value < lower or value > upper
                for value, lower, upper in zip(
                    self.projection.proposed_joint_positions,
                    limits.lower_joint_limits,
                    limits.upper_joint_limits,
                )
            )
            or expected_joint_delta
            > limits.maximum_joint_velocity_radians_per_second
            * protocol.motion_period_for(
                self.probe.requested_translation_meters
            )
        ):
            raise ValueError(
                "control resolution probe execution does not match its protocol"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe.to_dict(),
            "recorded_target_pose": list(self.recorded_target_pose.values),
            "start_reset": self.start_reset.to_dict(),
            "commanded_action": list(self.commanded_action.values),
            "target_pose": list(self.target_pose.values),
            "projection": self.projection.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionProbeExecution:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution probe execution must be an object")
        try:
            return cls(
                ControlResolutionProbePlan.from_dict(payload["probe"]),
                DroidPose(tuple(payload["recorded_target_pose"])),
                TrialResetState.from_dict(payload["start_reset"]),
                DroidAction(tuple(payload["commanded_action"])),
                DroidPose(tuple(payload["target_pose"])),
                SafetyProjectionAttempt.from_dict(payload["projection"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution probe execution is incomplete"
            ) from error


@dataclass(frozen=True)
class ControlResolutionProjectionFailure:
    """Exact pre-actuation probe rejected by the simulator safety gate."""

    probe: ControlResolutionProbePlan
    recorded_target_pose: DroidPose
    start_reset: TrialResetState
    commanded_action: DroidAction
    target_pose: DroidPose
    projection: SafetyProjectionAttempt
    motion_period_seconds: float

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        observation_id: int,
    ) -> None:
        expected_action = protocol.probe_action(
            self.start_reset.pose,
            self.recorded_target_pose,
            self.probe.requested_translation_meters,
        )
        expected_target = self.start_reset.pose.applied(expected_action)
        expected_delta = maximum_joint_position_delta(
            self.projection.proposed_joint_positions,
            self.start_reset.joint_positions,
        )
        velocity_rejected = (
            ControlGateReason.JOINT_VELOCITY_VIOLATION
            in self.projection.gate.reasons
        )
        if (
            protocol.probe_plan(self.probe.sample_index) != self.probe
            or self.probe.kind is not ControlResolutionProbeKind.TRANSLATION
            or self.commanded_action != expected_action
            or self.target_pose != expected_target
            or self.projection.gate.passed
            or not self.projection.gate.reasons
            or self.projection.gate.observation_id != observation_id
            or self.projection.gate.next_pose != self.target_pose
            or not isclose(
                self.projection.maximum_joint_delta_rad,
                expected_delta,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isclose(
                self.motion_period_seconds,
                protocol.motion_period_for(
                    self.probe.requested_translation_meters
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or (
                velocity_rejected
                and expected_delta
                <= protocol.safety_limits.maximum_joint_velocity_radians_per_second
                * self.motion_period_seconds
            )
        ):
            raise ValueError(
                "control resolution projection failure is inconsistent"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe": self.probe.to_dict(),
            "recorded_target_pose": list(self.recorded_target_pose.values),
            "start_reset": self.start_reset.to_dict(),
            "commanded_action": list(self.commanded_action.values),
            "target_pose": list(self.target_pose.values),
            "projection": self.projection.to_dict(),
            "motion_period_seconds": self.motion_period_seconds,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionProjectionFailure:
        if not isinstance(payload, Mapping):
            raise ValueError(
                "control resolution projection failure must be an object"
            )
        try:
            return cls(
                ControlResolutionProbePlan.from_dict(payload["probe"]),
                DroidPose(tuple(payload["recorded_target_pose"])),
                TrialResetState.from_dict(payload["start_reset"]),
                DroidAction(tuple(payload["commanded_action"])),
                DroidPose(tuple(payload["target_pose"])),
                SafetyProjectionAttempt.from_dict(payload["projection"]),
                float(payload["motion_period_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution projection failure is incomplete"
            ) from error


def _settlement_from_protocol_dict(
    payload: Mapping[str, Any],
) -> ControlResolutionSettlement:
    has_tracked = "tracked_error_settlement" in payload
    has_fixed = "settling_updates" in payload
    if has_tracked == has_fixed:
        raise ValueError("control resolution settlement variant is invalid")
    if has_tracked:
        return TrackedErrorSettlement.from_dict(
            payload["tracked_error_settlement"]
        )
    return FixedUpdateSettlement(payload["settling_updates"])


@dataclass(frozen=True)
class ControlResolutionProtocol:
    translation_magnitudes_meters: tuple[float, ...] = (0.0, 5e-4, 1e-3)
    repeats_per_magnitude: int = 3
    motion_period_seconds: float = 0.25
    motion_period_overrides: tuple[ControlResolutionMotionPeriod, ...] = (
        ControlResolutionMotionPeriod(5e-4, 0.5),
        ControlResolutionMotionPeriod(1e-3, 0.5),
    )
    baseline_policy: ControlResolutionBaselinePolicy | None = (
        ControlResolutionBaselinePolicy()
    )
    settlement: ControlResolutionSettlement = TrackedErrorSettlement()
    safety_limits: SimulatorSafetyLimits = SimulatorSafetyLimits()
    capture_tolerances: ResetEquivalenceTolerances = ResetEquivalenceTolerances()
    reset_tolerances: ResetEquivalenceTolerances = (
        CONTROL_RESOLUTION_RESET_TOLERANCES
    )

    def __post_init__(self) -> None:
        if (
            not self.translation_magnitudes_meters
            or self.translation_magnitudes_meters[0] != 0.0
            or tuple(sorted(set(self.translation_magnitudes_meters)))
            != self.translation_magnitudes_meters
            or not all(
                isfinite(value) and value >= 0.0
                for value in self.translation_magnitudes_meters
            )
            or self.repeats_per_magnitude < 2
            or not isfinite(self.motion_period_seconds)
            or self.motion_period_seconds != 1.0 / DROID_FPS
            or len(
                {
                    override.translation_meters
                    for override in self.motion_period_overrides
                }
            )
            != len(self.motion_period_overrides)
            or any(
                override.translation_meters
                not in self.translation_magnitudes_meters
                or override.seconds < self.motion_period_seconds
                for override in self.motion_period_overrides
            )
            or not isinstance(
                self.settlement,
                (FixedUpdateSettlement, TrackedErrorSettlement),
            )
            or (
                isinstance(self.settlement, TrackedErrorSettlement)
                and self.settlement.rollback_tracking_error_cap_radians
                is not None
                and self.settlement.rollback_tracking_error_cap_radians
                != self.reset_tolerances.maximum_joint_difference_radians
            )
        ):
            raise ValueError("control resolution protocol is invalid")

    @property
    def requested_translations(self) -> tuple[float, ...]:
        return tuple(
            plan.requested_translation_meters for plan in self.probe_plans
        )

    @property
    def probe_plans(self) -> tuple[ControlResolutionProbePlan, ...]:
        return tuple(
            ControlResolutionProbePlan.for_request(index, magnitude)
            for index, magnitude in enumerate(
                magnitude
                for magnitude in self.translation_magnitudes_meters
                for _ in range(self.repeats_per_magnitude)
            )
        )

    def probe_plan(self, sample_index: int) -> ControlResolutionProbePlan:
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < len(self.probe_plans)
        ):
            raise ValueError("control resolution probe index is invalid")
        return self.probe_plans[sample_index]

    def motion_period_for(self, translation_meters: float) -> float:
        if translation_meters not in self.translation_magnitudes_meters:
            raise ValueError("control resolution magnitude is not in the protocol")
        return next(
            (
                override.seconds
                for override in self.motion_period_overrides
                if override.translation_meters == translation_meters
            ),
            self.motion_period_seconds,
        )

    def safe_joint_motion_period(
        self,
        start_joint_positions: tuple[float, ...],
        target_joint_positions: tuple[float, ...],
        minimum_period_seconds: float,
    ) -> float:
        if (
            len(start_joint_positions) != 7
            or len(target_joint_positions) != 7
            or not isfinite(minimum_period_seconds)
            or minimum_period_seconds <= 0.0
        ):
            raise ValueError("control resolution joint motion period is invalid")
        maximum_delta = maximum_joint_position_delta(
            start_joint_positions,
            target_joint_positions,
        )
        return max(
            minimum_period_seconds,
            maximum_delta
            / self.safety_limits.maximum_joint_velocity_radians_per_second,
        )

    def probe_action(
        self,
        live_pose: DroidPose,
        recorded_target_pose: DroidPose,
        translation_meters: float,
    ) -> DroidAction:
        """Build the protocol retreat action from one exact live start."""

        if not isfinite(translation_meters) or translation_meters < 0.0:
            raise ValueError("control resolution probe magnitude is invalid")
        direction = retreat_direction(live_pose, recorded_target_pose)
        return DroidAction(
            (
                direction[0] * translation_meters,
                direction[1] * translation_meters,
                direction[2] * translation_meters,
                0.0,
                0.0,
                0.0,
                0.0,
            )
        )

    def validate_samples(
        self,
        reference_reset: TrialResetState,
        samples: tuple[ControlResolutionSample, ...],
        *,
        require_complete: bool,
    ) -> None:
        expected_translations = (
            self.requested_translations
            if require_complete
            else self.requested_translations[: len(samples)]
        )
        if (
            tuple(sample.index for sample in samples)
            != tuple(range(len(samples)))
            or tuple(sample.requested_translation_meters for sample in samples)
            != expected_translations
        ):
            raise ValueError("control resolution sample roster is invalid")
        for sample in samples:
            validate_reset_equivalence(
                reference_reset,
                sample.start_reset,
                tolerances=self.reset_tolerances,
            )
            validate_reset_equivalence(
                reference_reset,
                sample.rollback_reset,
                tolerances=self.reset_tolerances,
            )

    def validate_sample_execution(
        self,
        reference_reset: TrialResetState,
        sample: ControlResolutionSample,
        *,
        expected_attachment: bool = True,
        rollback_drive_target: ControlResolutionDriveTarget | None = None,
    ) -> None:
        expected_joint_delta = maximum_joint_position_delta(
            sample.projection.proposed_joint_positions,
            sample.start_reset.joint_positions,
        )
        probe = self.probe_plan(sample.index)
        rollback_target_positions = (
            probe.rollback_joint_target(
                rollback_drive_target,
                reference_reset,
            )
            if rollback_drive_target is not None
            else reference_reset.joint_positions
        )
        rollback_requested_joint_motion = maximum_joint_position_delta(
            sample.endpoint.safety.joint_positions,
            rollback_target_positions,
        )
        rollback_final_tracking_error = maximum_joint_position_delta(
            sample.rollback_reset.joint_positions,
            rollback_target_positions,
        )
        self.settlement.validate_evidence(
            sample.tracked_settlement,
            expected_joint_delta,
            sample.settlement_joint_error_radians,
            rollback_requested_joint_motion,
            rollback_final_tracking_error,
        )
        rollback_interlock = (
            sample.tracked_settlement.rollback_interlock
            if sample.tracked_settlement is not None
            else None
        )
        expected_rollback_command = (
            probe.drive_command(
                self.safe_joint_motion_period(
                    sample.endpoint.safety.joint_positions,
                    rollback_target_positions,
                    self.motion_period_for(sample.requested_translation_meters),
                )
                if probe.applies_drive_command
                else None
            )
            if rollback_drive_target is not None
            else None
        )
        rollback_period_valid = True
        if rollback_drive_target is not None and isinstance(
            self.settlement, TrackedErrorSettlement
        ):
            if sample.tracked_settlement is None:
                rollback_period_valid = False
            else:
                rollback_period_valid = (
                    sample.tracked_settlement.rollback_drive_command
                    == expected_rollback_command
                )
        interlocks = (sample.interlock, rollback_interlock)
        if any(
            evidence is not None
            and (
                evidence.collision_detected
                or evidence.maximum_contact_force_newtons
                > self.safety_limits.maximum_contact_force_newtons
            )
            for evidence in interlocks
        ) or (
            sample.endpoint.safety.collision_detected
            or sample.endpoint.safety.contact_force_newtons
            > self.safety_limits.maximum_contact_force_newtons
            or sample.endpoint.safety.plug_attached is not expected_attachment
            or not rollback_period_valid
        ):
            raise ValueError("control resolution sample failed safety")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "translation_magnitudes_meters": list(
                self.translation_magnitudes_meters
            ),
            "repeats_per_magnitude": self.repeats_per_magnitude,
            "motion_period_seconds": self.motion_period_seconds,
            "safety_limits": self.safety_limits.to_dict(),
            "capture_tolerances": self.capture_tolerances.to_dict(),
            "reset_tolerances": self.reset_tolerances.to_dict(),
            "translation_policy": "retreat_from_recorded_target",
            "rotation_policy": "hold_current_orientation",
            "gripper_policy": "hold_current_width",
        }
        if self.motion_period_overrides:
            payload["motion_period_overrides"] = [
                override.to_dict() for override in self.motion_period_overrides
            ]
        if self.baseline_policy is not None:
            payload["baseline_policy"] = self.baseline_policy.to_dict()
        payload.update(self.settlement.protocol_fields())
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionProtocol:
        if not isinstance(payload, dict):
            raise ValueError("control resolution protocol must be an object")
        try:
            protocol = cls(
                translation_magnitudes_meters=tuple(
                    float(value)
                    for value in payload["translation_magnitudes_meters"]
                ),
                repeats_per_magnitude=int(payload["repeats_per_magnitude"]),
                motion_period_seconds=float(payload["motion_period_seconds"]),
                motion_period_overrides=tuple(
                    ControlResolutionMotionPeriod.from_dict(override)
                    for override in payload.get("motion_period_overrides", ())
                ),
                baseline_policy=(
                    ControlResolutionBaselinePolicy.from_dict(
                        payload["baseline_policy"]
                    )
                    if "baseline_policy" in payload
                    else None
                ),
                settlement=_settlement_from_protocol_dict(payload),
                safety_limits=SimulatorSafetyLimits.from_dict(
                    payload["safety_limits"]
                ),
                capture_tolerances=ResetEquivalenceTolerances.from_dict(
                    payload["capture_tolerances"]
                ),
                reset_tolerances=ResetEquivalenceTolerances.from_dict(
                    payload["reset_tolerances"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution protocol is incomplete") from error
        if (
            payload.get("translation_policy")
            != "retreat_from_recorded_target"
            or payload.get("rotation_policy") != "hold_current_orientation"
            or payload.get("gripper_policy") != "hold_current_width"
        ):
            raise ValueError("control resolution protocol policy is invalid")
        return protocol


CONTROL_RESOLUTION_PROTOCOL = ControlResolutionProtocol()


@dataclass(frozen=True)
class ControlResolutionSettlementEvidence:
    requested_joint_motion_radians: float
    required_tracking_error_radians: float
    updates_used: int
    passing_tracking_errors_radians: tuple[float, ...]

    def __post_init__(self) -> None:
        scalars = (
            self.requested_joint_motion_radians,
            self.required_tracking_error_radians,
            *self.passing_tracking_errors_radians,
        )
        if (
            not all(isfinite(value) and value >= 0.0 for value in scalars)
            or self.required_tracking_error_radians <= 0.0
            or not self.passing_tracking_errors_radians
            or isinstance(self.updates_used, bool)
            or not isinstance(self.updates_used, int)
            or self.updates_used <= 0
            or any(
                error > self.required_tracking_error_radians
                for error in self.passing_tracking_errors_radians
            )
        ):
            raise ValueError("control resolution settlement evidence is invalid")

    @property
    def final_tracking_error_radians(self) -> float:
        return self.passing_tracking_errors_radians[-1]

    def validate(
        self,
        policy: TrackedErrorSettlement,
        cap_radians: float | None = None,
    ) -> None:
        if (
            not isclose(
                self.required_tracking_error_radians,
                policy.maximum_tracking_error(
                    self.requested_joint_motion_radians,
                    cap_radians,
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or len(self.passing_tracking_errors_radians)
            != policy.required_consecutive_updates
            or not policy.required_consecutive_updates
            <= self.updates_used
            <= policy.maximum_updates
        ):
            raise ValueError(
                "control resolution settlement does not match its policy"
            )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "requested_joint_motion_radians": (
                self.requested_joint_motion_radians
            ),
            "required_tracking_error_radians": (
                self.required_tracking_error_radians
            ),
            "updates_used": self.updates_used,
            "passing_tracking_errors_radians": list(
                self.passing_tracking_errors_radians
            ),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionSettlementEvidence:
        if not isinstance(payload, dict):
            raise ValueError("control resolution settlement must be an object")
        updates_used = payload.get("updates_used")
        if isinstance(updates_used, bool) or not isinstance(updates_used, int):
            raise ValueError("control resolution settlement updates must be an integer")
        try:
            return cls(
                requested_joint_motion_radians=float(
                    payload["requested_joint_motion_radians"]
                ),
                required_tracking_error_radians=float(
                    payload["required_tracking_error_radians"]
                ),
                updates_used=updates_used,
                passing_tracking_errors_radians=tuple(
                    float(value)
                    for value in payload["passing_tracking_errors_radians"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution settlement is incomplete"
            ) from error


@dataclass(frozen=True)
class ControlResolutionSettlementAttempt:
    """Complete tracking-error trace for one exhausted settlement window."""

    requested_joint_motion_radians: float
    required_tracking_error_radians: float
    tracking_errors_radians: tuple[float, ...]
    final_joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        values = (
            self.requested_joint_motion_radians,
            self.required_tracking_error_radians,
            *self.tracking_errors_radians,
        )
        if (
            not all(isfinite(value) and value >= 0.0 for value in values)
            or not all(isfinite(value) for value in self.final_joint_positions)
            or self.required_tracking_error_radians <= 0.0
            or not self.tracking_errors_radians
            or len(self.final_joint_positions) != 7
        ):
            raise ValueError("control resolution settlement attempt is invalid")

    def validate(
        self,
        policy: TrackedErrorSettlement,
        cap_radians: float | None = None,
    ) -> None:
        passing = tuple(
            error <= self.required_tracking_error_radians
            for error in self.tracking_errors_radians
        )
        if (
            len(self.tracking_errors_radians) != policy.maximum_updates
            or not isclose(
                self.required_tracking_error_radians,
                policy.maximum_tracking_error(
                    self.requested_joint_motion_radians,
                    cap_radians,
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or any(
                all(
                    passing[
                        end - policy.required_consecutive_updates + 1 : end + 1
                    ]
                )
                for end in range(
                    policy.required_consecutive_updates - 1,
                    len(passing),
                )
            )
        ):
            raise ValueError(
                "control resolution settlement attempt does not match its policy"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_joint_motion_radians": (
                self.requested_joint_motion_radians
            ),
            "required_tracking_error_radians": (
                self.required_tracking_error_radians
            ),
            "tracking_errors_radians": list(self.tracking_errors_radians),
            "final_joint_positions": list(self.final_joint_positions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionSettlementAttempt:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution settlement attempt must be an object")
        try:
            return cls(
                float(payload["requested_joint_motion_radians"]),
                float(payload["required_tracking_error_radians"]),
                tuple(float(value) for value in payload["tracking_errors_radians"]),
                tuple(float(value) for value in payload["final_joint_positions"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution settlement attempt is incomplete"
            ) from error


class ControlResolutionSettlementPhase(str, Enum):
    MOTION = "motion"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class ControlResolutionSettlementTimeoutTrace:
    """Common exact execution and tracking trace for a settlement timeout."""

    execution: ControlResolutionProbeExecution
    start_joint_positions: tuple[float, ...]
    target_joint_positions: tuple[float, ...]
    attempt: ControlResolutionSettlementAttempt
    interlock: ControlInterlockEvidence
    drive_command: ControlResolutionDriveCommand
    timing: ControlResolutionMotionTiming

    @property
    def probe(self) -> ControlResolutionProbePlan:
        return self.execution.probe

    def __post_init__(self) -> None:
        if (
            len(self.start_joint_positions) != 7
            or len(self.target_joint_positions) != 7
        ):
            raise ValueError("control resolution settlement failure is invalid")
        requested_motion = maximum_joint_position_delta(
            self.start_joint_positions,
            self.target_joint_positions,
        )
        if (
            not all(
                isfinite(value)
                for value in (
                    *self.start_joint_positions,
                    *self.target_joint_positions,
                )
            )
            or not isclose(
                self.attempt.requested_joint_motion_radians,
                requested_motion,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isclose(
                self.attempt.tracking_errors_radians[-1],
                maximum_joint_position_delta(
                    self.attempt.final_joint_positions,
                    self.target_joint_positions,
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("control resolution settlement failure is invalid")

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        completed_sample_count: int,
        drive_target: ControlResolutionDriveTarget,
        observation_id: int,
        reference_reset: TrialResetState,
        tracking_error_cap_radians: float | None = None,
    ) -> None:
        if not isinstance(protocol.settlement, TrackedErrorSettlement):
            raise ValueError("fixed settlement cannot carry a timeout attempt")
        self.attempt.validate(protocol.settlement, tracking_error_cap_radians)
        self.execution.validate(protocol, observation_id)
        validate_reset_equivalence(
            reference_reset,
            self.execution.start_reset,
            tolerances=protocol.reset_tolerances,
        )
        expected_probe = (
            protocol.probe_plans[completed_sample_count]
            if completed_sample_count < len(protocol.requested_translations)
            else None
        )
        if (
            self.probe != expected_probe
            or self.interlock.collision_detected
            or self.interlock.maximum_contact_force_newtons
            > protocol.safety_limits.maximum_contact_force_newtons
        ):
            raise ValueError(
                "control resolution settlement failure is not bound to its protocol step"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution": self.execution.to_dict(),
            "start_joint_positions": list(self.start_joint_positions),
            "target_joint_positions": list(self.target_joint_positions),
            "attempt": self.attempt.to_dict(),
            "interlock": self.interlock.to_dict(),
            "drive_command": self.drive_command.to_dict(),
            "timing": self.timing.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionSettlementTimeoutTrace:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution timeout trace must be an object")
        try:
            timing = ControlResolutionMotionTiming.from_dict(payload["timing"])
            return cls(
                ControlResolutionProbeExecution.from_dict(payload["execution"]),
                tuple(float(value) for value in payload["start_joint_positions"]),
                tuple(float(value) for value in payload["target_joint_positions"]),
                ControlResolutionSettlementAttempt.from_dict(payload["attempt"]),
                ControlInterlockEvidence.from_dict(payload["interlock"]),
                _drive_command_from_dict(payload["drive_command"]),
                timing,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution timeout trace is incomplete") from error


@dataclass(frozen=True)
class ControlResolutionMotionTimeout:
    trace: ControlResolutionSettlementTimeoutTrace
    rollback_outcome: ControlResolutionRollbackOutcome

    @property
    def probe(self) -> ControlResolutionProbePlan:
        return self.trace.probe

    @property
    def execution(self) -> ControlResolutionProbeExecution:
        return self.trace.execution

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        completed_sample_count: int,
        drive_target: ControlResolutionDriveTarget,
        observation_id: int,
        reference_reset: TrialResetState,
        expected_attachment: bool,
    ) -> None:
        self.trace.validate(
            protocol,
            completed_sample_count,
            drive_target,
            observation_id,
            reference_reset,
        )
        if (
            self.trace.start_joint_positions
            != self.execution.start_reset.joint_positions
            or self.trace.target_joint_positions
            != self.execution.projection.proposed_joint_positions
            or self.rollback_outcome.start_joint_positions
            != self.trace.attempt.final_joint_positions
            or self.trace.drive_command
            != self.probe.drive_command(
                protocol.motion_period_for(
                    self.probe.requested_translation_meters
                )
                if self.probe.applies_drive_command
                else None
            )
            or self.trace.timing.duration_seconds
            < protocol.motion_period_for(
                self.probe.requested_translation_meters
            )
        ):
            raise ValueError("motion timeout evidence is inconsistent")
        self.rollback_outcome.validate(
            protocol,
            self.probe,
            drive_target,
            reference_reset,
            expected_attachment,
            protocol.motion_period_for(
                self.probe.requested_translation_meters
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": ControlResolutionSettlementPhase.MOTION.value,
            **self.trace.to_dict(),
            "rollback_outcome": self.rollback_outcome.to_dict(),
        }


@dataclass(frozen=True)
class ControlResolutionRollbackTimeout:
    trace: ControlResolutionSettlementTimeoutTrace
    forward: ControlResolutionForwardEvidence

    @property
    def probe(self) -> ControlResolutionProbePlan:
        return self.trace.probe

    @property
    def execution(self) -> ControlResolutionProbeExecution:
        return self.trace.execution

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        completed_sample_count: int,
        drive_target: ControlResolutionDriveTarget,
        observation_id: int,
        reference_reset: TrialResetState,
        expected_attachment: bool,
    ) -> None:
        self.trace.validate(
            protocol,
            completed_sample_count,
            drive_target,
            observation_id,
            reference_reset,
            (
                protocol.settlement.rollback_tracking_error_cap_radians
                if isinstance(protocol.settlement, TrackedErrorSettlement)
                else None
            ),
        )
        self.forward.validate(protocol, self.execution, expected_attachment)
        expected_command = self._expected_drive_command(protocol, drive_target)
        expected_target = self.probe.rollback_joint_target(
            drive_target,
            reference_reset,
        )
        if (
            self.trace.start_joint_positions
            != self.forward.endpoint.safety.joint_positions
            or self.trace.target_joint_positions != expected_target
            or self.trace.drive_command != expected_command
        ):
            raise ValueError("rollback timeout evidence is inconsistent")

    def _expected_drive_command(
        self,
        protocol: ControlResolutionProtocol,
        drive_target: ControlResolutionDriveTarget,
    ) -> ControlResolutionDriveCommand:
        if not self.probe.applies_drive_command:
            return self.probe.drive_command(None)
        return self.probe.drive_command(
            protocol.safe_joint_motion_period(
                self.trace.start_joint_positions,
                drive_target.joint_positions,
                protocol.motion_period_for(
                    self.probe.requested_translation_meters
                ),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": ControlResolutionSettlementPhase.ROLLBACK.value,
            **self.trace.to_dict(),
            "forward": self.forward.to_dict(),
        }


ControlResolutionSettlementFailure = Union[
    ControlResolutionMotionTimeout,
    ControlResolutionRollbackTimeout,
]


def _settlement_failure_from_dict(payload: Any) -> ControlResolutionSettlementFailure:
    if not isinstance(payload, Mapping):
        raise ValueError("control resolution settlement failure must be an object")
    try:
        phase = ControlResolutionSettlementPhase(payload["phase"])
        trace = ControlResolutionSettlementTimeoutTrace.from_dict(payload)
        if phase is ControlResolutionSettlementPhase.MOTION:
            if "forward" in payload:
                raise ValueError("motion timeout cannot carry forward evidence")
            return ControlResolutionMotionTimeout(
                trace,
                _rollback_outcome_from_dict(payload["rollback_outcome"]),
            )
        if "rollback_outcome" in payload:
            raise ValueError("rollback timeout cannot carry recovery outcome")
        return ControlResolutionRollbackTimeout(
            trace,
            ControlResolutionForwardEvidence.from_dict(payload["forward"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "control resolution settlement failure is incomplete"
        ) from error


@dataclass(frozen=True)
class TrackedSettlementEvidence:
    motion: ControlResolutionSettlementEvidence
    rollback: ControlResolutionSettlementEvidence
    rollback_interlock: ControlInterlockEvidence
    rollback_drive_command: ControlResolutionDriveCommand = DriveCommandSkipped()

    def validate(
        self,
        policy: TrackedErrorSettlement,
        motion_requested_joint_motion_radians: float,
        motion_final_tracking_error_radians: float,
        rollback_requested_joint_motion_radians: float,
        rollback_final_tracking_error_radians: float,
    ) -> None:
        comparisons = (
            (
                self.motion,
                motion_requested_joint_motion_radians,
                motion_final_tracking_error_radians,
                None,
            ),
            (
                self.rollback,
                rollback_requested_joint_motion_radians,
                rollback_final_tracking_error_radians,
                policy.rollback_tracking_error_cap_radians,
            ),
        )
        for evidence, requested_motion, final_error, cap_radians in comparisons:
            evidence.validate(policy, cap_radians)
            if (
                not isclose(
                    evidence.requested_joint_motion_radians,
                    requested_motion,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not isclose(
                    evidence.final_tracking_error_radians,
                    final_error,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "control resolution settlement claims are inconsistent"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "motion": self.motion.to_dict(),
            "rollback": self.rollback.to_dict(),
            "rollback_interlock": self.rollback_interlock.to_dict(),
            "rollback_drive_command": self.rollback_drive_command.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        require_typed_drive_command: bool = False,
    ) -> TrackedSettlementEvidence:
        if not isinstance(payload, dict):
            raise ValueError("tracked settlement evidence must be an object")
        has_typed_command = "rollback_drive_command" in payload
        has_legacy_period = "rollback_command_period_seconds" in payload
        if (
            has_typed_command and has_legacy_period
        ) or (require_typed_drive_command and not has_typed_command):
            raise ValueError(
                "tracked settlement drive command generation is invalid"
            )
        try:
            return cls(
                motion=ControlResolutionSettlementEvidence.from_dict(
                    payload["motion"]
                ),
                rollback=ControlResolutionSettlementEvidence.from_dict(
                    payload["rollback"]
                ),
                rollback_interlock=ControlInterlockEvidence.from_dict(
                    payload["rollback_interlock"]
                ),
                rollback_drive_command=(
                    _drive_command_from_dict(payload["rollback_drive_command"])
                    if has_typed_command
                    else (
                        DriveCommandApplied(
                            float(payload["rollback_command_period_seconds"])
                        )
                        if has_legacy_period
                        else DriveCommandSkipped()
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("tracked settlement evidence is incomplete") from error


class ControlResolutionResetPhase(str, Enum):
    CAPTURE_TO_BASELINE = "capture_to_baseline"
    SAMPLE_START = "sample_start"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class ControlResolutionCaptureIdentity:
    source: ControlResolutionCaptureSourceIdentity
    observation_id: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, ControlResolutionCaptureSourceIdentity)
            or isinstance(self.observation_id, bool)
            or not isinstance(self.observation_id, int)
            or self.observation_id <= 0
        ):
            raise ValueError("control resolution capture identity is invalid")

    @property
    def reference_recording(self) -> str:
        return self.source.reference_recording

    @property
    def seed(self) -> int:
        return self.source.seed

    @property
    def context_index(self) -> int:
        return self.source.context_index

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.source.to_dict(),
            "observation_id": self.observation_id,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionCaptureIdentity:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution capture identity must be an object")
        observation_id = payload.get("observation_id")
        if isinstance(observation_id, bool) or not isinstance(observation_id, int):
            raise ValueError("control resolution capture identity counts are invalid")
        try:
            return cls(
                ControlResolutionCaptureSourceIdentity.from_dict(payload),
                observation_id,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution capture identity is incomplete"
            ) from error


@dataclass(frozen=True)
class RejectedControlResolutionReset:
    phase: ControlResolutionResetPhase
    sample_index: int | None
    reference: TrialResetState
    candidate: TrialResetState
    tolerances: ResetEquivalenceTolerances

    def __post_init__(self) -> None:
        requires_index = self.phase in (
            ControlResolutionResetPhase.SAMPLE_START,
            ControlResolutionResetPhase.ROLLBACK,
        )
        if (
            (requires_index and (self.sample_index is None or self.sample_index < 0))
            or (not requires_index and self.sample_index is not None)
            or self.measurement.passes(self.tolerances)
        ):
            raise ValueError("rejected control resolution reset is invalid")

    @property
    def measurement(self) -> ResetEquivalenceMeasurement:
        return ResetEquivalenceMeasurement.between(self.reference, self.candidate)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "sample_index": self.sample_index,
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "tolerances": self.tolerances.to_dict(),
            "measurement": self.measurement.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> RejectedControlResolutionReset:
        if not isinstance(payload, dict):
            raise ValueError("rejected control resolution reset must be an object")
        sample_index = payload.get("sample_index")
        if sample_index is not None and (
            isinstance(sample_index, bool) or not isinstance(sample_index, int)
        ):
            raise ValueError("rejected reset sample index must be an integer")
        try:
            rejected = cls(
                phase=ControlResolutionResetPhase(payload["phase"]),
                sample_index=sample_index,
                reference=TrialResetState.from_dict(payload["reference"]),
                candidate=TrialResetState.from_dict(payload["candidate"]),
                tolerances=ResetEquivalenceTolerances.from_dict(
                    payload["tolerances"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "rejected control resolution reset is incomplete"
            ) from error
        if payload.get("measurement") != rejected.measurement.to_dict():
            raise ValueError("rejected reset measurement is inconsistent")
        return rejected


@dataclass(frozen=True)
class ControlResolutionEndpoint:
    pose: DroidPose
    safety: ControlSafetySnapshot

    def to_dict(self) -> dict[str, Any]:
        return {"pose": list(self.pose.values), "safety": self.safety.to_dict()}

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionEndpoint:
        if not isinstance(payload, dict):
            raise ValueError("control resolution endpoint must be an object")
        try:
            return cls(
                DroidPose(tuple(payload["pose"])),
                ControlSafetySnapshot.from_dict(payload["safety"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution endpoint is incomplete") from error


@dataclass(frozen=True)
class ControlResolutionMotionTiming:
    started_at_sim_seconds: float
    settled_at_sim_seconds: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.started_at_sim_seconds)
            or not isfinite(self.settled_at_sim_seconds)
            or self.settled_at_sim_seconds <= self.started_at_sim_seconds
        ):
            raise ValueError("control resolution motion timing is invalid")

    @property
    def duration_seconds(self) -> float:
        return self.settled_at_sim_seconds - self.started_at_sim_seconds

    def to_dict(self) -> dict[str, float]:
        return {
            "motion_started_at_sim_seconds": self.started_at_sim_seconds,
            "motion_settled_at_sim_seconds": self.settled_at_sim_seconds,
            "settling_time_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Any,
    ) -> ControlResolutionMotionTiming:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution motion timing must be an object")
        try:
            timing = cls(
                float(payload["motion_started_at_sim_seconds"]),
                float(payload["motion_settled_at_sim_seconds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution motion timing is incomplete") from error
        if payload.get("settling_time_seconds") != timing.duration_seconds:
            raise ValueError("control resolution motion timing is inconsistent")
        return timing

    @classmethod
    def optional_from_sample_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> ControlResolutionMotionTiming | None:
        fields = (
            "motion_started_at_sim_seconds",
            "motion_settled_at_sim_seconds",
            "settling_time_seconds",
        )
        present = tuple(field in payload for field in fields)
        if not any(present):
            return None
        if not all(present):
            raise ValueError("control resolution motion timing is incomplete")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class ControlResolutionForwardEvidence:
    endpoint: ControlResolutionEndpoint
    settlement: ControlResolutionSettlementEvidence
    interlock: ControlInterlockEvidence
    timing: ControlResolutionMotionTiming

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        execution: ControlResolutionProbeExecution,
        expected_attachment: bool,
    ) -> None:
        if not isinstance(protocol.settlement, TrackedErrorSettlement):
            raise ValueError("fixed settlement cannot carry forward evidence")
        requested_motion = maximum_joint_position_delta(
            execution.projection.proposed_joint_positions,
            execution.start_reset.joint_positions,
        )
        final_error = maximum_joint_position_delta(
            self.endpoint.safety.joint_positions,
            execution.projection.proposed_joint_positions,
        )
        self.settlement.validate(protocol.settlement)
        if (
            not isclose(
                self.settlement.requested_joint_motion_radians,
                requested_motion,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isclose(
                self.settlement.final_tracking_error_radians,
                final_error,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or self.timing.duration_seconds
            < protocol.motion_period_for(
                execution.probe.requested_translation_meters
            )
            or self.interlock.collision_detected
            or self.interlock.maximum_contact_force_newtons
            > protocol.safety_limits.maximum_contact_force_newtons
            or self.endpoint.safety.collision_detected
            or self.endpoint.safety.contact_force_newtons
            > protocol.safety_limits.maximum_contact_force_newtons
            or self.endpoint.safety.plug_attached is not expected_attachment
        ):
            raise ValueError("control resolution forward evidence is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint.to_dict(),
            "settlement": self.settlement.to_dict(),
            "interlock": self.interlock.to_dict(),
            "timing": self.timing.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionForwardEvidence:
        if not isinstance(payload, Mapping):
            raise ValueError("control resolution forward evidence must be an object")
        try:
            timing = ControlResolutionMotionTiming.from_dict(payload["timing"])
            return cls(
                ControlResolutionEndpoint.from_dict(payload["endpoint"]),
                ControlResolutionSettlementEvidence.from_dict(
                    payload["settlement"]
                ),
                ControlInterlockEvidence.from_dict(payload["interlock"]),
                timing,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution forward evidence is incomplete"
            ) from error


@dataclass(frozen=True)
class ControlResolutionRollbackSuccess:
    start_joint_positions: tuple[float, ...]
    drive_command: ControlResolutionDriveCommand
    settlement: ControlResolutionSettlementEvidence
    interlock: ControlInterlockEvidence
    reset: TrialResetState

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        probe: ControlResolutionProbePlan,
        drive_target: ControlResolutionDriveTarget,
        reference_reset: TrialResetState,
        expected_attachment: bool,
        minimum_period_seconds: float,
    ) -> None:
        if not isinstance(protocol.settlement, TrackedErrorSettlement):
            raise ValueError("fixed settlement cannot carry rollback evidence")
        rollback_target = probe.rollback_joint_target(
            drive_target,
            reference_reset,
        )
        requested_motion = maximum_joint_position_delta(
            self.start_joint_positions,
            rollback_target,
        )
        final_error = maximum_joint_position_delta(
            self.reset.joint_positions,
            rollback_target,
        )
        self.settlement.validate(
            protocol.settlement,
            protocol.settlement.rollback_tracking_error_cap_radians,
        )
        expected_command = probe.drive_command(
            protocol.safe_joint_motion_period(
                self.start_joint_positions,
                drive_target.joint_positions,
                minimum_period_seconds,
            )
            if probe.applies_drive_command
            else None
        )
        validate_reset_equivalence(
            reference_reset,
            self.reset,
            tolerances=protocol.reset_tolerances,
        )
        if (
            len(self.start_joint_positions) != 7
            or self.drive_command != expected_command
            or not isclose(
                self.settlement.requested_joint_motion_radians,
                requested_motion,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isclose(
                self.settlement.final_tracking_error_radians,
                final_error,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or self.interlock.collision_detected
            or self.interlock.maximum_contact_force_newtons
            > protocol.safety_limits.maximum_contact_force_newtons
            or self.reset.plug_attached is not expected_attachment
        ):
            raise ValueError("control resolution rollback recovery is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "settled",
            "start_joint_positions": list(self.start_joint_positions),
            "drive_command": self.drive_command.to_dict(),
            "settlement": self.settlement.to_dict(),
            "interlock": self.interlock.to_dict(),
            "reset": self.reset.to_dict(),
        }


@dataclass(frozen=True)
class ControlResolutionRollbackFailure:
    start_joint_positions: tuple[float, ...]
    drive_command: ControlResolutionDriveCommand
    attempt: ControlResolutionSettlementAttempt | None
    interlock: ControlInterlockEvidence
    error: str

    def __post_init__(self) -> None:
        if (
            len(self.start_joint_positions) != 7
            or not all(isfinite(value) for value in self.start_joint_positions)
            or not self.error
        ):
            raise ValueError("control resolution rollback failure is invalid")

    def validate(
        self,
        protocol: ControlResolutionProtocol,
        probe: ControlResolutionProbePlan,
        drive_target: ControlResolutionDriveTarget,
        reference_reset: TrialResetState,
        expected_attachment: bool,
        minimum_period_seconds: float,
    ) -> None:
        del expected_attachment
        rollback_target = probe.rollback_joint_target(
            drive_target,
            reference_reset,
        )
        expected_command = probe.drive_command(
            protocol.safe_joint_motion_period(
                self.start_joint_positions,
                drive_target.joint_positions,
                minimum_period_seconds,
            )
            if probe.applies_drive_command
            else None
        )
        if self.drive_command != expected_command:
            raise ValueError("rollback recovery period is inconsistent")
        if self.attempt is not None:
            if not isinstance(protocol.settlement, TrackedErrorSettlement):
                raise ValueError("fixed settlement cannot carry rollback attempt")
            self.attempt.validate(protocol.settlement)
            requested_motion = maximum_joint_position_delta(
                self.start_joint_positions,
                rollback_target,
            )
            if not isclose(
                self.attempt.requested_joint_motion_radians,
                requested_motion,
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not isclose(
                self.attempt.tracking_errors_radians[-1],
                maximum_joint_position_delta(
                    self.attempt.final_joint_positions,
                    rollback_target,
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("rollback attempt target is inconsistent")
        if (
            self.interlock.collision_detected
            or self.interlock.maximum_contact_force_newtons
            > protocol.safety_limits.maximum_contact_force_newtons
        ) and self.attempt is not None:
            raise ValueError("unsafe rollback cannot claim a settlement timeout")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "start_joint_positions": list(self.start_joint_positions),
            "drive_command": self.drive_command.to_dict(),
            "attempt": self.attempt.to_dict() if self.attempt is not None else None,
            "interlock": self.interlock.to_dict(),
            "error": self.error,
        }


ControlResolutionRollbackOutcome = Union[
    ControlResolutionRollbackSuccess,
    ControlResolutionRollbackFailure,
]


def _rollback_outcome_from_dict(payload: Any) -> ControlResolutionRollbackOutcome:
    if not isinstance(payload, Mapping):
        raise ValueError("control resolution rollback outcome must be an object")
    try:
        status = payload["status"]
        start = tuple(float(value) for value in payload["start_joint_positions"])
        interlock = ControlInterlockEvidence.from_dict(payload["interlock"])
        drive_command = _drive_command_from_dict(payload["drive_command"])
        if status == "settled":
            if "attempt" in payload or "error" in payload:
                raise ValueError("settled rollback carries failure fields")
            return ControlResolutionRollbackSuccess(
                start,
                drive_command,
                ControlResolutionSettlementEvidence.from_dict(
                    payload["settlement"]
                ),
                interlock,
                TrialResetState.from_dict(payload["reset"]),
            )
        if status == "failed":
            if "settlement" in payload or "reset" in payload:
                raise ValueError("failed rollback carries settled fields")
            return ControlResolutionRollbackFailure(
                start,
                drive_command,
                (
                    ControlResolutionSettlementAttempt.from_dict(
                        payload["attempt"]
                    )
                    if payload.get("attempt") is not None
                    else None
                ),
                interlock,
                str(payload["error"]),
            )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("control resolution rollback outcome is incomplete") from error
    raise ValueError("control resolution rollback outcome status is invalid")


@dataclass(frozen=True)
class ControlResolutionSample:
    index: int
    requested_translation_meters: float
    start_reset: TrialResetState
    commanded_action: DroidAction
    target_pose: DroidPose
    projection: SafetyProjectionAttempt
    endpoint: ControlResolutionEndpoint
    interlock: ControlInterlockEvidence
    rollback_reset: TrialResetState
    tracked_settlement: TrackedSettlementEvidence | None = None
    motion_timing: ControlResolutionMotionTiming | None = None

    def __post_init__(self) -> None:
        scalars = (
            self.requested_translation_meters,
        )
        if (
            self.index < 0
            or not all(isfinite(value) and value >= 0.0 for value in scalars)
            or not self.projection.gate.passed
            or self.projection.scale != DroidActionScale.uniform(1.0)
        ):
            raise ValueError("control resolution sample is invalid")

    @property
    def actual_action(self) -> DroidAction:
        return action_between(self.start_reset.pose, self.endpoint.pose)

    @property
    def settlement_joint_error_radians(self) -> float:
        return maximum_joint_position_delta(
            self.endpoint.safety.joint_positions,
            self.projection.proposed_joint_positions,
        )

    def controller_tracking_error_radians(
        self,
        probe: ControlResolutionProbePlan,
        drive_target: ControlResolutionDriveTarget,
    ) -> float:
        return maximum_joint_position_delta(
            self.endpoint.safety.joint_positions,
            probe.controller_tracking_joint_target(
                drive_target,
                self.projection.proposed_joint_positions,
            ),
        )

    @property
    def actual_translation_meters(self) -> float:
        return float(np.linalg.norm(self.actual_action.values[:3]))

    @property
    def actual_orientation_drift_radians(self) -> float:
        return float(
            Rotation.from_euler("xyz", self.actual_action.values[3:6]).magnitude()
        )

    @property
    def settling_time_seconds(self) -> float | None:
        return self.motion_timing.duration_seconds if self.motion_timing else None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "index": self.index,
            "requested_translation_meters": self.requested_translation_meters,
            "start_reset": self.start_reset.to_dict(),
            "commanded_action": list(self.commanded_action.values),
            "target_pose": list(self.target_pose.values),
            "projection": self.projection.to_dict(),
            "endpoint": self.endpoint.to_dict(),
            "actual_action": list(self.actual_action.values),
            "settled_joint_tracking_error_radians": (
                self.settlement_joint_error_radians
            ),
            "interlock": self.interlock.to_dict(),
            "rollback_reset": self.rollback_reset.to_dict(),
        }
        if self.tracked_settlement is not None:
            payload["tracked_settlement"] = self.tracked_settlement.to_dict()
        if self.motion_timing is not None:
            payload.update(self.motion_timing.to_dict())
        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Any,
        *,
        require_typed_drive_command: bool = False,
    ) -> ControlResolutionSample:
        if not isinstance(payload, dict):
            raise ValueError("control resolution sample must be an object")
        try:
            sample = cls(
                index=int(payload["index"]),
                requested_translation_meters=float(
                    payload["requested_translation_meters"]
                ),
                start_reset=TrialResetState.from_dict(payload["start_reset"]),
                commanded_action=DroidAction(tuple(payload["commanded_action"])),
                target_pose=DroidPose(tuple(payload["target_pose"])),
                projection=SafetyProjectionAttempt.from_dict(payload["projection"]),
                endpoint=ControlResolutionEndpoint.from_dict(payload["endpoint"]),
                interlock=ControlInterlockEvidence.from_dict(payload["interlock"]),
                rollback_reset=TrialResetState.from_dict(payload["rollback_reset"]),
                tracked_settlement=(
                    TrackedSettlementEvidence.from_dict(
                        payload["tracked_settlement"],
                        require_typed_drive_command=require_typed_drive_command,
                    )
                    if payload.get("tracked_settlement") is not None
                    else None
                ),
                motion_timing=ControlResolutionMotionTiming.optional_from_sample_dict(
                    payload
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution sample is incomplete") from error
        if (
            not _reconstructed_metric_payload_matches(
                list(sample.actual_action.values),
                payload.get("actual_action"),
            )
            or not _reconstructed_metric_payload_matches(
                sample.settlement_joint_error_radians,
                payload.get("settled_joint_tracking_error_radians"),
            )
            or not _reconstructed_metric_payload_matches(
                sample.settling_time_seconds,
                payload.get("settling_time_seconds"),
            )
        ):
            raise ValueError("control resolution sample claims are inconsistent")
        return sample


def _control_resolution_capture(
    request: Any,
    state: Any,
    load: ControlResolutionLoad,
) -> tuple[ControlObservation, Any, TrialResetState]:
    from sim.control_session import ControlSession, ControlSessionState

    observation = ControlObservation.from_dict(request)
    session_state = ControlSessionState.from_dict(state)
    captured_reset = ControlSession.trial_context(
        observation,
        session_state,
    ).reset
    if load is ControlResolutionLoad.UNLOADED:
        captured_reset = TrialResetState(
            captured_reset.pose,
            captured_reset.joint_positions,
            captured_reset.collision_detected,
            captured_reset.contact_force_newtons,
            captured_reset.plug_position,
            False,
        )
    return observation, session_state, captured_reset


@dataclass(frozen=True)
class ControlResolutionFailureEvidence:
    session_id: str
    failed_at_unix_seconds: float
    reference_reset: TrialResetState | None
    completed_samples: tuple[ControlResolutionSample, ...]
    error: str
    rejected_reset: RejectedControlResolutionReset | None = None
    protocol: ControlResolutionProtocol = CONTROL_RESOLUTION_PROTOCOL
    load: ControlResolutionLoad = ControlResolutionLoad.ATTACHED
    baseline: ControlResolutionBaselineEvidence | None = None
    baseline_attempt: ControlResolutionBaselineAttempt | None = None
    capture_identity: ControlResolutionCaptureIdentity | None = None
    settlement_failure: ControlResolutionSettlementFailure | None = None
    projection_failure: ControlResolutionProjectionFailure | None = None

    def __post_init__(self) -> None:
        validate_recording_id(self.session_id)
        if (
            self.protocol.baseline_policy is None
            and self.capture_identity is not None
        ):
            raise ValueError(
                "legacy control resolution failure cannot carry capture identity"
            )
        if self.completed_samples and self.reference_reset is None:
            raise ValueError(
                "completed control resolution samples require a reference reset"
            )
        if (
            not isfinite(self.failed_at_unix_seconds)
            or self.failed_at_unix_seconds <= 0.0
            or not self.error
        ):
            raise ValueError("control resolution failure evidence is invalid")
        if self.baseline is not None and self.baseline_attempt is not None:
            raise ValueError("control resolution failure has two baseline variants")
        if self.baseline is not None:
            if self.protocol.baseline_policy is None:
                raise ValueError("legacy protocol cannot carry baseline evidence")
            self.baseline.validate(
                self.protocol.baseline_policy,
                self.load,
                self.protocol.safety_limits,
            )
            if self.reference_reset != self.baseline.reference_reset:
                raise ValueError("control resolution baseline reset is inconsistent")
        elif self.protocol.baseline_policy is not None and (
            self.reference_reset is not None or self.completed_samples
        ):
            raise ValueError(
                "current control resolution failure lost its baseline evidence"
            )
        if self.baseline_attempt is not None:
            if self.protocol.baseline_policy is None:
                raise ValueError("legacy protocol cannot carry a baseline attempt")
            self.baseline_attempt.validate(
                self.protocol.baseline_policy,
                self.load,
                self.protocol.safety_limits,
            )
            if (
                self.reference_reset is not None
                or self.completed_samples
                or self.rejected_reset is not None
            ):
                raise ValueError(
                    "failed baseline attempt cannot carry probe evidence"
                )
        if self.settlement_failure is not None:
            if (
                self.baseline is None
                or self.baseline.drive_target is None
                or self.reference_reset is None
                or self.capture_identity is None
                or self.baseline_attempt is not None
                or self.rejected_reset is not None
            ):
                raise ValueError(
                    "settlement failure requires one drive-bound stable baseline"
                )
            self.settlement_failure.validate(
                self.protocol,
                len(self.completed_samples),
                self.baseline.drive_target,
                self.capture_identity.observation_id,
                self.reference_reset,
                self.load.plug_attached,
            )
        if self.projection_failure is not None:
            if (
                self.baseline is None
                or self.reference_reset is None
                or self.capture_identity is None
                or self.baseline_attempt is not None
                or self.rejected_reset is not None
                or self.settlement_failure is not None
                or len(self.completed_samples)
                != self.projection_failure.probe.sample_index
            ):
                raise ValueError(
                    "projection failure requires the next stable-baseline probe"
                )
            validate_reset_equivalence(
                self.reference_reset,
                self.projection_failure.start_reset,
                tolerances=self.protocol.reset_tolerances,
            )
            self.projection_failure.validate(
                self.protocol,
                self.capture_identity.observation_id,
            )
        if self.reference_reset is None:
            if self.rejected_reset is not None:
                raise ValueError(
                    "rejected resolution reset requires an acquired reference reset"
                )
            return
        self.protocol.validate_samples(
            self.reference_reset,
            self.completed_samples,
            require_complete=False,
        )
        for sample in self.completed_samples:
            self.protocol.validate_sample_execution(
                self.reference_reset,
                sample,
                expected_attachment=self.load.plug_attached,
                rollback_drive_target=(
                    self.baseline.drive_target
                    if self.baseline is not None
                    else None
                ),
            )
        if self.rejected_reset is None:
            return
        capture_failure = (
            self.rejected_reset.phase
            is ControlResolutionResetPhase.CAPTURE_TO_BASELINE
        )
        expected_tolerances = (
            self.protocol.capture_tolerances
            if capture_failure
            else self.protocol.reset_tolerances
        )
        if (
            self.rejected_reset.tolerances != expected_tolerances
            or (
                capture_failure
                and (
                    self.completed_samples
                    or self.baseline is None
                    or self.rejected_reset.candidate
                    != self.baseline.initial_reset
                )
            )
            or (
                not capture_failure
                and (
                    self.rejected_reset.reference != self.reference_reset
                    or self.rejected_reset.sample_index
                    != len(self.completed_samples)
                )
            )
        ):
            raise ValueError("rejected reset is not bound to the failed protocol step")

    @property
    def diagnostic_only(self) -> bool:
        return True

    @property
    def multi_step_authority_granted(self) -> bool:
        return False

    @property
    def production_authority_granted(self) -> bool:
        return False

    def validate_capture(self, request: Any, state: Any) -> None:
        """Bind authenticated failure evidence to its raw captured session."""

        if self.capture_identity is None:
            raise ValueError(
                "legacy control resolution failure has no capture identity"
            )
        observation, session_state, captured_reset = _control_resolution_capture(
            request,
            state,
            self.load,
        )
        if (
            session_state.session_id != self.session_id
            or session_state.reference_recording
            != self.capture_identity.reference_recording
            or session_state.seed != self.capture_identity.seed
            or observation.warmup_frames != self.capture_identity.context_index
            or observation.observation_id != self.capture_identity.observation_id
            or session_state.execution_policy
            is not ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT
        ):
            raise ValueError(
                "control resolution failure is not bound to its capture"
            )
        if self.settlement_failure is not None:
            if (
                observation.target_pose
                != self.settlement_failure.execution.recorded_target_pose
            ):
                raise ValueError(
                    "control resolution failed probe target is not capture-bound"
                )
            validate_reset_equivalence(
                self.reference_reset,
                self.settlement_failure.execution.start_reset,
                tolerances=self.protocol.reset_tolerances,
            )
        if self.projection_failure is not None:
            if (
                observation.target_pose
                != self.projection_failure.recorded_target_pose
            ):
                raise ValueError(
                    "control resolution rejected probe target is not capture-bound"
                )
        capture_failure = (
            self.rejected_reset is not None
            and self.rejected_reset.phase
            is ControlResolutionResetPhase.CAPTURE_TO_BASELINE
        )
        if capture_failure:
            if self.rejected_reset.reference != captured_reset:
                raise ValueError(
                    "control resolution failure capture reset is inconsistent"
                )
            return
        initial_reset = (
            self.baseline.initial_reset
            if self.baseline is not None
            else (
                self.baseline_attempt.initial_reset
                if self.baseline_attempt is not None
                else None
            )
        )
        if initial_reset is not None:
            validate_reset_equivalence(
                captured_reset,
                initial_reset,
                tolerances=self.protocol.capture_tolerances,
            )

    def to_dict(self) -> dict[str, Any]:
        current = self.protocol.baseline_policy is not None
        authenticated = current and self.capture_identity is not None
        drive_bound = (
            (self.baseline is None or self.baseline.drive_target is not None)
            and (
                self.baseline_attempt is None
                or self.baseline_attempt.drive_target is not None
            )
        )
        return {
            "schema": (
                CONTROL_RESOLUTION_FAILURE_SCHEMA
                if authenticated and drive_bound
                else (
                    CONTROL_RESOLUTION_FAILURE_SCHEMA_V3
                    if authenticated
                    else (
                        CONTROL_RESOLUTION_FAILURE_SCHEMA_V2
                        if current
                        else CONTROL_RESOLUTION_FAILURE_SCHEMA_V1
                    )
                )
            ),
            "session_id": self.session_id,
            "failed_at_unix_seconds": self.failed_at_unix_seconds,
            "protocol": self.protocol.to_dict(),
            "reference_reset": (
                self.reference_reset.to_dict()
                if self.reference_reset is not None
                else None
            ),
            "completed_samples": [
                sample.to_dict() for sample in self.completed_samples
            ],
            "error": self.error,
            "rejected_reset": (
                self.rejected_reset.to_dict()
                if self.rejected_reset is not None
                else None
            ),
            **({"load": self.load.value} if current else {}),
            **(
                {"baseline": self.baseline.to_dict()}
                if self.baseline is not None
                else {}
            ),
            **(
                {"baseline_attempt": self.baseline_attempt.to_dict()}
                if self.baseline_attempt is not None
                else {}
            ),
            **(
                {"capture_identity": self.capture_identity.to_dict()}
                if self.capture_identity is not None
                else {}
            ),
            **(
                {"settlement_failure": self.settlement_failure.to_dict()}
                if self.settlement_failure is not None
                else {}
            ),
            **(
                {"projection_failure": self.projection_failure.to_dict()}
                if self.projection_failure is not None
                else {}
            ),
            "diagnostic_only": self.diagnostic_only,
            "multi_step_authority_granted": self.multi_step_authority_granted,
            "production_authority_granted": self.production_authority_granted,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionFailureEvidence:
        if (
            not isinstance(payload, dict)
            or payload.get("schema")
            not in (
                CONTROL_RESOLUTION_FAILURE_SCHEMA_V1,
                CONTROL_RESOLUTION_FAILURE_SCHEMA_V2,
                CONTROL_RESOLUTION_FAILURE_SCHEMA_V3,
                CONTROL_RESOLUTION_FAILURE_SCHEMA_V4,
                CONTROL_RESOLUTION_FAILURE_SCHEMA,
            )
        ):
            raise ValueError("control resolution failure schema is invalid")
        try:
            current = payload["schema"] != CONTROL_RESOLUTION_FAILURE_SCHEMA_V1
            authenticated = payload["schema"] in (
                CONTROL_RESOLUTION_FAILURE_SCHEMA_V3,
                CONTROL_RESOLUTION_FAILURE_SCHEMA_V4,
                CONTROL_RESOLUTION_FAILURE_SCHEMA,
            )
            drive_bound = payload["schema"] in (
                CONTROL_RESOLUTION_FAILURE_SCHEMA_V4,
                CONTROL_RESOLUTION_FAILURE_SCHEMA,
            )
            if current != ("load" in payload):
                raise ValueError("control resolution failure load is missing")
            if authenticated != ("capture_identity" in payload):
                raise ValueError(
                    "control resolution failure capture identity is missing"
                )
            evidence = cls(
                session_id=str(payload["session_id"]),
                failed_at_unix_seconds=float(payload["failed_at_unix_seconds"]),
                protocol=ControlResolutionProtocol.from_dict(payload["protocol"]),
                reference_reset=(
                    TrialResetState.from_dict(payload["reference_reset"])
                    if payload.get("reference_reset") is not None
                    else None
                ),
                completed_samples=tuple(
                    ControlResolutionSample.from_dict(
                        sample,
                        require_typed_drive_command=drive_bound,
                    )
                    for sample in payload["completed_samples"]
                ),
                error=str(payload["error"]),
                rejected_reset=(
                    RejectedControlResolutionReset.from_dict(
                        payload["rejected_reset"]
                    )
                    if payload.get("rejected_reset") is not None
                    else None
                ),
                load=ControlResolutionLoad(
                    payload.get("load", ControlResolutionLoad.ATTACHED.value)
                ),
                baseline=(
                    ControlResolutionBaselineEvidence.from_dict(
                        payload["baseline"]
                    )
                    if "baseline" in payload
                    else None
                ),
                baseline_attempt=(
                    ControlResolutionBaselineAttempt.from_dict(
                        payload["baseline_attempt"]
                    )
                    if "baseline_attempt" in payload
                    else None
                ),
                capture_identity=(
                    ControlResolutionCaptureIdentity.from_dict(
                        payload["capture_identity"]
                    )
                    if "capture_identity" in payload
                    else None
                ),
                settlement_failure=(
                    _settlement_failure_from_dict(
                        payload["settlement_failure"]
                    )
                    if "settlement_failure" in payload
                    else None
                ),
                projection_failure=(
                    ControlResolutionProjectionFailure.from_dict(
                        payload["projection_failure"]
                    )
                    if "projection_failure" in payload
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution failure evidence is incomplete"
            ) from error
        if current != (evidence.protocol.baseline_policy is not None):
            raise ValueError("control resolution failure protocol generation is invalid")
        baseline_drive_bound = (
            evidence.baseline.drive_target is not None
            if evidence.baseline is not None
            else (
                evidence.baseline_attempt.drive_target is not None
                if evidence.baseline_attempt is not None
                else None
            )
        )
        if (
            baseline_drive_bound is not None
            and drive_bound is not baseline_drive_bound
        ) or (not drive_bound and evidence.settlement_failure is not None):
            raise ValueError(
                "control resolution failure drive-target generation is invalid"
            )
        if (
            "projection_failure" in payload
            and payload["schema"] != CONTROL_RESOLUTION_FAILURE_SCHEMA
        ):
            raise ValueError(
                "control resolution projection failure generation is invalid"
            )
        if (
            payload.get("diagnostic_only") is not evidence.diagnostic_only
            or payload.get("multi_step_authority_granted")
            is not evidence.multi_step_authority_granted
            or payload.get("production_authority_granted")
            is not evidence.production_authority_granted
        ):
            raise ValueError("control resolution failure claims are inconsistent")
        return evidence


@dataclass(frozen=True)
class ControlResolutionTimingSummary:
    mean_seconds: float
    maximum_seconds: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.mean_seconds)
            or not isfinite(self.maximum_seconds)
            or self.mean_seconds <= 0.0
            or self.maximum_seconds < self.mean_seconds
        ):
            raise ValueError("control resolution timing summary is invalid")


@dataclass(frozen=True)
class ControlResolutionResponse:
    requested_translation_meters: float
    mean_realized_along_axis_meters: float
    maximum_translation_error_meters: float
    maximum_orientation_drift_radians: float
    maximum_joint_tracking_error_radians: float
    settling_time: ControlResolutionTimingSummary | None

    def to_dict(self) -> dict[str, float]:
        payload = {
            "requested_translation_meters": self.requested_translation_meters,
            "mean_realized_along_axis_meters": self.mean_realized_along_axis_meters,
            "maximum_translation_error_meters": self.maximum_translation_error_meters,
            "maximum_orientation_drift_radians": self.maximum_orientation_drift_radians,
            "maximum_joint_tracking_error_radians": (
                self.maximum_joint_tracking_error_radians
            ),
        }
        if self.settling_time is not None:
            payload["mean_settling_time_seconds"] = self.settling_time.mean_seconds
            payload["maximum_settling_time_seconds"] = (
                self.settling_time.maximum_seconds
            )
        return payload


@dataclass(frozen=True)
class ControlResolutionSummary:
    zero_translation_drift_meters: float
    zero_orientation_drift_radians: float
    zero_joint_tracking_error_radians: float
    maximum_zero_settling_time_seconds: float | None
    start_repeatability: ResetEquivalenceMeasurement
    rollback_repeatability: ResetEquivalenceMeasurement
    responses: tuple[ControlResolutionResponse, ...]

    @property
    def diagnostic_only(self) -> bool:
        return True

    @property
    def multi_step_authority_granted(self) -> bool:
        return False

    @property
    def production_authority_granted(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "zero_translation_drift_meters": self.zero_translation_drift_meters,
            "zero_orientation_drift_radians": self.zero_orientation_drift_radians,
            "zero_joint_tracking_error_radians": (
                self.zero_joint_tracking_error_radians
            ),
            "start_repeatability": self.start_repeatability.to_dict(),
            "rollback_repeatability": self.rollback_repeatability.to_dict(),
            "responses": [response.to_dict() for response in self.responses],
            "diagnostic_only": self.diagnostic_only,
            "multi_step_authority_granted": self.multi_step_authority_granted,
            "production_authority_granted": self.production_authority_granted,
        }
        if self.maximum_zero_settling_time_seconds is not None:
            payload["maximum_zero_settling_time_seconds"] = (
                self.maximum_zero_settling_time_seconds
            )
        return payload


@dataclass(frozen=True)
class ControlResolutionReport:
    session_id: str
    reference_recording: str
    seed: int
    context_index: int
    observation_id: int
    captured_pose: DroidPose
    recorded_target_pose: DroidPose
    reference_reset: TrialResetState
    samples: tuple[ControlResolutionSample, ...]
    protocol: ControlResolutionProtocol = CONTROL_RESOLUTION_PROTOCOL
    load: ControlResolutionLoad = ControlResolutionLoad.ATTACHED
    baseline: ControlResolutionBaselineEvidence | None = None

    def __post_init__(self) -> None:
        validate_recording_id(self.session_id)
        validate_recording_id(self.reference_recording)
        safety_limits = self.protocol.safety_limits
        if (
            self.seed < 0
            or self.context_index <= 0
            or self.observation_id <= 0
        ):
            raise ValueError("control resolution sample roster is invalid")
        self.protocol.validate_samples(
            self.reference_reset,
            self.samples,
            require_complete=True,
        )
        if self.baseline is not None:
            if self.protocol.baseline_policy is None:
                raise ValueError("legacy protocol cannot carry baseline evidence")
            self.baseline.validate(
                self.protocol.baseline_policy,
                self.load,
                safety_limits,
            )
            if self.reference_reset != self.baseline.reference_reset:
                raise ValueError("control resolution baseline reset is inconsistent")
        elif self.protocol.baseline_policy is not None:
            raise ValueError("current control resolution report has no stable baseline")
        elif any(sample.settling_time_seconds is not None for sample in self.samples):
            raise ValueError("legacy control resolution report has settling time")
        for sample in self.samples:
            probe = self.protocol.probe_plan(sample.index)
            if self.baseline is not None and sample.settling_time_seconds is None:
                raise ValueError("current control resolution sample has no settling time")
            drive_target = (
                self.baseline.drive_target
                if self.baseline is not None
                and self.baseline.drive_target is not None
                else ControlResolutionDriveTarget(
                    sample.projection.proposed_joint_positions,
                    0.0,
                )
            )
            ControlResolutionProbeExecution(
                probe,
                self.recorded_target_pose,
                sample.start_reset,
                sample.commanded_action,
                sample.target_pose,
                sample.projection,
            ).validate(
                self.protocol,
                self.observation_id,
            )
            if (
                sample.settling_time_seconds is not None
                and sample.settling_time_seconds
                < self.protocol.motion_period_for(
                    sample.requested_translation_meters
                )
            ):
                raise ValueError("control resolution command does not match its protocol")
            self.protocol.validate_sample_execution(
                self.reference_reset,
                sample,
                expected_attachment=self.load.plug_attached,
                rollback_drive_target=(
                    self.baseline.drive_target
                    if self.baseline is not None
                    else None
                ),
            )

    @property
    def summary(self) -> ControlResolutionSummary:
        zero = tuple(
            sample
            for sample in self.samples
            if self.protocol.probe_plan(sample.index).kind
            is ControlResolutionProbeKind.HOLD
        )
        responses = []
        for magnitude in self.protocol.translation_magnitudes_meters[1:]:
            matching = tuple(
                sample for sample in self.samples
                if sample.requested_translation_meters == magnitude
            )
            settling_times = tuple(
                sample.settling_time_seconds
                for sample in matching
                if sample.settling_time_seconds is not None
            )
            expected_translations = tuple(
                np.asarray(
                    self.protocol.probe_action(
                        sample.start_reset.pose,
                        self.recorded_target_pose,
                        magnitude,
                    ).values[:3],
                    dtype=np.float64,
                )
                for sample in matching
            )
            along_axis = tuple(
                float(
                    np.dot(
                        sample.actual_action.values[:3],
                        expected_translation / magnitude,
                    )
                )
                for sample, expected_translation in zip(
                    matching, expected_translations
                )
            )
            responses.append(
                ControlResolutionResponse(
                    requested_translation_meters=magnitude,
                    mean_realized_along_axis_meters=sum(along_axis) / len(along_axis),
                    maximum_translation_error_meters=max(
                        float(
                            np.linalg.norm(
                                np.asarray(sample.actual_action.values[:3])
                                - expected_translation
                            )
                        )
                        for sample, expected_translation in zip(
                            matching, expected_translations
                        )
                    ),
                    maximum_orientation_drift_radians=max(
                        sample.actual_orientation_drift_radians
                        for sample in matching
                    ),
                    maximum_joint_tracking_error_radians=max(
                        self._controller_tracking_error(sample)
                        for sample in matching
                    ),
                    settling_time=(
                        ControlResolutionTimingSummary(
                            sum(settling_times) / len(settling_times),
                            max(settling_times),
                        )
                        if settling_times
                        else None
                    ),
                )
            )
        return ControlResolutionSummary(
            zero_translation_drift_meters=max(
                sample.actual_translation_meters for sample in zero
            ),
            zero_orientation_drift_radians=max(
                sample.actual_orientation_drift_radians for sample in zero
            ),
            zero_joint_tracking_error_radians=max(
                self._controller_tracking_error(sample)
                for sample in zero
            ),
            maximum_zero_settling_time_seconds=(
                max(
                    sample.settling_time_seconds
                    for sample in zero
                    if sample.settling_time_seconds is not None
                )
                if self.baseline is not None
                else None
            ),
            start_repeatability=ResetEquivalenceMeasurement.worst_case(
                tuple(
                    ResetEquivalenceMeasurement.between(
                        self.reference_reset,
                        sample.start_reset,
                    )
                    for sample in self.samples
                )
            ),
            rollback_repeatability=ResetEquivalenceMeasurement.worst_case(
                tuple(
                    ResetEquivalenceMeasurement.between(
                        self.reference_reset,
                        sample.rollback_reset,
                    )
                    for sample in self.samples
                )
            ),
            responses=tuple(responses),
        )

    def _controller_tracking_error(
        self,
        sample: ControlResolutionSample,
    ) -> float:
        if self.baseline is None or self.baseline.drive_target is None:
            return sample.settlement_joint_error_radians
        return sample.controller_tracking_error_radians(
            self.protocol.probe_plan(sample.index),
            self.baseline.drive_target,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": (
                CONTROL_RESOLUTION_SCHEMA
                if self.baseline is not None
                and self.baseline.drive_target is not None
                else (
                    CONTROL_RESOLUTION_SCHEMA_V2
                    if self.baseline is not None
                    else CONTROL_RESOLUTION_SCHEMA_V1
                )
            ),
            "session_id": self.session_id,
            "reference_recording": self.reference_recording,
            "seed": self.seed,
            "context_index": self.context_index,
            "observation_id": self.observation_id,
            "captured_pose": list(self.captured_pose.values),
            "recorded_target_pose": list(self.recorded_target_pose.values),
            "protocol": self.protocol.to_dict(),
            "reference_reset": self.reference_reset.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "summary": self.summary.to_dict(),
            **(
                {
                    "load": self.load.value,
                    "baseline": self.baseline.to_dict(),
                }
                if self.baseline is not None
                else {}
            ),
        }

    def validate_capture(self, request: Any, state: Any) -> None:
        """Bind the diagnostic to the raw captured session it measured."""

        observation, session_state, captured_reset = _control_resolution_capture(
            request,
            state,
            self.load,
        )
        if (
            observation.observation_id != self.observation_id
            or observation.pose != self.captured_pose
            or observation.target_pose != self.recorded_target_pose
            or observation.warmup_frames != self.context_index
            or session_state.session_id != self.session_id
            or session_state.reference_recording != self.reference_recording
            or session_state.seed != self.seed
            or session_state.execution_policy
            is not ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT
        ):
            raise ValueError("control resolution report is not bound to its capture")
        validate_reset_equivalence(
            captured_reset,
            (
                self.baseline.initial_reset
                if self.baseline is not None
                else self.reference_reset
            ),
            tolerances=self.protocol.capture_tolerances,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlResolutionReport:
        if payload.get("schema") not in (
            CONTROL_RESOLUTION_SCHEMA_V1,
            CONTROL_RESOLUTION_SCHEMA_V2,
            CONTROL_RESOLUTION_SCHEMA,
        ):
            raise ValueError("control resolution schema is invalid")
        try:
            current = payload["schema"] != CONTROL_RESOLUTION_SCHEMA_V1
            drive_bound = payload["schema"] == CONTROL_RESOLUTION_SCHEMA
            if current != ("load" in payload and "baseline" in payload):
                raise ValueError("control resolution generation fields are invalid")
            report = cls(
                session_id=str(payload["session_id"]),
                reference_recording=str(payload["reference_recording"]),
                seed=int(payload["seed"]),
                context_index=int(payload["context_index"]),
                observation_id=int(payload["observation_id"]),
                captured_pose=DroidPose(tuple(payload["captured_pose"])),
                recorded_target_pose=DroidPose(
                    tuple(payload["recorded_target_pose"])
                ),
                protocol=ControlResolutionProtocol.from_dict(payload["protocol"]),
                reference_reset=TrialResetState.from_dict(payload["reference_reset"]),
                samples=tuple(
                    ControlResolutionSample.from_dict(
                        sample,
                        require_typed_drive_command=drive_bound,
                    )
                    for sample in payload["samples"]
                ),
                load=ControlResolutionLoad(
                    payload.get("load", ControlResolutionLoad.ATTACHED.value)
                ),
                baseline=(
                    ControlResolutionBaselineEvidence.from_dict(
                        payload["baseline"]
                    )
                    if "baseline" in payload
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution report is incomplete") from error
        if current != (report.protocol.baseline_policy is not None):
            raise ValueError("control resolution protocol generation is invalid")
        if current and (
            report.baseline is None
            or drive_bound is not (report.baseline.drive_target is not None)
        ):
            raise ValueError(
                "control resolution report drive-target generation is invalid"
            )
        if not _reconstructed_metric_payload_matches(
            report.summary.to_dict(),
            payload.get("summary"),
        ):
            raise ValueError("control resolution summary is inconsistent")
        return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate one insertion control-resolution report."
    )
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    report = ControlResolutionReport.from_dict(json.loads(args.report.read_text()))
    report.validate_capture(
        json.loads((args.report.parent / "request.json").read_text()),
        json.loads((args.report.parent / "state.json").read_text()),
    )
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
