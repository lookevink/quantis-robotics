"""Typed evidence for a reset-repeatable insertion control-resolution experiment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum
import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Mapping

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
    ControlInterlockEvidence,
    SafetyProjectionAttempt,
    SimulatorSafetyLimits,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.direct_safety import ControlSafetySnapshot
from jepa_wm.trial_equivalence import (
    ResetEquivalenceMeasurement,
    ResetEquivalenceTolerances,
    TrialResetState,
    validate_reset_equivalence,
)
from sim.recording import validate_recording_id


CONTROL_RESOLUTION_SCHEMA = "quantis.jepa_wm_control_resolution.v1"
CONTROL_RESOLUTION_FAILURE_SCHEMA = "quantis.jepa_wm_control_resolution_failure.v1"
CONTROL_RESOLUTION_RESET_TOLERANCES = ResetEquivalenceTolerances(
    maximum_translation_difference_meters=5e-4,
    maximum_rotation_difference_radians=1e-3,
    maximum_gripper_difference=1e-3,
    maximum_joint_difference_radians=5e-4,
    maximum_reset_contact_force_newtons=0.01,
    maximum_plug_position_difference_meters=5e-4,
)


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
class ControlResolutionProtocol:
    translation_magnitudes_meters: tuple[float, ...] = (0.0, 3e-5, 1e-4, 2e-4)
    repeats_per_magnitude: int = 3
    motion_period_seconds: float = 0.25
    settling_updates: int = 8
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
            or self.settling_updates <= 0
        ):
            raise ValueError("control resolution protocol is invalid")

    @property
    def requested_translations(self) -> tuple[float, ...]:
        return tuple(
            magnitude
            for magnitude in self.translation_magnitudes_meters
            for _ in range(self.repeats_per_magnitude)
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "translation_magnitudes_meters": list(
                self.translation_magnitudes_meters
            ),
            "repeats_per_magnitude": self.repeats_per_magnitude,
            "motion_period_seconds": self.motion_period_seconds,
            "settling_updates": self.settling_updates,
            "safety_limits": self.safety_limits.to_dict(),
            "capture_tolerances": self.capture_tolerances.to_dict(),
            "reset_tolerances": self.reset_tolerances.to_dict(),
            "translation_policy": "retreat_from_recorded_target",
            "rotation_policy": "hold_current_orientation",
            "gripper_policy": "hold_current_width",
        }

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
                settling_updates=int(payload["settling_updates"]),
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


class ControlResolutionResetPhase(str, Enum):
    CAPTURE_TO_BASELINE = "capture_to_baseline"
    SAMPLE_START = "sample_start"
    ROLLBACK = "rollback"


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
    def settled_joint_tracking_error_radians(self) -> float:
        return max(
            abs(actual - target)
            for actual, target in zip(
                self.endpoint.safety.joint_positions,
                self.projection.proposed_joint_positions,
            )
        )

    @property
    def actual_translation_meters(self) -> float:
        return float(np.linalg.norm(self.actual_action.values[:3]))

    @property
    def actual_orientation_drift_radians(self) -> float:
        return float(
            Rotation.from_euler("xyz", self.actual_action.values[3:6]).magnitude()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "requested_translation_meters": self.requested_translation_meters,
            "start_reset": self.start_reset.to_dict(),
            "commanded_action": list(self.commanded_action.values),
            "target_pose": list(self.target_pose.values),
            "projection": self.projection.to_dict(),
            "endpoint": self.endpoint.to_dict(),
            "actual_action": list(self.actual_action.values),
            "settled_joint_tracking_error_radians": (
                self.settled_joint_tracking_error_radians
            ),
            "interlock": self.interlock.to_dict(),
            "rollback_reset": self.rollback_reset.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionSample:
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
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution sample is incomplete") from error
        if (
            payload.get("actual_action") != list(sample.actual_action.values)
            or payload.get("settled_joint_tracking_error_radians")
            != sample.settled_joint_tracking_error_radians
        ):
            raise ValueError("control resolution sample claims are inconsistent")
        return sample


@dataclass(frozen=True)
class ControlResolutionFailureEvidence:
    session_id: str
    failed_at_unix_seconds: float
    reference_reset: TrialResetState | None
    completed_samples: tuple[ControlResolutionSample, ...]
    error: str
    rejected_reset: RejectedControlResolutionReset | None = None
    protocol: ControlResolutionProtocol = CONTROL_RESOLUTION_PROTOCOL

    def __post_init__(self) -> None:
        validate_recording_id(self.session_id)
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
                    or self.rejected_reset.candidate != self.reference_reset
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_RESOLUTION_FAILURE_SCHEMA,
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
            "diagnostic_only": self.diagnostic_only,
            "multi_step_authority_granted": self.multi_step_authority_granted,
            "production_authority_granted": self.production_authority_granted,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlResolutionFailureEvidence:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != CONTROL_RESOLUTION_FAILURE_SCHEMA
        ):
            raise ValueError("control resolution failure schema is invalid")
        try:
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
                    ControlResolutionSample.from_dict(sample)
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
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "control resolution failure evidence is incomplete"
            ) from error
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
class ControlResolutionResponse:
    requested_translation_meters: float
    mean_realized_along_axis_meters: float
    maximum_translation_error_meters: float
    maximum_orientation_drift_radians: float
    maximum_joint_tracking_error_radians: float

    def to_dict(self) -> dict[str, float]:
        return {
            "requested_translation_meters": self.requested_translation_meters,
            "mean_realized_along_axis_meters": self.mean_realized_along_axis_meters,
            "maximum_translation_error_meters": self.maximum_translation_error_meters,
            "maximum_orientation_drift_radians": self.maximum_orientation_drift_radians,
            "maximum_joint_tracking_error_radians": (
                self.maximum_joint_tracking_error_radians
            ),
        }


@dataclass(frozen=True)
class ControlResolutionSummary:
    zero_translation_drift_meters: float
    zero_orientation_drift_radians: float
    zero_joint_tracking_error_radians: float
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
        return {
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
        for sample in self.samples:
            expected_action = self.protocol.probe_action(
                sample.start_reset.pose,
                self.recorded_target_pose,
                sample.requested_translation_meters,
            )
            expected = np.asarray(
                expected_action.values,
                dtype=np.float64,
            )
            commanded = np.asarray(sample.commanded_action.values, dtype=np.float64)
            expected_target = sample.start_reset.pose.applied(sample.commanded_action)
            start_target_distance = float(
                np.linalg.norm(
                    np.asarray(sample.start_reset.pose.values[:3])
                    - np.asarray(self.recorded_target_pose.values[:3])
                )
            )
            probe_target_distance = float(
                np.linalg.norm(
                    np.asarray(sample.target_pose.values[:3])
                    - np.asarray(self.recorded_target_pose.values[:3])
                )
            )
            expected_joint_delta = max(
                abs(proposed - current)
                for proposed, current in zip(
                    sample.projection.proposed_joint_positions,
                    sample.start_reset.joint_positions,
                )
            )
            if (
                not np.allclose(commanded, expected, rtol=0.0, atol=1e-12)
                or sample.target_pose != expected_target
                or (
                    sample.requested_translation_meters > 0.0
                    and probe_target_distance <= start_target_distance
                )
                or sample.projection.gate.observation_id != self.observation_id
                or sample.projection.gate.next_pose != sample.target_pose
                or not isclose(
                    sample.projection.maximum_joint_delta_rad,
                    expected_joint_delta,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not safety_limits.action_bounds.accepts(
                    (sample.commanded_action,)
                )
                or any(
                    value < lower or value > upper
                    for value, lower, upper in zip(
                        sample.target_pose.values[:3],
                        safety_limits.minimum_workspace_xyz,
                        safety_limits.maximum_workspace_xyz,
                    )
                )
                or not 0.0 <= sample.target_pose.values[6] <= 1.0
                or any(
                    value < lower or value > upper
                    for value, lower, upper in zip(
                        sample.projection.proposed_joint_positions,
                        safety_limits.lower_joint_limits,
                        safety_limits.upper_joint_limits,
                    )
                )
                or expected_joint_delta
                > safety_limits.maximum_joint_velocity_radians_per_second
                * self.protocol.motion_period_seconds
            ):
                raise ValueError("control resolution command does not match its protocol")
            if (
                sample.interlock.collision_detected
                or sample.interlock.maximum_contact_force_newtons
                > safety_limits.maximum_contact_force_newtons
                or sample.endpoint.safety.collision_detected
                or sample.endpoint.safety.contact_force_newtons
                > safety_limits.maximum_contact_force_newtons
                or not sample.endpoint.safety.plug_attached
            ):
                raise ValueError("control resolution sample failed safety")

    @property
    def summary(self) -> ControlResolutionSummary:
        zero = tuple(
            sample for sample in self.samples
            if sample.requested_translation_meters == 0.0
        )
        responses = []
        for magnitude in self.protocol.translation_magnitudes_meters[1:]:
            matching = tuple(
                sample for sample in self.samples
                if sample.requested_translation_meters == magnitude
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
                        sample.settled_joint_tracking_error_radians
                        for sample in matching
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
                sample.settled_joint_tracking_error_radians for sample in zero
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CONTROL_RESOLUTION_SCHEMA,
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
        }

    def validate_capture(self, request: Any, state: Any) -> None:
        """Bind the diagnostic to the raw captured session it measured."""

        from sim.control_session import ControlSession, ControlSessionState

        observation = ControlObservation.from_dict(request)
        session_state = ControlSessionState.from_dict(state)
        captured_reset = ControlSession.trial_context(
            observation, session_state
        ).reset
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
            self.reference_reset,
            tolerances=self.protocol.capture_tolerances,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ControlResolutionReport:
        if payload.get("schema") != CONTROL_RESOLUTION_SCHEMA:
            raise ValueError("control resolution schema is invalid")
        try:
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
                    ControlResolutionSample.from_dict(sample)
                    for sample in payload["samples"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control resolution report is incomplete") from error
        if payload.get("summary") != report.summary.to_dict():
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
