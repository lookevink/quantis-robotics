"""Dependency-light joint-settlement policy and evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any


def _strict_number(payload: dict[str, Any], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number")
    return float(value)


@dataclass(frozen=True)
class TrackedJointSettlementPolicy:
    """Require bounded consecutive command-relative tracking passes."""

    absolute_tracking_floor_radians: float = 5e-4
    tracking_error_fraction_of_requested_motion: float = 0.25
    required_consecutive_updates: int = 2
    maximum_updates: int = 32

    def __post_init__(self) -> None:
        if (
            not isfinite(self.absolute_tracking_floor_radians)
            or self.absolute_tracking_floor_radians <= 0.0
            or not isfinite(self.tracking_error_fraction_of_requested_motion)
            or not 0.0 < self.tracking_error_fraction_of_requested_motion < 1.0
            or isinstance(self.required_consecutive_updates, bool)
            or not isinstance(self.required_consecutive_updates, int)
            or self.required_consecutive_updates <= 0
            or isinstance(self.maximum_updates, bool)
            or not isinstance(self.maximum_updates, int)
            or self.maximum_updates < self.required_consecutive_updates
        ):
            raise ValueError("tracked joint settlement policy is invalid")

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

    def to_dict(self) -> dict[str, float | int]:
        return {
            "absolute_tracking_floor_radians": self.absolute_tracking_floor_radians,
            "tracking_error_fraction_of_requested_motion": (
                self.tracking_error_fraction_of_requested_motion
            ),
            "required_consecutive_updates": self.required_consecutive_updates,
            "maximum_updates": self.maximum_updates,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> TrackedJointSettlementPolicy:
        if not isinstance(payload, dict):
            raise ValueError("tracked joint settlement policy must be an object")
        required_updates = payload.get("required_consecutive_updates")
        maximum_updates = payload.get("maximum_updates")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (required_updates, maximum_updates)
        ):
            raise ValueError("settlement update counts must be integers")
        try:
            return cls(
                absolute_tracking_floor_radians=_strict_number(
                    payload, "absolute_tracking_floor_radians"
                ),
                tracking_error_fraction_of_requested_motion=_strict_number(
                    payload, "tracking_error_fraction_of_requested_motion"
                ),
                required_consecutive_updates=required_updates,
                maximum_updates=maximum_updates,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("tracked joint settlement policy is incomplete") from error


@dataclass(frozen=True)
class JointSettlementEvidence:
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
            raise ValueError("joint settlement evidence is invalid")

    @property
    def final_tracking_error_radians(self) -> float:
        return self.passing_tracking_errors_radians[-1]

    def validate(
        self,
        policy: TrackedJointSettlementPolicy,
        cap_radians: float | None = None,
        *,
        expected_requested_motion_radians: float | None = None,
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
            or (
                expected_requested_motion_radians is not None
                and not isclose(
                    self.requested_joint_motion_radians,
                    expected_requested_motion_radians,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        ):
            raise ValueError("joint settlement evidence does not match its policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_joint_motion_radians": self.requested_joint_motion_radians,
            "required_tracking_error_radians": self.required_tracking_error_radians,
            "updates_used": self.updates_used,
            "passing_tracking_errors_radians": list(
                self.passing_tracking_errors_radians
            ),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> JointSettlementEvidence:
        if not isinstance(payload, dict):
            raise ValueError("joint settlement evidence must be an object")
        updates_used = payload.get("updates_used")
        values = payload.get("passing_tracking_errors_radians")
        if isinstance(updates_used, bool) or not isinstance(updates_used, int):
            raise ValueError("joint settlement updates must be an integer")
        if not isinstance(values, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in values
        ):
            raise ValueError("passing tracking errors must contain JSON numbers")
        try:
            return cls(
                requested_joint_motion_radians=_strict_number(
                    payload, "requested_joint_motion_radians"
                ),
                required_tracking_error_radians=_strict_number(
                    payload, "required_tracking_error_radians"
                ),
                updates_used=updates_used,
                passing_tracking_errors_radians=tuple(float(value) for value in values),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("joint settlement evidence is incomplete") from error


@dataclass(frozen=True)
class GripperSettlementCriterion:
    target_width_meters: float
    maximum_error_meters: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.target_width_meters)
            or self.target_width_meters < 0.0
            or not isfinite(self.maximum_error_meters)
            or self.maximum_error_meters <= 0.0
        ):
            raise ValueError("gripper settlement criterion is invalid")

    def error(self, actual_width_meters: float) -> float:
        if not isfinite(actual_width_meters) or actual_width_meters < 0.0:
            raise ValueError("gripper width is invalid")
        return abs(actual_width_meters - self.target_width_meters)


@dataclass(frozen=True)
class GripperSettlementTrace:
    errors_meters: tuple[float, ...]
    maximum_error_meters: float

    def __post_init__(self) -> None:
        if (
            not self.errors_meters
            or not all(isfinite(value) and value >= 0.0 for value in self.errors_meters)
            or not isfinite(self.maximum_error_meters)
            or self.maximum_error_meters <= 0.0
        ):
            raise ValueError("gripper settlement trace is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors_meters": list(self.errors_meters),
            "maximum_error_meters": self.maximum_error_meters,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> GripperSettlementTrace:
        if not isinstance(payload, dict):
            raise ValueError("gripper settlement trace must be an object")
        errors = payload.get("errors_meters")
        if not isinstance(errors, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in errors
        ):
            raise ValueError("gripper settlement errors are invalid")
        return cls(
            tuple(float(value) for value in errors),
            _strict_number(payload, "maximum_error_meters"),
        )


@dataclass(frozen=True)
class GripperSettlementMeasurement:
    target_width_meters: float
    end_width_meters: float
    trace: GripperSettlementTrace

    def __post_init__(self) -> None:
        if (
            not isfinite(self.target_width_meters)
            or self.target_width_meters < 0.0
            or not isfinite(self.end_width_meters)
            or self.end_width_meters < 0.0
            or not isclose(
                self.trace.errors_meters[-1],
                abs(self.end_width_meters - self.target_width_meters),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("gripper settlement measurement is invalid")

    def validate(
        self,
        expected_target_width_meters: float,
        expected_maximum_error_meters: float,
    ) -> None:
        if (
            not isclose(
                self.target_width_meters,
                expected_target_width_meters,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isclose(
                self.trace.maximum_error_meters,
                expected_maximum_error_meters,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("gripper settlement measurement is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_width_meters": self.target_width_meters,
            "end_width_meters": self.end_width_meters,
            "trace": self.trace.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> GripperSettlementMeasurement:
        if not isinstance(payload, dict):
            raise ValueError("gripper settlement measurement must be an object")
        return cls(
            _strict_number(payload, "target_width_meters"),
            _strict_number(payload, "end_width_meters"),
            GripperSettlementTrace.from_dict(payload.get("trace")),
        )


@dataclass(frozen=True)
class GripperTrackedJointSettlementEvidence:
    joint: JointSettlementEvidence
    gripper: GripperSettlementMeasurement

    def validate(
        self,
        policy: TrackedJointSettlementPolicy,
        *,
        expected_requested_motion_radians: float,
        expected_target_gripper_width_meters: float,
        expected_gripper_error_meters: float,
    ) -> None:
        self.joint.validate(
            policy,
            expected_requested_motion_radians=expected_requested_motion_radians,
        )
        self.gripper.validate(
            expected_target_gripper_width_meters,
            expected_gripper_error_meters,
        )
        if (
            len(self.gripper.trace.errors_meters)
            != policy.required_consecutive_updates
            or any(
                error > self.gripper.trace.maximum_error_meters
                for error in self.gripper.trace.errors_meters
            )
        ):
            raise ValueError("gripper settlement evidence does not match its policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_settlement": self.joint.to_dict(),
            "gripper_settlement": self.gripper.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> GripperTrackedJointSettlementEvidence:
        if not isinstance(payload, dict):
            raise ValueError("joint and gripper settlement must be an object")
        return cls(
            JointSettlementEvidence.from_dict(payload.get("joint_settlement")),
            GripperSettlementMeasurement.from_dict(
                payload.get("gripper_settlement")
            ),
        )


@dataclass(frozen=True)
class JointSettlementTrace:
    """Raw arm tracking trace shared by arm-only and arm+gripper attempts."""

    requested_joint_motion_radians: float
    required_tracking_error_radians: float
    tracking_errors_radians: tuple[float, ...]
    final_joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        scalars = (
            self.requested_joint_motion_radians,
            self.required_tracking_error_radians,
            *self.tracking_errors_radians,
            *self.final_joint_positions,
        )
        if (
            not all(isfinite(value) for value in scalars)
            or self.requested_joint_motion_radians < 0.0
            or self.required_tracking_error_radians <= 0.0
            or not self.tracking_errors_radians
            or any(value < 0.0 for value in self.tracking_errors_radians)
            or len(self.final_joint_positions) != 7
        ):
            raise ValueError("joint settlement trace is invalid")

    def validate_failed_attempt(
        self,
        policy: TrackedJointSettlementPolicy,
        cap_radians: float | None = None,
        *,
        expected_requested_motion_radians: float | None = None,
        expected_target_joint_positions: tuple[float, ...] | None = None,
        gripper: GripperSettlementTrace | None = None,
    ) -> None:
        if (
            expected_target_joint_positions is not None
            and len(expected_target_joint_positions) != 7
        ):
            raise ValueError("joint settlement target is invalid")
        if gripper is not None and len(gripper.errors_meters) != len(
            self.tracking_errors_radians
        ):
            raise ValueError("gripper settlement trace length is invalid")
        final_tracking_error = (
            max(
                abs(actual - target)
                for actual, target in zip(
                    self.final_joint_positions,
                    expected_target_joint_positions,
                )
            )
            if expected_target_joint_positions is not None
            else self.tracking_errors_radians[-1]
        )
        passing = tuple(
            tracking_error <= self.required_tracking_error_radians
            and (
                gripper is None
                or gripper.errors_meters[index] <= gripper.maximum_error_meters
            )
            for index, tracking_error in enumerate(self.tracking_errors_radians)
        )
        if (
            (
                expected_requested_motion_radians is not None
                and not isclose(
                    self.requested_joint_motion_radians,
                    expected_requested_motion_radians,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            or not isclose(
                self.required_tracking_error_radians,
                policy.maximum_tracking_error(
                    self.requested_joint_motion_radians,
                    cap_radians,
                ),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or len(self.tracking_errors_radians) != policy.maximum_updates
            or not isclose(
                self.tracking_errors_radians[-1],
                final_tracking_error,
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
            raise ValueError("joint settlement attempt does not match its policy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_joint_motion_radians": self.requested_joint_motion_radians,
            "required_tracking_error_radians": self.required_tracking_error_radians,
            "tracking_errors_radians": list(self.tracking_errors_radians),
            "final_joint_positions": list(self.final_joint_positions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> JointSettlementTrace:
        if not isinstance(payload, dict):
            raise ValueError("joint settlement attempt must be an object")
        errors = payload.get("tracking_errors_radians")
        positions = payload.get("final_joint_positions")
        if not isinstance(errors, (list, tuple)) or not isinstance(
            positions, (list, tuple)
        ) or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (*errors, *positions)
        ):
            raise ValueError("joint settlement attempt arrays are invalid")
        try:
            return cls(
                requested_joint_motion_radians=_strict_number(
                    payload, "requested_joint_motion_radians"
                ),
                required_tracking_error_radians=_strict_number(
                    payload, "required_tracking_error_radians"
                ),
                tracking_errors_radians=tuple(float(value) for value in errors),
                final_joint_positions=tuple(float(value) for value in positions),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("joint settlement trace is incomplete") from error


@dataclass(frozen=True, init=False)
class JointSettlementAttempt:
    """Arm-only exhausted settlement attempt."""

    trace: JointSettlementTrace

    def __init__(
        self,
        requested_joint_motion_radians: float,
        required_tracking_error_radians: float,
        tracking_errors_radians: tuple[float, ...],
        final_joint_positions: tuple[float, ...],
    ) -> None:
        object.__setattr__(
            self,
            "trace",
            JointSettlementTrace(
                requested_joint_motion_radians,
                required_tracking_error_radians,
                tracking_errors_radians,
                final_joint_positions,
            ),
        )

    requested_joint_motion_radians = property(
        lambda self: self.trace.requested_joint_motion_radians
    )
    required_tracking_error_radians = property(
        lambda self: self.trace.required_tracking_error_radians
    )
    tracking_errors_radians = property(
        lambda self: self.trace.tracking_errors_radians
    )
    final_joint_positions = property(lambda self: self.trace.final_joint_positions)

    def validate(
        self,
        policy: TrackedJointSettlementPolicy,
        cap_radians: float | None = None,
        *,
        expected_requested_motion_radians: float | None = None,
        expected_target_joint_positions: tuple[float, ...] | None = None,
    ) -> None:
        self.trace.validate_failed_attempt(
            policy,
            cap_radians,
            expected_requested_motion_radians=expected_requested_motion_radians,
            expected_target_joint_positions=expected_target_joint_positions,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.trace.to_dict()

    @classmethod
    def from_dict(cls, payload: Any) -> JointSettlementAttempt:
        if isinstance(payload, dict) and "gripper_settlement" in payload:
            raise ValueError("arm-only settlement has unexpected gripper evidence")
        trace = JointSettlementTrace.from_dict(payload)
        return cls(
            trace.requested_joint_motion_radians,
            trace.required_tracking_error_radians,
            trace.tracking_errors_radians,
            trace.final_joint_positions,
        )


@dataclass(frozen=True, init=False)
class GripperTrackedJointSettlementAttempt:
    """Exhausted arm settlement composed with its gripper predicate trace."""

    trace: JointSettlementTrace
    gripper: GripperSettlementMeasurement

    def __init__(
        self,
        requested_joint_motion_radians: float,
        required_tracking_error_radians: float,
        tracking_errors_radians: tuple[float, ...],
        final_joint_positions: tuple[float, ...],
        gripper: GripperSettlementMeasurement,
    ) -> None:
        if not isinstance(gripper, GripperSettlementMeasurement):
            raise ValueError("gripper settlement attempt is incomplete")
        object.__setattr__(
            self,
            "trace",
            JointSettlementTrace(
                requested_joint_motion_radians,
                required_tracking_error_radians,
                tracking_errors_radians,
                final_joint_positions,
            ),
        )
        object.__setattr__(self, "gripper", gripper)
        if len(gripper.trace.errors_meters) != len(tracking_errors_radians):
            raise ValueError("gripper settlement trace length is invalid")

    requested_joint_motion_radians = property(
        lambda self: self.trace.requested_joint_motion_radians
    )
    required_tracking_error_radians = property(
        lambda self: self.trace.required_tracking_error_radians
    )
    tracking_errors_radians = property(
        lambda self: self.trace.tracking_errors_radians
    )
    final_joint_positions = property(lambda self: self.trace.final_joint_positions)

    def validate(
        self,
        policy: TrackedJointSettlementPolicy,
        *,
        expected_requested_motion_radians: float,
        expected_target_joint_positions: tuple[float, ...],
        expected_target_gripper_width_meters: float,
        expected_gripper_error_meters: float,
    ) -> None:
        self.trace.validate_failed_attempt(
            policy,
            expected_requested_motion_radians=expected_requested_motion_radians,
            expected_target_joint_positions=expected_target_joint_positions,
            gripper=self.gripper.trace,
        )
        self.gripper.validate(
            expected_target_gripper_width_meters,
            expected_gripper_error_meters,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.trace.to_dict(),
            "gripper_settlement": self.gripper.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> GripperTrackedJointSettlementAttempt:
        if not isinstance(payload, dict) or "gripper_settlement" not in payload:
            raise ValueError("gripper settlement attempt is incomplete")
        trace = JointSettlementTrace.from_dict(payload)
        return cls(
            trace.requested_joint_motion_radians,
            trace.required_tracking_error_radians,
            trace.tracking_errors_radians,
            trace.final_joint_positions,
            GripperSettlementMeasurement.from_dict(
                payload["gripper_settlement"]
            ),
        )
