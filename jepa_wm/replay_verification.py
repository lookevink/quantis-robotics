"""Simulator-independent contract for visualization replay verification."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping


@dataclass(frozen=True)
class ReplayLimits:
    maximum_arm_error_rad: float = 0.01
    maximum_gripper_error_m: float = 0.003
    maximum_contact_force_newtons: float = 2.0

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                self.maximum_arm_error_rad,
                self.maximum_gripper_error_m,
                self.maximum_contact_force_newtons,
            )
        ):
            raise ValueError("replay limits are invalid")

    def to_dict(self) -> dict[str, float]:
        return {
            "maximum_arm_error_rad": self.maximum_arm_error_rad,
            "maximum_gripper_error_m": self.maximum_gripper_error_m,
            "maximum_contact_force_newtons": self.maximum_contact_force_newtons,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayLimits:
        try:
            return cls(
                maximum_arm_error_rad=float(payload["maximum_arm_error_rad"]),
                maximum_gripper_error_m=float(payload["maximum_gripper_error_m"]),
                maximum_contact_force_newtons=float(
                    payload["maximum_contact_force_newtons"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("replay limits are incomplete") from error


@dataclass(frozen=True)
class ReplayVerification:
    maximum_arm_error_rad: float
    maximum_gripper_error_m: float
    maximum_contact_force_newtons: float
    collision_detected: bool
    limits: ReplayLimits = ReplayLimits()

    def __post_init__(self) -> None:
        if (
            not all(
                isfinite(value) and value >= 0.0
                for value in (
                    self.maximum_arm_error_rad,
                    self.maximum_gripper_error_m,
                    self.maximum_contact_force_newtons,
                )
            )
            or not isinstance(self.collision_detected, bool)
        ):
            raise ValueError("replay verification is invalid")

    @property
    def tracking_passed(self) -> bool:
        return (
            self.maximum_arm_error_rad <= self.limits.maximum_arm_error_rad
            and self.maximum_gripper_error_m
            <= self.limits.maximum_gripper_error_m
        )

    @property
    def safety_passed(self) -> bool:
        return (
            not self.collision_detected
            and self.maximum_contact_force_newtons
            <= self.limits.maximum_contact_force_newtons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "replay_limits": self.limits.to_dict(),
            "replay_tracking_passed": self.tracking_passed,
            "maximum_replay_joint_error_rad": self.maximum_arm_error_rad,
            "maximum_replay_gripper_error_m": self.maximum_gripper_error_m,
            "replay_safety_passed": self.safety_passed,
            "maximum_replay_contact_force_newtons": (
                self.maximum_contact_force_newtons
            ),
            "replay_collision_detected": self.collision_detected,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ReplayVerification:
        limits_payload = payload.get("replay_limits")
        limits = (
            ReplayLimits.from_dict(limits_payload)
            if isinstance(limits_payload, Mapping)
            else ReplayLimits()
        )
        try:
            instance = cls(
                maximum_arm_error_rad=float(
                    payload["maximum_replay_joint_error_rad"]
                ),
                maximum_gripper_error_m=float(
                    payload["maximum_replay_gripper_error_m"]
                ),
                maximum_contact_force_newtons=float(
                    payload["maximum_replay_contact_force_newtons"]
                ),
                collision_detected=payload["replay_collision_detected"],
                limits=limits,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("replay verification is incomplete") from error
        tracking_claim = payload.get(
            "replay_tracking_passed", payload.get("tracking_passed")
        )
        if (
            tracking_claim is not instance.tracking_passed
            or payload.get("replay_safety_passed", instance.safety_passed)
            is not instance.safety_passed
        ):
            raise ValueError("replay verification claims are invalid")
        return instance
