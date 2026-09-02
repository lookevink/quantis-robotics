"""Bounded loaded-drive compensation for attached contact-grasp motion."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Sequence

from jepa_wm.action import DROID_FPS
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.joint_drive import JointDriveBiasCompensation, JointDriveTarget


CONTACT_GRASP_MAXIMUM_DRIVE_BIAS_RADIANS = 0.002


@dataclass(frozen=True)
class ContactGraspDrivePolicy:
    """Apply one measured attached-load bias without bypassing velocity limits."""

    drive_bias_compensation: JointDriveBiasCompensation = (
        JointDriveBiasCompensation(CONTACT_GRASP_MAXIMUM_DRIVE_BIAS_RADIANS)
    )
    control_period_seconds: float = 1.0 / DROID_FPS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.drive_bias_compensation, JointDriveBiasCompensation)
            or not isclose(
                self.drive_bias_compensation.maximum_bias_radians,
                CONTACT_GRASP_MAXIMUM_DRIVE_BIAS_RADIANS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not isclose(
                self.control_period_seconds,
                1.0 / DROID_FPS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("contact-grasp drive policy is invalid")

    def forward_drive_target(
        self,
        desired_joint_positions: Sequence[float],
        desired_gripper_width_meters: float,
        active_drive_target: JointDriveTarget,
        stable_joint_positions: Sequence[float],
        safety_limits: SimulatorSafetyLimits,
    ) -> JointDriveTarget:
        target = JointDriveTarget.for_command(
            self.drive_bias_compensation.compensated_joint_target(
                desired_joint_positions,
                active_drive_target,
                stable_joint_positions,
                safety_limits,
            ),
            desired_gripper_width_meters,
        )
        maximum_motion = (
            safety_limits.maximum_joint_velocity_radians_per_second
            * self.control_period_seconds
        )
        if max(
            abs(target_value - start_value)
            for target_value, start_value in zip(
                target.joint_positions,
                stable_joint_positions,
            )
        ) > maximum_motion:
            raise ValueError("compensated contact-grasp target exceeds velocity gate")
        return target

    def to_dict(self) -> dict[str, object]:
        return {
            "drive_bias_compensation": self.drive_bias_compensation.to_dict(),
            "control_period_seconds": self.control_period_seconds,
        }


CONTACT_GRASP_DRIVE_POLICY = ContactGraspDrivePolicy()
