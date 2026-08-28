"""Shared tracking and safety mechanics for visualization-only Isaac replays."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import MAX_GRIPPER_WIDTH_M
from jepa_wm.replay_verification import ReplayLimits, ReplayVerification
from sim.isaac_control_runtime import read_control_contact
from sim.isaac_demo_runtime import (
    Actuators,
    JointCommand,
    PlugAttachment,
    move_joint_command,
)
from sim.recording import RecordingLabel


def gripper_width_from_closedness(closedness: float) -> float:
    if not isfinite(closedness) or not 0.0 <= closedness <= 1.0:
        raise ValueError("replay gripper closedness is invalid")
    return (1.0 - closedness) * MAX_GRIPPER_WIDTH_M


def command_errors(
    actual: JointCommand, expected: JointCommand
) -> tuple[float, float]:
    return (
        float(np.max(np.abs(actual.arm_positions - expected.arm_positions))),
        abs(actual.gripper_width_m - expected.gripper_width_m),
    )


@dataclass
class ReplayTrackingMonitor:
    maximum_arm_error_rad: float = 0.0
    maximum_gripper_error_m: float = 0.0
    limits: ReplayLimits = ReplayLimits()

    def observe(self, actual: JointCommand, expected: JointCommand) -> None:
        arm_error, gripper_error = command_errors(actual, expected)
        self.maximum_arm_error_rad = max(self.maximum_arm_error_rad, arm_error)
        self.maximum_gripper_error_m = max(
            self.maximum_gripper_error_m, gripper_error
        )
        if (
            arm_error > self.limits.maximum_arm_error_rad
            or gripper_error > self.limits.maximum_gripper_error_m
        ):
            raise RuntimeError("visualization failed replay tracking")


@dataclass
class ReplaySafetyMonitor:
    maximum_contact_force_newtons: float = 0.0
    collision_detected: bool = False
    limits: ReplayLimits = ReplayLimits()

    def observe(self, collision: bool, contact_force_newtons: float) -> None:
        if not isfinite(contact_force_newtons) or contact_force_newtons < 0.0:
            raise RuntimeError("visualization contact reading is invalid")
        self.collision_detected = self.collision_detected or collision
        self.maximum_contact_force_newtons = max(
            self.maximum_contact_force_newtons, contact_force_newtons
        )
        if (
            collision
            or contact_force_newtons > self.limits.maximum_contact_force_newtons
        ):
            raise RuntimeError("visualization encountered unsafe contact")


@dataclass
class ReplayRuntime:
    actuators: Actuators
    attachment: PlugAttachment
    recorder: Any
    sensor: Any
    sample_period_seconds: float
    limits: ReplayLimits = ReplayLimits()
    tracking: ReplayTrackingMonitor = field(init=False)
    safety: ReplaySafetyMonitor = field(init=False)

    def __post_init__(self) -> None:
        if not isfinite(self.sample_period_seconds) or self.sample_period_seconds <= 0:
            raise ValueError("replay sample period is invalid")
        self.tracking = ReplayTrackingMonitor(limits=self.limits)
        self.safety = ReplaySafetyMonitor(limits=self.limits)

    def observe(self, expected: JointCommand) -> JointCommand:
        actual = self.actuators.actual_command()
        self.tracking.observe(actual, expected)
        self.safety.observe(*read_control_contact(self.sensor))
        return actual

    @property
    def verification(self) -> ReplayVerification:
        return ReplayVerification(
            maximum_arm_error_rad=self.tracking.maximum_arm_error_rad,
            maximum_gripper_error_m=self.tracking.maximum_gripper_error_m,
            maximum_contact_force_newtons=(
                self.safety.maximum_contact_force_newtons
            ),
            collision_detected=self.safety.collision_detected,
            limits=self.limits,
        )

    async def transition(
        self,
        start: JointCommand,
        end: JointCommand,
        *,
        frame_count: int,
        phase: RecordingLabel,
        stage: ObservationStage,
    ) -> JointCommand:
        """Replay and verify every rendered command in one smooth transition."""

        if frame_count <= 0:
            raise ValueError("replay frame count must be positive")
        for frame in range(1, frame_count + 1):
            progress = frame / frame_count
            blend = progress * progress * (3.0 - 2.0 * progress)
            target = JointCommand(
                start.arm_positions
                + (end.arm_positions - start.arm_positions) * blend,
                start.gripper_width_m
                + (end.gripper_width_m - start.gripper_width_m) * blend,
            )
            await move_joint_command(
                self.actuators,
                self.actuators.actual_command(),
                target,
                self.attachment,
                frame_count=1,
                phase=phase,
                stage=stage,
                recorder=self.recorder,
                sample_period_seconds=self.sample_period_seconds,
            )
            self.observe(target)
        return self.actuators.actual_command()
