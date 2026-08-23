"""Seeded, simulator-independent domain exploration plans for JEPA-WM."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random


ARM_JOINTS = 7
DOMAIN_DATASET_ID = "jepa_wm_domain_v1"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    HELD_OUT = "held_out"


class SegmentOutcome(str, Enum):
    STATIONARY = "stationary"
    EXPLORATION = "exploration"
    FAILED_GRASP = "failed_grasp"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class ExplorationTarget:
    arm_offset_radians: tuple[float, ...]
    gripper_width_m: float
    frames: int
    outcome: SegmentOutcome


@dataclass(frozen=True)
class ExplorationPlan:
    seed: int
    split: DatasetSplit
    targets: tuple[ExplorationTarget, ...]
    sample_period_seconds: float
    initial_arm_offset_radians: tuple[float, ...]
    camera_offset_m: tuple[float, float, float]
    scene_offset_m: tuple[float, float, float]
    socket_scale: float
    light_exposure_delta: float

    def metadata(self) -> dict[str, object]:
        return {
            "dataset": DOMAIN_DATASET_ID,
            "split": self.split.value,
            "seed": self.seed,
            "segments": len(self.targets),
            "sample_period_seconds": self.sample_period_seconds,
            "initial_arm_offset_radians": list(self.initial_arm_offset_radians),
            "camera_offset_m": list(self.camera_offset_m),
            "scene_offset_m": list(self.scene_offset_m),
            "socket_scale": self.socket_scale,
            "light_exposure_delta": self.light_exposure_delta,
            "segment_outcomes": [target.outcome.value for target in self.targets],
        }


def _rounded_uniform(generator: random.Random, lower: float, upper: float) -> float:
    return round(generator.uniform(lower, upper), 6)


def validate_sample_times(
    sample_times: tuple[float, ...],
    sample_period_seconds: float,
    *,
    tolerance_seconds: float = 0.002,
) -> None:
    if len(sample_times) < 2:
        raise ValueError("at least two simulation sample times are required")
    if sample_period_seconds <= 0.0 or tolerance_seconds < 0.0:
        raise ValueError("sample period must be positive and tolerance non-negative")
    for previous, current in zip(sample_times, sample_times[1:]):
        delta = current - previous
        if abs(delta - sample_period_seconds) > tolerance_seconds:
            raise ValueError(
                f"simulation sample cadence is invalid: {delta:.6f}s "
                f"differs from {sample_period_seconds:.6f}s"
            )


def build_exploration_plan(seed: int, split: DatasetSplit) -> ExplorationPlan:
    """Build a bounded plan that excites every Franka joint and the gripper."""

    if seed < 0:
        raise ValueError("exploration seed must be non-negative")
    generator = random.Random(seed)
    targets = [
        ExplorationTarget(
            (0.0,) * ARM_JOINTS,
            0.07,
            4,
            SegmentOutcome.STATIONARY,
        )
    ]
    for cycle in range(2):
        joint_order = list(range(ARM_JOINTS))
        generator.shuffle(joint_order)
        for sequence_index, joint_index in enumerate(joint_order):
            offsets = [0.0] * ARM_JOINTS
            magnitude = _rounded_uniform(generator, 0.04, 0.10)
            offsets[joint_index] = magnitude * generator.choice((-1.0, 1.0))
            targets.append(
                ExplorationTarget(
                    tuple(offsets),
                    0.025 if (cycle + sequence_index) % 2 == 0 else 0.07,
                    4,
                    SegmentOutcome.EXPLORATION,
                )
            )
    # Closing at the ready pose is an explicit failed grasp: the plug remains
    # unattached. The final open return is the corresponding recovery segment.
    targets.extend(
        (
            ExplorationTarget(
                (0.0,) * ARM_JOINTS,
                0.025,
                4,
                SegmentOutcome.FAILED_GRASP,
            ),
            ExplorationTarget(
                (0.0,) * ARM_JOINTS,
                0.07,
                4,
                SegmentOutcome.RECOVERY,
            ),
        )
    )
    return ExplorationPlan(
        seed=seed,
        split=split,
        targets=tuple(targets),
        sample_period_seconds=0.25,
        initial_arm_offset_radians=tuple(
            _rounded_uniform(generator, -0.01, 0.01) for _ in range(ARM_JOINTS)
        ),
        camera_offset_m=tuple(
            _rounded_uniform(generator, -0.012, 0.012) for _ in range(3)
        ),
        scene_offset_m=(
            0.0,
            _rounded_uniform(generator, -0.025, 0.025),
            _rounded_uniform(generator, -0.015, 0.015),
        ),
        socket_scale=_rounded_uniform(generator, 0.98, 1.02),
        light_exposure_delta=_rounded_uniform(generator, -0.4, 0.4),
    )
