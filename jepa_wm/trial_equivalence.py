"""State-equivalence policy for reset-controlled simulator trials."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import DroidPose, action_between
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation


@dataclass(frozen=True)
class ResetEquivalenceTolerances:
    maximum_translation_difference_meters: float = 5e-4
    maximum_rotation_difference_radians: float = 3e-3
    maximum_gripper_difference: float = 0.01
    maximum_joint_difference_radians: float = 1e-3
    maximum_reset_contact_force_newtons: float = 0.01
    maximum_plug_position_difference_meters: float = 5e-4

    def __post_init__(self) -> None:
        values = (
            self.maximum_translation_difference_meters,
            self.maximum_rotation_difference_radians,
            self.maximum_gripper_difference,
            self.maximum_joint_difference_radians,
            self.maximum_reset_contact_force_newtons,
            self.maximum_plug_position_difference_meters,
        )
        if not all(isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("reset equivalence tolerances must be finite and nonnegative")


@dataclass(frozen=True)
class TrialResetState:
    pose: DroidPose
    joint_positions: tuple[float, ...]
    collision_detected: bool
    contact_force_newtons: float
    plug_position: tuple[float, ...] | None = None
    plug_attached: bool = False

    def __post_init__(self) -> None:
        if (
            len(self.joint_positions) != 7
            or not all(isfinite(value) for value in self.joint_positions)
            or not isinstance(self.collision_detected, bool)
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
            or (
                self.plug_position is not None
                and (
                    len(self.plug_position) != 3
                    or not all(isfinite(value) for value in self.plug_position)
                )
            )
            or not isinstance(self.plug_attached, bool)
        ):
            raise ValueError("trial reset state is invalid")


@dataclass(frozen=True)
class ControlTrialContext:
    observation: ControlObservation
    reset: TrialResetState
    policy: ControlExecutionPolicy
    reference_recording: str
    seed: int
    previous_session_id: str | None


def validate_reset_equivalence(
    reference: TrialResetState,
    candidate: TrialResetState,
    *,
    tolerances: ResetEquivalenceTolerances = ResetEquivalenceTolerances(),
) -> None:
    delta = action_between(reference.pose, candidate.pose)
    translation = float(np.linalg.norm(delta.values[:3]))
    rotation = float(Rotation.from_euler("xyz", delta.values[3:6]).magnitude())
    if (
        translation > tolerances.maximum_translation_difference_meters
        or rotation > tolerances.maximum_rotation_difference_radians
        or abs(delta.values[6]) > tolerances.maximum_gripper_difference
        or not np.allclose(
            reference.joint_positions,
            candidate.joint_positions,
            rtol=0.0,
            atol=tolerances.maximum_joint_difference_radians,
        )
        or reference.plug_attached != candidate.plug_attached
        or (reference.plug_position is None) != (candidate.plug_position is None)
        or (
            reference.plug_position is not None
            and candidate.plug_position is not None
            and not np.allclose(
                reference.plug_position,
                candidate.plug_position,
                rtol=0.0,
                atol=tolerances.maximum_plug_position_difference_meters,
            )
        )
    ):
        raise ValueError("trials did not start from the same reset state")
    if (
        reference.collision_detected
        or candidate.collision_detected
        or reference.contact_force_newtons
        > tolerances.maximum_reset_contact_force_newtons
        or candidate.contact_force_newtons
        > tolerances.maximum_reset_contact_force_newtons
    ):
        raise ValueError("trial reset contains collision or contact")
