"""Fail-closed simulator safety policy for one proposed control action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import dist, isfinite, sqrt
from time import time
from typing import Any, Mapping, Sequence

from jepa_wm.action import (
    MAX_GRIPPER_WIDTH_M,
    DroidAction,
    DroidActionScale,
    DroidPose,
)
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_tracking import ActionTrackingLimits
from jepa_wm.planner import PlannerActionBounds


FRANKA_LOWER_JOINT_LIMITS = (
    -2.8973,
    -1.7628,
    -2.8973,
    -3.0718,
    -2.8973,
    -0.0175,
    -2.8973,
)
FRANKA_UPPER_JOINT_LIMITS = (
    2.8973,
    1.7628,
    2.8973,
    -0.0698,
    2.8973,
    3.7525,
    2.8973,
)

# Historical ordered projections remain readable in persisted shadow evidence.
LEGACY_ACTION_SCALES = (
    DroidActionScale(1.0, 0.25, 0.25),
    DroidActionScale(0.5, 0.125, 0.125),
    DroidActionScale.uniform(0.25),
    DroidActionScale.uniform(0.125),
)

# Preserve bounded gripper intent while independently reducing pose motion for IK.
# Every candidate still passes the complete simulator safety gate below.
ACTION_SCALES = (
    DroidActionScale(1.0, 0.25, 1.0),
    DroidActionScale(0.5, 0.125, 1.0),
    *LEGACY_ACTION_SCALES,
)

# Contact acquisition is a receding-horizon close, not a one-shot replay of the
# reference gripper delta. Keep small proposals above the measured translation
# noise floor, while bounding larger proposals to a one-millimetre command.
MAXIMUM_CONTACT_GRASP_TRANSLATION_COMMAND_METERS = 0.001
LEGACY_MAXIMUM_CONTACT_GRASP_COARSE_TRANSLATION_COMMAND_METERS = 0.005
MAXIMUM_CONTACT_GRASP_COARSE_TRANSLATION_COMMAND_METERS = 0.002
MAXIMUM_CONTACT_GRASP_FINE_CLOSURE_COMMAND_METERS = 0.0015
MINIMUM_CONTACT_GRASP_TRANSPORT_COMMAND_METERS = 0.0005
MAXIMUM_CONTACT_GRASP_TRANSPORT_COMMAND_METERS = 0.00075
CONTACT_GRASP_ACTION_SCALES = (
    DroidActionScale(0.25, 0.125, 0.125),
    DroidActionScale.uniform(0.125),
    DroidActionScale(0.0625, 0.125, 0.125),
    DroidActionScale(0.03125, 0.125, 0.125),
)
CONTACT_GRASP_FINE_ACTION_SCALES = (
    DroidActionScale.uniform(0.125),
    DroidActionScale(0.0625, 0.125, 0.125),
    DroidActionScale(0.03125, 0.125, 0.125),
)
CONTACT_GRASP_ULTRAFINE_ACTION_SCALES = (
    DroidActionScale(0.0625, 0.125, 0.125),
    DroidActionScale(0.03125, 0.125, 0.125),
)
# At the controller's minimum translation scale, fall back to orientation hold.
# Lula can occasionally return a remote arm branch for a tiny Cartesian command
# near the wrist's pitch singularity.  A half-scale rotation landed on opposite
# sides of the measured one-milliradian activity boundary and was not direction-
# observable.  The ordinary velocity gate rejects the remote branch, then this
# policy preserves translation and gripper intent without asking the drive to
# realize an ambiguous sub-resolution turn.
CONTACT_GRASP_MICRO_ACTION_SCALES = (
    DroidActionScale(0.03125, 0.125, 0.125),
    DroidActionScale(0.03125, 0.0, 0.125),
)
CONTACT_GRASP_ACTION_SCALE_LEVELS = (
    CONTACT_GRASP_ACTION_SCALES,
    CONTACT_GRASP_FINE_ACTION_SCALES,
    CONTACT_GRASP_ULTRAFINE_ACTION_SCALES,
    CONTACT_GRASP_MICRO_ACTION_SCALES,
)

# The authenticated acquisition trajectory contains native 4-FPS approach
# steps up to 6.81 mm. Before the demonstrated close phase, permit at most a
# 2 mm arm command and retain every smaller scale as an IK/velocity fallback.
# A guarded 4.15 mm command moved in the correct direction but missed its
# Cartesian target by 0.97 mm; quarter scale projects safely inside the
# unchanged 0.5 mm tracking gate, while half scale leaves only 15 micrometres.
# Gripper scaling remains independent, so a harmless opening correction can no
# longer collapse an otherwise resolvable arm command to 1/32 scale.
CONTACT_GRASP_COARSE_ACTION_SCALES = tuple(
    DroidActionScale(translation, 0.125, 0.125)
    for translation in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
) + (DroidActionScale(0.03125, 0.0, 0.125),)
CONTACT_GRASP_COARSE_ACTION_SCALE_POLICIES = tuple(
    CONTACT_GRASP_COARSE_ACTION_SCALES[index:]
    for index in range(len(CONTACT_GRASP_COARSE_ACTION_SCALES) - 1)
)

# A direction cosine is not observable when the permitted Cartesian tracking
# error is comparable to the commanded turn.  The unchanged 3 mrad rotation
# error and 0.5 cosine gates require a command of at least 3.464 mrad for the
# direction cone to remain meaningful.  Round upward rather than asking Isaac
# to distinguish a boundary-sized turn from drive/IK residuals.
MINIMUM_DIRECTION_OBSERVABLE_ROTATION_RADIANS = 3.5e-3
CONTACT_GRASP_ROTATION_SCALE_LEVELS = (0.125, 0.25, 0.5, 1.0)


def _rotation_resolved_policy(
    policy: Sequence[DroidActionScale],
    action: DroidAction,
    *,
    required: bool,
) -> tuple[DroidActionScale, ...]:
    """Use one observable learned turn, or explicitly hold orientation."""

    result = tuple(policy)
    if not required:
        return result
    rotation_norm = sqrt(sum(value * value for value in action.values[3:6]))
    rotation_scale = next(
        (
            scale
            for scale in CONTACT_GRASP_ROTATION_SCALE_LEVELS
            if rotation_norm * scale
            >= MINIMUM_DIRECTION_OBSERVABLE_ROTATION_RADIANS
        ),
        0.0,
    )
    return tuple(
        DroidActionScale(
            scale.translation,
            rotation_scale if scale.rotation > 0.0 else 0.0,
            scale.gripper,
        )
        for scale in result
    )

# Once the fixed-joint attachment is live, gripper intent no longer controls
# the approach scale. Preserve the authenticated active gripper target and let
# the learned Cartesian transport use up to a 0.75-millimetre command; every
# candidate still passes the ordinary IK, joint-delta, velocity, and workspace
# gates before a drive target is written.
CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES = tuple(
    tuple(DroidActionScale(scale.translation, scale.rotation, 0.0) for scale in policy)
    for policy in CONTACT_GRASP_ACTION_SCALE_LEVELS
)

# This directional first-step roster remains reconstruction-only for v2
# evidence. It made direction observable but predates the controller-resolution
# floor enforced by the current horizon-aware contract.
DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALES = tuple(
    DroidActionScale(translation, 0.125, 0.0)
    for translation in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
)
DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES = tuple(
    DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALES[index:]
    for index in range(len(DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALES))
)


def resolvable_contact_grasp_transport_action_scales(
    translation_norm: float,
) -> tuple[DroidActionScale, ...]:
    """Project a horizon into three ordered commands inside the measured band."""

    if (
        not isfinite(translation_norm)
        or translation_norm < MINIMUM_CONTACT_GRASP_TRANSPORT_COMMAND_METERS
    ):
        raise ValueError(
            "contact-grasp transport proposal is below controller resolution"
        )
    maximum_scale = min(
        1.0,
        MAXIMUM_CONTACT_GRASP_TRANSPORT_COMMAND_METERS / translation_norm,
    )
    minimum_scale = min(
        1.0,
        MINIMUM_CONTACT_GRASP_TRANSPORT_COMMAND_METERS / translation_norm,
    )
    scales = tuple(
        dict.fromkeys(
            (maximum_scale, (maximum_scale + minimum_scale) / 2.0, minimum_scale)
        )
    )
    return tuple(DroidActionScale(scale, 0.125, 0.0) for scale in scales)

# The demonstration closes from 44 mm to 18 mm across the contact window. The
# live controller preserves that learned direction but must keep Cartesian and
# gripper motion independently bounded. These rosters use the 0.25 gripper
# scale only when that produces at most a 1.5 mm width command; larger closure
# remains on the exercised 0.125 scale.
CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES = tuple(
    tuple(DroidActionScale(scale.translation, scale.rotation, 0.25) for scale in policy)
    for policy in CONTACT_GRASP_ACTION_SCALE_LEVELS
)

# Large positive closedness proposals still represent a receding-horizon
# grasp, not authority to close the whole recorded contact window at once.
# Reduce only the gripper component until the physical width command is within
# the same 1.5 mm bound; Cartesian approach scales and all ordinary gates stay
# unchanged.
CONTACT_GRASP_REDUCED_CLOSING_ACTION_SCALE_POLICIES = tuple(
    tuple(
        tuple(
            DroidActionScale(scale.translation, scale.rotation, gripper_scale)
            for scale in policy
        )
        for gripper_scale in (0.125, 0.0625, 0.03125, 0.015625)
    )
    for policy in CONTACT_GRASP_ACTION_SCALE_LEVELS
)


def _with_gripper_scale(
    policy: Sequence[DroidActionScale],
    gripper_scale: float,
) -> tuple[DroidActionScale, ...]:
    return tuple(
        DroidActionScale(scale.translation, scale.rotation, gripper_scale)
        for scale in policy
    )


def _bounded_closing_policy(
    policy: tuple[DroidActionScale, ...],
    gripper_delta: float,
) -> tuple[DroidActionScale, ...]:
    for gripper_scale in (0.25, 0.125, 0.0625, 0.03125, 0.015625):
        candidate = _with_gripper_scale(policy, gripper_scale)
        if (
            gripper_delta
            * gripper_scale
            * MAX_GRIPPER_WIDTH_M
            <= MAXIMUM_CONTACT_GRASP_FINE_CLOSURE_COMMAND_METERS
        ):
            return candidate
    return _with_gripper_scale(policy, 0.015625)


def _exact_coarse_translation_policy(
    translation_norm: float,
    coarse_limit: float,
    minimum_command: float | None = None,
    orientation_hold_fallback: bool = False,
) -> tuple[DroidActionScale, ...]:
    """Fill the safe coarse bound, retaining ordered IK fallbacks."""

    maximum_scale = min(1.0, coarse_limit / max(translation_norm, 1e-12))
    minimum_scale = min(
        maximum_scale,
        (
            minimum_command / max(translation_norm, 1e-12)
            if minimum_command is not None
            else 0.03125
        ),
    )
    translations = tuple(
        dict.fromkeys(
            (
                maximum_scale,
                maximum_scale / 2.0,
                maximum_scale / 4.0,
                maximum_scale / 8.0,
                maximum_scale / 16.0,
                minimum_scale,
            )
        )
    )
    scales = tuple(
        DroidActionScale(translation, 0.125, 0.125)
        for translation in translations
        if translation >= minimum_scale
    )
    hold = DroidActionScale(maximum_scale, 0.0, 0.125)
    ordered = (scales[0], hold, *scales[1:]) if orientation_hold_fallback else scales
    final_hold = DroidActionScale(minimum_scale, 0.0, 0.125)
    return (*ordered, *((final_hold,) if final_hold not in ordered else ()))


# These policies were exercised by earlier guarded contact-grasp checkpoints.
# They remain reconstruction-only so their persisted negative evidence stays
# readable after the magnitude-aware policy was introduced.
LEGACY_CONTACT_GRASP_ACTION_SCALE_POLICIES = (
    (
        DroidActionScale(0.5, 0.125, 0.25),
        DroidActionScale.uniform(0.25),
        DroidActionScale.uniform(0.125),
    ),
    (
        DroidActionScale(0.375, 0.125, 0.25),
        DroidActionScale.uniform(0.25),
        DroidActionScale.uniform(0.125),
    ),
    (
        DroidActionScale(0.25, 0.125, 0.25),
        DroidActionScale.uniform(0.25),
        DroidActionScale.uniform(0.125),
    ),
    (
        DroidActionScale(0.25, 0.125, 0.125),
        DroidActionScale.uniform(0.25),
        DroidActionScale.uniform(0.125),
    ),
    (
        DroidActionScale.uniform(0.125),
        DroidActionScale.uniform(0.25),
        DroidActionScale.uniform(0.125),
    ),
)


def contact_grasp_action_scales(
    action: DroidAction,
    *,
    attachment_acquired: bool = False,
    require_directional_transport_progress: bool = False,
    require_resolvable_transport: bool = False,
    coarse_acquisition: bool = False,
    maximum_coarse_translation_command_meters: float | None = None,
    require_resolvable_rotation: bool = False,
    exact_coarse_translation_projection: bool = False,
    coarse_orientation_hold_fallback: bool = False,
    minimum_coarse_translation_command_meters: float | None = None,
    resolution_floored_acquisition: bool = False,
    maximum_resolution_floored_translation_command_meters: float | None = None,
) -> tuple[DroidActionScale, ...]:
    """Bound approach motion and calibrate gripper closure independently."""

    if (
        not isinstance(coarse_acquisition, bool)
        or not isinstance(require_resolvable_rotation, bool)
        or not isinstance(exact_coarse_translation_projection, bool)
        or not isinstance(coarse_orientation_hold_fallback, bool)
        or not isinstance(resolution_floored_acquisition, bool)
    ):
        raise ValueError("contact-grasp acquisition scale phase is invalid")
    if minimum_coarse_translation_command_meters is not None and (
        isinstance(minimum_coarse_translation_command_meters, bool)
        or not isfinite(minimum_coarse_translation_command_meters)
        or minimum_coarse_translation_command_meters <= 0.0
    ):
        raise ValueError("contact-grasp acquisition resolution floor is invalid")
    if maximum_coarse_translation_command_meters is not None and (
        isinstance(maximum_coarse_translation_command_meters, bool)
        or not isfinite(maximum_coarse_translation_command_meters)
        or maximum_coarse_translation_command_meters <= 0.0
    ):
        raise ValueError("contact-grasp acquisition translation limit is invalid")
    if maximum_resolution_floored_translation_command_meters is not None and (
        isinstance(maximum_resolution_floored_translation_command_meters, bool)
        or not isfinite(maximum_resolution_floored_translation_command_meters)
        or maximum_resolution_floored_translation_command_meters <= 0.0
    ):
        raise ValueError("contact-grasp acquisition translation limit is invalid")
    if (
        minimum_coarse_translation_command_meters is not None
        and maximum_coarse_translation_command_meters is not None
        and minimum_coarse_translation_command_meters
        > maximum_coarse_translation_command_meters
    ):
        raise ValueError("contact-grasp acquisition resolution floor is invalid")
    if (
        minimum_coarse_translation_command_meters is not None
        and maximum_resolution_floored_translation_command_meters is not None
        and minimum_coarse_translation_command_meters
        > maximum_resolution_floored_translation_command_meters
    ):
        raise ValueError("contact-grasp acquisition resolution floor is invalid")
    translation_norm = sqrt(sum(value * value for value in action.values[:3]))
    if attachment_acquired:
        if require_resolvable_transport:
            if not require_directional_transport_progress:
                raise ValueError(
                    "resolvable contact-grasp transport must be directional"
                )
            return _rotation_resolved_policy(
                resolvable_contact_grasp_transport_action_scales(
                    translation_norm
                ),
                action,
                required=require_resolvable_rotation,
            )
        if (
            require_directional_transport_progress
            and translation_norm
            < ActionTrackingLimits().translation_activity_meters
        ):
            raise ValueError(
                "contact-grasp transport proposal is below tracking activity"
            )
        policies = (
            DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES
            if require_directional_transport_progress
            else CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES
        )
        for scales in policies:
            if (
                translation_norm * scales[0].translation
                <= MAXIMUM_CONTACT_GRASP_TRANSPORT_COMMAND_METERS
            ):
                return _rotation_resolved_policy(
                    scales,
                    action,
                    required=require_resolvable_rotation,
                )
        return _rotation_resolved_policy(
            policies[-1],
            action,
            required=require_resolvable_rotation,
        )

    # A current close-phase reopening is a gripper correction with incidental
    # Cartesian drift, not an approach command. Preserve the exercised micro
    # roster instead of enlarging that drift to the acquisition floor. The
    # historical coarse policies retain their versioned behavior below.
    if resolution_floored_acquisition and action.values[6] < 0.0:
        return _rotation_resolved_policy(
            CONTACT_GRASP_MICRO_ACTION_SCALES,
            action,
            required=require_resolvable_rotation,
        )

    if coarse_acquisition or resolution_floored_acquisition:
        coarse_limit = (
            maximum_resolution_floored_translation_command_meters
            if (
                resolution_floored_acquisition
                and not coarse_acquisition
                and maximum_resolution_floored_translation_command_meters
                is not None
            )
            else maximum_coarse_translation_command_meters
            if maximum_coarse_translation_command_meters is not None
            else MAXIMUM_CONTACT_GRASP_COARSE_TRANSLATION_COMMAND_METERS
        )
        if (
            minimum_coarse_translation_command_meters is not None
            and translation_norm < minimum_coarse_translation_command_meters
        ):
            raise ValueError(
                "contact-grasp coarse proposal is below controller resolution"
            )
        policies = CONTACT_GRASP_COARSE_ACTION_SCALE_POLICIES
        if exact_coarse_translation_projection:
            policies = (
                _exact_coarse_translation_policy(
                    translation_norm,
                    coarse_limit,
                    minimum_coarse_translation_command_meters,
                    coarse_orientation_hold_fallback,
                ),
            )
        for policy in policies:
            if (
                translation_norm * policy[0].translation
                <= coarse_limit
            ):
                selected = (
                    _bounded_closing_policy(policy, action.values[6])
                    if action.values[6] > 0.0
                    else policy
                )
                return _rotation_resolved_policy(
                    selected,
                    action,
                    required=require_resolvable_rotation,
                )
        return _rotation_resolved_policy(
            policies[-1],
            action,
            required=require_resolvable_rotation,
        )

    # Negative DROID gripper action decreases closedness and therefore opens the
    # fingers. A live post-contact opening request jumped from 0.509 mm back to
    # 0.943 mm as its raw norm crossed the magnitude boundary; realization
    # remained 0.298 mm and the unchanged tracking gate correctly rolled it
    # back. Reopening must not re-enlarge the approach step.
    if action.values[6] < 0.0:
        return _rotation_resolved_policy(
            CONTACT_GRASP_MICRO_ACTION_SCALES,
            action,
            required=require_resolvable_rotation,
        )

    for index, scales in enumerate(CONTACT_GRASP_ACTION_SCALE_LEVELS):
        if (
            translation_norm * scales[0].translation
            <= MAXIMUM_CONTACT_GRASP_TRANSLATION_COMMAND_METERS
        ):
            if action.values[6] <= 0.0:
                return _rotation_resolved_policy(
                    scales,
                    action,
                    required=require_resolvable_rotation,
                )
            closing_policies = (
                CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES[index],
                *CONTACT_GRASP_REDUCED_CLOSING_ACTION_SCALE_POLICIES[index],
            )
            for closing_policy in closing_policies:
                if (
                    action.values[6]
                    * closing_policy[0].gripper
                    * MAX_GRIPPER_WIDTH_M
                    <= MAXIMUM_CONTACT_GRASP_FINE_CLOSURE_COMMAND_METERS
                ):
                    return _rotation_resolved_policy(
                        closing_policy,
                        action,
                        required=require_resolvable_rotation,
                    )
            return _rotation_resolved_policy(
                closing_policies[-1],
                action,
                required=require_resolvable_rotation,
            )
    return _rotation_resolved_policy(
        CONTACT_GRASP_MICRO_ACTION_SCALES,
        action,
        required=require_resolvable_rotation,
    )


ORIENTATION_HOLD_ACTION_SCALES = tuple(
    DroidActionScale(scale.translation, 0.0, scale.gripper) for scale in ACTION_SCALES
)
# One guarded retry briefly persisted these positional slices before the scale
# roster was made genuinely magnitude-bounded. Retain them for reconstruction
# only; current target policies never return either tuple.
LEGACY_TRACKING_BOUNDED_ACTION_SCALES = ACTION_SCALES[1:]
LEGACY_TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES = ORIENTATION_HOLD_ACTION_SCALES[
    1:
]
TRACKING_BOUNDED_ACTION_SCALES = (
    DroidActionScale(0.75, 0.1875, 0.75),
    *(scale for scale in ACTION_SCALES if scale.translation <= 0.5),
)
TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES = (
    DroidActionScale(0.75, 0.0, 0.75),
    *(scale for scale in ORIENTATION_HOLD_ACTION_SCALES if scale.translation <= 0.5),
)

ACTION_SCALE_POLICIES = (ACTION_SCALES, LEGACY_ACTION_SCALES)
INSERTION_ACTION_SCALE_POLICIES = (
    *ACTION_SCALE_POLICIES,
    *CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES,
    *(
        policy
        for policies in CONTACT_GRASP_REDUCED_CLOSING_ACTION_SCALE_POLICIES
        for policy in policies
    ),
    *CONTACT_GRASP_ACTION_SCALE_LEVELS,
    *CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES,
    *DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES,
    *LEGACY_CONTACT_GRASP_ACTION_SCALE_POLICIES,
    ORIENTATION_HOLD_ACTION_SCALES,
    LEGACY_TRACKING_BOUNDED_ACTION_SCALES,
    LEGACY_TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES,
    TRACKING_BOUNDED_ACTION_SCALES,
    TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES,
)


def _projection_policy_for_attempts(
    attempted_scales: Sequence[DroidActionScale],
    policies: Sequence[tuple[DroidActionScale, ...]],
) -> tuple[DroidActionScale, ...]:
    scales = tuple(attempted_scales)
    for policy in policies:
        if scales and scales == policy[: len(scales)]:
            return policy
    raise ValueError("safety projection order is invalid")


def projection_policy_for_attempts(
    attempted_scales: Sequence[DroidActionScale],
) -> tuple[DroidActionScale, ...]:
    """Return the current or historical policy matching an ordered attempt prefix."""

    return _projection_policy_for_attempts(attempted_scales, ACTION_SCALE_POLICIES)


def insertion_projection_policy_for_attempts(
    attempted_scales: Sequence[DroidActionScale],
) -> tuple[DroidActionScale, ...]:
    """Return an insertion policy matching an ordered attempt prefix."""

    return _projection_policy_for_attempts(
        attempted_scales,
        INSERTION_ACTION_SCALE_POLICIES,
    )


def insertion_projection_policy_for_scale(
    scale: DroidActionScale,
) -> tuple[DroidActionScale, ...]:
    """Return the insertion projection policy containing one selected scale."""

    for policy in INSERTION_ACTION_SCALE_POLICIES:
        if scale in policy:
            return policy
    raise ValueError("safety projection scale is invalid")


def _finite_tuple(name: str, values: Sequence[float], count: int) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != count or not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain {count} finite values")
    return result


@dataclass(frozen=True)
class ControlInterlockEvidence:
    """Peak collision/contact evidence retained across one execution interval."""

    maximum_contact_force_newtons: float
    collision_detected: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.maximum_contact_force_newtons, bool)
            or not isfinite(self.maximum_contact_force_newtons)
            or self.maximum_contact_force_newtons < 0.0
            or not isinstance(self.collision_detected, bool)
        ):
            raise ValueError("control interlock evidence is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum_contact_force_newtons": self.maximum_contact_force_newtons,
            "collision_detected": self.collision_detected,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ControlInterlockEvidence:
        if not isinstance(payload, dict):
            raise ValueError("control interlock evidence must be an object")
        force = payload.get("maximum_contact_force_newtons")
        collision = payload.get("collision_detected")
        if (
            isinstance(force, bool)
            or not isinstance(force, (int, float))
            or not isinstance(collision, bool)
        ):
            raise ValueError("control interlock evidence is incomplete")
        return cls(float(force), collision)


@dataclass(frozen=True)
class SimulatorSafetyState:
    observed_joint_positions: tuple[float, ...]
    current_joint_positions: tuple[float, ...]
    proposed_joint_positions: tuple[float, ...]
    control_period_seconds: float
    contact_force_newtons: float
    collision_detected: bool

    def __post_init__(self) -> None:
        for field in (
            "observed_joint_positions",
            "current_joint_positions",
            "proposed_joint_positions",
        ):
            object.__setattr__(
                self, field, _finite_tuple(field, getattr(self, field), 7)
            )
        if (
            not isfinite(self.control_period_seconds)
            or self.control_period_seconds <= 0.0
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons < 0.0
        ):
            raise ValueError("simulator safety telemetry is invalid")


@dataclass(frozen=True)
class SimulatorSafetyLimits:
    minimum_workspace_xyz: tuple[float, ...] = (-0.10, -0.60, 0.00)
    maximum_workspace_xyz: tuple[float, ...] = (0.85, 0.60, 1.20)
    lower_joint_limits: tuple[float, ...] = FRANKA_LOWER_JOINT_LIMITS
    upper_joint_limits: tuple[float, ...] = FRANKA_UPPER_JOINT_LIMITS
    maximum_joint_velocity_radians_per_second: float = 0.5
    maximum_observation_joint_drift_radians: float = 0.002
    maximum_observation_plug_drift_meters: float = 0.002
    maximum_contact_force_newtons: float = 2.0
    maximum_observation_age_seconds: float = 3.0
    maximum_command_age_seconds: float = 2.5
    minimum_warmup_frames: int = 4
    action_bounds: PlannerActionBounds = PlannerActionBounds()

    def __post_init__(self) -> None:
        minimum = _finite_tuple("workspace minimum", self.minimum_workspace_xyz, 3)
        maximum = _finite_tuple("workspace maximum", self.maximum_workspace_xyz, 3)
        lower = _finite_tuple("joint lower limits", self.lower_joint_limits, 7)
        upper = _finite_tuple("joint upper limits", self.upper_joint_limits, 7)
        if any(left >= right for left, right in zip(minimum, maximum)):
            raise ValueError("workspace minimum must be below its maximum")
        if any(left >= right for left, right in zip(lower, upper)):
            raise ValueError("joint minimum must be below its maximum")
        scalars = (
            self.maximum_joint_velocity_radians_per_second,
            self.maximum_observation_joint_drift_radians,
            self.maximum_observation_plug_drift_meters,
            self.maximum_contact_force_newtons,
            self.maximum_observation_age_seconds,
            self.maximum_command_age_seconds,
        )
        if not all(isfinite(value) and value > 0.0 for value in scalars):
            raise ValueError("simulator safety limits must be finite and positive")
        if self.minimum_warmup_frames < 0:
            raise ValueError("minimum warm-up frames must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "minimum_workspace_xyz": list(self.minimum_workspace_xyz),
            "maximum_workspace_xyz": list(self.maximum_workspace_xyz),
            "lower_joint_limits": list(self.lower_joint_limits),
            "upper_joint_limits": list(self.upper_joint_limits),
            "maximum_joint_velocity_radians_per_second": (
                self.maximum_joint_velocity_radians_per_second
            ),
            "maximum_observation_joint_drift_radians": (
                self.maximum_observation_joint_drift_radians
            ),
            "maximum_observation_plug_drift_meters": (
                self.maximum_observation_plug_drift_meters
            ),
            "maximum_contact_force_newtons": self.maximum_contact_force_newtons,
            "maximum_observation_age_seconds": self.maximum_observation_age_seconds,
            "maximum_command_age_seconds": self.maximum_command_age_seconds,
            "minimum_warmup_frames": self.minimum_warmup_frames,
            "action_bounds": self.action_bounds.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SimulatorSafetyLimits:
        try:
            return cls(
                minimum_workspace_xyz=tuple(payload["minimum_workspace_xyz"]),
                maximum_workspace_xyz=tuple(payload["maximum_workspace_xyz"]),
                lower_joint_limits=tuple(payload["lower_joint_limits"]),
                upper_joint_limits=tuple(payload["upper_joint_limits"]),
                maximum_joint_velocity_radians_per_second=float(
                    payload["maximum_joint_velocity_radians_per_second"]
                ),
                maximum_observation_joint_drift_radians=float(
                    payload["maximum_observation_joint_drift_radians"]
                ),
                maximum_observation_plug_drift_meters=float(
                    payload["maximum_observation_plug_drift_meters"]
                ),
                maximum_contact_force_newtons=float(
                    payload["maximum_contact_force_newtons"]
                ),
                maximum_observation_age_seconds=float(
                    payload["maximum_observation_age_seconds"]
                ),
                maximum_command_age_seconds=float(
                    payload["maximum_command_age_seconds"]
                ),
                minimum_warmup_frames=int(payload["minimum_warmup_frames"]),
                action_bounds=PlannerActionBounds.from_dict(payload["action_bounds"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("simulator safety limits are incomplete") from error


class ControlGateReason(str, Enum):
    STALE_OBSERVATION = "stale_observation"
    OBSERVATION_MISMATCH = "observation_mismatch"
    PROPOSAL_MISMATCH = "proposal_mismatch"
    COMMAND_TIME_INVALID = "command_time_invalid"
    WARMUP_INCOMPLETE = "warmup_incomplete"
    ACTION_OUT_OF_BOUNDS = "action_out_of_bounds"
    WORKSPACE_VIOLATION = "workspace_violation"
    GRIPPER_VIOLATION = "gripper_violation"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"
    OBSERVATION_STATE_DRIFT = "observation_state_drift"
    JOINT_VELOCITY_VIOLATION = "joint_velocity_violation"
    IK_SOLUTION_FAILED = "ik_solution_failed"
    COLLISION_DETECTED = "collision_detected"
    FORCE_LIMIT_EXCEEDED = "force_limit_exceeded"
    TARGET_POSE_MISSING = "target_pose_missing"
    TARGET_PROGRESS_INSUFFICIENT = "target_progress_insufficient"
    DRIVE_TARGET_INVALID = "drive_target_invalid"


@dataclass(frozen=True)
class ProjectedTargetProgressPolicy:
    """Require one projected action to close a useful share of its target gap."""

    minimum_translation_error_reduction_fraction: float = 0.25

    def __post_init__(self) -> None:
        if (
            not isfinite(self.minimum_translation_error_reduction_fraction)
            or not 0.0 < self.minimum_translation_error_reduction_fraction <= 1.0
        ):
            raise ValueError("projected target-progress policy is invalid")

    def failure_reason(
        self,
        current: DroidPose,
        target: DroidPose | None,
        projected: DroidPose,
    ) -> ControlGateReason | None:
        if target is None:
            return ControlGateReason.TARGET_POSE_MISSING
        current_error = dist(current.values[:3], target.values[:3])
        projected_error = dist(projected.values[:3], target.values[:3])
        maximum_projected_error = current_error * (
            1.0 - self.minimum_translation_error_reduction_fraction
        )
        return (
            None
            if projected_error <= maximum_projected_error + 1e-12
            else ControlGateReason.TARGET_PROGRESS_INSUFFICIENT
        )

    def translation_bound_can_satisfy(
        self,
        current: DroidPose,
        target: DroidPose | None,
        maximum_translation_meters: float,
    ) -> bool:
        """Whether an ideally directed bounded translation can pass this gate."""

        if (
            isinstance(maximum_translation_meters, bool)
            or not isinstance(maximum_translation_meters, (int, float))
            or not isfinite(maximum_translation_meters)
            or maximum_translation_meters <= 0.0
        ):
            raise ValueError("target-progress translation bound is invalid")
        if target is None:
            return False
        required_reduction = (
            dist(current.values[:3], target.values[:3])
            * self.minimum_translation_error_reduction_fraction
        )
        return required_reduction <= maximum_translation_meters + 1e-12

    def apply(
        self,
        decision: ControlGateDecision,
        current: DroidPose,
        target: DroidPose | None,
    ) -> ControlGateDecision:
        """Apply target-progress policy without weakening an existing rejection."""
        if not decision.passed:
            return decision
        failure = self.failure_reason(current, target, decision.next_pose)
        return (
            decision
            if failure is None
            else ControlGateDecision(
                decision.observation_id,
                decision.next_pose,
                (failure,),
            )
        )


INSERTION_TARGET_PROGRESS = ProjectedTargetProgressPolicy()


@dataclass(frozen=True)
class ControlGateDecision:
    observation_id: int
    next_pose: DroidPose
    reasons: tuple[ControlGateReason, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons

    @classmethod
    def from_dict(cls, payload: Any) -> ControlGateDecision:
        if not isinstance(payload, dict):
            raise ValueError("control gate decision must be an object")
        try:
            decision = cls(
                observation_id=int(payload["observation_id"]),
                next_pose=DroidPose(tuple(payload["next_pose"])),
                reasons=tuple(
                    ControlGateReason(reason) for reason in payload["reasons"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("control gate decision is incomplete") from error
        if (
            isinstance(decision.observation_id, bool)
            or decision.observation_id <= 0
            or payload.get("passed") is not decision.passed
        ):
            raise ValueError("control gate decision is inconsistent")
        return decision

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "passed": self.passed,
            "next_pose": list(self.next_pose.values),
            "reasons": [reason.value for reason in self.reasons],
        }


@dataclass(frozen=True)
class SafetyProjectionAttempt:
    scale: DroidActionScale
    gate: ControlGateDecision
    maximum_joint_delta_rad: float
    proposed_joint_positions: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not isfinite(self.maximum_joint_delta_rad)
            or self.maximum_joint_delta_rad < 0.0
        ):
            raise ValueError("safety projection evidence is invalid")
        object.__setattr__(
            self,
            "proposed_joint_positions",
            _finite_tuple("proposed joint positions", self.proposed_joint_positions, 7),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.to_dict(),
            "gate": self.gate.to_dict(),
            "maximum_joint_delta_rad": self.maximum_joint_delta_rad,
            "proposed_joint_positions": list(self.proposed_joint_positions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> SafetyProjectionAttempt:
        if not isinstance(payload, dict):
            raise ValueError("safety projection attempt must be an object")
        try:
            return cls(
                DroidActionScale.from_payload(payload["scale"]),
                ControlGateDecision.from_dict(payload["gate"]),
                float(payload["maximum_joint_delta_rad"]),
                tuple(payload["proposed_joint_positions"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("safety projection attempt is incomplete") from error


class SimulatorControlGate:
    def __init__(self, limits: SimulatorSafetyLimits = SimulatorSafetyLimits()):
        self.limits = limits

    def evaluate(
        self,
        observation: ControlObservation,
        proposal: ProposedControl,
        state: SimulatorSafetyState,
        *,
        now_unix_seconds: float | None = None,
    ) -> ControlGateDecision:
        now = time() if now_unix_seconds is None else now_unix_seconds
        reasons = []
        if (
            now < observation.captured_at_unix_seconds
            or now - observation.captured_at_unix_seconds
            > self.limits.maximum_observation_age_seconds
        ):
            reasons.append(ControlGateReason.STALE_OBSERVATION)
        if proposal.observation_id != observation.observation_id:
            reasons.append(ControlGateReason.OBSERVATION_MISMATCH)
        if proposal.proposal != observation.expected_proposal:
            reasons.append(ControlGateReason.PROPOSAL_MISMATCH)
        if (
            proposal.created_at_unix_seconds < observation.captured_at_unix_seconds
            or proposal.created_at_unix_seconds > now
            or now - proposal.created_at_unix_seconds
            > self.limits.maximum_command_age_seconds
        ):
            reasons.append(ControlGateReason.COMMAND_TIME_INVALID)
        if observation.warmup_frames < self.limits.minimum_warmup_frames:
            reasons.append(ControlGateReason.WARMUP_INCOMPLETE)
        if not self.limits.action_bounds.accepts(proposal.actions):
            reasons.append(ControlGateReason.ACTION_OUT_OF_BOUNDS)
        try:
            next_pose = observation.pose.applied(proposal.first_action)
        except ValueError:
            next_pose = observation.pose
            reasons.append(ControlGateReason.GRIPPER_VIOLATION)
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                next_pose.values[:3],
                self.limits.minimum_workspace_xyz,
                self.limits.maximum_workspace_xyz,
            )
        ):
            reasons.append(ControlGateReason.WORKSPACE_VIOLATION)
        if not 0.0 <= next_pose.values[6] <= 1.0:
            reasons.append(ControlGateReason.GRIPPER_VIOLATION)
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                state.proposed_joint_positions,
                self.limits.lower_joint_limits,
                self.limits.upper_joint_limits,
            )
        ):
            reasons.append(ControlGateReason.JOINT_LIMIT_VIOLATION)
        maximum_joint_delta = (
            self.limits.maximum_joint_velocity_radians_per_second
            * state.control_period_seconds
        )
        if any(
            abs(current - observed)
            > self.limits.maximum_observation_joint_drift_radians
            for observed, current in zip(
                state.observed_joint_positions, state.current_joint_positions
            )
        ):
            reasons.append(ControlGateReason.OBSERVATION_STATE_DRIFT)
        if any(
            abs(proposed - current) > maximum_joint_delta
            for current, proposed in zip(
                state.current_joint_positions, state.proposed_joint_positions
            )
        ):
            reasons.append(ControlGateReason.JOINT_VELOCITY_VIOLATION)
        if state.collision_detected:
            reasons.append(ControlGateReason.COLLISION_DETECTED)
        if state.contact_force_newtons > self.limits.maximum_contact_force_newtons:
            reasons.append(ControlGateReason.FORCE_LIMIT_EXCEEDED)
        return ControlGateDecision(
            observation.observation_id, next_pose, tuple(reasons)
        )
