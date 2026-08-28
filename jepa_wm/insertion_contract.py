"""Shared identity, geometry, and capture contract for reach-and-insert."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import dist, isfinite, sqrt
from pathlib import Path
from typing import Any, Mapping

from jepa_wm.action import (
    ActionSelectionBounds,
    DroidAction,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.control_safety import (
    ACTION_SCALES,
    LEGACY_TRACKING_BOUNDED_ACTION_SCALES,
    LEGACY_TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES,
    ORIENTATION_HOLD_ACTION_SCALES,
    TRACKING_BOUNDED_ACTION_SCALES,
    TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES,
)
from jepa_wm.identifiers import validate_safe_identifier
from jepa_wm.insertion_layout import (
    ContactInsertionLayout,
    ContactInsertionSegment,
    ContactInsertionSpan,
)
from jepa_wm.trajectory import (
    RecordedRollout,
    RolloutProtocol,
    RolloutWindow,
    load_rollout_at,
)

INSERTION_TASK_ID = "reach_and_insert"
INSERTION_AXIS = (-1.0, 0.0, 0.0)
MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS = 0.0125
MINIMUM_CURRENT_FOLLOWUP_ACTION_HORIZON = 1
REARWARD_GRASP_OFFSET_METERS = 0.04
KINEMATIC_INSERTION_MODE = "kinematic_scripted_baseline"
CONTACT_AWARE_INSERTION_MODE = "contact_aware_scripted_baseline"
CONNECTOR_CONTACT_SENSOR_ID = "connector_tip"
COMPLIANT_COLLISION_PARTS = ("latch",)


class InsertionTargetOrigin(str, Enum):
    REFERENCE_CONTEXT = "reference_context"
    LIVE_OBSERVATION = "live_observation"


class InsertionLiveTargetMetric(str, Enum):
    EUCLIDEAN_DISTANCE = "euclidean_distance"
    FORWARD_PROJECTION = "forward_projection"


class InsertionProjectionScalePolicy(str, Enum):
    """Persisted compatibility semantics for large translation proposals."""

    LEGACY_POSITIONAL = "legacy_positional"
    TRACKING_BOUNDED = "tracking_bounded"


@dataclass(frozen=True)
class InsertionControlTargetPolicy:
    minimum_translation_meters: float = 5e-4
    orientation_hold_tolerance_radians: float | None = 1.25e-3
    minimum_action_horizon: int = 3
    maximum_action_horizon: int = 12
    camera: str = "wrist"
    action_bounds: ActionSelectionBounds = ActionSelectionBounds(
        minimum_action_norm=0.0
    )
    target_origin: InsertionTargetOrigin = InsertionTargetOrigin.REFERENCE_CONTEXT
    live_target_metric: InsertionLiveTargetMetric = (
        InsertionLiveTargetMetric.FORWARD_PROJECTION
    )
    maximum_full_scale_translation_meters: float | None = (
        MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS
    )
    projection_scale_policy: InsertionProjectionScalePolicy = (
        InsertionProjectionScalePolicy.TRACKING_BOUNDED
    )

    def __post_init__(self) -> None:
        validate_safe_identifier(self.camera)
        if (
            not isfinite(self.minimum_translation_meters)
            or self.minimum_translation_meters <= 0.0
            or (
                self.orientation_hold_tolerance_radians is not None
                and (
                    not isfinite(self.orientation_hold_tolerance_radians)
                    or self.orientation_hold_tolerance_radians <= 0.0
                )
            )
            or isinstance(self.minimum_action_horizon, bool)
            or not isinstance(self.minimum_action_horizon, int)
            or self.minimum_action_horizon <= 0
            or isinstance(self.maximum_action_horizon, bool)
            or not isinstance(self.maximum_action_horizon, int)
            or self.maximum_action_horizon < self.minimum_action_horizon
            or not isinstance(self.target_origin, InsertionTargetOrigin)
            or not isinstance(self.live_target_metric, InsertionLiveTargetMetric)
            or not isinstance(
                self.projection_scale_policy,
                InsertionProjectionScalePolicy,
            )
            or (
                self.maximum_full_scale_translation_meters is not None
                and (
                    not isfinite(self.maximum_full_scale_translation_meters)
                    or self.maximum_full_scale_translation_meters <= 0.0
                )
            )
        ):
            raise ValueError("insertion control target policy is invalid")

    def for_followup(self) -> InsertionControlTargetPolicy:
        """Select follow-up targets from the exact synchronized live pose."""

        return replace(
            self,
            target_origin=InsertionTargetOrigin.LIVE_OBSERVATION,
        )

    def for_current_followup(self) -> InsertionControlTargetPolicy:
        """Apply current fail-closed bounds when deriving a new follow-up."""

        return replace(
            self.for_followup(),
            maximum_full_scale_translation_meters=(
                self.maximum_full_scale_translation_meters
                or MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS
            ),
            projection_scale_policy=(
                InsertionProjectionScalePolicy.TRACKING_BOUNDED
            ),
        )

    def for_adaptive_followup(self) -> InsertionControlTargetPolicy:
        """Retain the first still-ahead live target under current bounds.

        A live controller may still be behind the immediately preceding
        reference frames.  Starting the signed forward search at horizon one
        avoids forcing a farther target whose required progress exceeds the
        bounded command.  This is a distinct generation so persisted current
        policies keep their historical reconstruction semantics.
        """

        return replace(
            self.for_current_followup(),
            minimum_action_horizon=MINIMUM_CURRENT_FOLLOWUP_ACTION_HORIZON,
        )

    def for_legacy_bounded_followup(self) -> InsertionControlTargetPolicy:
        """Reconstruct the brief cap-with-positional-roster generation."""

        return replace(
            self.for_followup(),
            maximum_full_scale_translation_meters=(
                self.maximum_full_scale_translation_meters
                or MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS
            ),
            projection_scale_policy=(
                InsertionProjectionScalePolicy.LEGACY_POSITIONAL
            ),
        )

    def authorizes_followup(self, candidate: InsertionControlTargetPolicy) -> bool:
        """Accept exact historical lineage or the current one-way tightening."""

        return candidate in (
            self.for_followup(),
            self.for_legacy_bounded_followup(),
            self.for_current_followup(),
            self.for_adaptive_followup(),
        )

    def projection_scales(
        self,
        current: DroidPose,
        target: DroidPose | None,
        proposed_action: DroidAction | None = None,
    ) -> tuple[DroidActionScale, ...]:
        """Hold orientation when its target error is below measured resolution."""

        if target is None or self.orientation_hold_tolerance_radians is None:
            return ACTION_SCALES
        relative = action_between(current, target)
        rotation_error = sqrt(sum(value * value for value in relative.values[3:6]))
        scales = (
            ORIENTATION_HOLD_ACTION_SCALES
            if rotation_error <= self.orientation_hold_tolerance_radians
            else ACTION_SCALES
        )
        if (
            proposed_action is not None
            and self.maximum_full_scale_translation_meters is not None
            and sqrt(sum(value * value for value in proposed_action.values[:3]))
            > self.maximum_full_scale_translation_meters
        ):
            if (
                self.projection_scale_policy
                is InsertionProjectionScalePolicy.LEGACY_POSITIONAL
            ):
                return (
                    LEGACY_TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES
                    if scales is ORIENTATION_HOLD_ACTION_SCALES
                    else LEGACY_TRACKING_BOUNDED_ACTION_SCALES
                )
            return (
                TRACKING_BOUNDED_ORIENTATION_HOLD_ACTION_SCALES
                if scales is ORIENTATION_HOLD_ACTION_SCALES
                else TRACKING_BOUNDED_ACTION_SCALES
            )
        return scales

    def select(
        self,
        recording: Path,
        *,
        context_index: int,
        current_pose: DroidPose | None = None,
    ) -> RecordedRollout:
        """Select the first bounded target beyond the configured resolution metric."""

        if (
            self.target_origin is InsertionTargetOrigin.LIVE_OBSERVATION
            and current_pose is None
        ):
            raise ValueError(
                "live insertion target selection requires its observation pose"
            )
        for horizon in range(
            self.minimum_action_horizon,
            self.maximum_action_horizon + 1,
        ):
            rollout = load_rollout_at(
                recording,
                camera=self.camera,
                context_index=context_index,
                protocol=RolloutProtocol(action_horizon=horizon),
                bounds=self.action_bounds,
            )
            if self.target_origin is InsertionTargetOrigin.LIVE_OBSERVATION:
                assert current_pose is not None
                if (
                    self.live_target_metric
                    is InsertionLiveTargetMetric.FORWARD_PROJECTION
                ):
                    reference_delta = tuple(
                        target - context
                        for target, context in zip(
                            rollout.target_pose.values[:3],
                            rollout.context_pose.values[:3],
                        )
                    )
                    reference_distance = sqrt(
                        sum(value * value for value in reference_delta)
                    )
                    translation_meters = (
                        sum(
                            (target - live) * direction
                            for target, live, direction in zip(
                                rollout.target_pose.values[:3],
                                current_pose.values[:3],
                                reference_delta,
                            )
                        )
                        / reference_distance
                        if reference_distance > 0.0
                        else 0.0
                    )
                else:
                    translation_meters = dist(
                        current_pose.values[:3],
                        rollout.target_pose.values[:3],
                    )
            else:
                translation_meters = dist(
                    rollout.context_pose.values[:3],
                    rollout.target_pose.values[:3],
                )
            if translation_meters >= self.minimum_translation_meters:
                return rollout
        raise ValueError(
            "insertion control target has no resolvable bounded-horizon goal"
        )

    def validate_observation(
        self,
        observation: ControlObservation,
        recording: Path,
        *,
        frame_root: Path,
    ) -> None:
        rollout = self.select(
            recording,
            context_index=observation.warmup_frames,
            current_pose=observation.pose,
        )
        expected = ControlTarget(
            rollout.target.path.relative_to(frame_root.resolve()),
            rollout.target_pose,
        )
        if observation.target != expected:
            raise ValueError("insertion control observation target is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "minimum_translation_meters": self.minimum_translation_meters,
            "minimum_action_horizon": self.minimum_action_horizon,
            "maximum_action_horizon": self.maximum_action_horizon,
            "camera": self.camera,
            "action_bounds": self.action_bounds.to_dict(),
            "target_origin": self.target_origin.value,
            "live_target_metric": self.live_target_metric.value,
            "maximum_full_scale_translation_meters": (
                self.maximum_full_scale_translation_meters
            ),
            "projection_scale_policy": self.projection_scale_policy.value,
        }
        if self.orientation_hold_tolerance_radians is not None:
            payload["orientation_hold_tolerance_radians"] = (
                self.orientation_hold_tolerance_radians
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionControlTargetPolicy:
        if not isinstance(payload, Mapping):
            raise ValueError("insertion control target policy must be an object")
        minimum_horizon = payload.get("minimum_action_horizon")
        maximum_horizon = payload.get("maximum_action_horizon")
        camera = payload.get("camera")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (minimum_horizon, maximum_horizon)
        ) or not isinstance(camera, str):
            raise ValueError("insertion target policy fields are invalid")
        try:
            return cls(
                minimum_translation_meters=float(
                    payload["minimum_translation_meters"]
                ),
                orientation_hold_tolerance_radians=(
                    float(payload["orientation_hold_tolerance_radians"])
                    if "orientation_hold_tolerance_radians" in payload
                    else None
                ),
                minimum_action_horizon=minimum_horizon,
                maximum_action_horizon=maximum_horizon,
                camera=camera,
                action_bounds=ActionSelectionBounds.from_dict(
                    payload["action_bounds"]
                ),
                target_origin=InsertionTargetOrigin(
                    payload.get(
                        "target_origin",
                        InsertionTargetOrigin.REFERENCE_CONTEXT.value,
                    )
                ),
                live_target_metric=InsertionLiveTargetMetric(
                    payload.get(
                        "live_target_metric",
                        InsertionLiveTargetMetric.EUCLIDEAN_DISTANCE.value,
                    )
                ),
                maximum_full_scale_translation_meters=(
                    float(payload["maximum_full_scale_translation_meters"])
                    if payload.get("maximum_full_scale_translation_meters")
                    is not None
                    else None
                ),
                projection_scale_policy=InsertionProjectionScalePolicy(
                    payload.get(
                        "projection_scale_policy",
                        InsertionProjectionScalePolicy.LEGACY_POSITIONAL.value,
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion control target policy is incomplete") from error


INSERTION_CONTROL_TARGET_POLICY = InsertionControlTargetPolicy()


def insertion_control_target_policy(
    execution_policy: ControlExecutionPolicy,
) -> InsertionControlTargetPolicy | None:
    return (
        INSERTION_CONTROL_TARGET_POLICY
        if execution_policy
        in (
            ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            ControlExecutionPolicy.INSERTION_RESET_TRIAL,
            ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
        )
        else None
    )


@dataclass(frozen=True)
class ContactInsertionRecordingContract(ContactInsertionLayout):
    attachment: str = "dynamic_fixed_joint"
    socket_scale: float = 1.05

    @property
    def insertion_command_window(self) -> RolloutWindow:
        return RolloutWindow(
            self.start_index(ContactInsertionSegment.INSERT) - 1,
            self.insertion_steps,
            1,
        )

    def instrumentation_metadata(
        self,
        compliant_collision_parts: tuple[str, ...],
    ) -> dict[str, Any]:
        return {
            "connector_collisions_enabled": True,
            "contact_sensor": CONNECTOR_CONTACT_SENSOR_ID,
            "compliant_collision_parts": list(compliant_collision_parts),
            "attachment": self.attachment,
            "socket_scale": self.socket_scale,
            "insertion_steps": self.insertion_steps,
            "expected_frames": self.frame_count,
        }

    def validate_instrumentation(self, payload: Mapping[str, Any]) -> None:
        expected = self.instrumentation_metadata(COMPLIANT_COLLISION_PARTS)
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("contact-aware insertion instrumentation is incomplete")


CONTACT_INSERTION_RECORDING = ContactInsertionRecordingContract()
