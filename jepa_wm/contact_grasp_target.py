"""Phase-locked reference targets for bounded contact-grasp control."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from jepa_wm.action import (
    ActionSelectionBounds,
    DroidAction,
    DroidPose,
    compose_actions,
    compose_transport_action,
)
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
)
from jepa_wm.task_windows import CONTACT_GRASP_PROPOSAL_WINDOW
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL, RecordedRollout, load_rollouts


LEGACY_CONTACT_GRASP_TARGET_POLICY_SCHEMA = (
    "quantis.jepa_wm_contact_grasp_target_policy.v1"
)
DIRECTIONAL_CONTACT_GRASP_TARGET_POLICY_SCHEMA = (
    "quantis.jepa_wm_contact_grasp_target_policy.v2"
)
HORIZON_CONTACT_GRASP_TARGET_POLICY_SCHEMA = (
    "quantis.jepa_wm_contact_grasp_target_policy.v3"
)
CONTACT_GRASP_TARGET_POLICY_SCHEMA = (
    "quantis.jepa_wm_contact_grasp_target_policy.v4"
)


class _TransportComposition(str, Enum):
    FIRST_ACTION = "first_action"
    FULL_HORIZON = "full_horizon"
    TRANSLATION_HORIZON = "translation_horizon"


@dataclass(frozen=True)
class _PolicyCapabilities:
    directional_progress: bool
    transport_composition: _TransportComposition


_POLICY_CAPABILITIES = {
    LEGACY_CONTACT_GRASP_TARGET_POLICY_SCHEMA: _PolicyCapabilities(
        False,
        _TransportComposition.FIRST_ACTION,
    ),
    DIRECTIONAL_CONTACT_GRASP_TARGET_POLICY_SCHEMA: _PolicyCapabilities(
        True,
        _TransportComposition.FIRST_ACTION,
    ),
    HORIZON_CONTACT_GRASP_TARGET_POLICY_SCHEMA: _PolicyCapabilities(
        True,
        _TransportComposition.FULL_HORIZON,
    ),
    CONTACT_GRASP_TARGET_POLICY_SCHEMA: _PolicyCapabilities(
        True,
        _TransportComposition.TRANSLATION_HORIZON,
    ),
}


def _frame_index(path: Path) -> int:
    if (
        path.is_absolute()
        or path.suffix != ".png"
        or not path.stem.startswith("frame_")
    ):
        raise ValueError("contact-grasp previous target is invalid")
    suffix = path.stem.removeprefix("frame_")
    if len(suffix) != 6 or not suffix.isdigit():
        raise ValueError("contact-grasp previous target is invalid")
    return int(suffix)


@dataclass(frozen=True)
class ContactGraspTargetStep:
    observation: ControlObservation
    plug_attached: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.observation, ControlObservation)
            or not isinstance(self.plug_attached, bool)
        ):
            raise ValueError("contact-grasp target step is invalid")


@dataclass(frozen=True)
class ContactGraspTargetPolicy:
    """Hold acquisition, then advance by measured reference-state progress."""

    schema: str = CONTACT_GRASP_TARGET_POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema not in _POLICY_CAPABILITIES:
            raise ValueError("contact-grasp target policy is invalid")

    @property
    def requires_directional_transport_progress(self) -> bool:
        return _POLICY_CAPABILITIES[self.schema].directional_progress

    @property
    def uses_horizon_transport_action(self) -> bool:
        return (
            _POLICY_CAPABILITIES[self.schema].transport_composition
            is not _TransportComposition.FIRST_ACTION
        )

    def action_for_execution(
        self,
        actions: Sequence[DroidAction],
        *,
        plug_attached: bool,
    ) -> DroidAction:
        """Resolve historical first-step or current horizon transport intent."""

        sequence = tuple(actions)
        if (
            len(sequence) != DROID_ROLLOUT_PROTOCOL.action_horizon
            or any(not isinstance(action, DroidAction) for action in sequence)
            or not isinstance(plug_attached, bool)
        ):
            raise ValueError("contact-grasp proposal action horizon is invalid")
        composition = _POLICY_CAPABILITIES[self.schema].transport_composition
        if (
            plug_attached
            and composition is _TransportComposition.TRANSLATION_HORIZON
        ):
            return compose_transport_action(sequence)
        if plug_attached and composition is _TransportComposition.FULL_HORIZON:
            return compose_actions(sequence)
        return sequence[0]

    @property
    def acquisition_target_index(self) -> int:
        return CONTACT_INSERTION_RECORDING.start_index(
            ContactInsertionSegment.GRASP_ATTACH
        )

    @property
    def transport_context_indices(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in CONTACT_GRASP_PROPOSAL_WINDOW.context_indices
            if index >= self.acquisition_target_index
        )

    @property
    def transport_target_indices(self) -> tuple[int, ...]:
        horizon = DROID_ROLLOUT_PROTOCOL.action_horizon
        return tuple(index + horizon for index in self.transport_context_indices)

    def next_target_index(
        self,
        *,
        live_pose: DroidPose,
        plug_attached: bool,
        previous_target: Path,
        reference_context_poses: Mapping[int, DroidPose],
    ) -> int:
        """Select a monotonic target from attachment and exact live pose."""

        if not isinstance(live_pose, DroidPose) or not isinstance(plug_attached, bool):
            raise ValueError("contact-grasp live state is invalid")
        expected_contexts = self.transport_context_indices
        if (
            tuple(reference_context_poses) != expected_contexts
            or any(
                not isinstance(reference_context_poses[index], DroidPose)
                for index in expected_contexts
            )
        ):
            raise ValueError("contact-grasp reference poses are invalid")
        previous_index = _frame_index(previous_target)
        allowed_previous = (
            self.acquisition_target_index,
            *self.transport_target_indices,
        )
        if previous_index not in allowed_previous:
            raise ValueError("contact-grasp previous target is invalid")
        if not plug_attached:
            if previous_index != self.acquisition_target_index:
                raise ValueError("contact-grasp previous target is invalid")
            return self.acquisition_target_index

        nearest_context = min(
            expected_contexts,
            key=lambda index: sum(
                (
                    live_pose.values[axis]
                    - reference_context_poses[index].values[axis]
                )
                ** 2
                for axis in range(3)
            ),
        )
        state_aligned_target = (
            nearest_context + DROID_ROLLOUT_PROTOCOL.action_horizon
        )
        return min(
            max(
                previous_index,
                state_aligned_target,
                self.transport_target_indices[0],
            ),
            self.transport_target_indices[-1],
        )

    def context_index_for_target(self, target: Path) -> int:
        """Return the proposal-conditioning context bound to one target."""

        target_index = _frame_index(target)
        if target_index not in (
            self.acquisition_target_index,
            *self.transport_target_indices,
        ):
            raise ValueError("contact-grasp target is outside the trained window")
        return target_index - DROID_ROLLOUT_PROTOCOL.action_horizon

    def select(
        self,
        recording: Path,
        *,
        frame_root: Path,
        live_pose: DroidPose,
        plug_attached: bool,
        previous_target: Path,
        camera: str = "wrist",
    ) -> ControlTarget:
        """Load and select the exact reference target for a live follow-up."""

        required = self._reference_rollouts(recording, camera=camera)
        target_index = self.next_target_index(
            live_pose=live_pose,
            plug_attached=plug_attached,
            previous_target=previous_target,
            reference_context_poses={
                index: required[index].context_pose
                for index in self.transport_context_indices
            },
        )
        return self._target(required, target_index, frame_root)

    def initial_target(
        self,
        recording: Path,
        *,
        frame_root: Path,
        camera: str = "wrist",
    ) -> ControlTarget:
        """Return the canonical acquisition target for a new live chain."""

        return self._target(
            self._reference_rollouts(recording, camera=camera),
            self.acquisition_target_index,
            frame_root,
        )

    def validate_observation_target(
        self,
        observation: ControlObservation,
        recording: Path,
        *,
        frame_root: Path,
        require_initial: bool,
        camera: str = "wrist",
    ) -> None:
        """Authenticate one persisted target against exact reference bytes."""

        target_index = _frame_index(observation.target_frame)
        allowed = (
            self.acquisition_target_index,
            *self.transport_target_indices,
        )
        if (
            target_index not in allowed
            or (
                require_initial
                and target_index != self.acquisition_target_index
            )
            or observation.warmup_frames
            != self.context_index_for_target(observation.target_frame)
            or observation.target
            != self._target(
                self._reference_rollouts(recording, camera=camera),
                target_index,
                frame_root,
            )
        ):
            raise ValueError("contact-grasp observation target is invalid")

    def validate_schedule(
        self,
        steps: Sequence[ContactGraspTargetStep],
    ) -> None:
        """Validate the phase-locked monotonic schedule without filesystem IO."""

        if (
            not steps
            or any(not isinstance(step, ContactGraspTargetStep) for step in steps)
        ):
            raise ValueError("contact-grasp target schedule is invalid")
        target_indices = tuple(
            _frame_index(step.observation.target_frame) for step in steps
        )
        allowed = (
            self.acquisition_target_index,
            *self.transport_target_indices,
        )
        if (
            target_indices[0] != self.acquisition_target_index
            or any(index not in allowed for index in target_indices)
            or any(
                current < previous
                for previous, current in zip(
                    target_indices, target_indices[1:]
                )
            )
            or any(
                not attached and index != self.acquisition_target_index
                for attached, index in zip(
                    (step.plug_attached for step in steps), target_indices
                )
            )
            or any(
                observation.warmup_frames
                != self.context_index_for_target(observation.target_frame)
                for observation in (step.observation for step in steps)
            )
        ):
            raise ValueError("contact-grasp target schedule is invalid")

    def validate_reference_schedule(
        self,
        steps: Sequence[ContactGraspTargetStep],
        recording: Path,
        *,
        frame_root: Path,
        camera: str = "wrist",
    ) -> None:
        """Replay a schedule against the exact authenticated reference bytes."""

        self.validate_schedule(steps)
        if steps[0].observation.target != self.initial_target(
            recording,
            frame_root=frame_root,
            camera=camera,
        ):
            raise ValueError("contact-grasp initial target is invalid")
        for previous, current, attached in zip(
            (step.observation for step in steps),
            (step.observation for step in steps[1:]),
            (step.plug_attached for step in steps[1:]),
        ):
            expected = self.select(
                recording,
                frame_root=frame_root,
                live_pose=current.pose,
                plug_attached=attached,
                previous_target=previous.target_frame,
                camera=camera,
            )
            if current.target != expected:
                raise ValueError("contact-grasp target schedule is invalid")

    def _reference_rollouts(
        self,
        recording: Path,
        *,
        camera: str,
    ) -> dict[int, RecordedRollout]:
        rollouts = load_rollouts(
            recording,
            camera=camera,
            bounds=ActionSelectionBounds(minimum_action_norm=0.0),
        )
        indexed = {rollout.context[0].index: rollout for rollout in rollouts}
        required_contexts = (
            self.acquisition_target_index - DROID_ROLLOUT_PROTOCOL.action_horizon,
            *self.transport_context_indices,
        )
        try:
            required = {index: indexed[index] for index in required_contexts}
        except KeyError as error:
            raise ValueError(
                "contact-grasp reference rollouts are incomplete"
            ) from error
        return required

    @staticmethod
    def _target(
        required: Mapping[int, RecordedRollout],
        target_index: int,
        frame_root: Path,
    ) -> ControlTarget:
        target_context = target_index - DROID_ROLLOUT_PROTOCOL.action_horizon
        rollout = required[target_context]
        return ControlTarget(
            rollout.target.path.relative_to(frame_root.resolve()),
            rollout.target_pose,
        )

    def to_dict(self) -> dict[str, str]:
        return {"schema": self.schema}

    @classmethod
    def from_dict(cls, payload: Any) -> ContactGraspTargetPolicy:
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema"}
            or payload["schema"] not in _POLICY_CAPABILITIES
        ):
            raise ValueError("contact-grasp target policy is invalid")
        return cls(str(payload["schema"]))


CONTACT_GRASP_TARGET_POLICY = ContactGraspTargetPolicy()
