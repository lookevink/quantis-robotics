"""Typed validation and metrics for chained simulator control rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import fsum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import DroidPose, action_between
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import SimulatorSafetyLimits
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
)
from sim.recording import validate_recording_id


ROLLOUT_SCHEMA = "quantis.jepa_wm_control_rollout.v1"
MAX_CONTROL_ROLLOUT_STEPS = 8
ORCHESTRATION_FAILED = "orchestration_failed"


@dataclass(frozen=True)
class PoseError:
    translation_meters: float
    rotation_radians: float
    gripper_closedness: float

    @classmethod
    def between(cls, pose: DroidPose, target: DroidPose) -> PoseError:
        delta = action_between(pose, target)
        return cls(
            float(np.linalg.norm(delta.values[:3])),
            float(Rotation.from_euler("xyz", delta.values[3:6]).magnitude()),
            abs(delta.values[6]),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "translation_meters": self.translation_meters,
            "rotation_radians": self.rotation_radians,
            "gripper_closedness": self.gripper_closedness,
        }


@dataclass(frozen=True)
class ControlStepSummary:
    state: ControlSessionState
    observation: ControlObservation
    response: ProposedControl
    result: ControlResult

    @classmethod
    def from_session(cls, session: ControlSession) -> ControlStepSummary:
        observation, state = session.load_capture()
        response = session.load_response()
        result = session.load_result()
        limits = SimulatorSafetyLimits()
        if (
            response.observation_id != observation.observation_id
            or response.proposal != observation.expected_proposal
            or result.gate.observation_id != observation.observation_id
            or result.inference_age_seconds > limits.maximum_observation_age_seconds
            or response.created_at_unix_seconds < observation.captured_at_unix_seconds
            or response.created_at_unix_seconds
            > observation.captured_at_unix_seconds + result.inference_age_seconds + 1e-6
        ):
            raise ValueError(
                f"control step identity or freshness is invalid: {session.session_id}"
            )
        if result.selected_action_scale is not None:
            commanded = result.selected_action_scale.apply(response.first_action)
            expected_pose = observation.pose.applied(commanded)
            if not np.allclose(
                expected_pose.values,
                result.gate.next_pose.values,
                rtol=0.0,
                atol=1e-8,
            ):
                raise ValueError(
                    f"control gate pose is not bound to its response: {session.session_id}"
                )
            if result.post_action is not None and (
                result.post_action.raw_proposed_action != response.first_action
                or result.post_action.commanded_action != commanded
            ):
                raise ValueError(
                    f"post-action evidence is not bound to its response: {session.session_id}"
                )
        return cls(state, observation, response, result)

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def status(self) -> ControlResultStatus:
        return self.result.status

    @property
    def post_action_pose(self) -> DroidPose | None:
        return self.result.post_action.pose if self.result.post_action else None

    def to_dict(self) -> dict[str, Any]:
        tracking = self.result.post_action.tracking if self.result.post_action else None
        return {
            "session": self.session_id,
            "observation_id": self.observation.observation_id,
            "status": self.status.value,
            "command_age_seconds": self.result.inference_age_seconds,
            "selected_action_scale": (
                self.result.selected_action_scale.to_dict()
                if self.result.selected_action_scale is not None
                else None
            ),
            "translation_cosine": (
                tracking.translation_cosine if tracking is not None else None
            ),
            "rotation_cosine": tracking.rotation_cosine if tracking is not None else None,
            "contact_force_newtons": (
                self.result.post_action.contact_force_newtons
                if self.result.post_action is not None
                else None
            ),
        }


@dataclass(frozen=True)
class IncompleteControlStepSummary:
    session_id: str
    reason: str

    def __post_init__(self) -> None:
        validate_recording_id(self.session_id)
        if not self.reason:
            raise ValueError("incomplete control step requires a reason")

    @property
    def status(self) -> str:
        return ORCHESTRATION_FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session_id,
            "observation_id": None,
            "status": self.status,
            "command_age_seconds": None,
            "selected_action_scale": None,
            "translation_cosine": None,
            "rotation_cosine": None,
            "contact_force_newtons": None,
            "reason": self.reason,
        }


RolloutStep = ControlStepSummary | IncompleteControlStepSummary


def _target_pose(
    data_root: Path,
    reference_recording: str,
    observation: ControlObservation,
) -> DroidPose:
    target = observation.target_frame
    if (
        target.is_absolute()
        or len(target.parts) != 4
        or target.parts[:3] != ("recordings", reference_recording, "wrist")
        or target.suffix != ".png"
        or not target.stem.startswith("frame_")
    ):
        raise ValueError("control target frame path is invalid")
    try:
        target_index = int(target.stem.removeprefix("frame_"))
    except ValueError as error:
        raise ValueError("control target frame path is invalid") from error
    steps_path = data_root / "recordings" / reference_recording / "steps.jsonl"
    for line in steps_path.read_text().splitlines():
        payload = json.loads(line)
        if payload.get("index") == target_index:
            return DroidPose(tuple(payload["end_effector_pose"]))
    raise ValueError("control target pose is missing from recording telemetry")


@dataclass(frozen=True)
class ControlRolloutReport:
    rollout_id: str
    reference_recording: str
    seed: int
    proposal: Path
    requested_steps: int
    steps: tuple[RolloutStep, ...]
    target_pose: DroidPose | None
    orchestration_error: str | None = None

    def __post_init__(self) -> None:
        validate_recording_id(self.rollout_id)
        validate_recording_id(self.reference_recording)
        if (
            self.seed < 0
            or not self.proposal.is_absolute()
            or isinstance(self.requested_steps, bool)
            or not len(self.steps)
            <= self.requested_steps
            <= MAX_CONTROL_ROLLOUT_STEPS
            or len({step.session_id for step in self.steps}) != len(self.steps)
        ):
            raise ValueError("control rollout contract is invalid")
        incomplete = [
            index
            for index, step in enumerate(self.steps)
            if isinstance(step, IncompleteControlStepSummary)
        ]
        if incomplete and (
            incomplete != [len(self.steps) - 1] or self.orchestration_error is None
        ):
            raise ValueError("control rollout has an invalid incomplete step")
        if self.orchestration_error is not None and not self.orchestration_error:
            raise ValueError("control rollout orchestration error is empty")
        complete = self.complete_steps
        if not complete:
            if self.target_pose is not None:
                raise ValueError("control rollout target has no complete observation")
            return
        if self.target_pose is None:
            raise ValueError("control rollout target pose is missing")
        for index, step in enumerate(complete):
            expected_previous = self.steps[index - 1].session_id if index else None
            if (
                step.state.previous_session_id != expected_previous
                or step.state.reference_recording != self.reference_recording
                or step.state.seed != self.seed
                or step.observation.expected_proposal != self.proposal
            ):
                raise ValueError("control rollout session chain or provenance is invalid")
        observations = tuple(step.observation for step in complete)
        if len({observation.observation_id for observation in observations}) != len(
            observations
        ):
            raise ValueError("control rollout observation IDs must be unique")
        if len({observation.target_frame for observation in observations}) != 1:
            raise ValueError("control rollout changed its target frame")
        initial_warmup = observations[0].warmup_frames
        if initial_warmup < 4 or any(
            observation.warmup_frames != initial_warmup + index
            for index, observation in enumerate(observations)
        ):
            raise ValueError("control rollout warm-up sequence is invalid")
        if any(
            current.captured_at_unix_seconds <= previous.captured_at_unix_seconds
            for previous, current in zip(observations, observations[1:])
        ):
            raise ValueError("control rollout capture times are not ordered")
        for previous, current in zip(observations, observations[1:]):
            expected_previous_action = action_between(previous.pose, current.pose)
            if not np.allclose(
                current.previous_action.values,
                expected_previous_action.values,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("control rollout previous-action chain is invalid")
        non_applied = [
            index
            for index, step in enumerate(complete)
            if step.status != ControlResultStatus.APPLIED
        ]
        if non_applied and non_applied != [len(self.steps) - 1]:
            raise ValueError("control rollout continued after a non-applied step")

    @property
    def complete_steps(self) -> tuple[ControlStepSummary, ...]:
        return tuple(
            step for step in self.steps if isinstance(step, ControlStepSummary)
        )

    @property
    def applied_steps(self) -> tuple[ControlStepSummary, ...]:
        return tuple(
            step
            for step in self.complete_steps
            if step.status == ControlResultStatus.APPLIED
        )

    @property
    def initial_goal_error(self) -> PoseError | None:
        if not self.complete_steps or self.target_pose is None:
            return None
        return PoseError.between(self.complete_steps[0].observation.pose, self.target_pose)

    @property
    def final_goal_error(self) -> PoseError | None:
        initial = self.initial_goal_error
        if initial is None or self.target_pose is None:
            return None
        final_pose = (
            self.applied_steps[-1].post_action_pose
            if self.applied_steps
            else self.complete_steps[0].observation.pose
        )
        if final_pose is None:
            raise AssertionError("validated applied step has no post-action pose")
        return PoseError.between(final_pose, self.target_pose)

    def to_dict(self) -> dict[str, Any]:
        initial = self.initial_goal_error
        final = self.final_goal_error
        ages = [step.result.inference_age_seconds for step in self.complete_steps]
        applied_count = len(self.applied_steps)
        return {
            "schema": ROLLOUT_SCHEMA,
            "rollout_id": self.rollout_id,
            "reference_recording": self.reference_recording,
            "seed": self.seed,
            "proposal": str(self.proposal),
            "requested_steps": self.requested_steps,
            "attempted_steps": len(self.steps),
            "complete_steps": len(self.complete_steps),
            "applied_steps": applied_count,
            "all_steps_applied": (
                applied_count == self.requested_steps
                and self.orchestration_error is None
            ),
            "orchestration_error": self.orchestration_error,
            "mean_command_age_seconds": fsum(ages) / len(ages) if ages else None,
            "maximum_command_age_seconds": max(ages) if ages else None,
            "initial_goal_error": initial.to_dict() if initial else None,
            "final_goal_error": final.to_dict() if final else None,
            "translation_progress_meters": (
                initial.translation_meters - final.translation_meters
                if initial and final
                else None
            ),
            "rotation_progress_radians": (
                initial.rotation_radians - final.rotation_radians
                if initial and final
                else None
            ),
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_sessions(
        cls,
        data_root: Path,
        rollout_id: str,
        session_ids: Sequence[str],
        *,
        reference_recording: str,
        seed: int,
        proposal: Path,
        requested_steps: int,
        orchestration_error: str | None = None,
    ) -> ControlRolloutReport:
        if not session_ids:
            raise ValueError("control rollout sessions must be non-empty")
        summaries: list[RolloutStep] = []
        for index, session_id in enumerate(session_ids):
            session = ControlSession.at(data_root / "control_sessions", session_id)
            try:
                summaries.append(ControlStepSummary.from_session(session))
            except ValueError as error:
                if orchestration_error is None or index != len(session_ids) - 1:
                    raise
                summaries.append(IncompleteControlStepSummary(session_id, str(error)))
        complete = tuple(
            step for step in summaries if isinstance(step, ControlStepSummary)
        )
        target = (
            _target_pose(data_root, reference_recording, complete[0].observation)
            if complete
            else None
        )
        return cls(
            rollout_id=rollout_id,
            reference_recording=reference_recording,
            seed=seed,
            proposal=proposal,
            requested_steps=requested_steps,
            steps=tuple(summaries),
            target_pose=target,
            orchestration_error=orchestration_error,
        )
