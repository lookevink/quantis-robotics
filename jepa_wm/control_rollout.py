"""Typed validation and metrics for chained simulator control rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import fsum, isfinite
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import DroidPose, action_between
from jepa_wm.control_tracking import (
    evaluate_action_tracking,
    tracking_limits_for_policy,
)
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.grasp_task import (
    GraspTaskStep,
    ReachAndGraspDecision,
    evaluate_reach_and_grasp,
)
from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.shadow_planning import ShadowSearchEvidence
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
)
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from sim.recording import validate_recording_id


ROLLOUT_SCHEMA = "quantis.jepa_wm_control_rollout.v1"
MAX_CONTROL_ROLLOUT_STEPS = 8


class IncompleteStepStatus(str, Enum):
    ORCHESTRATION_FAILED = "orchestration_failed"


class OrchestrationOperation(str, Enum):
    INITIAL_CONTROL_STEP = "initial_control_step"
    INITIAL_STATUS = "initial_status"
    FOLLOWUP_CAPTURE = "followup_capture"
    FOLLOWUP_INFERENCE = "followup_inference"
    FOLLOWUP_APPLY = "followup_apply"
    FOLLOWUP_STATUS = "followup_status"
    RESET_TRIAL_SOURCE_PREFLIGHT = "reset_trial_source_preflight"
    RESET_TRIAL_CAPTURE = "reset_trial_capture"
    RESET_TRIAL_BINDING = "reset_trial_binding"
    RESET_TRIAL_APPLY = "reset_trial_apply"

    @property
    def requires_step_index(self) -> bool:
        return self in (
            OrchestrationOperation.FOLLOWUP_CAPTURE,
            OrchestrationOperation.FOLLOWUP_INFERENCE,
            OrchestrationOperation.FOLLOWUP_APPLY,
            OrchestrationOperation.FOLLOWUP_STATUS,
        )

    @classmethod
    def parse_phase(
        cls, encoded_phase: str
    ) -> tuple[OrchestrationOperation, int | None]:
        for operation in cls:
            if not operation.requires_step_index and encoded_phase == operation.value:
                return operation, None
            prefix = f"{operation.value}_"
            if operation.requires_step_index and encoded_phase.startswith(prefix):
                encoded_index = encoded_phase[len(prefix) :]
                if encoded_index.isdigit():
                    return operation, int(encoded_index)
        raise ValueError("control rollout orchestration phase is malformed")


@dataclass(frozen=True)
class OrchestrationFailure:
    operation: OrchestrationOperation
    exit_code: int
    step_index: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.exit_code, bool)
            or not 1 <= self.exit_code <= 255
            or self.operation.requires_step_index != (self.step_index is not None)
            or (self.step_index is not None and self.step_index <= 0)
        ):
            raise ValueError("control rollout orchestration failure is invalid")

    @classmethod
    def parse(cls, value: str) -> OrchestrationFailure:
        try:
            phase, encoded_exit_code = value.rsplit(":exit_", 1)
            exit_code = int(encoded_exit_code)
            operation, step_index = OrchestrationOperation.parse_phase(phase)
            return cls(operation, exit_code, step_index)
        except (TypeError, ValueError) as error:
            raise ValueError("control rollout orchestration failure is malformed") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "step_index": self.step_index,
            "exit_code": self.exit_code,
        }


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
class ControlStepTiming:
    observation_age_seconds: float
    inference_latency_seconds: float
    command_age_seconds: float

    def __post_init__(self) -> None:
        values = (
            self.observation_age_seconds,
            self.inference_latency_seconds,
            self.command_age_seconds,
        )
        if not all(isfinite(value) and value >= 0.0 for value in values) or not np.isclose(
            self.observation_age_seconds,
            self.inference_latency_seconds + self.command_age_seconds,
            rtol=0.0,
            atol=1e-6,
        ):
            raise ValueError("control timing evidence is inconsistent")

    @classmethod
    def from_step(
        cls,
        observation: ControlObservation,
        response: ProposedControl,
        result: ControlResult,
    ) -> ControlStepTiming:
        inference_latency = (
            response.created_at_unix_seconds
            - observation.captured_at_unix_seconds
        )
        return cls(
            result.observation_age_seconds,
            inference_latency,
            result.observation_age_seconds - inference_latency,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "observation_age_seconds": self.observation_age_seconds,
            "inference_latency_seconds": self.inference_latency_seconds,
            "command_age_seconds": self.command_age_seconds,
        }


@dataclass(frozen=True)
class ControlStepSummary:
    state: ControlSessionState
    observation: ControlObservation
    response: ProposedControl
    result: ControlResult
    shadow: ShadowSearchEvidence | None = None
    shadow_safety: ShadowSafetyEvidence | None = None

    @classmethod
    def from_session(cls, session: ControlSession) -> ControlStepSummary:
        observation, state = session.load_capture()
        response = session.load_response()
        result = session.load_result()
        limits = SimulatorSafetyLimits()
        if state.execution_policy is ControlExecutionPolicy.INSERTION_RESET_TRIAL:
            binding = session.load_insertion_trial_binding(response)
            attempted_scales = tuple(
                attempt.scale for attempt in result.projection_attempts
            )
            try:
                binding.validate_attempted_projection_scales(attempted_scales)
            except ValueError as error:
                raise ValueError(
                    f"insertion trial exceeded its source projection: {session.session_id}"
                ) from error
            if (
                result.status is not ControlResultStatus.BLOCKED
                and result.execution_interlock is None
            ):
                raise ValueError(
                    f"insertion trial execution interlock is missing: {session.session_id}"
                )
            if result.insertion_trial_refresh is not None:
                observation, response = result.insertion_trial_refresh.authorize(
                    observation,
                    response,
                    state.require_safety_snapshot(),
                )
        elif result.insertion_trial_refresh is not None:
            raise ValueError(
                f"non-insertion result has execution refresh: {session.session_id}"
            )
        timing = ControlStepTiming.from_step(observation, response, result)
        if (
            response.observation_id != observation.observation_id
            or response.proposal != observation.expected_proposal
            or result.gate.observation_id != observation.observation_id
            or (
                result.status != ControlResultStatus.BLOCKED
                and (
                    timing.observation_age_seconds
                    > limits.maximum_observation_age_seconds
                    or timing.command_age_seconds
                    > limits.maximum_command_age_seconds
                )
            )
            or response.created_at_unix_seconds < observation.captured_at_unix_seconds
            or response.created_at_unix_seconds
            > observation.captured_at_unix_seconds
            + timing.observation_age_seconds
            + 1e-6
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
            if result.post_action is not None:
                actual = action_between(observation.pose, result.post_action.pose)
                tracking = evaluate_action_tracking(
                    commanded,
                    actual,
                    tracking_limits_for_policy(state.execution_policy),
                )
                if (
                    not np.allclose(
                        actual.values,
                        result.post_action.actual_action.values,
                        rtol=0.0,
                        atol=1e-10,
                    )
                    or tracking != result.post_action.tracking
                ):
                    raise ValueError(
                        f"post-action realization is inconsistent: {session.session_id}"
                    )
        shadow = session.load_shadow() if session.shadow_path.is_file() else None
        shadow_safety = (
            session.load_shadow_safety()
            if session.shadow_safety_path.is_file()
            else None
        )
        return cls(state, observation, response, result, shadow, shadow_safety)

    @property
    def session_id(self) -> str:
        return self.state.session_id

    @property
    def status(self) -> ControlResultStatus:
        return self.result.status

    @property
    def is_applied(self) -> bool:
        return self.status is ControlResultStatus.APPLIED and self.post_action_pose is not None

    @property
    def post_action_pose(self) -> DroidPose | None:
        return self.result.post_action.pose if self.result.post_action else None

    @property
    def timing(self) -> ControlStepTiming:
        return ControlStepTiming.from_step(
            self.observation,
            self.response,
            self.result,
        )

    def to_dict(self) -> dict[str, Any]:
        tracking = self.result.post_action.tracking if self.result.post_action else None
        return {
            "session": self.session_id,
            "observation_id": self.observation.observation_id,
            "status": self.status.value,
            "timing": self.timing.to_dict(),
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
            "plug_position": (
                list(self.result.post_action.plug_position)
                if self.result.post_action is not None
                and self.result.post_action.plug_position is not None
                else None
            ),
            "plug_attached": (
                self.result.post_action.plug_attached
                if self.result.post_action is not None
                else None
            ),
            "execution_error": self.result.execution_error,
            "execution_interlock": (
                self.result.execution_interlock.to_dict()
                if self.result.execution_interlock is not None
                else None
            ),
            "shadow_search": self.shadow.to_dict() if self.shadow is not None else None,
            "shadow_safety": (
                self.shadow_safety.to_dict()
                if self.shadow_safety is not None
                else None
            ),
        }


@dataclass(frozen=True)
class IncompleteControlStepSummary:
    session_id: str
    failure: OrchestrationFailure

    def __post_init__(self) -> None:
        validate_recording_id(self.session_id)

    @property
    def status(self) -> IncompleteStepStatus:
        return IncompleteStepStatus.ORCHESTRATION_FAILED

    def to_dict(self) -> dict[str, Any]:
        return {
            "session": self.session_id,
            "observation_id": None,
            "status": self.status.value,
            "timing": None,
            "selected_action_scale": None,
            "translation_cosine": None,
            "rotation_cosine": None,
            "contact_force_newtons": None,
            "failure": self.failure.to_dict(),
        }


RolloutStep = Union[ControlStepSummary, IncompleteControlStepSummary]


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
    orchestration_failure: OrchestrationFailure | None = None
    reference_task: str | None = None

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
            incomplete != [len(self.steps) - 1]
            or self.orchestration_failure is None
            or self.steps[incomplete[0]].failure != self.orchestration_failure
        ):
            raise ValueError("control rollout has an invalid incomplete step")
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
        target_frames = tuple(observation.target_frame for observation in observations)
        if self.reference_task == GRASP_TASK_ID:
            target_indices = tuple(
                int(frame.stem.removeprefix("frame_")) for frame in target_frames
            )
            if target_indices != tuple(
                range(target_indices[0], target_indices[0] + len(target_indices))
            ):
                raise ValueError("grasp control rollout target schedule is invalid")
        elif len(set(target_frames)) != 1:
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
        shadow_adapters = {
            step.shadow.adapter for step in complete if step.shadow is not None
        }
        if len(shadow_adapters) > 1:
            raise ValueError("control rollout changed its shadow action adapter")

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

    @property
    def reach_and_grasp(self) -> ReachAndGraspDecision | None:
        if not self.complete_steps:
            return None
        initial_state = self.complete_steps[0].state
        if initial_state.plug_position is None:
            return None
        evidence = [
            GraspTaskStep(
                tuple(initial_state.plug_position),
                initial_state.plug_attached,
                True,
                initial_state.collision_detected,
                initial_state.contact_force_newtons,
            )
        ]
        for step in self.complete_steps:
            post_action = step.result.post_action
            if post_action is None or post_action.plug_position is None:
                return None
            evidence.append(
                GraspTaskStep(
                    tuple(post_action.plug_position),
                    post_action.plug_attached,
                    post_action.tracking.passed,
                    post_action.collision_detected,
                    post_action.contact_force_newtons,
                )
            )
        return evaluate_reach_and_grasp(tuple(evidence)) if evidence else None

    def to_dict(self) -> dict[str, Any]:
        initial = self.initial_goal_error
        final = self.final_goal_error
        timings = [step.timing for step in self.complete_steps]
        shadows = [step.shadow for step in self.complete_steps if step.shadow is not None]
        shadow_safety = [
            step.shadow_safety
            for step in self.complete_steps
            if step.shadow_safety is not None
        ]
        applied_count = len(self.applied_steps)
        grasp = self.reach_and_grasp
        return {
            "schema": ROLLOUT_SCHEMA,
            "rollout_id": self.rollout_id,
            "reference_recording": self.reference_recording,
            "reference_task": self.reference_task,
            "seed": self.seed,
            "proposal": str(self.proposal),
            "requested_steps": self.requested_steps,
            "attempted_steps": len(self.steps),
            "complete_steps": len(self.complete_steps),
            "applied_steps": applied_count,
            "all_steps_applied": (
                applied_count == self.requested_steps
                and self.orchestration_failure is None
            ),
            "orchestration_failure": (
                self.orchestration_failure.to_dict()
                if self.orchestration_failure is not None
                else None
            ),
            "mean_observation_age_seconds": (
                fsum(timing.observation_age_seconds for timing in timings)
                / len(timings)
                if timings
                else None
            ),
            "maximum_observation_age_seconds": (
                max(timing.observation_age_seconds for timing in timings)
                if timings
                else None
            ),
            "mean_inference_latency_seconds": (
                fsum(timing.inference_latency_seconds for timing in timings)
                / len(timings)
                if timings
                else None
            ),
            "mean_command_age_seconds": (
                fsum(timing.command_age_seconds for timing in timings) / len(timings)
                if timings
                else None
            ),
            "shadow_searches": len(shadows),
            "shadow_adapter": str(shadows[0].adapter) if shadows else None,
            "shadow_gate_passes": sum(shadow.passes_shadow_gate for shadow in shadows),
            "shadow_safety_evaluations": len(shadow_safety),
            "shadow_safety_passes": sum(evidence.passed for evidence in shadow_safety),
            "mean_shadow_energy_improvement": (
                fsum(shadow.energy_improvement for shadow in shadows) / len(shadows)
                if shadows
                else None
            ),
            "mean_shadow_planning_seconds": (
                fsum(shadow.planning_seconds for shadow in shadows) / len(shadows)
                if shadows
                else None
            ),
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
            "reach_and_grasp": grasp.to_dict() if grasp is not None else None,
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
        orchestration_failure: OrchestrationFailure | None = None,
    ) -> ControlRolloutReport:
        if not session_ids:
            raise ValueError("control rollout sessions must be non-empty")
        summaries: list[RolloutStep] = []
        for index, session_id in enumerate(session_ids):
            session = ControlSession.at(data_root / "control_sessions", session_id)
            try:
                summaries.append(ControlStepSummary.from_session(session))
            except ValueError:
                if orchestration_failure is None or index != len(session_ids) - 1:
                    raise
                summaries.append(
                    IncompleteControlStepSummary(session_id, orchestration_failure)
                )
        complete = tuple(
            step for step in summaries if isinstance(step, ControlStepSummary)
        )
        reference_task = None
        if complete:
            manifest = json.loads(
                (
                    data_root
                    / "recordings"
                    / reference_recording
                    / "manifest.json"
                ).read_text()
            )
            metadata = manifest.get("metadata") if isinstance(manifest, dict) else None
            reference_task = metadata.get("task") if isinstance(metadata, dict) else None
        target_observation = (
            None
            if not complete
            else complete[-1]
            if reference_task == GRASP_TASK_ID
            else complete[0]
        )
        target = (
            _target_pose(data_root, reference_recording, target_observation.observation)
            if target_observation is not None
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
            orchestration_failure=orchestration_failure,
            reference_task=(reference_task if isinstance(reference_task, str) else None),
        )
