"""Authoritative task-state semantics for one control capture timeline."""

from __future__ import annotations

from dataclasses import dataclass

from jepa.contract import ObservationStage
from sim.control_capture_schedule import (
    ControlCaptureSchedule,
    ControlWarmupFramePlan,
)
from sim.control_context import RecordedControlStep
from sim.demo_sequence import Phase
from sim.recording import RecordingLabel, RecordingMoment


def control_context_recording_label(
    plug_attached: bool,
    task_index: int,
) -> RecordingLabel:
    """Resolve one task-indexed control context's evidence label."""

    if (
        not isinstance(plug_attached, bool)
        or isinstance(task_index, bool)
        or not isinstance(task_index, int)
        or task_index < 0
    ):
        raise ValueError("control context task index or attachment is invalid")
    if plug_attached:
        return RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP)
    if task_index == 0:
        return RecordingLabel(RecordingMoment.INITIAL)
    return RecordingLabel(RecordingMoment.MOTION, Phase.READY)


@dataclass(frozen=True)
class ControlTaskState:
    task_index: int
    plug_attached: bool
    recording_label: RecordingLabel
    observation_stage: ObservationStage
    frame_plan: ControlWarmupFramePlan

    @classmethod
    def from_step(
        cls,
        step: RecordedControlStep,
        frame_plan: ControlWarmupFramePlan,
        *,
        known_start: bool = False,
    ) -> ControlTaskState:
        if step.index != frame_plan.task_index:
            raise ValueError("control context does not match capture schedule")
        return cls(
            task_index=step.index,
            plug_attached=step.plug_attached,
            recording_label=(
                control_context_recording_label(step.plug_attached, step.index)
                if known_start or step.phase is None
                else step.phase
            ),
            observation_stage=(
                (
                    ObservationStage.CABLE_GRASPED
                    if step.plug_attached
                    else ObservationStage.APPROACHING_CABLE
                )
                if known_start or step.stage is None
                else step.stage
            ),
            frame_plan=frame_plan,
        )


@dataclass(frozen=True)
class ControlTaskTimeline:
    states: tuple[ControlTaskState, ...]
    initialization_task_index: int

    def __post_init__(self) -> None:
        if (
            not self.states
            or tuple(state.task_index for state in self.states)
            != tuple(range(len(self.states)))
            or self.initialization_task_index not in range(len(self.states))
        ):
            raise ValueError("control task timeline is invalid")

    @classmethod
    def from_context(
        cls,
        steps: tuple[RecordedControlStep, ...],
        schedule: ControlCaptureSchedule,
    ) -> ControlTaskTimeline:
        if len(steps) != len(schedule.frames):
            raise ValueError("control context does not match capture schedule")
        states = tuple(
            ControlTaskState.from_step(
                step,
                frame_plan,
                known_start=(
                    schedule.defer_camera_activation
                    and step.index == schedule.initialization_task_index
                ),
            )
            for step, frame_plan in zip(steps, schedule.frames)
        )
        return cls(states, schedule.initialization_task_index)

    def state(self, task_index: int) -> ControlTaskState:
        if (
            isinstance(task_index, bool)
            or not isinstance(task_index, int)
            or task_index not in range(len(self.states))
        ):
            raise ValueError("control task timeline index is invalid")
        return self.states[task_index]

    @property
    def initialization(self) -> ControlTaskState:
        return self.states[self.initialization_task_index]

    @property
    def replay(self) -> tuple[ControlTaskState, ...]:
        return self.states[self.initialization_task_index + 1 :]
