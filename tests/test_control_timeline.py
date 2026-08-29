from __future__ import annotations

from dataclasses import replace
import unittest

from jepa.contract import ObservationStage
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.insertion_layout import CONTACT_INSERTION_LAYOUT
from sim.control_capture_schedule import control_capture_schedule
from sim.control_context import ControlContextPurpose, RecordedControlStep
from sim.control_timeline import ControlTaskTimeline
from sim.demo_sequence import Phase
from sim.recording import RecordingLabel, RecordingMoment


def _step(
    index: int,
    *,
    attached: bool = False,
    phase: RecordingLabel | None = None,
    stage: ObservationStage | None = None,
) -> RecordedControlStep:
    return RecordedControlStep(
        index,
        (0.0,) * 7,
        0.04,
        attached,
        (-0.1, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0),
        phase,
        stage,
    )


class ControlTaskTimelineTest(unittest.TestCase):
    def test_derives_all_task_semantics_from_one_contiguous_timeline(self) -> None:
        schedule = control_capture_schedule(
            ControlExecutionPolicy.DIRECT,
            insertion_control=True,
            context_index=283,
            context_purpose=ControlContextPurpose.STANDARD,
        )
        steps = tuple(
            _step(
                index,
                attached=attached,
                phase=RecordingLabel.from_value(phase),
                stage=ObservationStage(stage),
            )
            for index, (phase, stage, attached) in enumerate(
                zip(
                    CONTACT_INSERTION_LAYOUT.phase_roster,
                    CONTACT_INSERTION_LAYOUT.stage_roster,
                    CONTACT_INSERTION_LAYOUT.attachment_roster,
                )
            )
        )

        timeline = ControlTaskTimeline.from_context(steps, schedule)

        self.assertEqual(len(timeline.states), 284)
        self.assertEqual(
            timeline.state(0).recording_label,
            RecordingLabel(RecordingMoment.INITIAL),
        )
        self.assertEqual(
            tuple(state.recording_label.value for state in timeline.states),
            CONTACT_INSERTION_LAYOUT.phase_roster,
        )
        self.assertEqual(
            tuple(state.observation_stage.value for state in timeline.states),
            CONTACT_INSERTION_LAYOUT.stage_roster,
        )
        self.assertEqual(
            timeline.state(114).recording_label,
            RecordingLabel(RecordingMoment.MOTION, Phase.PRE_INSERTION),
        )
        self.assertEqual(
            timeline.state(280).observation_stage,
            ObservationStage.PLUG_SEATED,
        )

    def test_known_start_retains_noninitial_task_semantics_without_replay(self) -> None:
        schedule = control_capture_schedule(
            ControlExecutionPolicy.DIRECT,
            insertion_control=True,
            context_index=110,
            context_purpose=ControlContextPurpose.CONTACT_GRASP,
        )

        steps = tuple(
            _step(
                index,
                attached=CONTACT_INSERTION_LAYOUT.attachment_roster[index],
                phase=RecordingLabel.from_value(
                    CONTACT_INSERTION_LAYOUT.phase_roster[index]
                ),
                stage=ObservationStage(CONTACT_INSERTION_LAYOUT.stage_roster[index]),
            )
            for index in range(111)
        )
        timeline = ControlTaskTimeline.from_context(steps, schedule)

        self.assertEqual(timeline.initialization.task_index, 110)
        self.assertEqual(
            timeline.initialization.recording_label,
            RecordingLabel(RecordingMoment.MOTION, Phase.READY),
        )
        self.assertEqual(timeline.replay, ())

    def test_rejects_context_and_schedule_drift_before_runtime_motion(self) -> None:
        schedule = control_capture_schedule(
            ControlExecutionPolicy.DIRECT,
            insertion_control=False,
            context_index=4,
            context_purpose=ControlContextPurpose.STANDARD,
        )
        steps = tuple(_step(index) for index in range(5))

        with self.assertRaisesRegex(ValueError, "does not match"):
            ControlTaskTimeline.from_context(
                steps[:-1] + (replace(steps[-1], index=5),),
                schedule,
            )


if __name__ == "__main__":
    unittest.main()
