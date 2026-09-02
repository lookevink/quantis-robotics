from __future__ import annotations

from types import SimpleNamespace
import unittest

from jepa_wm.grasp_to_insertion import GraspToInsertionReport
from jepa_wm.insertion_rollout import (
    GRASP_TO_INSERTION_ROLLOUT,
    InsertionRolloutPosition,
)
from jepa_wm.insertion_task import InsertionTarget, InsertionTaskStep
from sim.grasp_task import AttachmentMechanism, observe_grasp_acquisition


def _post_action(
    task_step: InsertionTaskStep | None,
    *,
    acquisition: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        insertion_task_step=task_step,
        plug_attached=task_step.plug_attached if task_step is not None else True,
        grasp_acquisition=(
            observe_grasp_acquisition(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                0.018,
            )
            if acquisition
            else None
        ),
        attachment_mechanism=(
            AttachmentMechanism.KINEMATIC_FOLLOW
            if acquisition or (task_step is not None and task_step.plug_attached)
            else None
        ),
        command_realization=(
            SimpleNamespace(passed=True) if task_step is not None else None
        ),
    )


def _step(
    session_id: str,
    task_step: InsertionTaskStep | None,
    *,
    position: int | None = None,
    acquisition: bool = False,
    initially_attached: bool = False,
) -> SimpleNamespace:
    state = (
        SimpleNamespace(
            plug_attached=initially_attached,
            resolved_insertion_rollout_position=lambda: InsertionRolloutPosition(
                position,
                GRASP_TO_INSERTION_ROLLOUT.maximum_steps,
            ),
        )
        if position is not None
        else SimpleNamespace(plug_attached=initially_attached)
    )
    return SimpleNamespace(
        session_id=session_id,
        state=state,
        result=SimpleNamespace(
            post_action=_post_action(task_step, acquisition=acquisition)
        ),
    )


class GraspToInsertionReportTest(unittest.TestCase):
    def _reports(self, *, with_geometry: bool):
        target = InsertionTarget((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        unattached = InsertionTaskStep(
            (0.1, 0.0, 0.0), (0.06, 0.0, 0.0), False, 0.0, True, False, 0.0
        )
        attached = InsertionTaskStep(
            (0.1, 0.0, 0.0), (0.06, 0.0, 0.0), True, 0.0, True, False, 0.0
        )
        seated = InsertionTaskStep(
            (0.0, 0.0, 0.0), (-0.04, 0.0, 0.0), True, 0.0, True, False, 0.5
        )
        grasp_steps = (
            _step("grasp-1", unattached if with_geometry else None),
            _step(
                "grasp-2",
                attached if with_geometry else None,
                acquisition=True,
            ),
        )
        approach_steps = tuple(
            _step(
                f"insertion-{index}",
                attached if with_geometry else None,
                position=index,
                initially_attached=True,
            )
            for index in range(1, GRASP_TO_INSERTION_ROLLOUT.maximum_steps - 3)
        )
        seated_steps = tuple(
            _step(
                f"insertion-{index}",
                seated if with_geometry else None,
                position=index,
                initially_attached=True,
            )
            for index in range(
                GRASP_TO_INSERTION_ROLLOUT.maximum_steps - 3,
                GRASP_TO_INSERTION_ROLLOUT.maximum_steps + 1,
            )
        )
        insertion_steps = (*approach_steps, *seated_steps)
        grasp = SimpleNamespace(
            requested_steps=192,
            applied_steps=grasp_steps,
            complete_steps=grasp_steps,
            orchestration_failure=None,
            reach_and_grasp=SimpleNamespace(passed=True),
            reference_recording="reference",
            seed=12600,
            current_wire_authenticated=True,
        )
        insertion = SimpleNamespace(
            requested_steps=GRASP_TO_INSERTION_ROLLOUT.maximum_steps,
            all_steps_applied=True,
            applied_steps=insertion_steps,
            complete_steps=insertion_steps,
            predecessor_session_id="grasp-2",
            reference_recording="reference",
            seed=12600,
            insertion_target=target if with_geometry else None,
            current_wire_authenticated=True,
        )
        return grasp, insertion

    def test_rejects_four_applied_actions_without_seating_evidence(self) -> None:
        grasp, insertion = self._reports(with_geometry=False)

        with self.assertRaisesRegex(ValueError, "invalid"):
            GraspToInsertionReport("run-1", grasp, insertion)

    def test_accepts_four_safe_seated_observations(self) -> None:
        grasp, insertion = self._reports(with_geometry=True)

        report = GraspToInsertionReport("run-1", grasp, insertion)

        self.assertTrue(report.insertion_decision.passed)
        self.assertEqual(len(report.insertion_decision.seated_indices), 4)

    def test_accepts_acquisition_on_the_first_grasp_action(self) -> None:
        grasp, insertion = self._reports(with_geometry=True)
        attached = grasp.complete_steps[1].result.post_action.insertion_task_step
        first = _step("grasp-1", attached, acquisition=True)
        retained = _step(
            "grasp-2",
            attached,
            initially_attached=True,
        )
        first_action_grasp = SimpleNamespace(
            **{
                **grasp.__dict__,
                "applied_steps": (first, retained),
                "complete_steps": (first, retained),
            }
        )

        report = GraspToInsertionReport("run-1", first_action_grasp, insertion)

        self.assertTrue(report.insertion_decision.passed)

    def test_rejects_legacy_insertion_wire_evidence(self) -> None:
        grasp, insertion = self._reports(with_geometry=True)
        legacy_insertion = SimpleNamespace(
            **{
                **insertion.__dict__,
                "current_wire_authenticated": False,
            }
        )

        with self.assertRaisesRegex(ValueError, "invalid"):
            GraspToInsertionReport("run-1", grasp, legacy_insertion)


if __name__ == "__main__":
    unittest.main()
