from __future__ import annotations

import unittest
from pathlib import Path

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.contact_grasp_acquisition_failure_replay import evaluate_replay_action
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl


class ContactGraspAcquisitionFailureReplayTest(unittest.TestCase):
    def _observation(self) -> ControlObservation:
        return ControlObservation(
            1,
            1.0,
            Path("context.png"),
            ControlTarget(
                Path("target.png"),
                DroidPose((0.2, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2)),
            ),
            Path("/tmp/proposal.pth"),
            DroidPose((0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 0.1)),
            DroidAction((0.0,) * 7),
            10,
        )

    def _control(self, action: DroidAction) -> ProposedControl:
        return ProposedControl(
            1,
            2.0,
            (action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )

    def test_requires_translation_progress_and_nonnegative_gripper(self) -> None:
        passed = evaluate_replay_action(
            self._observation(),
            self._control(DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01))),
        )
        opened = evaluate_replay_action(
            self._observation(),
            self._control(DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01))),
        )

        self.assertTrue(passed["passed"])
        self.assertFalse(opened["passed"])


if __name__ == "__main__":
    unittest.main()
