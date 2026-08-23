from pathlib import Path
import unittest

import numpy as np

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_safety import ControlGateReason, SimulatorSafetyLimits
from sim.isaac_control_execution import (
    ExecutionSafetyContext,
    select_safe_projection,
)
from sim.isaac_demo_kinematics import SolvedPose
from sim.isaac_demo_runtime import JointCommand


class ControlProjectionTest(unittest.TestCase):
    def test_falls_back_after_quarter_scale_ik_failure(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=100.0,
            context_frame=Path("context.png"),
            target_frame=Path("target.png"),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        proposal = ProposedControl(
            123,
            100.1,
            (
                DroidAction((0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)),
                DroidAction((0.0,) * 7),
                DroidAction((0.0,) * 7),
            ),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )
        calls = 0

        def solve(pose: DroidPose, warm_start: np.ndarray) -> SolvedPose:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("quarter-scale IK failed")
            joints = warm_start.copy()
            joints[0] += 0.001
            return SolvedPose(pose, joints, np.zeros(3), 0.04, 0.0, 0.0)

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
        )

        self.assertEqual(len(attempts), 2)
        self.assertIn(ControlGateReason.IK_SOLUTION_FAILED, attempts[0].gate.reasons)
        self.assertIsNotNone(selected)
        self.assertEqual(
            attempts[1].scale,
            DroidActionScale(0.5, 0.125, 0.125),
        )


if __name__ == "__main__":
    unittest.main()
