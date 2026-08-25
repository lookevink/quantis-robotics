from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from jepa_wm.action import DroidAction, DroidPose, MAX_GRIPPER_WIDTH_M
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlGateDecision,
    INSERTION_TARGET_PROGRESS,
    SafetyProjectionAttempt,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from sim.control_session import ControlSessionState
from sim.isaac_demo_runtime import JointCommand
from sim.isaac_insertion_safety import evaluate_direct_insertion_candidate


class DirectInsertionSafetyRuntimeTest(unittest.TestCase):
    def test_persists_live_safety_without_actuation(self) -> None:
        joints = (0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0)
        proposal_path = Path("/tmp/proposal.pth")
        observation = ControlObservation(
            9,
            100.0,
            Path("context.png"),
            ControlTarget(
                Path("target.png"),
                DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            ),
            proposal_path,
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            DroidAction((0.0,) * 7),
            43,
        )
        state = ControlSessionState(
            "insertion-session",
            "insertion-held-00",
            52600,
            "control-insertion-session",
            joints,
            False,
            0.25,
            plug_position=(0.4, 0.0, 0.5),
            plug_attached=True,
            current_gripper_width_m=0.04,
            execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
        )
        response = ProposedControl(
            9,
            100.1,
            (DroidAction((0.0,) * 7),) * 3,
            proposal_path,
            "a" * 64,
        )
        scale = ACTION_SCALES[0]
        attempt = SafetyProjectionAttempt(
            scale,
            ControlGateDecision(9, observation.pose, ()),
            0.0,
            joints,
        )
        session = Mock()
        session.load_capture.return_value = (observation, state)
        session.load_response.return_value = response
        stage = object()
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(
                actual_command=lambda: JointCommand(
                    np.asarray(joints),
                    (1.0 - observation.pose.values[6]) * MAX_GRIPPER_WIDTH_M,
                )
            ),
            attachment=SimpleNamespace(
                attached=True,
                world_pose=lambda: (np.asarray((0.4, 0.0, 0.5)), np.zeros(4)),
            ),
            sensor=object(),
        )
        omni = ModuleType("omni")
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
        omni.usd = usd

        with (
            patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}),
            patch("sim.isaac_insertion_safety.ControlSession.at", return_value=session),
            patch(
                "sim.isaac_insertion_safety.recording_task",
                return_value=INSERTION_TASK_ID,
            ),
            patch(
                "sim.isaac_insertion_safety.live_runtime_for",
                return_value=runtime,
            ),
            patch(
                "sim.isaac_insertion_safety.read_control_contact",
                return_value=(False, 0.25),
            ),
            patch(
                "sim.isaac_insertion_safety.select_safe_projection",
                return_value=((attempt,), object()),
            ) as select_projection,
            patch("sim.isaac_insertion_safety.time", return_value=100.2),
        ):
            payload = asyncio.run(
                evaluate_direct_insertion_candidate("insertion-session")
            )

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["authority"], "no_actuation")
        self.assertIs(
            select_projection.call_args.kwargs["target_progress"],
            INSERTION_TARGET_PROGRESS,
        )
        session.write_direct_safety.assert_called_once()
        self.assertFalse(session.claim_execution.called)
        self.assertFalse(session.write_result.called)


if __name__ == "__main__":
    unittest.main()
