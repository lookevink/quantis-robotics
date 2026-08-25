from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_safety import ControlGateReason, SimulatorSafetyLimits
from sim.isaac_control_execution import (
    ExecutionSafetyContext,
    capture_synchronized_post_action,
    select_safe_projection,
    synchronized_actual_command,
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
            target=ControlTarget(Path("target.png")),
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
            DroidActionScale(0.5, 0.125, 1.0),
        )


class FakeTimeline:
    def __init__(self) -> None:
        self.events: list[str] = []

    def play(self) -> None:
        self.events.append("play")

    def pause(self) -> None:
        self.events.append("pause")


class FakeArticulation:
    def __init__(self, valid: bool) -> None:
        self.valid = valid

    def is_physics_tensor_entity_valid(self) -> bool:
        return self.valid


class FakeActuators:
    def __init__(self, valid: bool) -> None:
        self.articulation = FakeArticulation(valid)
        self.command = object()

    def actual_command(self) -> object:
        if not self.articulation.valid:
            raise AssertionError("physics tensor is invalid")
        return self.command


class IsaacControlExecutionTest(unittest.TestCase):
    def test_reads_post_action_state_after_camera_capture_advances_physics(self) -> None:
        before = JointCommand(np.zeros(7), 0.04)
        after = JointCommand(np.ones(7), 0.02)
        actuators = FakeActuators(valid=True)
        actuators.command = before
        snapshot = object()

        async def capture(*_args: object) -> dict[str, object]:
            actuators.command = after
            return {"path": "/tmp/post.png", "shape": [512, 512, 4]}

        with (
            patch("sim.isaac_control_execution.capture_camera_frame", capture),
            patch(
                "sim.isaac_control_execution.read_control_contact",
                return_value=(False, 0.5),
            ),
            patch(
                "sim.isaac_control_execution.recording_snapshot",
                return_value=snapshot,
            ) as recording,
        ):
            captured = asyncio.run(
                capture_synchronized_post_action(
                    actuators,
                    object(),
                    object(),
                    Path("/tmp/post.png"),
                )
            )

        self.assertIs(captured.command, after)
        self.assertIs(captured.snapshot, snapshot)
        self.assertEqual(captured.contact_force_newtons, 0.5)
        recording.assert_called_once()
        self.assertIs(recording.call_args.args[2], after)

    def test_refreshes_a_stale_paused_physics_tensor_before_reading(self) -> None:
        actuators = FakeActuators(valid=False)
        timeline = FakeTimeline()

        async def advance() -> None:
            actuators.articulation.valid = True

        command = asyncio.run(
            synchronized_actual_command(actuators, timeline, advance)
        )

        self.assertIs(command, actuators.command)
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_does_not_advance_an_already_valid_tensor(self) -> None:
        actuators = FakeActuators(valid=True)
        timeline = FakeTimeline()
        advanced = False

        async def advance() -> None:
            nonlocal advanced
            advanced = True

        command = asyncio.run(
            synchronized_actual_command(actuators, timeline, advance)
        )

        self.assertIs(command, actuators.command)
        self.assertFalse(advanced)
        self.assertEqual(timeline.events, [])


if __name__ == "__main__":
    unittest.main()
