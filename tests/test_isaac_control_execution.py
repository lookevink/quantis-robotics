from __future__ import annotations

import asyncio
from dataclasses import replace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_safety import (
    ControlGateReason,
    INSERTION_TARGET_PROGRESS,
    SimulatorSafetyLimits,
)
from sim.isaac_control_execution import (
    ExecutionSafetyContext,
    capture_synchronized_post_action,
    select_safe_projection,
    rollback_control_command,
    settle_joint_command,
    synchronized_actual_command,
)
from sim.isaac_demo_kinematics import SolvedPose
from sim.isaac_demo_runtime import JointCommand


class ControlProjectionTest(unittest.TestCase):
    def test_target_progress_rejects_overshoot_until_quarter_scale(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=100.0,
            context_frame=Path("context.png"),
            target=ControlTarget(
                Path("target.png"),
                DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            ),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        proposal = ProposedControl(
            123,
            100.1,
            (
                DroidAction((0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
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

        def solve(pose: DroidPose, warm_start: np.ndarray) -> SolvedPose:
            return SolvedPose(pose, warm_start, np.zeros(3), 0.04, 0.0, 0.0)

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            target_progress=INSERTION_TARGET_PROGRESS,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(attempts[-1].scale, DroidActionScale.uniform(0.25))
        self.assertTrue(attempts[-1].gate.passed)
        self.assertTrue(
            all(
                attempt.gate.reasons
                == (ControlGateReason.TARGET_PROGRESS_INSUFFICIENT,)
                for attempt in attempts[:-1]
            )
        )

        missing_target = replace(
            observation,
            target=ControlTarget(Path("target.png")),
        )
        missing_attempts, missing_selection = select_safe_projection(
            replace(context, observation=missing_target),
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            target_progress=INSERTION_TARGET_PROGRESS,
        )
        self.assertIsNone(missing_selection)
        self.assertTrue(
            all(
                attempt.gate.reasons == (ControlGateReason.TARGET_POSE_MISSING,)
                for attempt in missing_attempts
            )
        )

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

        bounded_attempts, bounded = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            action_scales=(DroidActionScale(0.5, 0.125, 1.0),),
        )
        self.assertIsNotNone(bounded)
        self.assertEqual(
            tuple(attempt.scale for attempt in bounded_attempts),
            (DroidActionScale(0.5, 0.125, 1.0),),
        )


class FakeTimeline:
    def __init__(self, articulation: FakeArticulation | None = None) -> None:
        self.events: list[str] = []
        self.articulation = articulation

    def play(self) -> None:
        self.events.append("play")

    def pause(self) -> None:
        self.events.append("pause")
        if self.articulation is not None:
            self.articulation.valid = False


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

        observed = 0

        def observe_safety() -> object:
            nonlocal observed
            observed += 1
            return object()

        async def capture(
            *_args: object, observe_safety=None
        ) -> dict[str, object]:
            self.assertIsNotNone(observe_safety)
            observe_safety()
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
                    observe_safety=observe_safety,
                )
            )

        self.assertIs(captured.command, after)
        self.assertIs(captured.snapshot, snapshot)
        self.assertEqual(captured.contact_force_newtons, 0.5)
        self.assertEqual(observed, 1)
        recording.assert_called_once()
        self.assertIs(recording.call_args.args[2], after)

    def test_refreshes_a_stale_paused_physics_tensor_before_reading(self) -> None:
        actuators = FakeActuators(valid=False)
        timeline = FakeTimeline(actuators.articulation)

        async def advance() -> None:
            actuators.articulation.valid = True

        command = asyncio.run(
            synchronized_actual_command(actuators, timeline, advance)
        )

        self.assertIs(command, actuators.command)
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_pauses_when_command_tensor_resume_fails(self) -> None:
        actuators = FakeActuators(valid=False)
        timeline = FakeTimeline()
        timeline.play = Mock(side_effect=RuntimeError("resume failed"))

        async def advance() -> None:
            raise AssertionError("failed resume must not advance")

        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            asyncio.run(
                synchronized_actual_command(actuators, timeline, advance)
            )

        self.assertEqual(timeline.events, ["pause"])

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

    def test_settling_polls_safety_after_every_physics_update(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)
        updates = 0
        observations = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1

        def observe_safety() -> object:
            nonlocal observations
            observations += 1
            if observations == 2:
                raise RuntimeError("transient force")
            return object()

        with self.assertRaisesRegex(RuntimeError, "transient force"):
            asyncio.run(
                settle_joint_command(
                    actuators,
                    np.ones(7),
                    advance,
                    observe_safety=observe_safety,
                )
            )

        self.assertEqual(updates, 2)
        self.assertEqual(observations, 2)

    def test_rollback_is_observed_and_verified_after_its_physics_update(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.ones(7), 0.02)
        actuators.apply = lambda command: setattr(actuators, "command", command)
        events = []

        async def advance() -> None:
            events.append("advance")

        def observe_safety() -> object:
            events.append("observe")
            return object()

        attachment = type("Attachment", (), {"attached": True})()
        asyncio.run(
            rollback_control_command(
                actuators,
                target,
                attachment,
                advance,
                expected_attachment=True,
                observe_safety=observe_safety,
            )
        )

        self.assertEqual(events, ["advance", "observe"])
        self.assertIs(actuators.command, target)


if __name__ == "__main__":
    unittest.main()
