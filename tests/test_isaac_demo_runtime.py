from __future__ import annotations

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from sim.isaac_demo_runtime import (
    Actuators,
    FixedJointPlugMotion,
    ContactReading,
    JointCommand,
    PlugAttachment,
    PlugCollisionPolicy,
    _advance_sample,
    move_joint_command,
    move_joint_command_over_physics_steps,
    recording_snapshot,
    resume_live_simulation,
)
from sim.isaac_exploration import prepare_recording_stage
from jepa.contract import ObservationStage
from sim.demo_sequence import Phase
from sim.recording import RecordingLabel, RecordingMoment


class _RigidPrim:
    def __init__(self) -> None:
        self.positions = None

    def set_world_poses(self, *, positions) -> None:
        self.positions = positions

    def get_world_poses(self):
        return (
            np.asarray(self.positions, dtype=np.float64),
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        )


class PlugAttachmentTest(unittest.TestCase):
    def test_follow_updates_the_bound_physics_body(self) -> None:
        rigid = _RigidPrim()
        attachment = PlugAttachment(
            motion=FixedJointPlugMotion(
                prim=object(),
                hand_prim=object(),
                rigid_prim=rigid,
                fixed_joint=object(),
                hand_to_plug_offset=np.array([0.04, 0.0, 0.0]),
            ),
            collisions=PlugCollisionPolicy([]),
        )

        attachment.follow(np.array([0.1, -0.2, 1.3]))

        self.assertIsNone(rigid.positions)
        rigid.positions = [[0.14, -0.2, 1.3]]
        position, orientation = attachment.world_pose()
        np.testing.assert_allclose(position, [0.14, -0.2, 1.3])
        np.testing.assert_allclose(orientation, [1.0, 0.0, 0.0, 0.0])


class SafetyInterlockTest(unittest.IsolatedAsyncioTestCase):
    def test_resume_live_simulation_enables_app_driven_updates(self) -> None:
        timeline = SimpleNamespace(playing=False)
        timeline.set_auto_update = Mock()
        timeline.is_playing = lambda: timeline.playing
        timeline.play = Mock()

        self.assertTrue(resume_live_simulation(timeline))

        timeline.set_auto_update.assert_called_once_with(True)
        timeline.play.assert_called_once_with()

    @staticmethod
    def _omni_modules(app, timeline, *, simulation_time=None):
        omni = ModuleType("omni")
        kit = ModuleType("omni.kit")
        kit_app = ModuleType("omni.kit.app")
        timeline_module = ModuleType("omni.timeline")
        isaacsim = ModuleType("isaacsim")
        core = ModuleType("isaacsim.core")
        simulation_manager = ModuleType("isaacsim.core.simulation_manager")
        kit_app.get_app = lambda: app
        timeline_module.get_timeline_interface = lambda: timeline
        simulation_manager.SimulationManager = SimpleNamespace(
            get_simulation_time=(
                simulation_time
                if simulation_time is not None
                else timeline.get_current_time
            )
        )
        omni.kit = kit
        omni.timeline = timeline_module
        kit.app = kit_app
        isaacsim.core = core
        core.simulation_manager = simulation_manager
        return {
            "omni": omni,
            "omni.kit": kit,
            "omni.kit.app": kit_app,
            "omni.timeline": timeline_module,
            "isaacsim": isaacsim,
            "isaacsim.core": core,
            "isaacsim.core.simulation_manager": simulation_manager,
        }

    async def test_uses_physics_time_when_the_presentation_clock_is_frozen(
        self,
    ) -> None:
        timeline = SimpleNamespace(get_current_time=lambda: 19.5)
        physics = SimpleNamespace(current=2.0)

        class _App:
            updates = 0

            async def next_update_async(self) -> None:
                self.updates += 1
                physics.current += 0.01

        app = _App()
        with patch.dict(
            "sys.modules",
            self._omni_modules(
                app,
                timeline,
                simulation_time=lambda: physics.current,
            ),
        ):
            await _advance_sample(0.02)

        self.assertEqual(app.updates, 2)

    async def test_polls_during_sample_and_stops_on_intermediate_force(self) -> None:
        class _Timeline:
            current = 0.0

            def get_current_time(self) -> float:
                return self.current

        timeline = _Timeline()

        class _App:
            updates = 0

            async def next_update_async(self) -> None:
                self.updates += 1
                timeline.current += 0.01

        app = _App()
        modules = self._omni_modules(app, timeline)

        def observe() -> ContactReading:
            if app.updates == 2:
                raise RuntimeError("force exceeded")
            return ContactReading()

        with patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "force exceeded"):
                await _advance_sample(0.25, observe)

        self.assertEqual(app.updates, 2)

    async def test_one_physics_update_waits_for_simulation_time_to_advance(self) -> None:
        class _Timeline:
            current = 0.0

            def get_current_time(self) -> float:
                return self.current

        timeline = _Timeline()

        class _App:
            updates = 0

            async def next_update_async(self) -> None:
                self.updates += 1
                if self.updates == 3:
                    timeline.current = 0.25

        app = _App()

        def observe() -> ContactReading:
            return ContactReading(False, 1.0 if app.updates == 2 else 0.0)

        with patch.dict("sys.modules", self._omni_modules(app, timeline)):
            reading = await _advance_sample(None, observe)

        self.assertEqual(app.updates, 3)
        self.assertEqual(reading.force_newtons, 1.0)

    def test_rejects_invalid_contact_reading_before_peak_aggregation(self) -> None:
        for value in (float("nan"), -0.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "contact reading"):
                    ContactReading(False, value)

    async def test_preserves_transient_force_until_the_recorded_frame(self) -> None:
        class _Timeline:
            current = 0.0

            def get_current_time(self) -> float:
                return self.current

        timeline = _Timeline()

        class _App:
            updates = 0

            async def next_update_async(self) -> None:
                self.updates += 1
                timeline.current += 0.05

        app = _App()

        def observe() -> ContactReading:
            return ContactReading(False, 1.5 if app.updates == 2 else 0.0)

        with patch.dict("sys.modules", self._omni_modules(app, timeline)):
            reading = await _advance_sample(0.25, observe)

        self.assertFalse(reading.collision_detected)
        self.assertEqual(reading.force_newtons, 1.5)


class DriveOnlyMotionTest(unittest.IsolatedAsyncioTestCase):
    async def test_recorded_motion_settles_each_target_over_physics_steps(self) -> None:
        physics = SimpleNamespace(current=0.0)
        applied: list[float] = []
        actuators = Mock()

        def apply(command: JointCommand) -> None:
            applied.append(float(command.arm_positions[0]))

        actuators.apply_drive_command.side_effect = apply
        actuators.actual_command.return_value = JointCommand(
            np.full(7, 0.25), 0.04
        )
        attachment = Mock(hand_prim=object())
        recorder = Mock(capture_current=AsyncMock())

        class _SimulationManager:
            @staticmethod
            def get_physics_dt() -> float:
                return 0.0625

            @staticmethod
            def step(*, steps, callback, update_fabric) -> None:
                self.assertFalse(update_fabric)
                for index in range(steps):
                    physics.current += 0.0625
                    callback(index + 1, steps)

        class _RenderingManager:
            render_async = AsyncMock()

        simulation_manager = ModuleType("isaacsim.core.simulation_manager")
        simulation_manager.SimulationManager = _SimulationManager
        rendering_manager = ModuleType("isaacsim.core.rendering_manager")
        rendering_manager.RenderingManager = _RenderingManager

        recorded_arm_positions: list[np.ndarray] = []

        def snapshot(*args, **_kwargs):
            recorded_arm_positions.append(args[2].arm_positions.copy())
            return SimpleNamespace(simulation_time_seconds=physics.current)

        with (
            patch.dict(
                "sys.modules",
                {
                    "isaacsim.core.simulation_manager": simulation_manager,
                    "isaacsim.core.rendering_manager": rendering_manager,
                },
            ),
            patch(
                "sim.isaac_demo_runtime.physics_simulation_time_seconds",
                side_effect=lambda: physics.current,
            ),
            patch(
                "sim.isaac_demo_runtime.world_pose",
                return_value=(np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
            ),
            patch("sim.isaac_demo_runtime.recording_snapshot", side_effect=snapshot),
        ):
            sample_times = await move_joint_command_over_physics_steps(
                actuators,
                JointCommand(np.zeros(7), 0.04),
                JointCommand(np.ones(7), 0.04),
                attachment,
                frame_count=1,
                phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                stage=ObservationStage.CABLE_GRASPED,
                recorder=recorder,
                sample_period_seconds=0.25,
            )

        self.assertEqual(applied, [1.0])
        np.testing.assert_allclose(recorded_arm_positions, [np.full(7, 0.25)])
        self.assertEqual(sample_times, (0.25,))
        _RenderingManager.render_async.assert_awaited_once_with()
        recorder.capture_current.assert_awaited_once()
        actuators.articulation.set_dof_positions.assert_not_called()

    async def test_recording_cadence_is_configured_before_stage_reset(self) -> None:
        events: list[object] = []

        class _RenderingManager:
            @staticmethod
            def get_dt() -> float:
                events.append("get_dt")
                return 1.0 / 60.0

            @staticmethod
            def set_dt(period_seconds: float) -> None:
                events.append(("set_dt", period_seconds))

        async def reset_stage() -> None:
            events.append("reset_stage")

        rendering_manager = ModuleType("isaacsim.core.rendering_manager")
        rendering_manager.RenderingManager = _RenderingManager
        with (
            patch.dict(
                "sys.modules",
                {"isaacsim.core.rendering_manager": rendering_manager},
            ),
            patch("sim.isaac_demo_runtime.reset_stage", reset_stage),
        ):
            original_period = await prepare_recording_stage(0.25)

        self.assertEqual(original_period, 1.0 / 60.0)
        self.assertEqual(
            events,
            ["get_dt", ("set_dt", 0.25), "reset_stage"],
        )

    def test_recording_snapshot_uses_the_physics_clock(self) -> None:
        hand = Mock()
        hand.GetStage.return_value.GetPrimAtPath.return_value = object()
        attachment = Mock(hand_prim=hand, attached=True)
        attachment.world_pose.return_value = (
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        omni = ModuleType("omni")
        timeline = ModuleType("omni.timeline")
        timeline.get_timeline_interface = Mock(
            return_value=SimpleNamespace(get_current_time=Mock(return_value=2.0))
        )
        omni.timeline = timeline

        with (
            patch.dict(
                "sys.modules",
                {"omni": omni, "omni.timeline": timeline},
            ),
            patch(
                "sim.isaac_demo_runtime.world_pose",
                return_value=(np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
            ),
            patch(
                "sim.isaac_demo_runtime.physics_simulation_time_seconds",
                return_value=0.25,
            ),
            patch(
                "sim.isaac_demo_runtime.DroidPose.from_world_poses",
                return_value=Mock(),
            ),
        ):
            snapshot = recording_snapshot(
                RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                ObservationStage.CABLE_GRASPED,
                JointCommand(np.zeros(7), 0.04),
                attachment,
            )

        self.assertEqual(snapshot.simulation_time_seconds, 0.25)

    async def test_recorded_motion_captures_the_timestamped_current_frame(self) -> None:
        actuators = Mock()
        actuators.actual_command.return_value = JointCommand(np.zeros(7), 0.04)
        attachment = Mock(hand_prim=object())
        recorder = Mock()
        recorder.capture = AsyncMock()
        recorder.capture_current = AsyncMock()
        snapshot = SimpleNamespace(simulation_time_seconds=1.25)

        async def advance(_period, _observer):
            return ContactReading()

        with (
            patch("sim.isaac_demo_runtime._advance_sample", advance),
            patch(
                "sim.isaac_demo_runtime.world_pose",
                return_value=(np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
            ),
            patch(
                "sim.isaac_demo_runtime.recording_snapshot",
                return_value=snapshot,
            ),
        ):
            sample_times = await move_joint_command(
                actuators,
                JointCommand(np.zeros(7), 0.04),
                JointCommand(np.full(7, 0.001), 0.04),
                attachment,
                frame_count=1,
                phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                stage=ObservationStage.CABLE_GRASPED,
                recorder=recorder,
                sample_period_seconds=0.25,
            )

        self.assertEqual(sample_times, (1.25,))
        recorder.capture_current.assert_awaited_once_with(snapshot)
        recorder.capture.assert_not_awaited()

    def test_explicit_reset_retains_direct_state_initialization(self) -> None:
        articulation = Mock()
        arm_attributes = [Mock() for _ in range(7)]
        finger_attributes = [Mock(), Mock()]
        actuators = Actuators(
            articulation,
            arm_attributes,
            finger_attributes,
        )
        command = JointCommand(np.full(7, 0.001), 0.04)
        drive_target = JointCommand(np.full(7, 0.002), 0.06)

        actuators.set_reset_state(command, drive_target=drive_target)

        self.assertEqual(articulation.set_dof_positions.call_count, 2)
        for attribute in arm_attributes:
            attribute.Set.assert_called_once_with(float(np.rad2deg(0.002)))
        for attribute in finger_attributes:
            attribute.Set.assert_called_once_with(0.03)
        np.testing.assert_array_equal(
            articulation.set_dof_positions.call_args_list[0].kwargs["positions"],
            command.arm_positions,
        )
        np.testing.assert_array_equal(
            articulation.set_dof_positions.call_args_list[1].kwargs["positions"],
            np.asarray([0.02, 0.02]),
        )
        articulation.set_dof_velocities.assert_called_once()
        velocity_call = articulation.set_dof_velocities.call_args
        np.testing.assert_array_equal(
            velocity_call.kwargs["velocities"], np.zeros(9, dtype=np.float64)
        )
        np.testing.assert_array_equal(
            velocity_call.kwargs["dof_indices"], np.arange(9)
        )

    async def test_runtime_motion_never_sets_articulation_state_directly(self) -> None:
        articulation = Mock()
        articulation.get_dof_positions.return_value = np.asarray(
            [0.001] * 7 + [0.02, 0.02], dtype=np.float64
        )
        arm_attributes = [Mock() for _ in range(7)]
        finger_attributes = [Mock(), Mock()]
        actuators = Actuators(
            articulation,
            arm_attributes,
            finger_attributes,
        )
        attachment = Mock(hand_prim=object())

        async def advance(_period, _observer):
            return ContactReading()

        with (
            patch("sim.isaac_demo_runtime._advance_sample", advance),
            patch(
                "sim.isaac_demo_runtime.world_pose",
                return_value=(np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
            ),
            patch(
                "sim.isaac_demo_runtime.recording_snapshot",
                return_value=SimpleNamespace(simulation_time_seconds=None),
            ),
        ):
            await move_joint_command(
                actuators,
                JointCommand(np.zeros(7), 0.04),
                JointCommand(np.full(7, 0.001), 0.04),
                attachment,
                frame_count=1,
                phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                stage=ObservationStage.CABLE_GRASPED,
                recorder=None,
            )

        articulation.set_dof_positions.assert_not_called()
        self.assertEqual(articulation.set_dof_position_targets.call_count, 2)
        arm_target = articulation.set_dof_position_targets.call_args_list[0]
        np.testing.assert_allclose(
            arm_target.args[0],
            np.full(7, 0.001),
        )
        np.testing.assert_array_equal(
            arm_target.kwargs["dof_indices"],
            np.arange(7),
        )
        finger_target = articulation.set_dof_position_targets.call_args_list[1]
        np.testing.assert_allclose(finger_target.args[0], np.asarray([0.02]))
        np.testing.assert_array_equal(
            finger_target.kwargs["dof_indices"],
            np.asarray([7]),
        )
        for attribute in (*arm_attributes, *finger_attributes):
            attribute.Set.assert_called_once()


if __name__ == "__main__":
    unittest.main()
