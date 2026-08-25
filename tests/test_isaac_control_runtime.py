from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sim.isaac_control_runtime import (
    ControlContactSensors,
    LiveControlRuntime,
    LiveContactInterlock,
    read_contact,
    read_control_contact,
    refresh_live_control_runtime,
    synchronized_control_safety_snapshot,
)


class _Timeline:
    def __init__(self, *, playing: bool) -> None:
        self.playing = playing
        self.events: list[str] = []

    def is_playing(self) -> bool:
        return self.playing

    def play(self) -> None:
        self.playing = True
        self.events.append("play")

    def pause(self) -> None:
        self.playing = False
        self.events.append("pause")


class _Reading:
    def __init__(self, value: float, *, in_contact: bool = False) -> None:
        self.is_valid = True
        self.in_contact = in_contact
        self.value = value


class _Sensor:
    def __init__(self, value: float, *, in_contact: bool = False) -> None:
        self.reading = _Reading(value, in_contact=in_contact)

    def get_sensor_reading(self) -> _Reading:
        return self.reading


class ContactReadingTest(unittest.TestCase):
    def test_recreates_every_tensor_backed_runtime_wrapper(self) -> None:
        prims = ModuleType("isaacsim.core.experimental.prims")
        articulation_token = object()
        rigid_token = object()
        prims.Articulation = lambda path: (articulation_token, path)
        prims.RigidPrim = lambda path: (rigid_token, path)
        isaacsim = ModuleType("isaacsim")
        core = ModuleType("isaacsim.core")
        experimental = ModuleType("isaacsim.core.experimental")
        isaacsim.core = core
        core.experimental = experimental
        experimental.prims = prims
        stage = object()
        attachment = Mock()
        refreshed_attachment = Mock()
        attachment.with_refreshed_physics.return_value = refreshed_attachment
        old_runtime = LiveControlRuntime(
            "session",
            stage,
            Mock(),
            attachment,
            ControlContactSensors(Mock(), Mock()),
        )
        refreshed_actuators = Mock()
        refreshed_sensors = Mock()

        with (
            patch.dict(
                sys.modules,
                {
                    "isaacsim": isaacsim,
                    "isaacsim.core": core,
                    "isaacsim.core.experimental": experimental,
                    "isaacsim.core.experimental.prims": prims,
                },
            ),
            patch(
                "sim.isaac_control_runtime.create_actuators",
                return_value=refreshed_actuators,
            ) as create,
            patch(
                "sim.isaac_control_runtime.control_contact_sensors",
                return_value=refreshed_sensors,
            ) as sensors,
        ):
            refreshed = refresh_live_control_runtime(old_runtime)

        create.assert_called_once_with(stage, (articulation_token, "/World/Franka_R"))
        attachment.with_refreshed_physics.assert_called_once_with(
            (rigid_token, "/World/RJ45_Plug")
        )
        sensors.assert_called_once_with(
            stage, create=False, include_connector=True
        )
        self.assertIs(refreshed.actuators, refreshed_actuators)
        self.assertIs(refreshed.attachment, refreshed_attachment)
        self.assertIs(refreshed.sensor, refreshed_sensors)

    def test_pauses_when_resuming_the_timeline_fails(self) -> None:
        timeline = Mock()
        timeline.is_playing.return_value = False
        timeline.play.side_effect = RuntimeError("resume failed")

        async def advance() -> None:
            raise AssertionError("failed resume must not advance")

        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            asyncio.run(
                synchronized_control_safety_snapshot(
                    timeline,
                    Mock(),
                    Mock(),
                    Mock(),
                    advance,
                )
            )

        timeline.pause.assert_called_once_with()

    def test_refreshes_and_reads_all_physics_state_before_pause(self) -> None:
        timeline = _Timeline(playing=False)
        actuators = Mock()
        attachment = Mock(attached=True)
        refreshed = False

        async def advance() -> None:
            nonlocal refreshed
            refreshed = True

        def actual_command():
            if not refreshed or not timeline.is_playing():
                raise RuntimeError("articulation backend is unavailable")
            return SimpleNamespace(arm_positions=(0.0,) * 7, gripper_width_m=0.018)

        def world_pose():
            if not refreshed or not timeline.is_playing():
                raise RuntimeError("rigid-body backend is unavailable")
            return ((0.1, 0.2, 0.3), (1.0, 0.0, 0.0, 0.0))

        actuators.actual_command.side_effect = actual_command
        attachment.world_pose.side_effect = world_pose
        with patch(
            "sim.isaac_control_runtime.read_control_contact",
            return_value=(False, 0.0),
        ):
            state = asyncio.run(
                synchronized_control_safety_snapshot(
                    timeline,
                    actuators,
                    attachment,
                    Mock(),
                    advance,
                )
            )

        self.assertEqual(state.plug_position, (0.1, 0.2, 0.3))
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_live_interlock_retains_the_peak_that_triggers_abort(self) -> None:
        sensor = _Sensor(0.5)
        interlock = LiveContactInterlock(sensor, 2.0, "test motion")

        interlock.observe()
        sensor.reading = _Reading(2.5, in_contact=True)
        with self.assertRaisesRegex(RuntimeError, "test motion exceeded"):
            interlock.observe()

        self.assertEqual(interlock.evidence.maximum_contact_force_newtons, 2.5)
        self.assertTrue(interlock.evidence.collision_detected)

    def test_rejects_non_finite_force(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid force"):
            read_contact(_Sensor(float("nan")))

    def test_rejects_negative_force(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid force"):
            read_contact(_Sensor(-0.1))

    def test_uses_connector_force_without_treating_expected_contact_as_collision(self) -> None:
        collision, force = read_control_contact(
            ControlContactSensors(
                _Sensor(0.2),
                _Sensor(1.5, in_contact=True),
            )
        )

        self.assertFalse(collision)
        self.assertEqual(force, 1.5)


if __name__ == "__main__":
    unittest.main()
