from __future__ import annotations

import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from sim.isaac_control_runtime import (
    ControlContactSensors,
    LiveContactInterlock,
    read_contact,
    read_control_contact,
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
