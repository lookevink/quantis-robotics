from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jepa_wm.control_safety import ControlInterlockEvidence, SimulatorSafetyLimits
from sim.isaac_control_runtime import (
    ControlContactSensors,
    LiveControlRuntime,
    LiveContactInterlock,
    LiveInsertionInterlock,
    read_contact,
    read_control_contact,
    refresh_live_control_runtime,
    synchronized_control_safety_snapshot,
    synchronized_insertion_safety_snapshot,
    synchronized_insertion_resolution_runtime,
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
    def test_insertion_interlock_aborts_immediately_on_attachment_loss(self) -> None:
        attachment = SimpleNamespace(attached=False)
        interlock = LiveInsertionInterlock(
            LiveContactInterlock(_Sensor(0.0), 2.0, "test motion"),
            attachment,
            True,
            "test motion",
        )

        with self.assertRaisesRegex(RuntimeError, "attachment state changed"):
            interlock.observe()

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

    def test_insertion_rebuilds_wrappers_after_physics_ready_update(self) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(), Mock()
        )
        refreshed_runtime = LiveControlRuntime(
            "session", old_runtime.stage, Mock(), Mock(), Mock()
        )
        captured = Mock()
        live = Mock()
        physics_ready = False
        events: list[str] = []

        async def advance() -> None:
            nonlocal physics_ready
            events.append("advance")
            physics_ready = True

        def refresh(runtime: LiveControlRuntime) -> LiveControlRuntime:
            if not physics_ready:
                raise RuntimeError("physics tensor view is not ready")
            events.append("refresh")
            self.assertIs(runtime, old_runtime)
            return refreshed_runtime

        pose = Mock()
        drive_target = Mock()

        def read(runtime):
            events.append("read")
            self.assertIs(runtime, refreshed_runtime)
            return live, pose, drive_target

        def observe():
            events.append(
                "observe resumed" if "refresh" not in events else "observe refreshed"
            )
            return SimpleNamespace(
                collision_detected=False,
                force_newtons=0.0,
            )

        with (
            patch(
                "sim.isaac_control_runtime.refresh_live_control_runtime",
                side_effect=refresh,
            ),
            patch(
                "sim.isaac_control_runtime._control_safety_pose_and_drive_target",
                side_effect=read,
            ),
            patch(
                "sim.isaac_control_runtime.LiveContactInterlock.observe",
                side_effect=observe,
            ),
        ):
            synchronized = asyncio.run(
                synchronized_insertion_safety_snapshot(
                    old_runtime,
                    timeline,
                    advance,
                    captured,
                    SimulatorSafetyLimits(),
                    operation="test insertion refresh",
                )
            )

        self.assertEqual(
            events,
            [
                "advance",
                "observe resumed",
                "refresh",
                "observe refreshed",
                "read",
            ],
        )
        self.assertIs(synchronized.runtime, refreshed_runtime)
        self.assertIs(synchronized.safety, live)
        self.assertIs(synchronized.pose, pose)
        self.assertIs(synchronized.active_drive_target, drive_target)
        live.validate_continuity.assert_called_once_with(
            captured, SimulatorSafetyLimits()
        )
        captured.validate_contact_continuity.assert_called_once_with(
            ControlInterlockEvidence(0.0, False)
        )
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_resolution_refresh_defers_bounded_drift_to_stable_baseline(self) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(), Mock()
        )
        refreshed_runtime = LiveControlRuntime(
            "session", old_runtime.stage, Mock(), Mock(attached=True), Mock()
        )
        captured = Mock(plug_attached=True)
        live = Mock(plug_attached=True)

        async def advance() -> None:
            return None

        with (
            patch(
                "sim.isaac_control_runtime.refresh_live_control_runtime",
                return_value=refreshed_runtime,
            ),
            patch(
                "sim.isaac_control_runtime._control_safety_snapshot",
                return_value=live,
            ),
            patch(
                "sim.isaac_control_runtime.LiveContactInterlock.observe",
                return_value=SimpleNamespace(
                    collision_detected=False,
                    force_newtons=0.0,
                ),
            ) as observe,
        ):
            synchronized = asyncio.run(
                synchronized_insertion_resolution_runtime(
                    old_runtime,
                    timeline,
                    advance,
                    captured,
                    SimulatorSafetyLimits(),
                    operation="test resolution refresh",
                )
            )

        self.assertIs(synchronized.runtime, refreshed_runtime)
        self.assertIs(synchronized.safety, live)
        live.validate_continuity.assert_not_called()
        self.assertEqual(observe.call_count, 2)
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_resolution_refresh_rejects_attachment_change(self) -> None:
        timeline = _Timeline(playing=False)
        runtime = LiveControlRuntime("session", object(), Mock(), Mock(), Mock())
        live = Mock(plug_attached=False)

        async def advance() -> None:
            return None

        with (
            patch(
                "sim.isaac_control_runtime.refresh_live_control_runtime",
                return_value=runtime,
            ),
            patch(
                "sim.isaac_control_runtime._control_safety_snapshot",
                return_value=live,
            ),
            patch(
                "sim.isaac_control_runtime.LiveContactInterlock.observe"
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "attachment changed"):
                asyncio.run(
                    synchronized_insertion_resolution_runtime(
                        runtime,
                        timeline,
                        advance,
                        Mock(plug_attached=True),
                        SimulatorSafetyLimits(),
                        operation="test resolution refresh",
                    )
                )

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
