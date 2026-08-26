from __future__ import annotations

import asyncio
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from jepa_wm.control_safety import ControlInterlockEvidence, SimulatorSafetyLimits
from jepa_wm.insertion_refresh import ControlSafetySnapshot
from jepa_wm.joint_drive import JointDriveTarget
from sim.isaac_control_runtime import (
    ControlContactSensors,
    LiveControlRuntime,
    LiveContactInterlock,
    LiveInsertionInterlock,
    read_contact,
    read_control_contact,
    refresh_live_control_runtime,
    synchronized_control_safety_snapshot,
    synchronized_insertion_frame_capture,
    synchronized_insertion_safety_snapshot,
    synchronized_insertion_resolution_runtime,
)
from sim.isaac_demo_runtime import FixedJointPlugMotion, PlugAttachment


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
        old_offset = np.asarray((0.1, 0.2, 0.3))
        old_motion = SimpleNamespace(
            prim=object(),
            hand_prim=object(),
            rigid_prim=object(),
            fixed_joint=object(),
            hand_to_plug_offset=old_offset,
        )
        collision_attributes = [object()]
        attachment = SimpleNamespace(
            motion=old_motion,
            collisions=SimpleNamespace(
                collision_attributes=collision_attributes,
                excluded_collision_paths=frozenset({"/World/RJ45_Plug/Body"}),
            ),
        )
        old_runtime = LiveControlRuntime(
            "session",
            stage,
            Mock(),
            attachment,
            SimpleNamespace(hand=Mock(), connector=Mock()),
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
        sensors.assert_called_once_with(
            stage, create=False, include_connector=True
        )
        self.assertIs(refreshed.actuators, refreshed_actuators)
        self.assertIsInstance(refreshed.attachment, PlugAttachment)
        self.assertIsInstance(refreshed.attachment.motion, FixedJointPlugMotion)
        self.assertEqual(
            refreshed.attachment.motion.rigid_prim,
            (rigid_token, "/World/RJ45_Plug"),
        )
        self.assertIsNot(refreshed.attachment.motion.hand_to_plug_offset, old_offset)
        np.testing.assert_array_equal(
            refreshed.attachment.motion.hand_to_plug_offset,
            old_offset,
        )
        self.assertIsNot(refreshed.attachment.collisions, attachment.collisions)
        self.assertEqual(
            refreshed.attachment.collisions.collision_attributes,
            collision_attributes,
        )
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
        drive_target = JointDriveTarget((0.0,) * 7, 0.04)

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

    def test_insertion_frame_capture_stays_live_and_interlocked_until_read(self) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(attached=True), Mock()
        )
        refreshed_runtime = LiveControlRuntime(
            "session", old_runtime.stage, Mock(), Mock(attached=True), Mock()
        )
        captured = Mock(plug_attached=True)
        live = Mock()
        pose = Mock()
        drive_target = JointDriveTarget((0.0,) * 7, 0.018)
        refreshed_runtime.actuators.current_command.return_value = SimpleNamespace(
            arm_positions=np.zeros(7),
            gripper_width_m=drive_target.gripper_width_m,
        )
        refreshed_runtime.actuators.actual_command.return_value = SimpleNamespace(
            arm_positions=np.zeros(7),
            gripper_width_m=drive_target.gripper_width_m,
        )
        events: list[str] = []

        async def advance() -> None:
            events.append("readiness update")

        async def capture(observe_safety) -> None:
            self.assertTrue(timeline.is_playing())
            events.append("camera update")
            observe_safety()

        def refresh(runtime: LiveControlRuntime) -> LiveControlRuntime:
            self.assertIs(runtime, old_runtime)
            events.append("refresh")
            return refreshed_runtime

        def read(runtime: LiveControlRuntime):
            self.assertTrue(timeline.is_playing())
            self.assertIs(runtime, refreshed_runtime)
            events.append("read")
            return live, pose, drive_target

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
                return_value=SimpleNamespace(
                    collision_detected=False,
                    force_newtons=0.0,
                ),
            ),
        ):
            synchronized = asyncio.run(
                synchronized_insertion_frame_capture(
                    old_runtime,
                    timeline,
                    advance,
                    captured,
                    SimulatorSafetyLimits(),
                    capture,
                    expected_active_drive_target=drive_target,
                    operation="test insertion follow-up frame",
                )
            )

        self.assertEqual(
            events,
            ["readiness update", "refresh", "camera update", "read"],
        )
        self.assertIs(synchronized.safety, live)
        live.validate_followup_continuity.assert_called_once_with(
            captured,
            drive_target.gripper_width_m,
            SimulatorSafetyLimits(),
        )
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_insertion_frame_capture_accepts_settling_toward_active_target(self) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(attached=True), Mock()
        )
        refreshed_runtime = LiveControlRuntime(
            "session", old_runtime.stage, Mock(), Mock(attached=True), Mock()
        )
        captured = ControlSafetySnapshot(
            (
                0.8433861136436462,
                -1.3164855241775513,
                -1.1596380472183228,
                -2.6030569076538086,
                0.4332136809825897,
                3.494144916534424,
                -0.7353372573852539,
            ),
            0.01801662240177393,
            (-0.07707861810922623, -0.24581295251846313, 1.3099448680877686),
            0.0,
            False,
            True,
        )
        live = ControlSafetySnapshot(
            (
                0.84379643201828,
                -1.3162521123886108,
                -1.1597918272018433,
                -2.6027512550354004,
                0.4331771433353424,
                3.4940218925476074,
                -0.7353372573852539,
            ),
            0.01802041195333004,
            (-0.07722251117229462, -0.24587148427963257, 1.309919834136963),
            0.0,
            False,
            True,
        )
        active_target = JointDriveTarget(
            (
                0.8438659343383477,
                -1.3155067699643368,
                -1.1587890608769889,
                -2.602214382216666,
                0.43314604049138433,
                3.493953366889305,
                -0.735345115973845,
            ),
            0.01802057959139347,
        )
        refreshed_runtime.actuators.current_command.return_value = SimpleNamespace(
            arm_positions=np.asarray(active_target.joint_positions),
            gripper_width_m=active_target.gripper_width_m,
        )
        refreshed_runtime.actuators.actual_command.return_value = SimpleNamespace(
            arm_positions=np.asarray(live.joint_positions),
            gripper_width_m=live.gripper_width_m,
        )

        async def advance() -> None:
            return None

        async def capture(_observe_safety) -> None:
            return None

        with (
            patch(
                "sim.isaac_control_runtime.refresh_live_control_runtime",
                return_value=refreshed_runtime,
            ),
            patch(
                "sim.isaac_control_runtime._control_safety_pose_and_drive_target",
                return_value=(live, Mock(), active_target),
            ) as read,
            patch(
                "sim.isaac_control_runtime.LiveContactInterlock.observe",
                return_value=SimpleNamespace(
                    collision_detected=False,
                    force_newtons=0.0,
                ),
            ),
        ):
            synchronized = asyncio.run(
                synchronized_insertion_frame_capture(
                    old_runtime,
                    timeline,
                    advance,
                    captured,
                    SimulatorSafetyLimits(),
                    capture,
                    expected_active_drive_target=active_target,
                    operation="test insertion follow-up settling",
                )
            )
            changed_target = JointDriveTarget(
                active_target.joint_positions,
                active_target.gripper_width_m + 0.0001,
            )
            read.return_value = (live, Mock(), changed_target)
            with self.assertRaisesRegex(RuntimeError, "state changed"):
                asyncio.run(
                    synchronized_insertion_frame_capture(
                        old_runtime,
                        _Timeline(playing=False),
                        advance,
                        captured,
                        SimulatorSafetyLimits(),
                        capture,
                        expected_active_drive_target=active_target,
                        operation="test changed insertion drive target",
                    )
                )

        self.assertEqual(synchronized.safety, live)
        self.assertEqual(synchronized.active_drive_target, active_target)

    def test_insertion_frame_capture_waits_for_active_gripper_target(self) -> None:
        timeline = _Timeline(playing=False)
        update_count = 0
        active_target = JointDriveTarget(
            (
                0.8438414332563189,
                -1.315508101544882,
                -1.1587810713937188,
                -2.602226899073789,
                0.43315143339259177,
                3.4939389858194185,
                -0.7352948488082698,
            ),
            0.018020467832684517,
        )
        captured = ControlSafetySnapshot(
            (
                0.8433670997619629,
                -1.3164868354797363,
                -1.1596312522888184,
                -2.603066921234131,
                0.4332185685634613,
                3.494137763977051,
                -0.7353372573852539,
            ),
            0.018016536720097065,
            (-0.07707371562719345, -0.2458123117685318, 1.3099448680877686),
            0.0,
            False,
            True,
        )
        unsettled = ControlSafetySnapshot(
            captured.joint_positions,
            captured.gripper_width_m,
            captured.plug_position,
            0.0,
            False,
            True,
        )
        settled = ControlSafetySnapshot(
            (
                0.8437719345092773,
                -1.3162533044815063,
                -1.1597838401794434,
                -2.6027638912200928,
                0.4331825375556946,
                3.4940075874328613,
                -0.7353372573852539,
            ),
            0.018020231276750565,
            (-0.07722251117229462, -0.24587148427963257, 1.309919834136963),
            0.0,
            False,
            True,
        )
        actuators = Mock()
        actuators.current_command.return_value = SimpleNamespace(
            arm_positions=np.asarray(active_target.joint_positions),
            gripper_width_m=active_target.gripper_width_m,
        )
        actuators.actual_command.side_effect = lambda: SimpleNamespace(
            arm_positions=np.asarray(active_target.joint_positions),
            gripper_width_m=(
                captured.gripper_width_m
                if update_count == 1
                else settled.gripper_width_m
            ),
        )
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(attached=True), Mock()
        )
        refreshed_runtime = LiveControlRuntime(
            "session", old_runtime.stage, actuators, Mock(attached=True), Mock()
        )

        async def advance() -> None:
            nonlocal update_count
            update_count += 1

        async def capture(_observe_safety) -> None:
            return None

        def read(_runtime: LiveControlRuntime):
            snapshot = unsettled if update_count == 1 else settled
            return snapshot, Mock(), active_target

        with (
            patch(
                "sim.isaac_control_runtime.refresh_live_control_runtime",
                return_value=refreshed_runtime,
            ),
            patch(
                "sim.isaac_control_runtime._control_safety_pose_and_drive_target",
                side_effect=read,
            ),
            patch(
                "sim.isaac_control_runtime.LiveContactInterlock.observe",
                return_value=SimpleNamespace(
                    collision_detected=False,
                    force_newtons=0.0,
                ),
            ),
        ):
            synchronized = asyncio.run(
                synchronized_insertion_frame_capture(
                    old_runtime,
                    timeline,
                    advance,
                    captured,
                    SimulatorSafetyLimits(),
                    capture,
                    expected_active_drive_target=active_target,
                    operation="test insertion follow-up gripper settling",
                )
            )

        self.assertEqual(update_count, 2)
        self.assertEqual(synchronized.safety, settled)

    def test_insertion_frame_capture_bounds_gripper_settling_before_camera(self) -> None:
        timeline = _Timeline(playing=False)
        update_count = 0
        camera_called = False
        active_target = JointDriveTarget((0.0,) * 7, 0.02)
        captured = ControlSafetySnapshot(
            (0.0,) * 7,
            0.018,
            (0.0, 0.0, 0.0),
            0.0,
            False,
            True,
        )
        actuators = Mock()
        actuators.current_command.return_value = SimpleNamespace(
            arm_positions=np.zeros(7),
            gripper_width_m=active_target.gripper_width_m,
        )
        actuators.actual_command.return_value = SimpleNamespace(
            arm_positions=np.zeros(7),
            gripper_width_m=captured.gripper_width_m,
        )
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(attached=True), Mock()
        )
        refreshed_runtime = LiveControlRuntime(
            "session", old_runtime.stage, actuators, Mock(attached=True), Mock()
        )

        async def advance() -> None:
            nonlocal update_count
            update_count += 1

        async def capture(_observe_safety) -> None:
            nonlocal camera_called
            camera_called = True

        with (
            patch(
                "sim.isaac_control_runtime.refresh_live_control_runtime",
                return_value=refreshed_runtime,
            ),
            patch(
                "sim.isaac_control_runtime.LiveContactInterlock.observe",
                return_value=SimpleNamespace(
                    collision_detected=False,
                    force_newtons=0.0,
                ),
            ) as observe,
        ):
            with self.assertRaisesRegex(RuntimeError, "gripper did not settle"):
                asyncio.run(
                    synchronized_insertion_frame_capture(
                        old_runtime,
                        timeline,
                        advance,
                        captured,
                        SimulatorSafetyLimits(),
                        capture,
                        expected_active_drive_target=active_target,
                        operation="test bounded insertion follow-up settling",
                    )
                )

        self.assertEqual(update_count, 9)
        self.assertEqual(observe.call_count, 10)
        self.assertFalse(camera_called)
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

    def test_uses_cross_generation_connector_sensor_structure(self) -> None:
        collision, force = read_control_contact(
            SimpleNamespace(
                hand=_Sensor(0.2),
                connector=_Sensor(1.5, in_contact=True),
            )
        )

        self.assertFalse(collision)
        self.assertEqual(force, 1.5)


if __name__ == "__main__":
    unittest.main()
