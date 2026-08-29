from __future__ import annotations

import asyncio
from contextlib import contextmanager
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import sim.isaac_control_runtime as control_runtime

from jepa_wm.control_safety import ControlInterlockEvidence, SimulatorSafetyLimits
from jepa_wm.insertion_refresh import (
    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS,
    ControlSafetySnapshot,
)
from jepa_wm.joint_drive import JointDriveTarget
from sim.isaac_control_runtime import (
    ControlContactSensors,
    LiveControlRuntime,
    LiveContactInterlock,
    LiveInsertionInterlock,
    read_contact,
    read_control_contact,
    synchronized_control_safety_snapshot,
    synchronized_contact_grasp_execution_runtime,
    synchronized_contact_grasp_safety_snapshot,
    synchronized_insertion_frame_capture,
    synchronized_insertion_safety_snapshot,
    synchronized_insertion_resolution_runtime,
)


class _Timeline:
    def __init__(self, *, playing: bool) -> None:
        self.playing = playing
        self.events: list[str] = []
        self.auto_update = False

    def is_playing(self) -> bool:
        return self.playing

    def play(self) -> None:
        self.playing = True
        self.events.append("play")

    def set_auto_update(self, value: bool) -> None:
        self.auto_update = value

    def pause(self) -> None:
        self.playing = False
        self.events.append("pause")


class _DeferredPauseTimeline(_Timeline):
    """Model Isaac applying pause on the next application update."""

    def __init__(self) -> None:
        super().__init__(playing=True)
        self.pause_requested = False

    def pause(self) -> None:
        self.pause_requested = True
        self.events.append("pause")

    def commit_pending_pause(self) -> None:
        if self.pause_requested:
            self.playing = False
            self.pause_requested = False


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


@contextmanager
def _retain_test_runtime():
    """Keep focused tests independent of the Isaac articulation module."""

    def refresh(runtime: LiveControlRuntime) -> LiveControlRuntime:
        actuators = runtime.actuators
        if not isinstance(actuators.arm_attributes, list):
            actuators.arm_attributes = []
        if not isinstance(actuators.finger_attributes, list):
            actuators.finger_attributes = []
        return LiveControlRuntime(
            runtime.session_id,
            runtime.stage,
            SimpleNamespace(
                articulation=SimpleNamespace(
                    is_physics_tensor_entity_valid=lambda: True
                ),
                arm_attributes=list(actuators.arm_attributes),
                finger_attributes=list(actuators.finger_attributes),
                current_command=actuators.current_command,
                actual_command=actuators.actual_command,
            ),
            runtime.attachment,
            runtime.sensor,
        )

    with (
        patch.object(
            control_runtime,
            "repair_invalid_live_control_physics_view",
        ),
        patch.object(
            control_runtime,
            "refresh_live_control_articulation",
            side_effect=refresh,
        ),
    ):
        yield


class ContactReadingTest(unittest.TestCase):
    def test_initial_contact_grasp_holds_stable_captured_state_despite_target_bias(
        self,
    ) -> None:
        captured = ControlSafetySnapshot(
            (0.003,) + (0.0,) * 6,
            0.0402,
            (0.0, 0.0, 0.0),
            0.0,
            False,
            False,
        )
        live = ControlSafetySnapshot(
            captured.joint_positions,
            captured.gripper_width_m,
            captured.plug_position,
            0.0,
            False,
            False,
        )
        drive_target = JointDriveTarget((0.0,) * 7, 0.04)

        live.validate_initial_contact_grasp_continuity(
            captured,
            maximum_gripper_error_meters=(
                MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
            ),
        )
        with self.assertRaisesRegex(ValueError, "active drive target"):
            live.validate_followup_continuity(
                captured,
                drive_target,
                maximum_gripper_error_meters=(
                    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
                ),
            )

        drifted = ControlSafetySnapshot(
            captured.joint_positions,
            captured.gripper_width_m + 3e-4,
            captured.plug_position,
            0.0,
            False,
            False,
        )
        with self.assertRaisesRegex(ValueError, "initial contact-grasp"):
            drifted.validate_initial_contact_grasp_continuity(
                captured,
                maximum_gripper_error_meters=(
                    MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
                ),
            )

    def test_contact_grasp_continuity_accepts_settling_to_unchanged_drive_target(
        self,
    ) -> None:
        captured = ControlSafetySnapshot(
            (0.003,) + (0.0,) * 6,
            0.0402,
            (0.0, 0.0, 0.0),
            0.0,
            False,
            True,
        )
        live = ControlSafetySnapshot(
            (0.0,) * 7,
            0.04,
            captured.plug_position,
            0.0,
            False,
            True,
        )
        drive_target = JointDriveTarget((0.0,) * 7, 0.04)

        live.validate_followup_continuity(
            captured,
            drive_target,
            maximum_gripper_error_meters=(
                MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
            ),
        )

    def test_contact_grasp_gripper_floor_does_not_weaken_insertion_continuity(self) -> None:
        captured = ControlSafetySnapshot(
            (0.0,) * 7,
            0.04,
            (0.0, 0.0, 0.0),
            0.0,
            False,
            False,
        )
        live = ControlSafetySnapshot(
            captured.joint_positions,
            0.0402,
            captured.plug_position,
            0.0,
            False,
            False,
        )

        drive_target = JointDriveTarget(captured.joint_positions, 0.04)
        with self.assertRaisesRegex(ValueError, "active drive target"):
            live.validate_followup_continuity(captured, drive_target)
        live.validate_followup_continuity(
            captured,
            drive_target,
            maximum_gripper_error_meters=(
                MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
            ),
        )

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

    def test_synchronized_snapshot_waits_for_deferred_pause_to_commit(self) -> None:
        timeline = _DeferredPauseTimeline()
        actuators = Mock()
        attachment = Mock()
        sensors = Mock()
        snapshot = Mock()
        updates: list[str] = []

        async def advance() -> None:
            updates.append("update")
            timeline.commit_pending_pause()

        with patch(
            "sim.isaac_control_runtime._control_safety_snapshot",
            return_value=snapshot,
        ):
            actual = asyncio.run(
                synchronized_control_safety_snapshot(
                    timeline,
                    actuators,
                    attachment,
                    sensors,
                    advance,
                )
            )

        self.assertIs(actual, snapshot)
        self.assertFalse(timeline.is_playing())
        self.assertEqual(updates, ["update"])
        self.assertEqual(timeline.events, ["pause"])

    def test_pauses_when_resuming_the_timeline_fails(self) -> None:
        timeline = _Timeline(playing=False)
        timeline.play = Mock(side_effect=RuntimeError("resume failed"))
        timeline.pause = Mock()

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

    def test_recreates_only_the_tensor_backed_articulation_wrapper(self) -> None:
        prims = ModuleType("isaacsim.core.experimental.prims")
        articulation_token = object()
        prims.Articulation = lambda path: (articulation_token, path)
        isaacsim = ModuleType("isaacsim")
        core = ModuleType("isaacsim.core")
        experimental = ModuleType("isaacsim.core.experimental")
        isaacsim.core = core
        core.experimental = experimental
        experimental.prims = prims

        class Attribute:
            def __init__(self, value: float) -> None:
                self.value = value

            def Get(self) -> float:
                return self.value

        arm_attributes = [Attribute(float(value)) for value in range(7)]
        finger_attributes = [Attribute(0.01), Attribute(0.01)]
        attachment = object()
        sensor = object()
        old_runtime = LiveControlRuntime(
            "session",
            object(),
            control_runtime.Actuators(
                articulation=object(),
                arm_attributes=arm_attributes,
                finger_attributes=finger_attributes,
            ),
            attachment,
            sensor,
        )

        with patch.dict(
            sys.modules,
            {
                "isaacsim": isaacsim,
                "isaacsim.core": core,
                "isaacsim.core.experimental": experimental,
                "isaacsim.core.experimental.prims": prims,
            },
        ):
            refreshed = control_runtime.refresh_live_control_articulation(
                old_runtime
            )

        self.assertIsNot(refreshed, old_runtime)
        self.assertEqual(
            refreshed.actuators.articulation,
            (articulation_token, "/World/Franka_R"),
        )
        for refreshed_attribute, old_attribute in zip(
            refreshed.actuators.arm_attributes, arm_attributes
        ):
            self.assertIs(refreshed_attribute, old_attribute)
        for refreshed_attribute, old_attribute in zip(
            refreshed.actuators.finger_attributes, finger_attributes
        ):
            self.assertIs(refreshed_attribute, old_attribute)
        self.assertIs(refreshed.attachment, attachment)
        self.assertIs(refreshed.sensor, sensor)
        self.assertEqual(
            control_runtime._active_drive_target(refreshed),
            control_runtime._active_drive_target(old_runtime),
        )

    def test_live_physics_repair_leaves_a_valid_view_untouched(self) -> None:
        view = SimpleNamespace(is_valid=True)

        class SimulationManager:
            invalidate_physics = Mock()

            @classmethod
            def get_physics_simulation_view(cls):
                return view

        simulation_manager = ModuleType("isaacsim.core.simulation_manager")
        simulation_manager.SimulationManager = SimulationManager

        with patch.dict(
            sys.modules,
            {"isaacsim.core.simulation_manager": simulation_manager},
        ):
            control_runtime.repair_invalid_live_control_physics_view()

        SimulationManager.invalidate_physics.assert_not_called()

    def test_insertion_replaces_stale_articulation_after_resume(self) -> None:
        timeline = _Timeline(playing=False)
        events: list[str] = []
        arm_attributes = [object() for _ in range(7)]
        finger_attributes = [object(), object()]
        old_articulation = object()
        view = SimpleNamespace(is_valid=False)

        class NewArticulation:
            def is_physics_tensor_entity_valid(self) -> bool:
                return view.is_valid

        new_articulation = NewArticulation()
        actuators = SimpleNamespace(
            articulation=old_articulation,
            arm_attributes=arm_attributes,
            finger_attributes=finger_attributes,
        )
        stage = object()
        attachment = Mock()
        sensor = Mock()
        old_runtime = LiveControlRuntime(
            "session", stage, actuators, attachment, sensor
        )
        captured = Mock()
        live = Mock()
        physics_ready = False

        async def advance() -> None:
            nonlocal physics_ready
            events.append("advance")
            view.is_valid = True
            physics_ready = True

        pose = Mock()
        drive_target = JointDriveTarget((0.0,) * 7, 0.04)

        def read(runtime):
            if not physics_ready:
                raise RuntimeError("physics backend is not ready")
            if runtime is old_runtime:
                raise Exception("Failed to get DOF positions from backend")
            events.append("read")
            self.assertIs(runtime.actuators.articulation, new_articulation)
            for refreshed_attribute, old_attribute in zip(
                runtime.actuators.arm_attributes, arm_attributes
            ):
                self.assertIs(refreshed_attribute, old_attribute)
            for refreshed_attribute, old_attribute in zip(
                runtime.actuators.finger_attributes, finger_attributes
            ):
                self.assertIs(refreshed_attribute, old_attribute)
            return live, pose, drive_target

        def observe():
            events.append("observe live")
            return SimpleNamespace(
                collision_detected=False,
                force_newtons=0.0,
            )

        def articulation(path):
            events.append("refresh")
            self.assertEqual(path, "/World/Franka_R")
            self.assertTrue(view.is_valid)
            return new_articulation

        class SimulationManager:
            @classmethod
            def get_physics_simulation_view(cls):
                return view

            @classmethod
            def invalidate_physics(cls) -> None:
                events.append("invalidate")
                view.is_valid = False

        prims = ModuleType("isaacsim.core.experimental.prims")
        prims.Articulation = articulation
        simulation_manager = ModuleType("isaacsim.core.simulation_manager")
        simulation_manager.SimulationManager = SimulationManager
        isaacsim = ModuleType("isaacsim")
        core = ModuleType("isaacsim.core")
        experimental = ModuleType("isaacsim.core.experimental")
        isaacsim.core = core
        core.experimental = experimental
        core.simulation_manager = simulation_manager
        experimental.prims = prims

        with (
            patch.dict(
                sys.modules,
                {
                    "isaacsim": isaacsim,
                    "isaacsim.core": core,
                    "isaacsim.core.experimental": experimental,
                    "isaacsim.core.experimental.prims": prims,
                    "isaacsim.core.simulation_manager": simulation_manager,
                },
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
                    operation="test insertion resume",
                )
            )

        self.assertEqual(
            events,
            [
                "invalidate",
                "advance",
                "refresh",
                "observe live",
                "read",
            ],
        )
        self.assertIsNot(synchronized.runtime, old_runtime)
        self.assertEqual(synchronized.runtime.session_id, old_runtime.session_id)
        self.assertIs(synchronized.runtime.stage, stage)
        self.assertIs(synchronized.runtime.attachment, attachment)
        self.assertIs(synchronized.runtime.sensor, sensor)
        self.assertIs(
            control_runtime.live_runtime_for("session", stage),
            synchronized.runtime,
        )
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

    def test_insertion_refresh_rejects_changed_live_ownership(self) -> None:
        arm_attributes = [object() for _ in range(7)]
        finger_attributes = [object(), object()]
        runtime = LiveControlRuntime(
            "session",
            object(),
            SimpleNamespace(
                articulation=object(),
                arm_attributes=arm_attributes,
                finger_attributes=finger_attributes,
            ),
            Mock(),
            Mock(),
        )

        async def advance() -> None:
            return None

        changed_runtimes = {
            "no-op": runtime,
            "session": LiveControlRuntime(
                "different-session",
                runtime.stage,
                runtime.actuators,
                runtime.attachment,
                runtime.sensor,
            ),
            "drive attribute": LiveControlRuntime(
                runtime.session_id,
                runtime.stage,
                SimpleNamespace(
                    articulation=object(),
                    arm_attributes=[object(), *arm_attributes[1:]],
                    finger_attributes=finger_attributes,
                ),
                runtime.attachment,
                runtime.sensor,
            ),
            "finger drive attribute": LiveControlRuntime(
                runtime.session_id,
                runtime.stage,
                SimpleNamespace(
                    articulation=object(),
                    arm_attributes=arm_attributes,
                    finger_attributes=[object(), finger_attributes[1]],
                ),
                runtime.attachment,
                runtime.sensor,
            ),
            "invalid articulation": LiveControlRuntime(
                runtime.session_id,
                runtime.stage,
                SimpleNamespace(
                    articulation=SimpleNamespace(
                        is_physics_tensor_entity_valid=lambda: False
                    ),
                    arm_attributes=arm_attributes,
                    finger_attributes=finger_attributes,
                ),
                runtime.attachment,
                runtime.sensor,
            ),
        }

        for name, changed_runtime in changed_runtimes.items():
            with (
                self.subTest(name=name),
                patch.object(
                    control_runtime,
                    "refresh_live_control_articulation",
                    return_value=changed_runtime,
                ),
                patch.object(
                    control_runtime,
                    "repair_invalid_live_control_physics_view",
                ),
                self.assertRaisesRegex(
                    RuntimeError, "live insertion articulation refresh failed"
                ),
            ):
                asyncio.run(
                    synchronized_insertion_safety_snapshot(
                        runtime,
                        _Timeline(playing=False),
                        advance,
                        Mock(),
                        SimulatorSafetyLimits(),
                        operation="test rejected insertion refresh",
                    )
                )

    def test_contact_grasp_resume_accepts_target_relative_gripper_settling(
        self,
    ) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(attached=False), Mock()
        )
        captured = ControlSafetySnapshot(
            (0.0,) * 7,
            0.0325559638440609,
            (0.0, 0.0, 0.0),
            0.0,
            False,
            False,
        )
        live = ControlSafetySnapshot(
            (8.8e-6,) + (0.0,) * 6,
            0.03255075588822365,
            (0.0, 0.0, 0.0),
            0.0,
            False,
            False,
        )
        drive_target = JointDriveTarget((0.0,) * 7, 0.03242833912372589)

        async def advance() -> None:
            return None

        with (
            _retain_test_runtime(),
            patch(
                "sim.isaac_control_runtime._control_safety_pose_and_drive_target",
                return_value=(live, Mock(), drive_target),
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
                synchronized_contact_grasp_safety_snapshot(
                    old_runtime,
                    timeline,
                    advance,
                    captured,
                    SimulatorSafetyLimits(),
                    expected_active_drive_target=drive_target,
                    operation="test contact-grasp resume",
                    maximum_gripper_error_meters=(
                        MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
                    ),
                )
            )

        self.assertIs(synchronized.safety, live)
        self.assertEqual(synchronized.active_drive_target, drive_target)
        self.assertEqual(timeline.events, ["play", "pause"])

        execution_timeline = _Timeline(playing=False)
        with (
            _retain_test_runtime(),
            patch(
                "sim.isaac_control_runtime._control_safety_pose_and_drive_target",
                return_value=(live, Mock(), drive_target),
            ),
            patch(
                "sim.isaac_control_runtime.LiveContactInterlock.observe",
                return_value=SimpleNamespace(
                    collision_detected=False,
                    force_newtons=0.0,
                ),
            ),
        ):
            execution = asyncio.run(
                synchronized_contact_grasp_execution_runtime(
                    old_runtime,
                    execution_timeline,
                    advance,
                    captured,
                    SimulatorSafetyLimits(),
                    expected_active_drive_target=drive_target,
                    operation="test contact-grasp execution resume",
                    maximum_gripper_error_meters=(
                        MAXIMUM_CONTACT_GRASP_GRIPPER_ERROR_METERS
                    ),
                )
            )

        self.assertIs(execution.safety, live)
        self.assertTrue(execution_timeline.is_playing())
        self.assertEqual(execution_timeline.events, ["play"])

    def test_insertion_frame_capture_stays_live_and_interlocked_until_read(self) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(attached=True), Mock()
        )
        captured = Mock(plug_attached=True)
        live = Mock()
        pose = Mock()
        drive_target = JointDriveTarget((0.0,) * 7, 0.018)
        old_runtime.actuators.current_command.return_value = SimpleNamespace(
            arm_positions=np.zeros(7),
            gripper_width_m=drive_target.gripper_width_m,
        )
        old_runtime.actuators.actual_command.return_value = SimpleNamespace(
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

        def read(runtime: LiveControlRuntime):
            self.assertTrue(timeline.is_playing())
            self.assertIsNot(runtime, old_runtime)
            self.assertIs(runtime.stage, old_runtime.stage)
            self.assertIs(runtime.attachment, old_runtime.attachment)
            self.assertIs(runtime.sensor, old_runtime.sensor)
            events.append("read")
            return live, pose, drive_target

        with (
            _retain_test_runtime(),
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
            [
                "readiness update",
                "camera update",
                "read",
            ],
        )
        self.assertIs(synchronized.safety, live)
        live.validate_followup_continuity.assert_called_once_with(
            captured,
            drive_target,
            SimulatorSafetyLimits(),
            maximum_gripper_error_meters=1e-6,
        )
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_insertion_frame_capture_accepts_settling_toward_active_target(self) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(attached=True), Mock()
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
        old_runtime.actuators.current_command.return_value = SimpleNamespace(
            arm_positions=np.asarray(active_target.joint_positions),
            gripper_width_m=active_target.gripper_width_m,
        )
        old_runtime.actuators.actual_command.return_value = SimpleNamespace(
            arm_positions=np.asarray(live.joint_positions),
            gripper_width_m=live.gripper_width_m,
        )

        async def advance() -> None:
            return None

        async def capture(_observe_safety) -> None:
            return None

        with (
            _retain_test_runtime(),
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
                active_target.gripper_width_m - 1.5320256352424622e-6
                if update_count < 18
                else settled.gripper_width_m
            ),
        )
        old_runtime = LiveControlRuntime(
            "session", object(), actuators, Mock(attached=True), Mock()
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
            _retain_test_runtime(),
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

        self.assertEqual(update_count, 18)
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
            "session", object(), actuators, Mock(attached=True), Mock()
        )

        async def advance() -> None:
            nonlocal update_count
            update_count += 1

        async def capture(_observe_safety) -> None:
            nonlocal camera_called
            camera_called = True

        with (
            _retain_test_runtime(),
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

        self.assertEqual(update_count, 97)
        self.assertEqual(observe.call_count, 97)
        self.assertFalse(camera_called)
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_resolution_resume_defers_bounded_drift_to_stable_baseline(self) -> None:
        timeline = _Timeline(playing=False)
        old_runtime = LiveControlRuntime(
            "session", object(), Mock(), Mock(), Mock()
        )
        captured = Mock(plug_attached=True)
        live = Mock(plug_attached=True)

        async def advance() -> None:
            return None

        with (
            _retain_test_runtime(),
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
                    operation="test resolution resume",
                )
            )

        self.assertIsNot(synchronized.runtime, old_runtime)
        self.assertIs(synchronized.runtime.stage, old_runtime.stage)
        self.assertIs(synchronized.runtime.attachment, old_runtime.attachment)
        self.assertIs(synchronized.runtime.sensor, old_runtime.sensor)
        self.assertIs(synchronized.safety, live)
        live.validate_continuity.assert_not_called()
        self.assertEqual(observe.call_count, 1)
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_resolution_resume_rejects_attachment_change(self) -> None:
        timeline = _Timeline(playing=False)
        runtime = LiveControlRuntime("session", object(), Mock(), Mock(), Mock())
        live = Mock(plug_attached=False)

        async def advance() -> None:
            return None

        with (
            _retain_test_runtime(),
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
                        operation="test resolution resume",
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
