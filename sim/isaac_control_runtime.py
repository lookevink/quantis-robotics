"""Shared Isaac runtime primitives for simulator control workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

from jepa_wm.action import DroidPose
from jepa_wm.control_safety import ControlInterlockEvidence, SimulatorSafetyLimits
from jepa_wm.insertion_refresh import (
    MAXIMUM_SYNCHRONIZED_GRIPPER_ERROR_METERS,
    ControlSafetySnapshot,
)
from jepa_wm.joint_drive import JointDriveTarget
from sim.isaac_demo_runtime import (
    Actuators,
    FixedJointPlugMotion,
    KinematicPlugMotion,
    PlugAttachment,
    PlugCollisionPolicy,
    resume_live_simulation,
)
from sim.isaac_demo_runtime import ContactReading
from sim.isaac_demo_scene import PLUG_PATH, ROBOT_PATH, world_pose


CONTACT_SENSOR_NAME = "QuantisControlContact"
CONNECTOR_CONTACT_SENSOR_NAME = "QuantisConnectorContact"
MAXIMUM_INSERTION_GRIPPER_SETTLEMENT_UPDATES = 96


@dataclass(frozen=True)
class ContactSensorSpec:
    path: str
    radius: float
    missing_message: str
    translations: tuple[tuple[float, float, float], ...] | None = None


HAND_CONTACT_SENSOR = ContactSensorSpec(
    f"{ROBOT_PATH}/panda_hand/{CONTACT_SENSOR_NAME}",
    0.08,
    "control contact sensor is missing",
)
CONNECTOR_CONTACT_SENSOR = ContactSensorSpec(
    f"{PLUG_PATH}/{CONNECTOR_CONTACT_SENSOR_NAME}",
    0.025,
    "connector contact sensor is missing",
    ((0.0, 0.0, 0.0),),
)


@dataclass(frozen=True)
class ControlContactSensors:
    """One control interlock reading across the hand and optional connector."""

    hand: Any
    connector: Any | None = None


@dataclass
class LiveContactInterlock:
    """Poll one live sensor boundary while retaining its interval peak."""

    sensors: ControlContactSensors | Any
    maximum_contact_force_newtons: float
    operation: str
    initial_reading: ContactReading = ContactReading()
    _peak: ContactReading = field(default_factory=ContactReading, init=False)

    def __post_init__(self) -> None:
        if (
            not isfinite(self.maximum_contact_force_newtons)
            or self.maximum_contact_force_newtons <= 0.0
            or not self.operation
        ):
            raise ValueError("live contact interlock configuration is invalid")
        self._peak = self.initial_reading

    def observe(self) -> ContactReading:
        collision, force = read_control_contact(self.sensors)
        reading = ContactReading(collision, force)
        self._peak = self._peak.peak(reading)
        if collision or force > self.maximum_contact_force_newtons:
            raise RuntimeError(
                f"{self.operation} exceeded its live safety limit: "
                f"collision={collision}, force={force:.3f} N"
            )
        return reading

    @property
    def evidence(self) -> ControlInterlockEvidence:
        return ControlInterlockEvidence(
            self._peak.force_newtons,
            self._peak.collision_detected,
        )


@dataclass
class LiveInsertionInterlock:
    """Poll contact and fail immediately if the expected load detaches."""

    contact: LiveContactInterlock
    attachment: PlugAttachment
    expected_attachment: bool
    operation: str

    def observe(self) -> ContactReading:
        reading = self.contact.observe()
        if self.attachment.attached is not self.expected_attachment:
            raise RuntimeError(f"{self.operation} attachment state changed")
        return reading

    @property
    def evidence(self) -> ControlInterlockEvidence:
        return self.contact.evidence


@dataclass(frozen=True)
class LiveControlRuntime:
    """Session-bound Isaac objects retained across reloads and timeline pauses."""

    session_id: str
    stage: Any
    actuators: Actuators
    attachment: PlugAttachment
    sensor: Any


@dataclass(frozen=True)
class SynchronizedInsertionRuntime:
    runtime: LiveControlRuntime
    safety: ControlSafetySnapshot
    pose: DroidPose | None = None
    active_drive_target: JointDriveTarget | None = None


class _ContactGraspContinuity(Enum):
    INITIAL_CAPTURE = "initial_capture"
    ACTIVE_TARGET = "active_target"


@dataclass(frozen=True)
class _ControlSafetyHandles:
    actuators: Actuators
    attachment: PlugAttachment
    sensors: ControlContactSensors | Any


def _control_safety_snapshot(
    actuators: Actuators,
    attachment: PlugAttachment,
    sensors: ControlContactSensors | Any,
) -> ControlSafetySnapshot:
    collision_detected, contact_force = read_control_contact(sensors)
    current = actuators.actual_command()
    return ControlSafetySnapshot(
        joint_positions=tuple(float(value) for value in current.arm_positions),
        gripper_width_m=current.gripper_width_m,
        plug_position=tuple(float(value) for value in attachment.world_pose()[0]),
        contact_force_newtons=contact_force,
        collision_detected=collision_detected,
        plug_attached=attachment.attached,
    )


def _control_safety_and_pose(
    runtime: LiveControlRuntime,
) -> tuple[ControlSafetySnapshot, DroidPose]:
    current = runtime.actuators.actual_command()
    collision_detected, contact_force = read_control_contact(runtime.sensor)
    plug_position, _ = runtime.attachment.world_pose()
    hand_position, hand_orientation = world_pose(runtime.attachment.hand_prim)
    robot = runtime.stage.GetPrimAtPath(ROBOT_PATH)
    base_position, base_orientation = world_pose(robot)
    return (
        ControlSafetySnapshot(
            joint_positions=tuple(float(value) for value in current.arm_positions),
            gripper_width_m=current.gripper_width_m,
            plug_position=tuple(float(value) for value in plug_position),
            contact_force_newtons=contact_force,
            collision_detected=collision_detected,
            plug_attached=runtime.attachment.attached,
        ),
        DroidPose.from_world_poses(
            base_position,
            base_orientation,
            hand_position,
            hand_orientation,
            current.gripper_width_m,
        ),
    )


def _control_safety_pose_and_drive_target(
    runtime: LiveControlRuntime,
) -> tuple[ControlSafetySnapshot, DroidPose, JointDriveTarget]:
    safety, pose = _control_safety_and_pose(runtime)
    return safety, pose, _active_drive_target(runtime)


def _active_drive_target(runtime: LiveControlRuntime) -> JointDriveTarget:
    active = runtime.actuators.current_command()
    return JointDriveTarget(
        tuple(float(value) for value in active.arm_positions),
        active.gripper_width_m,
    )


async def _settle_insertion_frame_capture_gripper(
    runtime: LiveControlRuntime,
    advance: Any,
    observe_safety: Any,
    expected_active_drive_target: JointDriveTarget,
    operation: str,
    maximum_gripper_error_meters: float,
) -> None:
    """Wait bounded observed updates for the unchanged gripper drive target."""

    for update_index in range(
        MAXIMUM_INSERTION_GRIPPER_SETTLEMENT_UPDATES + 1
    ):
        if _active_drive_target(runtime) != expected_active_drive_target:
            raise RuntimeError(f"{operation} active drive target changed")
        actual = runtime.actuators.actual_command()
        if (
            abs(
                actual.gripper_width_m
                - expected_active_drive_target.gripper_width_m
            )
            <= maximum_gripper_error_meters
        ):
            return
        if update_index == MAXIMUM_INSERTION_GRIPPER_SETTLEMENT_UPDATES:
            break
        await advance()
        observe_safety()
    raise RuntimeError(f"{operation} gripper did not settle to its active target")


async def _synchronized_live_read(
    timeline: Any,
    advance: Any,
    state: Any,
    read: Any,
    *,
    refresh_after_resume: Any | None = None,
    observe_safety: Any | None = None,
    before_read: Any | None = None,
    pause_on_success: bool = True,
) -> tuple[Any, Any]:
    """Own resume, observed read, and requested terminal lifecycle."""

    completed = False
    try:
        readiness_update = resume_live_simulation(timeline)
        if readiness_update:
            await advance()
            if refresh_after_resume is not None:
                state = refresh_after_resume(state)
        if observe_safety is not None:
            observe_safety(state)
        if before_read is not None:
            await before_read(
                state,
                (
                    (lambda: observe_safety(state))
                    if observe_safety is not None
                    else None
                ),
            )
        result = state, read(state)
        completed = True
        return result
    finally:
        if pause_on_success or not completed:
            await pause_control_timeline(timeline, advance)


async def pause_control_timeline(
    timeline: Any,
    advance: Any,
) -> None:
    """Pause Isaac and process a deferred pause before returning."""

    timeline.pause()
    if timeline.is_playing():
        await advance()
    if timeline.is_playing():
        raise RuntimeError("control timeline did not pause")


async def synchronized_control_safety_snapshot(
    timeline: Any,
    actuators: Actuators,
    attachment: PlugAttachment,
    sensors: ControlContactSensors | Any,
    advance: Any,
    *,
    observe_safety: Any | None = None,
) -> ControlSafetySnapshot:
    """Read every physics-backed control value while live, then pause."""

    handles = _ControlSafetyHandles(actuators, attachment, sensors)
    _, snapshot = await _synchronized_live_read(
        timeline,
        advance,
        handles,
        lambda value: _control_safety_snapshot(
            value.actuators, value.attachment, value.sensors
        ),
        observe_safety=(
            (lambda _: observe_safety())
            if observe_safety is not None
            else None
        ),
    )
    return snapshot


async def _synchronized_insertion_runtime(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    limits: SimulatorSafetyLimits,
    *,
    operation: str,
    validate_resumed: Any | None = None,
    include_pose: bool = False,
    expected_attachment: bool | None = None,
    before_read: Any | None = None,
    pause_on_success: bool = True,
) -> SynchronizedInsertionRuntime:
    """Own the shared paused insertion resume and interlock lifecycle."""

    def interlock_for(value: LiveControlRuntime) -> Any:
        contact = LiveContactInterlock(
            value.sensor,
            limits.maximum_contact_force_newtons,
            operation,
        )
        if expected_attachment is None:
            return contact
        return LiveInsertionInterlock(
            contact,
            value.attachment,
            expected_attachment,
            operation,
        )

    live_interlock = interlock_for(runtime)
    continuity_validated = False

    def refresh_after_resume(value: LiveControlRuntime) -> LiveControlRuntime:
        refreshed = refresh_live_control_articulation(value)
        if (
            refreshed is value
            or refreshed.actuators.articulation
            is value.actuators.articulation
            or not _retains_live_control_ownership(value, refreshed)
        ):
            raise RuntimeError("live insertion articulation refresh failed")
        return refreshed

    def observe_safety(value: LiveControlRuntime) -> None:
        nonlocal continuity_validated
        if not _retains_live_control_ownership(runtime, value):
            raise RuntimeError("live insertion runtime identity changed")
        live_interlock.observe()
        if not continuity_validated and validate_resumed is not None:
            validate_resumed(live_interlock.evidence)
        continuity_validated = True

    read = (
        _control_safety_pose_and_drive_target
        if include_pose
        else lambda value: (
            _control_safety_snapshot(
                value.actuators,
                value.attachment,
                value.sensor,
            ),
            None,
            None,
        )
    )
    runtime, live = await _synchronized_live_read(
        timeline,
        advance,
        runtime,
        read,
        refresh_after_resume=refresh_after_resume,
        observe_safety=observe_safety,
        before_read=before_read,
        pause_on_success=pause_on_success,
    )
    safety, pose, active_drive_target = live
    return SynchronizedInsertionRuntime(
        runtime,
        safety,
        pose,
        active_drive_target,
    )


async def _synchronized_insertion_safety_snapshot(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    operation: str,
    pause_on_success: bool = True,
) -> SynchronizedInsertionRuntime:
    """Refresh, interlock, and require strict capture continuity."""

    synchronized = await _synchronized_insertion_runtime(
        runtime,
        timeline,
        advance,
        limits,
        operation=operation,
        validate_resumed=captured.validate_contact_continuity,
        include_pose=True,
        pause_on_success=pause_on_success,
    )
    try:
        synchronized.safety.validate_continuity(captured, limits)
    except ValueError as error:
        if not pause_on_success:
            await pause_control_timeline(timeline, advance)
        raise RuntimeError("live insertion state changed after capture") from error
    return synchronized


async def synchronized_insertion_safety_snapshot(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    operation: str,
) -> SynchronizedInsertionRuntime:
    """Read one strict insertion snapshot and commit the terminal pause."""

    return await _synchronized_insertion_safety_snapshot(
        runtime,
        timeline,
        advance,
        captured,
        limits,
        operation=operation,
    )


async def synchronized_insertion_execution_runtime(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    operation: str,
) -> SynchronizedInsertionRuntime:
    """Return a validated insertion runtime live for immediate execution."""

    return await _synchronized_insertion_safety_snapshot(
        runtime,
        timeline,
        advance,
        captured,
        limits,
        operation=operation,
        pause_on_success=False,
    )


async def _synchronized_contact_grasp_safety_snapshot(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    expected_active_drive_target: JointDriveTarget,
    operation: str,
    maximum_gripper_error_meters: float,
    continuity: _ContactGraspContinuity,
    pause_on_success: bool = True,
) -> SynchronizedInsertionRuntime:
    """Refresh contact-grasp state relative to its unchanged drive target."""

    synchronized = await _synchronized_insertion_runtime(
        runtime,
        timeline,
        advance,
        limits,
        operation=operation,
        validate_resumed=captured.validate_contact_continuity,
        include_pose=True,
        expected_attachment=captured.plug_attached,
        pause_on_success=pause_on_success,
    )
    try:
        if synchronized.active_drive_target != expected_active_drive_target:
            raise ValueError("live contact-grasp drive target changed after capture")
        if continuity is _ContactGraspContinuity.INITIAL_CAPTURE:
            synchronized.safety.validate_initial_contact_grasp_continuity(
                captured,
                limits,
                maximum_gripper_error_meters=maximum_gripper_error_meters,
            )
        else:
            synchronized.safety.validate_followup_continuity(
                captured,
                expected_active_drive_target,
                limits,
                maximum_gripper_error_meters=maximum_gripper_error_meters,
            )
    except ValueError as error:
        if not pause_on_success:
            await pause_control_timeline(timeline, advance)
        raise RuntimeError("live contact-grasp state changed after capture") from error
    return synchronized


async def synchronized_contact_grasp_safety_snapshot(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    expected_active_drive_target: JointDriveTarget,
    operation: str,
    maximum_gripper_error_meters: float,
) -> SynchronizedInsertionRuntime:
    """Read one target-relative grasp snapshot and commit the terminal pause."""

    return await _synchronized_contact_grasp_safety_snapshot(
        runtime,
        timeline,
        advance,
        captured,
        limits,
        expected_active_drive_target=expected_active_drive_target,
        operation=operation,
        maximum_gripper_error_meters=maximum_gripper_error_meters,
        continuity=_ContactGraspContinuity.ACTIVE_TARGET,
    )


async def synchronized_initial_contact_grasp_execution_runtime(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    expected_active_drive_target: JointDriveTarget,
    operation: str,
    maximum_gripper_error_meters: float,
) -> SynchronizedInsertionRuntime:
    """Return the initial captured-arm grasp runtime live for execution."""

    return await _synchronized_contact_grasp_safety_snapshot(
        runtime,
        timeline,
        advance,
        captured,
        limits,
        expected_active_drive_target=expected_active_drive_target,
        operation=operation,
        maximum_gripper_error_meters=maximum_gripper_error_meters,
        continuity=_ContactGraspContinuity.INITIAL_CAPTURE,
        pause_on_success=False,
    )


async def synchronized_contact_grasp_execution_runtime(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    expected_active_drive_target: JointDriveTarget,
    operation: str,
    maximum_gripper_error_meters: float,
) -> SynchronizedInsertionRuntime:
    """Return a validated grasp runtime live for immediate execution."""

    return await _synchronized_contact_grasp_safety_snapshot(
        runtime,
        timeline,
        advance,
        captured,
        limits,
        expected_active_drive_target=expected_active_drive_target,
        operation=operation,
        maximum_gripper_error_meters=maximum_gripper_error_meters,
        continuity=_ContactGraspContinuity.ACTIVE_TARGET,
        pause_on_success=False,
    )


async def synchronized_insertion_frame_capture(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    capture: Any,
    *,
    expected_active_drive_target: JointDriveTarget,
    operation: str,
    maximum_gripper_error_meters: float = (
        MAXIMUM_SYNCHRONIZED_GRIPPER_ERROR_METERS
    ),
) -> SynchronizedInsertionRuntime:
    """Render and read one interlocked insertion frame before pausing."""

    async def capture_before_read(
        live_runtime: LiveControlRuntime,
        observe: Any,
    ) -> None:
        if observe is None:
            raise RuntimeError("insertion frame capture has no live interlock")
        await _settle_insertion_frame_capture_gripper(
            live_runtime,
            advance,
            observe,
            expected_active_drive_target,
            operation,
            maximum_gripper_error_meters,
        )
        await capture(observe)

    synchronized = await _synchronized_insertion_runtime(
        runtime,
        timeline,
        advance,
        limits,
        operation=operation,
        validate_resumed=captured.validate_contact_continuity,
        include_pose=True,
        expected_attachment=captured.plug_attached,
        before_read=capture_before_read,
    )
    try:
        if synchronized.active_drive_target != expected_active_drive_target:
            raise ValueError("live insertion drive target changed during frame capture")
        synchronized.safety.validate_followup_continuity(
            captured,
            expected_active_drive_target,
            limits,
            maximum_gripper_error_meters=maximum_gripper_error_meters,
        )
    except ValueError as error:
        raise RuntimeError("live insertion state changed during frame capture") from error
    return synchronized


async def synchronized_insertion_resolution_runtime(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    operation: str,
) -> SynchronizedInsertionRuntime:
    """Resume safely while deferring bounded settling to the baseline contract."""

    synchronized = await _synchronized_insertion_runtime(
        runtime,
        timeline,
        advance,
        limits,
        operation=operation,
    )
    if synchronized.safety.plug_attached is not captured.plug_attached:
        raise RuntimeError(
            "live insertion attachment changed before resolution baseline"
        )
    return synchronized


_live_runtime: LiveControlRuntime | None = None


def bind_live_runtime(
    session_id: str,
    stage: Any,
    actuators: Actuators,
    attachment: PlugAttachment,
    sensor: Any,
) -> LiveControlRuntime:
    global _live_runtime
    _live_runtime = LiveControlRuntime(
        session_id, stage, actuators, attachment, sensor
    )
    return _live_runtime


def restore_live_runtime_handoff(
    handoff: Any,
) -> LiveControlRuntime:
    """Wrap retained low-level physics handles in current-generation owners."""

    retained_motion = handoff.attachment.motion
    if retained_motion.kind == "fixed_joint":
        motion = FixedJointPlugMotion(
            retained_motion.prim,
            retained_motion.hand_prim,
            retained_motion.rigid_prim,
            retained_motion.fixed_joint,
            retained_motion.hand_to_plug_offset,
        )
    elif retained_motion.kind == "kinematic":
        motion = KinematicPlugMotion(
            retained_motion.prim,
            retained_motion.hand_prim,
            retained_motion.hand_to_plug_offset,
        )
    else:
        raise RuntimeError("live control plug motion handoff is invalid")

    return bind_live_runtime(
        handoff.session_id,
        handoff.stage,
        Actuators(
            handoff.actuators.articulation,
            list(handoff.actuators.arm_attributes),
            list(handoff.actuators.finger_attributes),
        ),
        PlugAttachment(
            motion,
            PlugCollisionPolicy(
                list(handoff.attachment.collision_attributes),
                handoff.attachment.excluded_collision_paths,
            ),
        ),
        ControlContactSensors(
            handoff.sensor.hand,
            handoff.sensor.connector,
        ),
    )


def refresh_live_control_articulation(
    runtime: LiveControlRuntime,
) -> LiveControlRuntime:
    """Replace only the tensor-backed articulation after a paused resume."""

    from isaacsim.core.experimental.prims import Articulation

    return bind_live_runtime(
        runtime.session_id,
        runtime.stage,
        Actuators(
            Articulation(ROBOT_PATH),
            list(runtime.actuators.arm_attributes),
            list(runtime.actuators.finger_attributes),
        ),
        runtime.attachment,
        runtime.sensor,
    )


def _retains_live_control_ownership(
    original: LiveControlRuntime,
    refreshed: LiveControlRuntime,
) -> bool:
    """Allow only the tensor-backed articulation owner to be replaced."""

    original_arm = original.actuators.arm_attributes
    refreshed_arm = refreshed.actuators.arm_attributes
    original_fingers = original.actuators.finger_attributes
    refreshed_fingers = refreshed.actuators.finger_attributes
    return (
        refreshed.session_id == original.session_id
        and refreshed.stage is original.stage
        and refreshed.attachment is original.attachment
        and refreshed.sensor is original.sensor
        and len(refreshed_arm) == len(original_arm)
        and all(
            refreshed_attribute is original_attribute
            for refreshed_attribute, original_attribute in zip(
                refreshed_arm, original_arm
            )
        )
        and len(refreshed_fingers) == len(original_fingers)
        and all(
            refreshed_attribute is original_attribute
            for refreshed_attribute, original_attribute in zip(
                refreshed_fingers, original_fingers
            )
        )
    )


def _contact_sensor_components(sensors: Any) -> tuple[Any, Any | None]:
    """Read raw or cross-generation composite sensor structure fail-closed."""

    if hasattr(sensors, "hand") or hasattr(sensors, "connector"):
        try:
            hand = sensors.hand
            connector = sensors.connector
        except AttributeError as error:
            raise RuntimeError("control contact sensor set is malformed") from error
        if not callable(getattr(hand, "get_sensor_reading", None)) or (
            connector is not None
            and not callable(getattr(connector, "get_sensor_reading", None))
        ):
            raise RuntimeError("control contact sensor set is malformed")
        return hand, connector
    if not callable(getattr(sensors, "get_sensor_reading", None)):
        raise RuntimeError("control contact sensor is malformed")
    return sensors, None


def live_runtime_for(session_id: str, stage: Any) -> LiveControlRuntime | None:
    runtime = _live_runtime
    if runtime is None or runtime.session_id != session_id or runtime.stage is not stage:
        return None
    return runtime


def _contact_sensor(stage: Any, spec: ContactSensorSpec, *, create: bool) -> Any:
    from isaacsim.sensors.experimental.physics import Contact, ContactSensor

    prim = stage.GetPrimAtPath(spec.path)
    if create and not prim.IsValid():
        arguments = {
            "min_threshold": 0.0,
            "max_threshold": 1000.0,
            "radius": spec.radius,
        }
        if spec.translations is not None:
            arguments["translations"] = [list(value) for value in spec.translations]
        return ContactSensor(
            Contact.create(spec.path, **arguments)
        )
    if not prim.IsValid():
        raise RuntimeError(spec.missing_message)
    return ContactSensor(spec.path)


def contact_sensor(stage: Any, *, create: bool) -> Any:
    return _contact_sensor(stage, HAND_CONTACT_SENSOR, create=create)


def connector_contact_sensor(stage: Any, *, create: bool) -> Any:
    """Measure plug contacts in a tip-local radius under its rigid body."""

    return _contact_sensor(stage, CONNECTOR_CONTACT_SENSOR, create=create)


def control_contact_sensors(
    stage: Any,
    *,
    create: bool,
    include_connector: bool = False,
) -> ControlContactSensors:
    return ControlContactSensors(
        contact_sensor(stage, create=create),
        connector_contact_sensor(stage, create=create)
        if include_connector
        else None,
    )


def read_contact(sensor: Any) -> tuple[bool, float]:
    reading = sensor.get_sensor_reading()
    if not reading.is_valid:
        raise RuntimeError("control contact sensor has no valid physics reading")
    force = float(reading.value)
    if not isfinite(force) or force < 0.0:
        raise RuntimeError("control contact sensor returned an invalid force")
    return bool(reading.in_contact), force


def read_control_contact(sensors: ControlContactSensors | Any) -> tuple[bool, float]:
    hand, connector = _contact_sensor_components(sensors)
    hand_collision, hand_force = read_contact(hand)
    if connector is None:
        return hand_collision, hand_force
    _, connector_force = read_contact(connector)
    return hand_collision, max(hand_force, connector_force)
