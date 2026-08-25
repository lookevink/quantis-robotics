"""Shared Isaac runtime primitives for simulator control workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any

from jepa_wm.control_safety import ControlInterlockEvidence, SimulatorSafetyLimits
from jepa_wm.direct_safety import ControlSafetySnapshot
from sim.isaac_demo_runtime import Actuators, PlugAttachment, create_actuators
from sim.isaac_demo_runtime import ContactReading
from sim.isaac_demo_scene import PLUG_PATH, ROBOT_PATH


CONTACT_SENSOR_NAME = "QuantisControlContact"
CONNECTOR_CONTACT_SENSOR_NAME = "QuantisConnectorContact"


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


@dataclass(frozen=True)
class LiveControlRuntime:
    """Session-bound Isaac objects whose tensor handles can be refreshed."""

    session_id: str
    stage: Any
    actuators: Actuators
    attachment: PlugAttachment
    sensor: Any


@dataclass(frozen=True)
class SynchronizedInsertionRuntime:
    runtime: LiveControlRuntime
    safety: ControlSafetySnapshot


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


async def _synchronized_live_read(
    timeline: Any,
    advance: Any,
    state: Any,
    read: Any,
    *,
    refresh: Any | None = None,
    observe_after_advance: Any | None = None,
    observe_safety: Any | None = None,
) -> tuple[Any, Any]:
    """Own the pause-sensitive resume, refresh, observe, read, pause lifecycle."""

    resume = not timeline.is_playing()
    try:
        if resume:
            timeline.play()
            await advance()
            if observe_after_advance is not None:
                observe_after_advance(state)
            if refresh is not None:
                state = refresh(state)
        if observe_safety is not None:
            observe_safety(state)
        return state, read(state)
    finally:
        timeline.pause()


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


async def synchronized_insertion_safety_snapshot(
    runtime: LiveControlRuntime,
    timeline: Any,
    advance: Any,
    captured: ControlSafetySnapshot,
    limits: SimulatorSafetyLimits,
    *,
    operation: str,
) -> SynchronizedInsertionRuntime:
    """Refresh, interlock, and rebind one paused insertion runtime."""

    resumed_interlock = LiveContactInterlock(
        runtime.sensor,
        limits.maximum_contact_force_newtons,
        operation,
    )

    def observe_resumed_state(_: LiveControlRuntime) -> None:
        resumed_interlock.observe()
        captured.validate_contact_continuity(resumed_interlock.evidence)

    def observe_safety(value: LiveControlRuntime) -> None:
        interlock = LiveContactInterlock(
            value.sensor,
            limits.maximum_contact_force_newtons,
            operation,
        )
        interlock.observe()

    runtime, live = await _synchronized_live_read(
        timeline,
        advance,
        runtime,
        lambda value: _control_safety_snapshot(
            value.actuators, value.attachment, value.sensor
        ),
        refresh=refresh_live_control_runtime,
        observe_after_advance=observe_resumed_state,
        observe_safety=observe_safety,
    )
    try:
        live.validate_continuity(captured, limits)
    except ValueError as error:
        raise RuntimeError("live insertion state changed after capture") from error
    return SynchronizedInsertionRuntime(runtime, live)


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


def refresh_live_control_runtime(runtime: LiveControlRuntime) -> LiveControlRuntime:
    """Recreate experimental physics wrappers invalidated by a timeline pause."""

    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    actuators = create_actuators(runtime.stage, Articulation(ROBOT_PATH))
    attachment = runtime.attachment.with_refreshed_physics(RigidPrim(PLUG_PATH))
    sensor = control_contact_sensors(
        runtime.stage,
        create=False,
        include_connector=(
            isinstance(runtime.sensor, ControlContactSensors)
            and runtime.sensor.connector is not None
        ),
    )
    return bind_live_runtime(
        runtime.session_id,
        runtime.stage,
        actuators,
        attachment,
        sensor,
    )


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
    if not isinstance(sensors, ControlContactSensors):
        return read_contact(sensors)
    hand_collision, hand_force = read_contact(sensors.hand)
    if sensors.connector is None:
        return hand_collision, hand_force
    _, connector_force = read_contact(sensors.connector)
    return hand_collision, max(hand_force, connector_force)
