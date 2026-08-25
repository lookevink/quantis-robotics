"""Shared Isaac runtime primitives for simulator control workflows."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from sim.isaac_demo_runtime import Actuators, PlugAttachment
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


@dataclass(frozen=True)
class LiveControlRuntime:
    """Session-bound Isaac objects whose tensor handles survive server calls."""

    session_id: str
    stage: Any
    actuators: Actuators
    attachment: PlugAttachment
    sensor: Any


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
