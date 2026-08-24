"""Shared Isaac runtime primitives for simulator control workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sim.isaac_demo_runtime import Actuators, PlugAttachment
from sim.isaac_demo_scene import ROBOT_PATH


CONTACT_SENSOR_NAME = "QuantisControlContact"


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


def contact_sensor(stage: Any, *, create: bool) -> Any:
    from isaacsim.sensors.experimental.physics import Contact, ContactSensor

    path = f"{ROBOT_PATH}/panda_hand/{CONTACT_SENSOR_NAME}"
    prim = stage.GetPrimAtPath(path)
    if create and not prim.IsValid():
        return ContactSensor(
            Contact.create(path, min_threshold=0.0, max_threshold=1000.0, radius=0.08)
        )
    if not prim.IsValid():
        raise RuntimeError("control contact sensor is missing")
    return ContactSensor(path)


def read_contact(sensor: Any) -> tuple[bool, float]:
    reading = sensor.get_sensor_reading()
    if not reading.is_valid:
        raise RuntimeError("control contact sensor has no valid physics reading")
    return bool(reading.in_contact), float(reading.value)
