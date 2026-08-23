"""Shared Isaac runtime primitives for simulator control workflows."""

from __future__ import annotations

from typing import Any

from sim.isaac_demo_scene import ROBOT_PATH


CONTACT_SENSOR_NAME = "QuantisControlContact"


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
