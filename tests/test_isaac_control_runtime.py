from __future__ import annotations

import unittest

from sim.isaac_control_runtime import (
    ControlContactSensors,
    read_contact,
    read_control_contact,
)


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
