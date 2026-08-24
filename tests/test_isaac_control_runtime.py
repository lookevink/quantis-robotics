from __future__ import annotations

import unittest

from sim.isaac_control_runtime import read_contact


class _Reading:
    def __init__(self, value: float) -> None:
        self.is_valid = True
        self.in_contact = False
        self.value = value


class _Sensor:
    def __init__(self, value: float) -> None:
        self.reading = _Reading(value)

    def get_sensor_reading(self) -> _Reading:
        return self.reading


class ContactReadingTest(unittest.TestCase):
    def test_rejects_non_finite_force(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid force"):
            read_contact(_Sensor(float("nan")))

    def test_rejects_negative_force(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid force"):
            read_contact(_Sensor(-0.1))


if __name__ == "__main__":
    unittest.main()
