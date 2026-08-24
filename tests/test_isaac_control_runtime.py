from __future__ import annotations

import unittest

from sim.isaac_control_runtime import bind_live_runtime, live_runtime_for


class LiveControlRuntimeTest(unittest.TestCase):
    def test_reuses_objects_only_for_the_bound_session_and_stage(self) -> None:
        stage = object()
        actuators = object()
        attachment = object()
        sensor = object()

        runtime = bind_live_runtime(
            "session-01", stage, actuators, attachment, sensor
        )

        self.assertIs(live_runtime_for("session-01", stage), runtime)
        self.assertIsNone(live_runtime_for("session-02", stage))
        self.assertIsNone(live_runtime_for("session-01", object()))

    def test_rebinding_transfers_the_runtime_to_the_followup_session(self) -> None:
        stage = object()
        actuators = object()
        attachment = object()
        sensor = object()
        bind_live_runtime("session-01", stage, actuators, attachment, sensor)

        runtime = bind_live_runtime(
            "session-02", stage, actuators, attachment, sensor
        )

        self.assertIsNone(live_runtime_for("session-01", stage))
        self.assertIs(live_runtime_for("session-02", stage), runtime)


if __name__ == "__main__":
    unittest.main()
