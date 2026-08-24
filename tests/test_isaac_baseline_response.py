from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from jepa_wm.control_baselines import NonModelBaselinePolicy
from sim.isaac_baseline_response import persist_baseline_response


class BaselineResponseServiceTest(unittest.TestCase):
    @patch("sim.isaac_baseline_response.build_baseline_response")
    @patch("sim.isaac_baseline_response.load_held_out_reference")
    @patch("sim.isaac_baseline_response.ControlSession.at")
    def test_builds_and_persists_zero_response_in_resident_process(
        self,
        session_at: MagicMock,
        load_reference: MagicMock,
        build_response: MagicMock,
    ) -> None:
        session = session_at.return_value
        observation = MagicMock()
        state = MagicMock(reference_recording="held-00", seed=11400)
        session.load_capture.return_value = (observation, state)
        response = build_response.return_value
        response.to_dict.return_value = {"response": True}
        control_root = Path("/data/control_sessions")

        result = persist_baseline_response(
            "zero-session", "zero", control_root=control_root
        )

        session_at.assert_called_once_with(control_root, "zero-session")
        load_reference.assert_called_once_with(
            Path("/data/recordings/held-00"), 11400
        )
        build_response.assert_called_once_with(
            observation,
            NonModelBaselinePolicy.ZERO,
            scripted_actions=None,
        )
        session.write_response.assert_called_once_with(response)
        self.assertEqual(result, {"response": True})


if __name__ == "__main__":
    unittest.main()
