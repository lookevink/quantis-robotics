from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from sim.isaac_candidate_binding import (
    persist_experimental_candidate_response,
    prepare_experimental_candidate_source,
)


class CandidateBindingServiceTest(unittest.TestCase):
    @patch.dict("sim.isaac_candidate_binding._PREPARED_SOURCES._sources", clear=True)
    @patch("sim.isaac_candidate_binding.ControlSession.at")
    def test_preflights_source_before_live_capture(self, session_at: MagicMock) -> None:
        source = session_at.return_value
        evidence = source.load_candidate_source_evidence.return_value
        shadow = evidence.shadow
        shadow.observation_id = 91
        shadow.passes_shadow_gate = True
        safety = evidence.safety
        safety.passed = True

        result = prepare_experimental_candidate_source(
            "source-session", control_root=Path("/tmp/control-sessions")
        )

        self.assertEqual(result["observation_id"], 91)
        self.assertTrue(result["shadow_gate_passed"])
        self.assertTrue(result["safety_passed"])

    @patch.dict("sim.isaac_candidate_binding._PREPARED_SOURCES._sources", clear=True)
    @patch("sim.isaac_candidate_binding.build_experimental_candidate_response")
    @patch("sim.isaac_candidate_binding.ControlSession.at")
    def test_timestamps_and_persists_after_source_validation(
        self,
        session_at: MagicMock,
        build_response: MagicMock,
    ) -> None:
        session = MagicMock()
        source = MagicMock()
        session_at.side_effect = (session, source)
        observation = MagicMock()
        shadow = MagicMock()
        safety = MagicMock()
        binding = MagicMock()
        response = MagicMock()
        session.load_capture.return_value = (observation, MagicMock())
        source_evidence = source.load_candidate_source_evidence.return_value
        source_evidence.shadow = shadow
        source_evidence.safety = safety
        build_response.return_value = (binding, response)
        binding.to_dict.return_value = {"binding": True}
        response.to_dict.return_value = {"response": True}

        result = persist_experimental_candidate_response(
            "candidate-session",
            "source-session",
            control_root=Path("/tmp/control-sessions"),
        )

        build_response.assert_called_once_with(
            execution_session_id="candidate-session",
            source_session_id="source-session",
            observation=observation,
            shadow=shadow,
            safety=safety,
        )
        session.write_candidate_binding.assert_called_once_with(
            binding, source_evidence
        )
        session.write_response.assert_called_once_with(response)
        self.assertEqual(result, {"binding": {"binding": True}, "response": {"response": True}})


if __name__ == "__main__":
    unittest.main()
