from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

from sim.isaac_insertion_trial import (
    persist_insertion_trial_response,
    prepare_insertion_trial_source,
)


class InsertionTrialServiceTest(unittest.TestCase):
    @patch.dict("sim.isaac_insertion_trial._PREPARED_SOURCES._sources", clear=True)
    @patch("sim.isaac_insertion_trial.ControlSession.at")
    def test_preflights_source_before_live_reset(self, session_at: MagicMock) -> None:
        evidence = session_at.return_value.load_insertion_trial_source_evidence.return_value
        evidence.safety.passed = True
        evidence.safety.observation_id = 91
        evidence.safety.proposal.to_dict.return_value = {"proposal": True}
        evidence.safety.selected_action_scale.to_dict.return_value = {"scale": True}

        result = prepare_insertion_trial_source(
            "source-session", control_root=Path("/tmp/control-sessions")
        )

        self.assertEqual(result["observation_id"], 91)
        self.assertTrue(result["safety_passed"])
        self.assertEqual(result["proposal"], {"proposal": True})

    @patch.dict("sim.isaac_insertion_trial._PREPARED_SOURCES._sources", clear=True)
    @patch("sim.isaac_insertion_trial.build_insertion_trial_response")
    @patch("sim.isaac_insertion_trial.ControlSession.at")
    def test_persists_binding_and_fresh_response_after_capture(
        self,
        session_at: MagicMock,
        build_response: MagicMock,
    ) -> None:
        session = MagicMock()
        source = MagicMock()
        session_at.side_effect = (session, source)
        observation = MagicMock()
        state = MagicMock()
        execution = MagicMock()
        source_evidence = source.load_insertion_trial_source_evidence.return_value
        binding = MagicMock()
        response = MagicMock()
        session.load_capture.return_value = (observation, state)
        session.trial_context.return_value = execution
        build_response.return_value = (binding, response)
        binding.to_dict.return_value = {"binding": True}
        response.to_dict.return_value = {"response": True}

        result = persist_insertion_trial_response(
            "trial-session",
            "source-session",
            control_root=Path("/tmp/control-sessions"),
        )

        build_response.assert_called_once_with(
            execution_session_id="trial-session",
            source_session_id="source-session",
            execution=execution,
            source=source_evidence,
        )
        session.write_insertion_trial_binding.assert_called_once_with(
            binding, source_evidence
        )
        session.write_response.assert_called_once_with(response)
        self.assertEqual(result, {"binding": {"binding": True}, "response": {"response": True}})


if __name__ == "__main__":
    unittest.main()
