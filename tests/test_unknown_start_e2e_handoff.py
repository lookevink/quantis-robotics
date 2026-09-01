from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jepa_wm.control_policy import ControlExecutionPolicy
from sim.control_session import ControlResultStatus
from sim.isaac_control_followup import _contact_grasp_followup_policy


class UnknownStartE2EHandoffTest(unittest.TestCase):
    def _fixture(self):
        target = object()
        proposal = Path("/tmp/contact-grasp.pth")
        previous_observation = SimpleNamespace(
            target=target,
            warmup_frames=108,
            expected_proposal=proposal,
        )
        previous_state = SimpleNamespace(
            execution_policy=ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
            reference_recording="contact-held-00",
            seed=12600,
            recording="unknown-start-reset-v6-62605",
        )
        post_action = SimpleNamespace(
            tracking=SimpleNamespace(passed=True),
            collision_detected=False,
            contact_force_newtons=0.0,
        )
        response = object()
        previous_step = SimpleNamespace(
            state=previous_state,
            observation=previous_observation,
            response=response,
            result=SimpleNamespace(
                session_id="unknown-start-live-action-v7-62605",
                status=ControlResultStatus.APPLIED,
                gate=SimpleNamespace(passed=True),
                post_action=post_action,
            ),
        )
        policy = object()
        source_state = SimpleNamespace(
            reference_recording=previous_state.reference_recording,
            seed=previous_state.seed,
            recording=previous_state.recording,
            require_current_contact_grasp_policy=Mock(return_value=policy),
        )
        source_observation = SimpleNamespace(
            target=target,
            warmup_frames=previous_observation.warmup_frames,
            expected_proposal=proposal,
        )
        candidate_session = Mock()
        candidate_session.load_candidate_binding.return_value = SimpleNamespace(
            source_session_id="unknown-start-shadow-canary-v5-62605"
        )
        source_session = Mock()
        source_session.load_capture.return_value = (
            source_observation,
            source_state,
        )
        return previous_step, policy, candidate_session, source_session

    def test_promotes_only_the_bound_candidate_source_policy(self) -> None:
        previous, policy, candidate_session, source_session = self._fixture()
        with patch(
            "sim.isaac_control_followup.ControlSession.at",
            side_effect=(candidate_session, source_session),
        ):
            resolved = _contact_grasp_followup_policy(previous)

        self.assertIs(resolved, policy)
        candidate_session.load_candidate_binding.assert_called_once_with(
            previous.response
        )

    def test_rejects_a_candidate_target_different_from_its_source(self) -> None:
        previous, _, candidate_session, source_session = self._fixture()
        source_observation, source_state = source_session.load_capture.return_value
        source_session.load_capture.return_value = (
            SimpleNamespace(
                target=object(),
                warmup_frames=source_observation.warmup_frames,
                expected_proposal=source_observation.expected_proposal,
            ),
            source_state,
        )
        with patch(
            "sim.isaac_control_followup.ControlSession.at",
            side_effect=(candidate_session, source_session),
        ), self.assertRaisesRegex(ValueError, "not bound"):
            _contact_grasp_followup_policy(previous)


if __name__ == "__main__":
    unittest.main()
