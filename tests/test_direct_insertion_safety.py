from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.action import DROID_FPS, DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
    SafetyProjectionAttempt,
    SimulatorControlGate,
    SimulatorSafetyState,
)
from jepa_wm.direct_safety import (
    ControlSafetySnapshot,
    DirectInsertionSafetyEvidence,
)
from jepa_wm.training_artifact import ArtifactIdentity
from sim.control_session import ControlSession, ControlSessionState


_FINGERPRINT = "a" * 64
_JOINTS = (0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0)


def _attempt(*, passed: bool = True) -> SafetyProjectionAttempt:
    gate = ControlGateDecision(
        9,
        DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
        () if passed else (ControlGateReason.COLLISION_DETECTED,),
    )
    return SafetyProjectionAttempt(ACTION_SCALES[0], gate, 0.0, _JOINTS)


def _evidence(*, passed: bool = True) -> DirectInsertionSafetyEvidence:
    attempt = _attempt(passed=passed)
    return DirectInsertionSafetyEvidence(
        observation_id=9,
        evaluated_at_unix_seconds=100.2,
        proposed_actions=(DroidAction((0.0,) * 7),) * 3,
        proposal=ArtifactIdentity(Path("/tmp/proposal.pth"), _FINGERPRINT),
        attempts=(attempt,),
        selected_action_scale=attempt.scale if passed else None,
        live_state=ControlSafetySnapshot(
            _JOINTS,
            0.04,
            (0.4, 0.0, 0.5),
            0.25,
            False,
            True,
        ),
    )


class DirectInsertionSafetyEvidenceTest(unittest.TestCase):
    def test_round_trips_no_actuation_evidence(self) -> None:
        evidence = _evidence()

        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.to_dict()["authority"], "no_actuation")
        self.assertEqual(
            DirectInsertionSafetyEvidence.from_dict(evidence.to_dict()), evidence
        )

    def test_rejects_tampered_authority_and_numeric_strings(self) -> None:
        for field, value, nested in (
            ("authority", "command", False),
            ("evaluated_at_unix_seconds", "100.2", False),
            ("contact_force_newtons", "0.25", True),
        ):
            with self.subTest(field=field):
                payload = _evidence().to_dict()
                (payload["live_state"] if nested else payload)[field] = value
                with self.assertRaises(ValueError):
                    DirectInsertionSafetyEvidence.from_dict(payload)

    def test_attachment_is_required_for_a_passing_task_claim(self) -> None:
        payload = _evidence().to_dict()
        payload["live_state"]["plug_attached"] = False
        payload["passed"] = False

        evidence = DirectInsertionSafetyEvidence.from_dict(payload)

        self.assertFalse(evidence.passed)


class DirectInsertionSafetySessionTest(unittest.TestCase):
    def test_binds_gate_and_terminalizes_session_without_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = ControlSession.at(Path(temp_dir), "insertion-session")
            proposal_path = Path("/tmp/proposal.pth")
            observation = ControlObservation(
                observation_id=9,
                captured_at_unix_seconds=100.0,
                context_frame=Path("context.png"),
                target=ControlTarget(
                    Path("target.png"),
                    DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                ),
                expected_proposal=proposal_path,
                pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                previous_action=DroidAction((0.0,) * 7),
                warmup_frames=43,
            )
            state = ControlSessionState(
                "insertion-session",
                "insertion-held-00",
                52600,
                "control-insertion-session",
                _JOINTS,
                False,
                0.25,
                plug_position=(0.4, 0.0, 0.5),
                plug_attached=True,
                current_gripper_width_m=0.04,
                execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            )
            response = ProposedControl(
                9,
                100.1,
                (DroidAction((0.0,) * 7),) * 3,
                proposal_path,
                _FINGERPRINT,
            )
            gate = SimulatorControlGate().evaluate(
                observation,
                response,
                SimulatorSafetyState(
                    _JOINTS,
                    _JOINTS,
                    _JOINTS,
                    1.0 / DROID_FPS,
                    0.25,
                    False,
                ),
                now_unix_seconds=100.2,
            )
            attempt = SafetyProjectionAttempt(ACTION_SCALES[0], gate, 0.0, _JOINTS)
            evidence = DirectInsertionSafetyEvidence(
                observation_id=9,
                evaluated_at_unix_seconds=100.2,
                proposed_actions=response.actions,
                proposal=ArtifactIdentity(proposal_path, _FINGERPRINT),
                attempts=(attempt,),
                selected_action_scale=attempt.scale,
                live_state=ControlSafetySnapshot(
                    _JOINTS,
                    0.04,
                    (0.4, 0.0, 0.5),
                    0.25,
                    False,
                    True,
                ),
            )
            session.write_capture(observation, state)
            session.write_response(response)

            with self.assertRaisesRegex(ValueError, "cannot be executed"):
                session.claim_execution()
            session.write_direct_safety(evidence)

            self.assertEqual(session.load_direct_safety(), evidence)
            self.assertFalse(session.execution_path.exists())
            self.assertFalse(session.result_path.exists())
            payload = json.loads(session.direct_safety_path.read_text())
            payload["live_state"]["plug_position"][0] += 0.01
            session.direct_safety_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not bound"):
                session.load_direct_safety()


if __name__ == "__main__":
    unittest.main()
