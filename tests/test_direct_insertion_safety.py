from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.action import DROID_FPS, DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlInterlockEvidence,
    ControlGateDecision,
    ControlGateReason,
    SafetyProjectionAttempt,
    SimulatorControlGate,
    SimulatorSafetyState,
)
from jepa_wm.direct_safety import DirectInsertionSafetyEvidence
from jepa_wm.insertion_refresh import (
    ControlSafetySnapshot,
    InsertionEvaluationRefresh,
)
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.joint_drive import JointDriveTarget
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
    evaluation = InsertionEvaluationRefresh(
        100.2,
        ControlSafetySnapshot(
            _JOINTS,
            0.04,
            (0.4, 0.0, 0.5),
            0.25,
            False,
            True,
        ),
        DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
    )
    return DirectInsertionSafetyEvidence(
        observation_id=9,
        evaluation=evaluation,
        proposed_actions=(DroidAction((0.0,) * 7),) * 3,
        proposal=ArtifactIdentity(Path("/tmp/proposal.pth"), _FINGERPRINT),
        attempts=(attempt,),
        selected_action_scale=attempt.scale if passed else None,
        active_drive_target=JointDriveTarget(_JOINTS, 0.04),
    )


class DirectInsertionSafetyEvidenceTest(unittest.TestCase):
    def test_capture_snapshot_rejects_resumed_contact_changes(self) -> None:
        captured = _evidence().live_state

        captured.validate_contact_continuity(ControlInterlockEvidence(0.25, False))
        for collision, force in (
            (True, 0.25),
            (False, 0.251),
        ):
            with self.subTest(collision=collision, force=force):
                with self.assertRaisesRegex(ValueError, "contact state changed"):
                    captured.validate_contact_continuity(
                        ControlInterlockEvidence(force, collision)
                    )

    def test_round_trips_no_actuation_evidence(self) -> None:
        evidence = _evidence()

        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.to_dict()["authority"], "no_actuation")
        self.assertEqual(
            DirectInsertionSafetyEvidence.from_dict(evidence.to_dict()), evidence
        )

    def test_current_evidence_requires_its_evaluated_drive_target(self) -> None:
        for field in ("active_drive_target", "evaluation"):
            with self.subTest(field=field):
                payload = _evidence().to_dict()
                del payload[field]

                with self.assertRaisesRegex(ValueError, "incomplete"):
                    DirectInsertionSafetyEvidence.from_dict(payload)

    def test_reads_v2_without_upgrading_its_refresh_claim(self) -> None:
        payload = _evidence().to_dict()
        payload["schema"] = "quantis.jepa_wm_direct_insertion_safety.v2"
        evaluation = payload.pop("evaluation")
        payload["evaluated_at_unix_seconds"] = evaluation[
            "refreshed_at_unix_seconds"
        ]
        payload["live_state"] = evaluation["live_state"]

        legacy = DirectInsertionSafetyEvidence.from_dict(payload)

        self.assertIsNone(legacy.live_pose)
        self.assertEqual(legacy.active_drive_target, _evidence().active_drive_target)

        payload["live_pose"] = list(_evidence().live_pose.values)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DirectInsertionSafetyEvidence.from_dict(payload)

    def test_rejects_tampered_authority_and_numeric_strings(self) -> None:
        for field, value, nested in (
            ("authority", "command", False),
            ("refreshed_at_unix_seconds", "100.2", False),
            ("contact_force_newtons", "0.25", True),
        ):
            with self.subTest(field=field):
                payload = _evidence().to_dict()
                evaluation = payload["evaluation"]
                target = (
                    evaluation["live_state"]
                    if nested
                    else (payload if field == "authority" else evaluation)
                )
                target[field] = value
                with self.assertRaises(ValueError):
                    DirectInsertionSafetyEvidence.from_dict(payload)

    def test_attachment_is_required_for_a_passing_task_claim(self) -> None:
        payload = _evidence().to_dict()
        payload["evaluation"]["live_state"]["plug_attached"] = False
        payload["passed"] = False

        evidence = DirectInsertionSafetyEvidence.from_dict(payload)

        self.assertFalse(evidence.passed)


class DirectInsertionSafetySessionTest(unittest.TestCase):
    @patch("sim.control_session.validate_observation_target")
    def test_binds_gate_and_terminalizes_session_without_execution(
        self,
        _validate_target,
    ) -> None:
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
                active_drive_target=JointDriveTarget(_JOINTS, 0.04),
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
                evaluation=InsertionEvaluationRefresh(
                    100.2,
                    ControlSafetySnapshot(
                        _JOINTS,
                        0.04,
                        (0.4, 0.0, 0.5),
                        0.25,
                        False,
                        True,
                    ),
                    observation.pose,
                ),
                proposed_actions=response.actions,
                proposal=ArtifactIdentity(proposal_path, _FINGERPRINT),
                attempts=(attempt,),
                selected_action_scale=attempt.scale,
                active_drive_target=JointDriveTarget(_JOINTS, 0.04),
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
            payload["evaluation"]["live_state"]["plug_position"][0] += 0.01
            session.direct_safety_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not bound"):
                session.load_direct_safety()

            payload = evidence.to_dict()
            payload["schema"] = "quantis.jepa_wm_direct_insertion_safety.v1"
            del payload["active_drive_target"]
            evaluation = payload.pop("evaluation")
            payload["evaluated_at_unix_seconds"] = evaluation[
                "refreshed_at_unix_seconds"
            ]
            payload["live_state"] = evaluation["live_state"]
            session.direct_safety_path.write_text(json.dumps(payload))
            self.assertIsNone(session.load_direct_safety().active_drive_target)

            payload = evidence.to_dict()
            payload["active_drive_target"]["joint_positions"][0] += 0.001
            session.direct_safety_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "not bound"):
                session.load_direct_safety()


if __name__ == "__main__":
    unittest.main()
