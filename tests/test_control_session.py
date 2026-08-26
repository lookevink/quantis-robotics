import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.action import ActionSelectionBounds, DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.insertion_contract import InsertionControlTargetPolicy
from jepa_wm.insertion_rollout import InsertionRolloutPosition
from jepa_wm.joint_drive import JointDriveTarget
from sim.control_session import (
    ControlExecutionPolicy,
    ControlSession,
    ControlSessionState,
)


class ControlSessionTest(unittest.TestCase):
    def test_round_trips_only_a_typed_bounded_insertion_rollout_position(self) -> None:
        state = ControlSessionState(
            "insertion-rollout",
            "held-reference",
            52600,
            "control-insertion-rollout",
            (0.0,) * 7,
            False,
            0.0,
            execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            insertion_target_policy=InsertionControlTargetPolicy(),
            insertion_rollout_position=InsertionRolloutPosition(3, 4),
        )

        self.assertEqual(ControlSessionState.from_dict(state.to_dict()), state)
        malformed = state.to_dict()
        malformed["insertion_rollout_position"]["step_index"] = True
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ControlSessionState.from_dict(malformed)
        with self.assertRaisesRegex(ValueError, "invalid"):
            InsertionRolloutPosition(1, 5)

    def test_insertion_target_policy_is_persisted_and_required_for_its_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            session = ControlSession.at(
                data_root / "control_sessions",
                "insertion-target",
            )
            observation = ControlObservation(
                123,
                100.0,
                Path("context.png"),
                ControlTarget(
                    Path("recordings/held-reference/wrist/frame_000048.png"),
                    DroidPose((0.4005, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                ),
                Path("/tmp/proposal.pth"),
                DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                DroidAction((0.0,) * 7),
                43,
            )
            target_policy = InsertionControlTargetPolicy(
                minimum_translation_meters=4e-4,
                maximum_action_horizon=6,
                camera="overhead",
                action_bounds=ActionSelectionBounds(minimum_action_norm=0.0),
            )
            state = ControlSessionState(
                "insertion-target",
                "held-reference",
                52600,
                "control-insertion-target",
                (0.0,) * 7,
                False,
                0.0,
                execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
                insertion_target_policy=target_policy,
                active_drive_target=JointDriveTarget((0.001,) * 7, 0.04),
            )
            session.write_capture(observation, state)

            with patch.object(
                InsertionControlTargetPolicy,
                "validate_observation",
            ) as validate:
                restored_observation, restored_state = session.load_capture()

            self.assertEqual(restored_observation, observation)
            self.assertEqual(restored_state, state)
            validate.assert_called_once_with(
                observation,
                data_root / "recordings" / "held-reference",
                frame_root=data_root,
            )

            stripped = state.to_dict()
            del stripped["insertion_target_policy"]
            session.state_path.write_text(json.dumps(stripped))
            with patch(
                "sim.control_session.validate_observation_target",
                side_effect=ValueError("legacy target is inconsistent"),
            ):
                with self.assertRaisesRegex(ValueError, "legacy target"):
                    session.load_capture()

    def test_persists_only_a_response_bound_to_the_captured_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = ControlSession.at(Path(temp_dir), "session-a")
            observation = ControlObservation(
                observation_id=123,
                captured_at_unix_seconds=100.0,
                context_frame=Path("context.png"),
                target=ControlTarget(Path("target.png")),
                expected_proposal=Path("/tmp/proposal.pth"),
                pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                previous_action=DroidAction((0.0,) * 7),
                warmup_frames=4,
            )
            state = ControlSessionState(
                "session-a",
                "held-reference",
                11400,
                "control-session-a",
                (0.0,) * 7,
                False,
                0.0,
            )
            session.write_capture(observation, state)
            actions = (DroidAction((0.0,) * 7),) * 3

            with self.assertRaisesRegex(ValueError, "not bound"):
                session.write_response(
                    ProposedControl(123, 99.0, actions, Path("/tmp/proposal.pth"))
                )
            session.write_response(
                ProposedControl(123, 100.1, actions, Path("/tmp/proposal.pth"))
            )

            self.assertEqual(session.load_response().observation_id, 123)
            with self.assertRaisesRegex(ValueError, "already has"):
                session.write_response(
                    ProposedControl(123, 100.2, actions, Path("/tmp/proposal.pth"))
                )

    def test_rejects_state_copied_from_another_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = ControlSession.at(Path(temp_dir), "session-b")
            observation = ControlObservation(
                observation_id=123,
                captured_at_unix_seconds=100.0,
                context_frame=Path("context.png"),
                target=ControlTarget(Path("target.png")),
                expected_proposal=Path("/tmp/proposal.pth"),
                pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                previous_action=DroidAction((0.0,) * 7),
                warmup_frames=4,
            )
            state = ControlSessionState(
                "session-a",
                "held-reference",
                11400,
                "control-session-a",
                (0.0,) * 7,
                False,
                0.0,
            )
            session.write_capture(observation, state)
            session.response_path.write_text(
                json.dumps(
                    ProposedControl(
                        123,
                        100.1,
                        (DroidAction((0.0,) * 7),) * 3,
                        Path("/tmp/proposal.pth"),
                    ).to_dict()
                )
            )

            with self.assertRaisesRegex(ValueError, "different session"):
                session.load()

    def test_experimental_candidate_sessions_fail_closed_without_their_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = ControlSession.at(root, "candidate-session")
            proposal = Path("/tmp/experimental_shadow_candidate.pth")
            observation = ControlObservation(
                123,
                100.0,
                Path("context.png"),
                ControlTarget(Path("target.png")),
                proposal,
                DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                DroidAction((0.0,) * 7),
                4,
            )
            state = ControlSessionState(
                "candidate-session",
                "held-reference",
                11400,
                "control-candidate-session",
                (0.0,) * 7,
                False,
                0.0,
                execution_policy=ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
            )
            session.write_capture(observation, state)
            session.write_response(
                ProposedControl(
                    123,
                    100.1,
                    (DroidAction((0.0,) * 7),) * 3,
                    proposal,
                )
            )

            with self.assertRaisesRegex(ValueError, "candidate evidence"):
                session.load()

    def test_normal_sessions_reject_an_experimental_candidate_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = ControlSession.at(root, "direct-session")
            proposal = Path("/tmp/direct.pth")
            observation = ControlObservation(
                123,
                100.0,
                Path("context.png"),
                ControlTarget(Path("target.png")),
                proposal,
                DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                DroidAction((0.0,) * 7),
                4,
            )
            state = ControlSessionState(
                "direct-session",
                "held-reference",
                11400,
                "control-direct-session",
                (0.0,) * 7,
                False,
                0.0,
            )
            session.write_capture(observation, state)
            session.write_response(
                ProposedControl(
                    123,
                    100.1,
                    (DroidAction((0.0,) * 7),) * 3,
                    proposal,
                )
            )
            session.candidate_binding_path.write_text("{}")

            with self.assertRaisesRegex(ValueError, "non-experimental"):
                session.load()

    @patch("sim.control_session.validate_observation_target")
    def test_insertion_reset_trials_fail_closed_without_their_binding(
        self,
        _validate_target,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = ControlSession.at(Path(temp_dir), "insertion-trial")
            proposal = Path("/tmp/proposal.pth")
            observation = ControlObservation(
                123,
                100.0,
                Path("context.png"),
                ControlTarget(Path("target.png")),
                proposal,
                DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                DroidAction((0.0,) * 7),
                43,
            )
            state = ControlSessionState(
                "insertion-trial",
                "held-reference",
                52600,
                "control-insertion-trial",
                (0.0,) * 7,
                False,
                0.0,
                execution_policy=ControlExecutionPolicy.INSERTION_RESET_TRIAL,
                plug_position=(0.4, 0.0, 0.5),
                plug_attached=True,
            )
            session.write_capture(observation, state)
            session.write_response(
                ProposedControl(
                    123,
                    100.1,
                    (DroidAction((0.0,) * 7),) * 3,
                    proposal,
                    "a" * 64,
                )
            )

            with self.assertRaisesRegex(ValueError, "insertion trial evidence"):
                session.load()


if __name__ == "__main__":
    unittest.main()
