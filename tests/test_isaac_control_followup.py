from __future__ import annotations

from dataclasses import replace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlGateDecision,
    ControlInterlockEvidence,
    SafetyProjectionAttempt,
)
from jepa_wm.insertion_refresh import (
    ControlSafetySnapshot,
    InsertionEvaluationRefresh,
)
from jepa_wm.insertion_rollout import InsertionRolloutPosition
from jepa_wm.insertion_contract import INSERTION_CONTROL_TARGET_POLICY
from jepa_wm.insertion_trial import (
    InsertionTrialDriveEvidence,
    InsertionTrialPostActionEvidence,
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.joint_settlement import JointSettlementEvidence
from jepa_wm.target_progress import RealizedTargetProgressDecision
from jepa_wm.control_tracking import ActionTrackingDecision
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSessionState,
    InsertionFollowupLineage,
    PostActionEvidence,
)
from sim.isaac_control_followup import (
    build_insertion_followup_capture,
    validate_followup_continuity,
    verify_insertion_demo_rollout_result,
    verify_insertion_two_step_result,
)
from sim.isaac_demo_runtime import JointCommand


class FollowupContinuityTest(unittest.TestCase):
    def _previous(self) -> PostActionEvidence:
        action = DroidAction((0.0,) * 7)
        return PostActionEvidence(
            action,
            action,
            action,
            ActionTrackingDecision(0.0, 0.0, 0.0, 0.0, 0.0, ()),
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
            0.0,
            0.0,
            False,
            {"path": "/tmp/post.png", "shape": [512, 512, 4]},
        )

    def test_accepts_the_same_live_articulation(self) -> None:
        previous = self._previous()
        validate_followup_continuity(
            previous,
            JointCommand(np.asarray(previous.joint_positions), 0.04),
            previous.pose,
        )

    def test_rejects_a_reset_or_changed_live_articulation(self) -> None:
        previous = self._previous()
        with self.subTest("joint drift"):
            joints = np.asarray(previous.joint_positions)
            joints[0] += 0.01
            with self.assertRaisesRegex(ValueError, "live stage"):
                validate_followup_continuity(
                    previous,
                    JointCommand(joints, 0.04),
                    previous.pose,
                )
        with self.subTest("Cartesian drift"):
            with self.assertRaisesRegex(ValueError, "live stage"):
                validate_followup_continuity(
                    previous,
                    JointCommand(np.asarray(previous.joint_positions), 0.04),
                    DroidPose((0.41, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                )

    def test_rejects_a_lost_connector_attachment(self) -> None:
        previous = self._previous()
        previous = PostActionEvidence(
            previous.raw_proposed_action,
            previous.commanded_action,
            previous.actual_action,
            previous.tracking,
            previous.pose,
            previous.joint_positions,
            previous.maximum_joint_tracking_error_rad,
            previous.contact_force_newtons,
            previous.collision_detected,
            previous.frame,
            (0.4, 0.0, 0.5),
            True,
        )

        with self.assertRaisesRegex(ValueError, "live stage"):
            validate_followup_continuity(
                previous,
                JointCommand(np.asarray(previous.joint_positions), 0.04),
                previous.pose,
                current_plug_position=(0.4, 0.0, 0.5),
                current_plug_attached=False,
            )

    def test_terminal_verifier_uses_the_authenticated_proposal_path(self) -> None:
        first_step = Mock()
        first_step.observation.expected_proposal = Path("/tmp/actual-proposal.pth")
        first_step.state.resolved_insertion_rollout_position.return_value = (
            InsertionRolloutPosition(1, 2)
        )
        second_step = Mock()
        second_step.state.resolved_insertion_rollout_position.return_value = (
            InsertionRolloutPosition(2, 2)
        )
        report = Mock()
        report.to_dict.return_value = {"all_steps_applied": True}
        with (
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                side_effect=(first_step, second_step),
            ),
            patch(
                "jepa_wm.control_rollout.ControlRolloutReport.from_sessions",
                return_value=report,
            ) as from_sessions,
            patch("sim.isaac_control_followup.ControlSession.at"),
        ):
            result = verify_insertion_two_step_result(
                "run-action1",
                "run-action2",
                "reference",
                52600,
            )

        self.assertEqual(result["status"], "two_step_applied")
        self.assertEqual(
            from_sessions.call_args.kwargs["proposal"],
            Path("/tmp/actual-proposal.pth"),
        )
        report.require_all_steps_applied.assert_called_once_with()

    def test_demo_verifier_requires_the_complete_persisted_four_step_roster(self) -> None:
        steps = []
        for step_index in range(1, 5):
            step = Mock()
            step.observation.expected_proposal = Path("/tmp/actual-proposal.pth")
            step.state.insertion_rollout_position = InsertionRolloutPosition(
                step_index,
                4,
            )
            step.state.resolved_insertion_rollout_position.return_value = (
                step.state.insertion_rollout_position
            )
            steps.append(step)
        report = Mock()
        report.to_dict.return_value = {"all_steps_applied": True}
        with (
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                side_effect=steps,
            ),
            patch(
                "jepa_wm.control_rollout.ControlRolloutReport.from_sessions",
                return_value=report,
            ) as from_sessions,
            patch("sim.isaac_control_followup.ControlSession.at"),
        ):
            result = verify_insertion_demo_rollout_result(
                "run-action1,run-action2,run-action3,run-action4",
                "reference",
                52600,
            )

        self.assertEqual(result["status"], "demo_rollout_applied")
        self.assertEqual(from_sessions.call_args.kwargs["requested_steps"], 4)
        report.require_all_steps_applied.assert_called_once_with()

    def test_builds_a_followup_capture_from_the_exact_applied_drive_target(self) -> None:
        previous = self._previous()
        previous = PostActionEvidence(
            previous.raw_proposed_action,
            previous.commanded_action,
            previous.actual_action,
            previous.tracking,
            previous.pose,
            previous.joint_positions,
            previous.maximum_joint_tracking_error_rad,
            previous.contact_force_newtons,
            previous.collision_detected,
            previous.frame,
            (0.4, 0.0, 0.5),
            True,
            InsertionTrialPostActionEvidence(
                JointSettlementEvidence(0.001, 0.0005, 2, (0.0004, 0.0003)),
                RealizedTargetProgressDecision(
                    0.001, 0.0005, 0.5, 0.0001, 0.0001, False, ()
                ),
            ),
        )
        drive = InsertionTrialDriveEvidence(
            JointDriveTarget(tuple(previous.joint_positions), 0.04),
            JointDriveTarget(tuple(value + 0.001 for value in previous.joint_positions), 0.04),
        )
        gate = ControlGateDecision(9, previous.pose, ())
        result = ControlResult(
            ControlResultStatus.APPLIED,
            "insertion-trial-previous",
            gate,
            (SafetyProjectionAttempt(ACTION_SCALES[0], gate, 0.001, previous.joint_positions),),
            ACTION_SCALES[0],
            0.1,
            0.0,
            0.0,
            0.0,
            previous,
            execution_interlock=ControlInterlockEvidence(0.0, False),
            insertion_trial_refresh=InsertionEvaluationRefresh(
                101.0,
                ControlSafetySnapshot(
                    previous.joint_positions,
                    0.04,
                    previous.plug_position,
                    0.0,
                    False,
                    True,
                ),
                previous.pose,
            ),
            insertion_trial_drive=drive,
        )
        prior_observation = ControlObservation(
            9,
            100.0,
            Path("prior.png"),
            ControlTarget(Path("target-48.png"), previous.pose),
            Path("/tmp/proposal.pth"),
            previous.pose,
            DroidAction((0.0,) * 7),
            43,
        )
        prior_state = ControlSessionState(
            "insertion-trial-previous",
            "insertion-held-00",
            52600,
            "control-insertion-trial-previous",
            previous.joint_positions,
            False,
            0.0,
            execution_policy=ControlExecutionPolicy.INSERTION_RESET_TRIAL,
            plug_position=previous.plug_position,
            plug_attached=True,
            current_gripper_width_m=0.04,
            insertion_target_policy=INSERTION_CONTROL_TARGET_POLICY,
            active_drive_target=drive.active_target,
            insertion_rollout_position=InsertionRolloutPosition(1, 4),
        )
        current = ControlSafetySnapshot(
            previous.joint_positions,
            0.04,
            previous.plug_position,
            0.0,
            False,
            True,
        )

        lineage = InsertionFollowupLineage(
            prior_observation,
            prior_state,
            result,
        )
        observation, state = build_insertion_followup_capture(
            "insertion-followup-safety",
            lineage,
            captured_at_unix_seconds=102.0,
            context_frame=Path("followup.png"),
            target=ControlTarget(Path("target-49.png"), previous.pose),
            current=current,
            current_pose=previous.pose,
            active_drive_target=drive.forward_target,
            target_policy=INSERTION_CONTROL_TARGET_POLICY.for_followup(),
        )

        self.assertEqual(observation.previous_action, previous.actual_action)
        self.assertEqual(observation.warmup_frames, 44)
        self.assertEqual(state.previous_session_id, result.session_id)
        self.assertEqual(state.active_drive_target, drive.forward_target)
        self.assertEqual(
            state.insertion_rollout_position,
            InsertionRolloutPosition(2, 4),
        )
        self.assertEqual(
            state.execution_policy,
            ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
        )
        continued = InsertionFollowupLineage(
            prior_observation,
            replace(
                prior_state,
                previous_session_id="insertion-trial-before-previous",
                insertion_rollout_position=InsertionRolloutPosition(2, 4),
            ),
            result,
        )
        self.assertEqual(continued.rollout_position.step_index, 2)
        with self.assertRaisesRegex(ValueError, "maximum"):
            InsertionFollowupLineage(
                prior_observation,
                replace(
                    prior_state,
                    previous_session_id="insertion-trial-before-previous",
                    insertion_rollout_position=InsertionRolloutPosition(4, 4),
                ),
                result,
            )


if __name__ == "__main__":
    unittest.main()
