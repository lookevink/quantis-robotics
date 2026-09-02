from __future__ import annotations

from dataclasses import replace
import sys
from types import ModuleType, SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
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
    InsertionTrialRollbackEvidence,
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
    BLOCKED_IK_DIAGNOSTIC_FINGERPRINTS,
    build_insertion_followup_capture,
    diagnose_contact_grasp_active_rotation_ik,
    diagnose_contact_grasp_blocked_ik,
    diagnose_contact_grasp_blocked_ik_tolerances,
    diagnose_contact_grasp_execution_ik,
    diagnose_contact_grasp_settlement_rollback,
    restore_grasp_transition_retry,
    restore_insertion_rollback_retry,
    validate_followup_continuity,
    verify_insertion_demo_rollout_result,
    verify_insertion_two_step_result,
)
from sim.isaac_demo_runtime import JointCommand


class FollowupContinuityTest(unittest.TestCase):
    def test_blocked_ik_tolerance_diagnostic_replays_every_scale_without_motion(
        self,
    ) -> None:
        pose = DroidPose((0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        joints = (0.0,) * 7
        raw = DroidAction((0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        scales = (DroidActionScale.uniform(1.0), DroidActionScale.uniform(0.5))
        attempts = tuple(
            SafetyProjectionAttempt(
                scale,
                ControlGateDecision(
                    7,
                    pose.applied(scale.apply(raw)),
                    (ControlGateReason.IK_SOLUTION_FAILED,),
                ),
                0.0,
                joints,
            )
            for scale in scales
        )
        step = SimpleNamespace(
            result=SimpleNamespace(
                status=ControlResultStatus.BLOCKED,
                gate=SimpleNamespace(
                    reasons=(ControlGateReason.IK_SOLUTION_FAILED,)
                ),
                selected_action_scale=None,
                post_action=None,
                insertion_trial_refresh=InsertionEvaluationRefresh(
                    100.0,
                    ControlSafetySnapshot(
                        joints,
                        0.04,
                        (0.0, 0.0, 1.0),
                        0.0,
                        False,
                        True,
                    ),
                    pose,
                ),
                projection_attempts=attempts,
            )
        )
        tolerance_result = ({"orientation_tolerance_radians": 0.001, "solved": True},)

        from jepa_wm.control_rollout import ControlStepSummary

        with (
            patch(
                "sim.isaac_control_followup.ControlSession.at",
                return_value=object(),
            ),
            patch.object(
                ControlStepSummary,
                "from_session",
                return_value=step,
            ),
            patch(
                "sim.isaac_control_followup.diagnose_droid_pose_orientation_tolerances",
                return_value=tolerance_result,
            ) as diagnose,
        ):
            evidence = diagnose_contact_grasp_blocked_ik_tolerances(
                "grasp-to-insertion-run-grasp-01"
            )

        self.assertEqual(evidence["status"], "diagnosed_no_actuation")
        self.assertFalse(evidence["simulator_action_applied"])
        self.assertEqual(len(evidence["attempts"]), 2)
        self.assertEqual(diagnose.call_count, 2)

    def test_blocked_ik_tolerance_diagnostic_rejects_historical_fk_failure(
        self,
    ) -> None:
        pose = DroidPose((0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        joints = (0.0,) * 7
        attempt = SafetyProjectionAttempt(
            DroidActionScale.uniform(1.0),
            ControlGateDecision(
                7,
                pose,
                (ControlGateReason.IK_SOLUTION_FAILED,),
            ),
            0.001,
            (0.001, *joints[1:]),
        )
        step = SimpleNamespace(
            result=SimpleNamespace(
                status=ControlResultStatus.BLOCKED,
                gate=SimpleNamespace(
                    reasons=(ControlGateReason.IK_SOLUTION_FAILED,)
                ),
                selected_action_scale=None,
                post_action=None,
                insertion_trial_refresh=InsertionEvaluationRefresh(
                    100.0,
                    ControlSafetySnapshot(
                        joints,
                        0.04,
                        (0.0, 0.0, 1.0),
                        0.0,
                        False,
                        True,
                    ),
                    pose,
                ),
                projection_attempts=(attempt,),
            )
        )

        from jepa_wm.control_rollout import ControlStepSummary

        with (
            patch(
                "sim.isaac_control_followup.ControlSession.at",
                return_value=object(),
            ),
            patch.object(
                ControlStepSummary,
                "from_session",
                return_value=step,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "source is invalid"):
                diagnose_contact_grasp_blocked_ik_tolerances(
                    "grasp-to-insertion-run-grasp-01"
                )

    def test_active_rotation_ik_diagnostic_probes_scale_grid_without_motion(
        self,
    ) -> None:
        from jepa_wm.control_rollout import ControlStepSummary

        pose = DroidPose((0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        joints = (0.0,) * 7
        raw = DroidAction((0.001, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0))
        observation = ControlObservation(
            7,
            99.0,
            Path("context.png"),
            ControlTarget(Path("target.png")),
            Path("/tmp/proposal.pth"),
            pose,
            DroidAction((0.0,) * 7),
            4,
        )
        response = ProposedControl(
            7,
            99.5,
            (raw, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        source_attempts = (
            SafetyProjectionAttempt(
                DroidActionScale(0.5, 1.0, 0.0),
                ControlGateDecision(
                    7,
                    pose.applied(DroidActionScale(0.5, 1.0, 0.0).apply(raw)),
                    (ControlGateReason.IK_SOLUTION_FAILED,),
                ),
                0.0,
                joints,
            ),
            SafetyProjectionAttempt(
                DroidActionScale(0.25, 1.0, 0.0),
                ControlGateDecision(
                    7,
                    pose.applied(DroidActionScale(0.25, 1.0, 0.0).apply(raw)),
                    (ControlGateReason.JOINT_VELOCITY_VIOLATION,),
                ),
                2.0,
                (2.0, *joints[1:]),
                pose,
            ),
        )
        refresh = SimpleNamespace(
            refreshed_at_unix_seconds=100.0,
            live_pose=pose,
            live_state=SimpleNamespace(
                joint_positions=joints,
                gripper_width_m=0.02,
                plug_attached=True,
                collision_detected=False,
                contact_force_newtons=0.0,
            ),
            authorize_target_relative=Mock(return_value=(observation, response)),
        )
        policy = Mock(
            requires_resolvable_rotation=True,
        )
        policy.action_for_execution.return_value = raw
        step = SimpleNamespace(
            observation=observation,
            response=response,
            state=SimpleNamespace(
                plug_attached=True,
                current_joint_positions=joints,
                active_drive_target=object(),
                require_current_contact_grasp_policy=Mock(return_value=policy),
                require_safety_snapshot=Mock(return_value=object()),
            ),
            result=SimpleNamespace(
                status=ControlResultStatus.BLOCKED,
                gate=SimpleNamespace(
                    reasons=(ControlGateReason.JOINT_VELOCITY_VIOLATION,)
                ),
                selected_action_scale=None,
                post_action=None,
                insertion_trial_refresh=refresh,
                projection_attempts=source_attempts,
            ),
        )

        def project(_safety, _proposal, scale, **_kwargs):
            next_pose = pose.applied(scale.apply(raw))
            decision = ControlGateDecision(7, next_pose, ())
            attempt = SafetyProjectionAttempt(
                scale,
                decision,
                0.001,
                joints,
                next_pose,
                0.00075,
            )
            selected = SimpleNamespace(
                solved_pose=SimpleNamespace(
                    position_error_m=1e-6,
                    orientation_error_rad=1e-4,
                )
            )
            return attempt, selected

        with (
            patch(
                "sim.isaac_control_followup.ControlSession.at",
                return_value=object(),
            ),
            patch.object(ControlStepSummary, "from_session", return_value=step),
            patch(
                "sim.isaac_control_execution.project_control_candidate",
                side_effect=project,
            ) as project_candidate,
            patch(
                "sim.isaac_control_followup.diagnose_droid_pose_orientation_tolerances",
                return_value=({"solved": True},),
            ),
        ):
            evidence = diagnose_contact_grasp_active_rotation_ik(
                "grasp-to-insertion-run-grasp-15"
            )

        self.assertFalse(evidence["simulator_action_applied"])
        self.assertEqual(len(evidence["attempts"]), 10)
        self.assertEqual(project_candidate.call_count, 10)
        self.assertTrue(evidence["attempts"][0]["direction_observable"])
        self.assertFalse(evidence["attempts"][1]["direction_observable"])
        self.assertEqual(
            evidence["attempts"][0]["ik_orientation_tolerance_radians"],
            0.00075,
        )
        self.assertTrue(evidence["attempts"][0]["command_realization"]["passed"])

    def test_blocked_ik_diagnostic_finds_a_local_branch_without_motion(self) -> None:
        pose = DroidPose((0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        raw = DroidAction((0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1))
        joints = (0.0,) * 7
        expected = JointDriveTarget(joints, 0.04)
        scales = (
            DroidActionScale(0.375, 0.0, 0.0625),
            DroidActionScale(0.25, 0.0, 0.0625),
        )
        attempts = tuple(
            SafetyProjectionAttempt(
                scale,
                ControlGateDecision(
                    7,
                    pose.applied(scale.apply(raw)),
                    (ControlGateReason.JOINT_VELOCITY_VIOLATION,),
                ),
                1.0,
                (1.0,) * 7,
            )
            for scale in (*scales, *scales)
        )
        refresh = InsertionEvaluationRefresh(
            100.0,
            ControlSafetySnapshot(joints, 0.04, (0.0, 0.0, 1.0), 0.0, False, False),
            pose,
        )
        policy = Mock()
        policy.action_for_execution.return_value = raw
        observation = ControlObservation(
            7,
            99.0,
            Path("control_sessions/source/context.png"),
            ControlTarget(Path("recordings/reference/wrist/frame_000113.png")),
            Path("/tmp/proposal.pth"),
            pose,
            DroidAction((0.0,) * 7),
            97,
        )
        response = ProposedControl(
            7,
            99.5,
            (raw,) * 3,
            Path("/tmp/proposal.pth"),
        )
        step = SimpleNamespace(
            observation=observation,
            response=response,
            state=SimpleNamespace(
                plug_attached=False,
                active_drive_target=expected,
                current_joint_positions=joints,
                require_safety_snapshot=Mock(return_value=refresh.live_state),
                require_current_contact_grasp_policy=Mock(return_value=policy),
            ),
            result=SimpleNamespace(
                status=ControlResultStatus.BLOCKED,
                gate=SimpleNamespace(
                    reasons=(ControlGateReason.JOINT_VELOCITY_VIOLATION,)
                ),
                selected_action_scale=None,
                post_action=None,
                insertion_trial_refresh=refresh,
                projection_attempts=attempts,
            ),
        )
        actual = JointCommand(np.full(7, 1e-4), 0.04001)
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(actual_command=Mock(return_value=actual)),
            attachment=SimpleNamespace(attached=False),
            sensor=object(),
        )
        solved = SimpleNamespace(
            arm_positions=np.full(7, 2e-3),
            position_error_m=1e-5,
            orientation_error_rad=1e-4,
        )
        stage = object()
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
        omni = ModuleType("omni")
        omni.usd = usd
        session = SimpleNamespace(
            path=Path("/tmp/v34-session"),
            execution_path=Path("/tmp/v34-execution-does-not-exist"),
        )
        projected = SimpleNamespace(solved_pose=solved)
        projected_attempts = iter(
            SafetyProjectionAttempt(
                scale,
                ControlGateDecision(7, pose.applied(scale.apply(raw)), ()),
                0.002,
                tuple(solved.arm_positions),
            )
            for scale in scales
        )

        def project(*args, **kwargs):
            del args, kwargs
            return next(projected_attempts), projected

        with (
            patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}),
            patch(
                "sim.isaac_control_followup.ControlSession.at",
                return_value=session,
            ),
            patch(
                "sim.isaac_control_followup.artifact_fingerprint",
                side_effect=lambda path: BLOCKED_IK_DIAGNOSTIC_FINGERPRINTS[
                    path.name
                ],
            ),
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                return_value=step,
            ),
            patch(
                "sim.isaac_control_execution.project_control_candidate",
                side_effect=project,
            ) as project_candidate,
            patch(
                "sim.isaac_control_followup.live_runtime_for",
                return_value=runtime,
            ),
            patch(
                "sim.isaac_control_followup.current_drive_target",
                return_value=expected,
            ),
            patch(
                "sim.isaac_control_followup.read_control_contact",
                return_value=(False, 0.0),
            ),
        ):
            evidence = diagnose_contact_grasp_blocked_ik(
                "unknown-start-e2e-v34-62605-grasp-001"
            )

        self.assertTrue(evidence["diagnostic_passed"])
        self.assertFalse(evidence["simulator_action_applied"])
        self.assertEqual(len(evidence["attempts"]), 2)
        self.assertEqual(project_candidate.call_count, 2)

    def test_settlement_rollback_diagnostic_is_read_only_and_exact(self) -> None:
        captured = JointDriveTarget((0.0,) * 7, 0.04)
        expected = JointDriveTarget.for_command(
            captured.joint_positions,
            captured.gripper_width_m,
        )
        step = SimpleNamespace(
            result=SimpleNamespace(
                status=ControlResultStatus.ROLLBACK_FAILED,
                gate=SimpleNamespace(passed=True),
                selected_action_scale=object(),
                post_action=None,
                execution_error=(
                    "RuntimeError: contact-grasp command did not satisfy its "
                    "tracking gates within its bounded timeout: "
                    "tracking_reasons=['translation_error'], "
                    "translation_error_meters=0.000510000; "
                    "rollback verification failed: "
                    "RuntimeError: rollback command did not settle: "
                    "arm_error=0.001117 rad, gripper_error=0.000007 m"
                ),
                insertion_trial_refresh=SimpleNamespace(
                    live_state=SimpleNamespace(
                        joint_positions=captured.joint_positions,
                        gripper_width_m=captured.gripper_width_m,
                        plug_attached=False,
                    )
                ),
            ),
            state=SimpleNamespace(
                execution_policy=ControlExecutionPolicy.DIRECT,
                contact_grasp_target_policy=object(),
            ),
        )
        actual = JointCommand(np.full(7, 5e-4), 0.04001)
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(actual_command=Mock(return_value=actual)),
            attachment=SimpleNamespace(attached=False),
            sensor=object(),
        )
        stage = object()
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
        omni = ModuleType("omni")
        omni.usd = usd
        with (
            patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}),
            patch("sim.isaac_control_followup.ControlSession.at"),
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                return_value=step,
            ),
            patch(
                "sim.isaac_control_followup.live_runtime_for",
                return_value=runtime,
            ),
            patch(
                "sim.isaac_control_followup.current_drive_target",
                return_value=expected,
            ),
            patch(
                "sim.isaac_control_followup.read_control_contact",
                return_value=(False, 0.0),
            ),
        ):
            evidence = diagnose_contact_grasp_settlement_rollback("session-005")

        self.assertTrue(evidence["diagnostic_passed"])
        self.assertTrue(evidence["active_target_matches"])
        self.assertFalse(evidence["simulator_action_applied"])
        self.assertAlmostEqual(evidence["maximum_joint_error_rad"], 5e-4)

    def test_settlement_rollback_diagnostic_rejects_unrelated_failure(self) -> None:
        step = SimpleNamespace(
            result=SimpleNamespace(
                status=ControlResultStatus.ROLLBACK_FAILED,
                gate=SimpleNamespace(passed=True),
                selected_action_scale=object(),
                post_action=None,
                execution_error=(
                    "RuntimeError: contact interlock failed; rollback verification "
                    "failed: RuntimeError: rollback command did not settle"
                ),
                insertion_trial_refresh=SimpleNamespace(live_state=object()),
            ),
            state=SimpleNamespace(
                execution_policy=ControlExecutionPolicy.DIRECT,
                contact_grasp_target_policy=object(),
            ),
        )
        with (
            patch("sim.isaac_control_followup.ControlSession.at"),
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                return_value=step,
            ),
            self.assertRaisesRegex(ValueError, "settlement rollback source"),
        ):
            diagnose_contact_grasp_settlement_rollback("session-005")

    def test_execution_ik_diagnostic_probes_exact_rotation_without_motion(self) -> None:
        start = DroidPose((0.3, -0.2, 0.5, 0.0, 0.0, 0.0, 0.7))
        target = start.applied(
            DroidAction((0.0005, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0))
        )
        joints = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)
        step = SimpleNamespace(
            result=SimpleNamespace(
                status=ControlResultStatus.ROLLED_BACK_EXECUTION,
                gate=SimpleNamespace(passed=True, next_pose=target),
                selected_action_scale=object(),
                post_action=None,
                insertion_trial_drive=InsertionTrialDriveEvidence(
                    JointDriveTarget(joints, 0.04),
                    JointDriveTarget(joints, 0.04),
                ),
                execution_error=(
                    "RuntimeError: contact-grasp command stopped making "
                    "realizable progress: completion_reasons="
                    "['rotation_underrealized']"
                ),
                insertion_trial_refresh=SimpleNamespace(
                    live_pose=start,
                    live_state=SimpleNamespace(
                        joint_positions=joints,
                        plug_attached=True,
                    ),
                ),
            )
        )
        attempts = (
            {
                "orientation_tolerance_radians": 0.0005,
                "solved": True,
                "arm_positions": list(joints),
            },
        )
        with (
            patch("sim.isaac_control_followup.ControlSession.at"),
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                return_value=step,
            ),
            patch(
                "sim.isaac_control_followup.diagnose_droid_pose_orientation_tolerances",
                return_value=attempts,
            ) as diagnose,
            patch(
                "sim.isaac_control_followup.diagnose_joint_target_realization",
                return_value={"command_realization": {"passed": True}},
            ) as diagnose_drive,
        ):
            evidence = diagnose_contact_grasp_execution_ik("session-rotation")

        diagnose.assert_called_once()
        self.assertEqual(diagnose.call_args.args[0], start)
        self.assertEqual(diagnose.call_args.args[1], target)
        np.testing.assert_allclose(diagnose.call_args.args[2], joints)
        diagnose_drive.assert_called_once()
        self.assertIn("compensated_drive_target", evidence["attempts"][0])
        self.assertAlmostEqual(evidence["commanded_rotation_norm_radians"], 0.004)
        self.assertFalse(evidence["simulator_action_applied"])

    def test_execution_ik_diagnostic_rejects_translation_only_plateau(self) -> None:
        start = DroidPose((0.3, -0.2, 0.5, 0.0, 0.0, 0.0, 0.7))
        step = SimpleNamespace(
            result=SimpleNamespace(
                status=ControlResultStatus.ROLLED_BACK_EXECUTION,
                gate=SimpleNamespace(
                    passed=True,
                    next_pose=start.applied(
                        DroidAction((0.0005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                    ),
                ),
                selected_action_scale=object(),
                post_action=None,
                insertion_trial_drive=InsertionTrialDriveEvidence(
                    JointDriveTarget(
                        (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5), 0.04
                    ),
                    JointDriveTarget(
                        (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5), 0.04
                    ),
                ),
                execution_error=(
                    "RuntimeError: contact-grasp command stopped making "
                    "realizable progress: completion_reasons="
                    "['translation_underrealized']"
                ),
                insertion_trial_refresh=SimpleNamespace(
                    live_pose=start,
                    live_state=SimpleNamespace(
                        joint_positions=(
                            0.0,
                            -0.5,
                            0.0,
                            -2.0,
                            0.0,
                            1.5,
                            0.5,
                        ),
                        plug_attached=True,
                    ),
                ),
            )
        )
        with (
            patch("sim.isaac_control_followup.ControlSession.at"),
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                return_value=step,
            ),
            self.assertRaisesRegex(ValueError, "execution IK diagnostic source"),
        ):
            diagnose_contact_grasp_execution_ik("session-translation")

    def test_insertion_retry_rebinds_only_an_exact_settled_safe_rollback(
        self,
    ) -> None:
        drive_target = JointDriveTarget((0.0,) * 7, 0.03)
        rollback = object.__new__(InsertionTrialRollbackEvidence)
        object.__setattr__(rollback, "drive_target", drive_target)
        object.__setattr__(rollback, "plug_attached", True)
        previous = Mock()
        rolled_back = Mock()
        rolled_back.result.status = ControlResultStatus.ROLLED_BACK_PROGRESS
        rolled_back.result.insertion_trial_rollback = rollback
        rolled_back.state.previous_session_id = "previous-insertion"
        runtime = SimpleNamespace(
            actuators=object(), attachment=object(), sensor=object()
        )
        stage = object()
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
        omni = ModuleType("omni")
        omni.usd = usd
        with (
            patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}),
            patch("sim.isaac_control_followup.ControlSession.at"),
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                side_effect=(previous, rolled_back),
            ),
            patch(
                "sim.isaac_control_followup.InsertionFollowupLineage",
                return_value=SimpleNamespace(active_drive_target=drive_target),
            ),
            patch(
                "sim.isaac_control_followup.live_runtime_for",
                return_value=runtime,
            ) as live_runtime,
            patch("sim.isaac_control_followup.bind_live_runtime") as bind_runtime,
        ):
            evidence = restore_insertion_rollback_retry(
                "previous-insertion", "rolled-back-insertion"
            )

        self.assertEqual(evidence["status"], "insertion_rollback_retry_ready")
        live_runtime.assert_called_once_with("rolled-back-insertion", stage)
        bind_runtime.assert_called_once_with(
            "previous-insertion",
            stage,
            runtime.actuators,
            runtime.attachment,
            runtime.sensor,
        )

    def test_retry_rebinds_only_one_exact_settled_tracking_rollback(self) -> None:
        drive_target = JointDriveTarget((0.0,) * 7, 0.03)
        rollback = object.__new__(InsertionTrialRollbackEvidence)
        object.__setattr__(rollback, "drive_target", drive_target)
        object.__setattr__(rollback, "plug_attached", True)
        grasp = Mock()
        rolled_back = Mock()
        rolled_back.result.status = ControlResultStatus.ROLLED_BACK_TRACKING
        rolled_back.result.insertion_trial_rollback = rollback
        rolled_back.state.previous_session_id = "grasp-session"
        runtime = SimpleNamespace(
            actuators=object(), attachment=object(), sensor=object()
        )
        stage = object()
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
        omni = ModuleType("omni")
        omni.usd = usd
        with (
            patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}),
            patch("sim.isaac_control_followup.ControlSession.at"),
            patch(
                "jepa_wm.control_rollout.ControlStepSummary.from_session",
                side_effect=(grasp, rolled_back),
            ),
            patch(
                "sim.isaac_control_followup.GraspToInsertionLineage",
                return_value=SimpleNamespace(active_drive_target=drive_target),
            ),
            patch(
                "sim.isaac_control_followup.live_runtime_for",
                return_value=runtime,
            ) as live_runtime,
            patch("sim.isaac_control_followup.bind_live_runtime") as bind_runtime,
        ):
            evidence = restore_grasp_transition_retry(
                "grasp-session", "rolled-back-session"
            )

        self.assertEqual(evidence["status"], "grasp_transition_retry_ready")
        live_runtime.assert_called_once_with("rolled-back-session", stage)
        bind_runtime.assert_called_once_with(
            "grasp-session",
            stage,
            runtime.actuators,
            runtime.attachment,
            runtime.sensor,
        )

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
            expected_proposal=Path("/tmp/parent-proposal.pth"),
        )

        self.assertEqual(
            observation.expected_proposal,
            Path("/tmp/parent-proposal.pth"),
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
        extended = InsertionFollowupLineage(
            prior_observation,
            replace(
                prior_state,
                previous_session_id="insertion-trial-before-previous",
                insertion_rollout_position=InsertionRolloutPosition(4, 4),
            ),
            result,
            8,
        )
        self.assertEqual(
            extended.followup_position,
            InsertionRolloutPosition(5, 8),
        )


if __name__ == "__main__":
    unittest.main()
