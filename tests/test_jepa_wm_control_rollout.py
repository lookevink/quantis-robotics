from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    DroidAction,
    DroidActionScale,
    DroidPose,
    action_between,
)
from jepa_wm.action_prior import ActionPriorConfig
from jepa_wm.control_protocol import (
    LEGACY_CONTROL_SCHEMA,
    ControlObservation,
    ControlTarget,
    ProposedControl,
)
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_rollout import (
    ControlRolloutReport,
    ControlStepSummary,
    OrchestrationFailure,
    OrchestrationOperation,
    _contact_grasp_retained_direction,
    _projection_scale_policy_matches,
)
from jepa_wm.contact_grasp_target import (
    CONTACT_GRASP_TARGET_POLICY,
    LEGACY_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    ContactGraspTargetPolicy,
)
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
    ControlInterlockEvidence,
    SafetyProjectionAttempt,
    contact_grasp_action_scales,
)
from jepa_wm.control_tracking import (
    ActionTrackingDecision,
    evaluate_action_tracking,
    evaluate_command_realization,
    tracking_limits_for_policy,
)
from jepa_wm.joint_settlement import (
    GripperSettlementMeasurement,
    GripperSettlementTrace,
    GripperTrackedJointSettlementEvidence,
    JointSettlementEvidence,
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.insertion_trial import (
    InsertionTrialBinding,
    InsertionTrialDriveEvidence,
    InsertionTrialPostActionEvidence,
    InsertionTrialRollbackEvidence,
    InsertionTrialRollbackFailure,
    InsertionTrialRollbackFailureReason,
)
from jepa_wm.insertion_refresh import ControlSafetySnapshot, InsertionEvaluationRefresh
from jepa_wm.insertion_contract import InsertionControlTargetPolicy
from jepa_wm.insertion_rollout import InsertionRolloutPosition
from jepa_wm.planner import CEMConfig
from jepa_wm.planner_readiness import FirstActionThresholds
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
    ActionResponseTrial,
    CalibrationIdentity,
    TaskProgressObjective,
)
from jepa_wm.shadow_planning import (
    ShadowPlanningRequest,
    ShadowSearchConfig,
    plan_shadow_candidates,
)
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.target_progress import RealizedTargetProgressPolicy
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
    PostActionEvidence,
)


class ControlRolloutTest(unittest.TestCase):
    def test_projection_scale_reconstruction_accepts_only_roundoff(self) -> None:
        expected = (DroidActionScale(0.3492566783773001, 1.0, 0.125),)

        self.assertTrue(
            _projection_scale_policy_matches(
                (DroidActionScale(0.3492566783773002, 1.0, 0.125),),
                expected,
            )
        )
        self.assertFalse(
            _projection_scale_policy_matches(
                (DroidActionScale(0.3492566783783001, 1.0, 0.125),),
                expected,
            )
        )
        self.assertFalse(_projection_scale_policy_matches((), expected))

    def test_reconstructs_reset_candidate_refresh_as_candidate_evidence(self) -> None:
        session_id = "unknown-start-candidate"
        joints = (0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0)
        pose = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        action = DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        observation = ControlObservation(
            123,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png"), pose.applied(action)),
            Path("/tmp/proposal.pth"),
            pose,
            DroidAction((0.0,) * 7),
            4,
        )
        response = ProposedControl(
            123,
            100.1,
            (action,) * 3,
            Path("/tmp/proposal.pth"),
        )
        state = ControlSessionState(
            session_id,
            "held-reference",
            12600,
            "unknown-reset",
            joints,
            False,
            0.0,
            execution_policy=ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
            plug_position=(0.0, 0.0, 1.0),
            current_gripper_width_m=0.04,
            active_drive_target=JointDriveTarget(joints, 0.04),
        )
        refresh = InsertionEvaluationRefresh(
            100.15,
            state.require_safety_snapshot(),
            pose,
        )
        gate = ControlGateDecision(123, pose.applied(action), ())
        actual = action_between(pose, pose.applied(action))
        tracking = evaluate_action_tracking(
            action,
            actual,
            tracking_limits_for_policy(
                ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
            ),
        )
        result = ControlResult(
            ControlResultStatus.APPLIED,
            session_id,
            gate,
            (SafetyProjectionAttempt(ACTION_SCALES[0], gate, 0.0, joints),),
            ACTION_SCALES[0],
            0.2,
            0.0,
            0.0,
            0.0,
            PostActionEvidence(
                action,
                action,
                actual,
                tracking,
                pose.applied(action),
                joints,
                0.0,
                0.0,
                False,
                {"path": "/tmp/post.png", "shape": [512, 512, 4]},
                (0.0, 0.0, 1.0),
                False,
            ),
            execution_interlock=ControlInterlockEvidence(0.0, False),
            insertion_trial_refresh=refresh,
        )
        session = Mock(session_id=session_id)
        session.load_capture.return_value = (observation, state)
        session.load_response.return_value = response
        session.load_result.return_value = result
        session.shadow_path = Path("/tmp/no-shadow")
        session.shadow_safety_path = Path("/tmp/no-shadow-safety")

        summary = ControlStepSummary.from_session(session)

        self.assertEqual(summary.result.status, ControlResultStatus.APPLIED)
        session.load_candidate_binding.assert_called_once_with(response)

    def test_current_contact_grasp_derives_reference_transport_direction(self) -> None:
        acquisition = SimpleNamespace(
            state=SimpleNamespace(
                plug_attached=False,
                contact_grasp_target_policy=CONTACT_GRASP_TARGET_POLICY,
            ),
            result=SimpleNamespace(
                post_action=SimpleNamespace(plug_attached=True)
            ),
            observation=SimpleNamespace(
                target_pose=DroidPose(
                    (0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
                )
            ),
        )
        retained = SimpleNamespace(
            state=SimpleNamespace(
                plug_attached=True,
                contact_grasp_target_policy=CONTACT_GRASP_TARGET_POLICY,
            ),
            result=SimpleNamespace(
                post_action=SimpleNamespace(plug_attached=True)
            ),
            observation=SimpleNamespace(
                target_pose=DroidPose(
                    (0.43, 0.01, 0.5, 0.0, 0.0, 0.0, 0.5)
                )
            ),
        )

        np.testing.assert_allclose(
            _contact_grasp_retained_direction((acquisition, retained)),
            (0.03, 0.01, 0.0),
            rtol=0.0,
            atol=1e-12,
        )

        legacy = ContactGraspTargetPolicy(
            LEGACY_CONTACT_GRASP_TARGET_POLICY_SCHEMA
        )
        acquisition.state.contact_grasp_target_policy = legacy
        retained.state.contact_grasp_target_policy = legacy
        self.assertIsNone(
            _contact_grasp_retained_direction((acquisition, retained))
        )

    def test_current_contact_grasp_reconstructs_a_floored_close_scale(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            session = ControlSession.at(root / "control_sessions", "session-1")
            session.path.mkdir(parents=True)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            wrist = reference / "wrist"
            wrist.mkdir()
            steps = []
            for index in range(129):
                frame = wrist / f"frame_{index:06d}.png"
                frame.write_bytes(b"frame")
                steps.append(
                    {
                        "index": index,
                        "end_effector_pose": [
                            0.4 + 0.0002 * index,
                            0.0,
                            0.5,
                            0.0,
                            0.0,
                            0.0,
                            0.75,
                        ],
                        "frames": {"wrist": f"wrist/{frame.name}"},
                    }
                )
            (reference / "manifest.json").write_text(
                json.dumps(
                    {
                        "metadata": {"task": "reach_and_insert"},
                        "action": ACTION_RECORDING_CONTRACT.to_dict(),
                        "cameras": ["wrist"],
                        "frames": len(steps),
                    }
                )
            )
            (reference / "steps.jsonl").write_text(
                "\n".join(json.dumps(step) for step in steps) + "\n"
            )
            policy = ContactGraspTargetPolicy.for_scene_translation(
                (0.0, 0.0, 0.0)
            )
            pose = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.75))
            observation = ControlObservation(
                101,
                100.0,
                Path("control_sessions/session-1/context.png"),
                ControlTarget(
                    Path("recordings/reference/wrist/frame_000097.png"),
                    DroidPose((0.4194, 0.0, 0.5, 0.0, 0.0, 0.0, 0.75)),
                ),
                Path("/tmp/proposal.pth"),
                pose,
                DroidAction((0.0,) * 7),
                94,
            )
            actions = (
                DroidAction(
                    (
                        0.0014771344,
                        0.0003514159,
                        0.0007743249,
                        -0.0033661555,
                        0.0001008337,
                        -0.0012960136,
                        0.0041647237,
                    )
                ),
            ) * 3
            response = ProposedControl(
                101,
                100.2,
                actions,
                Path("/tmp/proposal.pth"),
            )
            joints = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)
            drive_target = JointDriveTarget(joints, 0.02)
            state = ControlSessionState(
                "session-1",
                "reference",
                12601,
                "control-recording",
                joints,
                False,
                0.0,
                "session-0",
                plug_position=(0.0, 0.0, 1.0),
                plug_attached=False,
                current_gripper_width_m=0.02,
                active_drive_target=drive_target,
                contact_grasp_target_policy=policy,
            )
            live_state = ControlSafetySnapshot(
                joints,
                0.02,
                (0.0, 0.0, 1.0),
                0.0,
                False,
                False,
            )
            refresh = InsertionEvaluationRefresh(100.3, live_state, pose)
            raw = policy.action_for_execution(
                actions,
                plug_attached=False,
            )
            scale = contact_grasp_action_scales(
                raw,
                coarse_acquisition=False,
                maximum_coarse_translation_command_meters=(
                    policy.coarse_acquisition_maximum_translation_meters
                ),
                require_resolvable_rotation=True,
                exact_coarse_translation_projection=True,
                coarse_orientation_hold_fallback=True,
                minimum_coarse_translation_command_meters=0.0005,
                resolution_floored_acquisition=True,
                maximum_resolution_floored_translation_command_meters=(
                    policy.fine_acquisition_maximum_translation_meters
                ),
            )[0]
            commanded = scale.apply(raw)
            post_pose = pose.applied(commanded)
            gate = ControlGateDecision(101, post_pose, ())
            actual = action_between(pose, post_pose)
            tracking = evaluate_action_tracking(commanded, actual)
            result = ControlResult(
                ControlResultStatus.APPLIED,
                "session-1",
                gate,
                (SafetyProjectionAttempt(scale, gate, 0.0, joints),),
                scale,
                0.01,
                0.0,
                0.0,
                0.0,
                PostActionEvidence(
                    raw,
                    commanded,
                    actual,
                    tracking,
                    post_pose,
                    joints,
                    0.0,
                    0.0,
                    False,
                    {"path": "/tmp/post.png", "shape": [512, 512, 4]},
                    (0.0, 0.0, 1.0),
                    False,
                ),
                execution_interlock=ControlInterlockEvidence(0.0, False),
                insertion_trial_refresh=refresh,
            )
            session.request_path.write_text(json.dumps(observation.to_dict()))
            session.response_path.write_text(json.dumps(response.to_dict()))
            session.state_path.write_text(json.dumps(state.to_dict()))
            session.result_path.write_text(json.dumps(result.to_dict()))

            with patch(
                "jepa_wm.control_rollout.contact_grasp_action_scales",
                wraps=contact_grasp_action_scales,
            ) as scale_policy:
                summary = ControlStepSummary.from_session(session)

        self.assertEqual(summary.result.selected_action_scale, scale)
        self.assertEqual(summary.result.post_action.raw_proposed_action, raw)
        self.assertEqual(
            scale_policy.call_args.kwargs["exact_coarse_translation_projection"],
            policy.uses_exact_coarse_translation_projection,
        )
        self.assertEqual(
            scale_policy.call_args.kwargs["coarse_orientation_hold_fallback"],
            policy.uses_coarse_orientation_hold_fallback,
        )
        self.assertEqual(
            scale_policy.call_args.kwargs[
                "minimum_coarse_translation_command_meters"
            ],
            policy.minimum_coarse_translation_command_meters,
        )
        self.assertTrue(
            scale_policy.call_args.kwargs["resolution_floored_acquisition"]
        )
        self.assertEqual(
            scale_policy.call_args.kwargs[
                "maximum_resolution_floored_translation_command_meters"
            ],
            policy.fine_acquisition_maximum_translation_meters,
        )

    def test_parses_reset_trial_preflight_failure(self) -> None:
        failure = OrchestrationFailure.parse(
            "reset_trial_source_preflight:exit_7"
        )

        self.assertEqual(
            failure.operation,
            OrchestrationOperation.RESET_TRIAL_SOURCE_PREFLIGHT,
        )
        self.assertEqual(failure.exit_code, 7)
        self.assertIsNone(failure.step_index)

    def test_reconstructs_insertion_settlement_and_realized_progress(self) -> None:
        session_id = "insertion-trial-52600-c43"
        joints = (0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0)
        captured = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04))
        start = DroidPose((0.4001, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04))
        target = DroidPose((0.4007, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04))
        realized = DroidPose((0.4004, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04))
        action = DroidAction((0.0003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        observation = ControlObservation(
            10,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png"), target),
            Path("/tmp/proposal.pth"),
            captured,
            DroidAction((0.0,) * 7),
            43,
        )
        response = ProposedControl(
            10,
            100.2,
            (action,) * 3,
            Path("/tmp/proposal.pth"),
            "a" * 64,
        )
        state = ControlSessionState(
            session_id,
            "insertion-held-00",
            52600,
            "control-insertion-trial",
            joints,
            False,
            0.0,
            execution_policy=ControlExecutionPolicy.INSERTION_RESET_TRIAL,
            plug_position=(0.4, 0.0, 0.5),
            plug_attached=True,
            current_gripper_width_m=0.04,
            active_drive_target=JointDriveTarget(joints, 0.04),
        )
        scale = ACTION_SCALES[0]
        gate = ControlGateDecision(10, start.applied(action), ())
        proposed_joints = (0.002, *joints[1:])
        attempt = SafetyProjectionAttempt(scale, gate, 0.002, proposed_joints)
        settlement = JointSettlementEvidence(
            0.002,
            0.0005,
            2,
            (0.0004, 0.0003),
        )
        actual_action = action_between(start, realized)
        progress = RealizedTargetProgressPolicy().evaluate(start, target, realized)
        post_action = PostActionEvidence(
            action,
            action,
            actual_action,
            evaluate_action_tracking(
                action,
                actual_action,
                tracking_limits_for_policy(
                    ControlExecutionPolicy.INSERTION_RESET_TRIAL
                ),
            ),
            realized,
            proposed_joints,
            0.0,
            0.0,
            False,
            {"path": "/tmp/post.png", "shape": [512, 512, 4]},
            plug_position=(0.4, 0.0, 0.5),
            plug_attached=True,
            insertion_trial=InsertionTrialPostActionEvidence(
                settlement,
                progress,
            ),
            command_realization=evaluate_command_realization(
                action,
                actual_action,
            ),
        )
        result = ControlResult(
            ControlResultStatus.APPLIED,
            session_id,
            gate,
            (attempt,),
            scale,
            0.5,
            0.0,
            0.0,
            0.0,
            post_action,
            execution_interlock=ControlInterlockEvidence(0.0, False),
            insertion_trial_refresh=InsertionEvaluationRefresh(
                100.3,
                state.require_safety_snapshot(),
                start,
            ),
            insertion_trial_drive=InsertionTrialDriveEvidence(
                state.active_drive_target,
                JointDriveTarget.for_command(
                    proposed_joints,
                    (1.0 - gate.next_pose.values[-1]) * 0.08,
                ),
            ),
        )
        binding = InsertionTrialBinding(
            session_id,
            "insertion-safety-52600-c43",
            9,
            10,
            ArtifactIdentity(Path("/tmp/proposal.pth"), "a" * 64),
            response.actions,
            scale,
            source_active_drive_target=state.active_drive_target,
            source_safety_refreshed_at_unix_seconds=100.2,
        )
        session = Mock(session_id=session_id)
        session.load_capture.return_value = (observation, state)
        session.load_response.return_value = response
        session.load_result.return_value = result
        session.load_insertion_trial_binding.return_value = binding
        session.shadow_path = Path("/tmp/no-shadow")
        session.shadow_safety_path = Path("/tmp/no-shadow-safety")

        summary = ControlStepSummary.from_session(session)

        self.assertEqual(
            summary.result.post_action.insertion_trial.joint_settlement,
            settlement,
        )
        self.assertEqual(
            summary.result.post_action.insertion_trial.realized_target_progress,
            progress,
        )
        self.assertEqual(summary.observation.pose, captured)

        session.load_result.return_value = replace(
            result,
            insertion_trial_drive=None,
        )
        with self.assertRaisesRegex(ValueError, "drive evidence is missing"):
            ControlStepSummary.from_session(session)

        drive_rejected_gate = ControlGateDecision(
            gate.observation_id,
            gate.next_pose,
            (ControlGateReason.DRIVE_TARGET_INVALID,),
        )
        session.load_result.return_value = replace(
            result,
            status=ControlResultStatus.BLOCKED,
            gate=drive_rejected_gate,
            selected_action_scale=None,
            post_action=None,
            execution_interlock=None,
            insertion_trial_drive=None,
        )
        with self.assertRaisesRegex(ValueError, "drive rejection is inconsistent"):
            ControlStepSummary.from_session(session)

        rejected_state = replace(
            state,
            active_drive_target=JointDriveTarget(
                tuple(value - 0.003 for value in joints),
                0.04,
            ),
        )
        session.load_capture.return_value = (observation, rejected_state)
        self.assertEqual(
            ControlStepSummary.from_session(session).result.gate.reasons,
            (ControlGateReason.DRIVE_TARGET_INVALID,),
        )
        session.load_capture.return_value = (observation, state)

        session.load_result.return_value = replace(
            result,
            insertion_trial_drive=replace(
                result.insertion_trial_drive,
                active_target=replace(
                    result.insertion_trial_drive.active_target,
                    joint_positions=(0.0001, *joints[1:]),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "active drive target is inconsistent"):
            ControlStepSummary.from_session(session)

        session.load_result.return_value = replace(
            result,
            insertion_trial_drive=replace(
                result.insertion_trial_drive,
                forward_target=replace(
                    result.insertion_trial_drive.forward_target,
                    joint_positions=(0.0019, *joints[1:]),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "forward drive target is inconsistent"):
            ControlStepSummary.from_session(session)

        session.load_result.return_value = result

        for unsafe_post_action in (
            replace(post_action, collision_detected=True),
            replace(post_action, contact_force_newtons=2.1),
            replace(post_action, plug_attached=False),
        ):
            with self.subTest(unsafe=unsafe_post_action):
                session.load_result.return_value = replace(
                    result,
                    post_action=unsafe_post_action,
                )
                with self.assertRaisesRegex(ValueError, "status is inconsistent"):
                    ControlStepSummary.from_session(session)

        drifted_joints = (0.0026, *proposed_joints[1:])
        session.load_result.return_value = replace(
            result,
            post_action=replace(
                post_action,
                joint_positions=drifted_joints,
                maximum_joint_tracking_error_rad=0.0006,
            ),
        )
        with self.assertRaisesRegex(ValueError, "status is inconsistent"):
            ControlStepSummary.from_session(session)

        session.load_result.return_value = replace(
            result,
            post_action=replace(
                post_action,
                insertion_trial=replace(
                    post_action.insertion_trial,
                    joint_settlement=replace(
                        settlement,
                        requested_joint_motion_radians=0.0019,
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "post-action evidence is inconsistent"):
            ControlStepSummary.from_session(session)

        session.load_result.return_value = replace(
            result,
            post_action=replace(
                post_action,
                insertion_trial=replace(
                    post_action.insertion_trial,
                    realized_target_progress=replace(
                        progress,
                        realized_translation_error_meters=0.0004,
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "post-action evidence is inconsistent"):
            ControlStepSummary.from_session(session)

        insufficient_pose = DroidPose(
            (0.4002, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04)
        )
        insufficient_action = action_between(start, insufficient_pose)
        insufficient_progress = RealizedTargetProgressPolicy().evaluate(
            start, target, insufficient_pose
        )
        rollback_settlement = JointSettlementEvidence(
            0.002,
            0.0005,
            2,
            (0.0002, 0.0001),
        )
        rollback_failure = InsertionTrialRollbackFailure(
            proposed_joints,
            joints,
            proposed_joints,
            True,
            InsertionTrialRollbackFailureReason.DRIVE_COMMAND_REJECTED,
            ControlInterlockEvidence(0.0, False),
            False,
            "RuntimeError: drive command rejected",
            drive_target=state.active_drive_target,
        )
        session.load_result.return_value = replace(
            result,
            status=ControlResultStatus.ROLLBACK_FAILED,
            post_action=replace(post_action, collision_detected=True),
            execution_error="RuntimeError: rollback failed",
            insertion_trial_rollback=rollback_failure,
        )
        self.assertEqual(
            ControlStepSummary.from_session(session).result.status,
            ControlResultStatus.ROLLBACK_FAILED,
        )
        rolled_back_post = replace(
            post_action,
            actual_action=insufficient_action,
            tracking=evaluate_action_tracking(
                action,
                insufficient_action,
                tracking_limits_for_policy(
                    ControlExecutionPolicy.INSERTION_RESET_TRIAL
                ),
            ),
            pose=insufficient_pose,
            insertion_trial=InsertionTrialPostActionEvidence(
                settlement,
                insufficient_progress,
            ),
            command_realization=evaluate_command_realization(
                action,
                insufficient_action,
            ),
        )
        rolled_back_result = replace(
            result,
            status=ControlResultStatus.ROLLED_BACK_TRACKING,
            post_action=rolled_back_post,
            insertion_trial_rollback=InsertionTrialRollbackEvidence(
                proposed_joints,
                joints,
                (0.0001, *joints[1:]),
                GripperTrackedJointSettlementEvidence(
                    rollback_settlement,
                    GripperSettlementMeasurement(
                        0.04,
                        0.04,
                        GripperSettlementTrace((0.0, 0.0), 1e-3),
                    ),
                ),
                True,
                state.active_drive_target,
            ),
        )
        session.load_result.return_value = rolled_back_result

        self.assertEqual(
            ControlResult.from_dict(rolled_back_result.to_dict()),
            rolled_back_result,
        )

        rolled_back = ControlStepSummary.from_session(session)

        self.assertFalse(
            rolled_back.result.post_action.insertion_trial.realized_target_progress.passed
        )
        self.assertEqual(
            rolled_back.result.insertion_trial_rollback.joint_settlement,
            rollback_settlement,
        )
        rollback = rolled_back_result.insertion_trial_rollback
        session.load_result.return_value = replace(
            rolled_back_result,
            insertion_trial_rollback=replace(
                rollback,
                drive_target=replace(
                    state.active_drive_target,
                    joint_positions=(0.0001, *joints[1:]),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "rollback evidence is inconsistent"):
            ControlStepSummary.from_session(session)

        session.load_result.return_value = replace(
            rolled_back_result,
            insertion_trial_rollback=replace(
                rollback,
                settlement=replace(
                    rollback.settlement,
                    gripper=replace(
                        rollback.settlement.gripper,
                        target_width_meters=0.05,
                        end_width_meters=0.05,
                    ),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "gripper settlement"):
            ControlStepSummary.from_session(session)

    def _write_step(
        self,
        root: Path,
        session_id: str,
        *,
        previous_session_id: str | None,
        pose_x: float,
        post_x: float,
        observation_id: int | None = None,
        target_frame: str = "recordings/reference/wrist/frame_000007.png",
        warmup_frames: int = 4,
        captured_at: float = 100.0,
        previous_action_x: float = 0.0,
        target_pose: DroidPose | None = None,
        insertion_target_policy: InsertionControlTargetPolicy | None = None,
        execution_policy: ControlExecutionPolicy = ControlExecutionPolicy.DIRECT,
        insertion_rollout_position: InsertionRolloutPosition | None = None,
    ) -> None:
        session = root / "control_sessions" / session_id
        session.mkdir(parents=True)
        observation = ControlObservation(
            observation_id=(
                100 + int(session_id[-1])
                if observation_id is None
                else observation_id
            ),
            captured_at_unix_seconds=captured_at,
            context_frame=Path(f"control_sessions/{session_id}/context.png"),
            target=ControlTarget(
                Path(target_frame),
                target_pose
                or DroidPose((0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            ),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((pose_x, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((previous_action_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            warmup_frames=warmup_frames,
        )
        state = ControlSessionState(
            session_id,
            "reference",
            11400,
            "control-recording",
            (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
            False,
            0.0,
            previous_session_id,
            execution_policy=execution_policy,
            insertion_target_policy=insertion_target_policy,
            insertion_rollout_position=insertion_rollout_position,
        )
        raw_action = DroidAction(
            (post_x - pose_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        response = ProposedControl(
            observation.observation_id,
            captured_at + 0.2,
            (raw_action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        scale = DroidActionScale(1.0, 0.25, 0.25)
        gate = ControlGateDecision(
            observation.observation_id,
            observation.pose.applied(scale.apply(raw_action)),
            (),
        )
        tracking = ActionTrackingDecision(1.0, 0.0, 0.0, 0.0, 0.0, ())
        result = ControlResult(
            ControlResultStatus.APPLIED,
            session_id,
            gate,
            (
                SafetyProjectionAttempt(
                    scale, gate, 0.01, state.current_joint_positions
                ),
            ),
            scale,
            1.0,
            0.0,
            0.0,
            0.0,
            PostActionEvidence(
                raw_action,
                scale.apply(raw_action),
                scale.apply(raw_action),
                tracking,
                DroidPose((post_x, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                state.current_joint_positions,
                0.0,
                0.0,
                False,
                {"path": "/tmp/post.png", "shape": [512, 512, 4]},
                command_realization=evaluate_command_realization(
                    scale.apply(raw_action),
                    scale.apply(raw_action),
                ),
            ),
        )
        (session / "request.json").write_text(json.dumps(observation.to_dict()))
        (session / "response.json").write_text(json.dumps(response.to_dict()))
        (session / "state.json").write_text(json.dumps(state.to_dict()))
        (session / "result.json").write_text(json.dumps(result.to_dict()))

    def _report(
        self,
        root: Path,
        sessions: tuple[str, ...],
        *,
        requested_steps: int,
        orchestration_failure: OrchestrationFailure | None = None,
    ) -> dict:
        return ControlRolloutReport.from_sessions(
            root,
            "rollout-1",
            sessions,
            reference_recording="reference",
            seed=11400,
            proposal=Path("/tmp/proposal.pth"),
            requested_steps=requested_steps,
            orchestration_failure=orchestration_failure,
        ).to_dict()

    def _write_reference(self, root: Path) -> None:
        reference = root / "recordings" / "reference"
        reference.mkdir(parents=True)
        (reference / "manifest.json").write_text("{}")
        (reference / "steps.jsonl").write_text(
            json.dumps(
                {
                    "index": 7,
                    "end_effector_pose": [
                        0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                    ],
                }
            )
            + "\n"
        )

    def test_terminal_summary_retains_execution_interlock_peak(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_step(
                root,
                "session-1",
                previous_session_id=None,
                pose_x=0.4,
                post_x=0.401,
            )
            session = ControlSession.at(root / "control_sessions", "session-1")
            applied = session.load_result()
            rolled_back = replace(
                applied,
                status=ControlResultStatus.ROLLED_BACK_EXECUTION,
                post_action=None,
                execution_error="RuntimeError: transient force",
                execution_interlock=ControlInterlockEvidence(2.5, True),
            )
            session.result_path.write_text(json.dumps(rolled_back.to_dict()))

            summary = ControlStepSummary.from_session(session).to_dict()

            self.assertEqual(
                summary["execution_interlock"],
                {
                    "maximum_contact_force_newtons": 2.5,
                    "collision_detected": True,
                },
            )
            self.assertEqual(
                summary["execution_error"], "RuntimeError: transient force"
            )

    def test_rejects_tampered_command_realization_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-1",
                previous_session_id=None,
                pose_x=0.4,
                post_x=0.401,
            )
            session = ControlSession.at(root / "control_sessions", "session-1")
            result = session.load_result()
            realization = result.post_action.command_realization
            self.assertIsNotNone(realization)
            tampered = replace(
                result.post_action,
                command_realization=replace(
                    realization,
                    translation_fraction=realization.translation_fraction + 0.1,
                ),
            )
            session.result_path.write_text(
                json.dumps(replace(result, post_action=tampered).to_dict())
            )

            with self.assertRaisesRegex(ValueError, "realization is inconsistent"):
                ControlStepSummary.from_session(session)

    def test_legacy_wire_evidence_cannot_promote_a_grasp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.4,
                post_x=0.401,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.401,
                post_x=0.431,
                target_frame="recordings/reference/wrist/frame_000007.png",
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.001,
            )
            for index, plug_position in enumerate((0.0, 0.03)):
                session = root / "control_sessions" / f"session-{index}"
                state = json.loads((session / "state.json").read_text())
                state["plug_position"] = [0.0, 0.0, 0.0]
                state["plug_attached"] = index > 0
                (session / "state.json").write_text(json.dumps(state))
                result = json.loads((session / "result.json").read_text())
                result["post_action_plug_position"] = [plug_position, 0.0, 0.0]
                result["post_action_plug_attached"] = True
                (session / "result.json").write_text(json.dumps(result))

            current = self._report(
                root,
                ("session-0", "session-1"),
                requested_steps=2,
            )
            self.assertTrue(current["reach_and_grasp"]["passed"])

            for index in range(2):
                session = root / "control_sessions" / f"session-{index}"
                for filename in ("request.json", "response.json"):
                    path = session / filename
                    payload = json.loads(path.read_text())
                    payload["schema"] = LEGACY_CONTROL_SCHEMA
                    path.write_text(json.dumps(payload))

            legacy = self._report(
                root,
                ("session-0", "session-1"),
                requested_steps=2,
            )

            self.assertIsNone(legacy["reach_and_grasp"])

    def test_legacy_wire_evidence_is_not_currently_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.4,
                post_x=0.401,
            )
            current = ControlRolloutReport.from_sessions(
                root,
                "rollout-1",
                ("session-0",),
                reference_recording="reference",
                seed=11400,
                proposal=Path("/tmp/proposal.pth"),
                requested_steps=1,
            )
            self.assertTrue(current.current_wire_authenticated)

            response_path = (
                root / "control_sessions" / "session-0" / "response.json"
            )
            response = json.loads(response_path.read_text())
            response["schema"] = LEGACY_CONTROL_SCHEMA
            response_path.write_text(json.dumps(response))
            legacy = ControlRolloutReport.from_sessions(
                root,
                "rollout-1",
                ("session-0",),
                reference_recording="reference",
                seed=11400,
                proposal=Path("/tmp/proposal.pth"),
                requested_steps=1,
            )

            self.assertFalse(legacy.current_wire_authenticated)

    def test_summarizes_a_provenance_bound_rollout_and_goal_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "manifest.json").write_text("{}")
            (reference / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 7,
                        "end_effector_pose": [
                            0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        ],
                    }
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
                target_pose=DroidPose(
                    (0.45, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
                ),
            )

            report = self._report(
                root, ("session-0", "session-1"), requested_steps=2
            )

            self.assertTrue(report["all_steps_applied"])

            self.assertEqual(report["applied_steps"], 2)
            self.assertAlmostEqual(report["translation_progress_meters"], 0.02)

    def test_four_step_report_reconstructs_the_complete_lineage_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            for index in range(4):
                self._write_step(
                    root,
                    f"session-{index}",
                    previous_session_id=(
                        f"session-{index - 1}" if index else None
                    ),
                    pose_x=0.40 + 0.01 * index,
                    post_x=0.41 + 0.01 * index,
                    warmup_frames=4 + index,
                    captured_at=100.0 + index,
                    previous_action_x=0.01 if index else 0.0,
                )

            report = self._report(
                root,
                tuple(f"session-{index}" for index in range(4)),
                requested_steps=4,
            )

            self.assertTrue(report["all_steps_applied"])
            with self.assertRaisesRegex(ValueError, "session chain"):
                self._report(
                    root,
                    ("session-1", "session-2"),
                    requested_steps=2,
                )

    def test_report_can_authenticate_an_explicit_external_predecessor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "insertion-1",
                previous_session_id="grasp-8",
                pose_x=0.40,
                post_x=0.41,
            )
            self._write_step(
                root,
                "insertion-2",
                previous_session_id="insertion-1",
                pose_x=0.41,
                post_x=0.42,
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            report = ControlRolloutReport.from_sessions(
                root,
                "rollout-1",
                ("insertion-1", "insertion-2"),
                reference_recording="reference",
                seed=11400,
                proposal=Path("/tmp/proposal.pth"),
                requested_steps=2,
                predecessor_session_id="grasp-8",
            )

            self.assertTrue(report.all_steps_applied)
            self.assertEqual(report.predecessor_session_id, "grasp-8")
            self.assertEqual(report.to_dict()["predecessor_session_id"], "grasp-8")
            with self.assertRaisesRegex(ValueError, "session chain"):
                ControlRolloutReport.from_sessions(
                    root,
                    "rollout-1",
                    ("insertion-1", "insertion-2"),
                    reference_recording="reference",
                    seed=11400,
                    proposal=Path("/tmp/proposal.pth"),
                    requested_steps=2,
                    predecessor_session_id="wrong-grasp",
                )

    def test_contact_grasp_report_authenticates_predecessor_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "manifest.json").write_text(
                json.dumps({"metadata": {"task": "reach_and_insert"}})
            )
            proposal = Path("/tmp/proposal.pth")
            pose = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))

            def observation(session: str, frame: int, artifact: Path) -> ControlObservation:
                return ControlObservation(
                    frame,
                    100.0,
                    Path(f"control_sessions/{session}/context.png"),
                    ControlTarget(
                        Path(
                            "recordings/reference/wrist/"
                            f"frame_{frame:06d}.png"
                        )
                    ),
                    artifact,
                    pose,
                    DroidAction((0.0,) * 7),
                    frame - 16,
                )

            current_observation = observation("current", 113, proposal)
            predecessor_observation = observation("predecessor", 113, proposal)

            def summary(observed, state):
                value = object.__new__(ControlStepSummary)
                object.__setattr__(value, "observation", observed)
                object.__setattr__(value, "state", state)
                object.__setattr__(value, "response", None)
                object.__setattr__(value, "result", None)
                object.__setattr__(value, "shadow", None)
                object.__setattr__(value, "shadow_safety", None)
                return value

            current = summary(
                current_observation,
                SimpleNamespace(
                    reference_recording="reference",
                    seed=11400,
                    plug_attached=False,
                ),
            )

            def predecessor(
                *,
                target_frame: int = 113,
                seed: int = 11400,
                artifact: Path = proposal,
            ) -> ControlStepSummary:
                return summary(
                    observation("predecessor", target_frame, artifact),
                    SimpleNamespace(
                        reference_recording="reference",
                        seed=seed,
                        plug_attached=False,
                    ),
                )

            def reconstruct(source: ControlStepSummary):
                policy = Mock()

                def validate(*args, previous_step, **kwargs):
                    del args, kwargs
                    if previous_step.observation.target_frame != Path(
                        "recordings/reference/wrist/frame_000113.png"
                    ):
                        raise ValueError("contact-grasp target schedule is invalid")

                policy.validate_reference_schedule.side_effect = validate
                with (
                    patch(
                        "jepa_wm.control_rollout.ControlStepSummary.from_session",
                        side_effect=(current, source),
                    ),
                    patch(
                        "jepa_wm.control_rollout._contact_grasp_target_policy",
                        return_value=policy,
                    ),
                    patch(
                        "jepa_wm.control_rollout._contact_grasp_target_steps",
                        return_value=(object(),),
                    ),
                    patch(
                        "jepa_wm.control_rollout._target_pose",
                        return_value=pose,
                    ),
                    patch.object(ControlRolloutReport, "__post_init__"),
                ):
                    report = ControlRolloutReport.from_sessions(
                        root,
                        "rollout-1",
                        ("current",),
                        reference_recording="reference",
                        seed=11400,
                        proposal=proposal,
                        requested_steps=1,
                        predecessor_session_id="predecessor",
                    )
                return report, policy

            report, policy = reconstruct(predecessor())

            self.assertEqual(report.predecessor_session_id, "predecessor")
            previous_step = policy.validate_reference_schedule.call_args.kwargs[
                "previous_step"
            ]
            self.assertEqual(previous_step.observation, predecessor_observation)
            with self.assertRaisesRegex(ValueError, "target schedule"):
                reconstruct(predecessor(target_frame=129))
            with self.assertRaisesRegex(ValueError, "predecessor provenance"):
                reconstruct(predecessor(seed=11401))
            with self.assertRaisesRegex(ValueError, "predecessor provenance"):
                reconstruct(predecessor(artifact=Path("/tmp/other.pth")))

    def test_insertion_followup_accepts_policy_selected_target_and_bounded_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "manifest.json").write_text(
                json.dumps({"metadata": {"task": "reach_and_insert"}})
            )
            (reference / "steps.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "index": index,
                            "end_effector_pose": [
                                pose_x, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                            ],
                        }
                    )
                    for index, pose_x in ((7, 0.43), (9, 0.432))
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                target_frame="recordings/reference/wrist/frame_000007.png",
                insertion_target_policy=InsertionControlTargetPolicy(),
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.4101,
                post_x=0.42,
                target_frame="recordings/reference/wrist/frame_000009.png",
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.0101,
                insertion_target_policy=InsertionControlTargetPolicy(),
            )

            report = self._report(
                root, ("session-0", "session-1"), requested_steps=2
            )

            self.assertTrue(report["all_steps_applied"])

            for session_id in ("session-0", "session-1"):
                state_path = root / "control_sessions" / session_id / "state.json"
                payload = json.loads(state_path.read_text())
                del payload["insertion_target_policy"]
                state_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "target contract"):
                self._report(root, ("session-0", "session-1"), requested_steps=2)

    def test_integrated_insertion_report_uses_its_authenticated_168_action_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "manifest.json").write_text(
                json.dumps({"metadata": {"task": "reach_and_insert"}})
            )
            (reference / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 7,
                        "end_effector_pose": [
                            0.43,
                            0.0,
                            0.5,
                            0.0,
                            0.0,
                            0.0,
                            0.5,
                        ],
                    }
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                insertion_target_policy=InsertionControlTargetPolicy(),
                execution_policy=(
                    ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION
                ),
                insertion_rollout_position=InsertionRolloutPosition(1, 168),
            )

            session = root / "control_sessions" / "session-0"
            step = ControlStepSummary(
                ControlSessionState.from_dict(
                    json.loads((session / "state.json").read_text())
                ),
                ControlObservation.from_dict(
                    json.loads((session / "request.json").read_text())
                ),
                ProposedControl.from_dict(
                    json.loads((session / "response.json").read_text())
                ),
                ControlResult.from_dict(
                    json.loads((session / "result.json").read_text())
                ),
            )
            report = ControlRolloutReport(
                "rollout-1",
                "reference",
                11400,
                Path("/tmp/proposal.pth"),
                168,
                (step,),
                step.observation.target_pose,
                reference_task="reach_and_insert",
            ).to_dict()

            self.assertEqual(report["requested_steps"], 168)
            self.assertFalse(report["all_steps_applied"])
            wrong_position = replace(
                step,
                state=replace(
                    step.state,
                    insertion_rollout_position=InsertionRolloutPosition(2, 168),
                ),
            )
            with self.assertRaisesRegex(ValueError, "positions"):
                ControlRolloutReport(
                    "rollout-1",
                    "reference",
                    11400,
                    Path("/tmp/proposal.pth"),
                    168,
                    (wrong_position,),
                    wrong_position.observation.target_pose,
                    reference_task="reach_and_insert",
                )

    def test_terminal_gate_rejects_blocked_or_rolled_back_second_action(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
            )
            report = ControlRolloutReport.from_sessions(
                root,
                "rollout-1",
                ("session-0", "session-1"),
                reference_recording="reference",
                seed=11400,
                proposal=Path("/tmp/proposal.pth"),
                requested_steps=2,
            )

            report.require_all_steps_applied()
            for status in (
                ControlResultStatus.BLOCKED,
                ControlResultStatus.ROLLED_BACK_PROGRESS,
            ):
                failed_report = deepcopy(report)
                object.__setattr__(failed_report.steps[-1].result, "status", status)
                with self.assertRaisesRegex(ValueError, "every requested step"):
                    failed_report.require_all_steps_applied()

    def test_summarizes_shadow_search_and_counterfactual_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            session = root / "control_sessions" / "session-0"
            response = ProposedControl.from_dict(
                json.loads((session / "response.json").read_text())
            )
            observation = ControlObservation.from_dict(
                json.loads((session / "request.json").read_text())
            )
            target = np.asarray([action.values for action in response.actions]) * 0.8
            shadow = plan_shadow_candidates(
                observation_id=response.observation_id,
                direct_actions=response.actions,
                score=lambda candidates: np.square(
                    candidates - target[None, :, :]
                ).sum(axis=(1, 2)),
                proposal=response.proposal,
                adapter=Path("/tmp/adapter.pth"),
                config=ShadowSearchConfig(
                    planner=CEMConfig(iterations=5, samples=160, elites=12, seed=7),
                    prior=ActionPriorConfig(penalty_weight=1e-12),
                    first_action_thresholds=FirstActionThresholds(
                        minimum_active_cosine=-1.0
                    ),
                ),
            )
            (session / "shadow_request.json").write_text(
                json.dumps(
                    ShadowPlanningRequest(
                        observation,
                        response,
                        Path("/tmp/adapter.pth"),
                        shadow.config.planner,
                    ).to_dict()
                )
            )
            (session / "shadow.json").write_text(json.dumps(shadow.to_dict()))
            scale = DroidActionScale(1.0, 0.25, 0.25)
            gate = ControlGateDecision(
                response.observation_id,
                observation.pose.applied(scale.apply(shadow.planned.actions[0])),
                (),
            )
            safety = ShadowSafetyEvidence(
                response.observation_id,
                response.created_at_unix_seconds + 1.0,
                response.created_at_unix_seconds,
                shadow.planned.actions,
                (
                    SafetyProjectionAttempt(
                        scale,
                        gate,
                        0.0,
                        (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
                    ),
                ),
                scale,
            )
            (session / "shadow_safety.json").write_text(
                json.dumps(safety.to_dict())
            )

            report = self._report(root, ("session-0",), requested_steps=1)

            self.assertEqual(report["shadow_searches"], 1)
            self.assertEqual(
                report["shadow_gate_passes"], int(shadow.passes_shadow_gate)
            )
            self.assertEqual(report["shadow_safety_passes"], 1)
            self.assertGreater(report["mean_shadow_energy_improvement"], 0.0)

            safety_path = session / "shadow_safety.json"
            safety_payload = json.loads(safety_path.read_text())
            safety_payload["attempts"][0]["gate"]["next_pose"][0] += 0.01
            safety_path.write_text(json.dumps(safety_payload))
            with self.assertRaisesRegex(ValueError, "gate evidence"):
                self._report(root, ("session-0",), requested_steps=1)
            safety_path.write_text(json.dumps(safety.to_dict()))

            safety_payload = safety.to_dict()
            safety_payload["counterfactual_as_of_unix_seconds"] += 0.01
            safety_path.write_text(json.dumps(safety_payload))
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._report(root, ("session-0",), requested_steps=1)

            safety_path.write_text(json.dumps(safety.to_dict()))

            shadow_request_path = session / "shadow_request.json"
            shadow_request = json.loads(shadow_request_path.read_text())
            shadow_request["expected_adapter"] = "/tmp/other-adapter.pth"
            shadow_request_path.write_text(json.dumps(shadow_request))
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._report(root, ("session-0",), requested_steps=1)

            shadow_request["expected_adapter"] = "/tmp/adapter.pth"
            shadow_request["expected_planner"]["seed"] += 1
            shadow_request_path.write_text(json.dumps(shadow_request))
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._report(root, ("session-0",), requested_steps=1)

    def test_preserves_requested_count_when_a_rollout_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "manifest.json").write_text("{}")
            (reference / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 7,
                        "end_effector_pose": [
                            0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        ],
                    }
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )

            report = self._report(root, ("session-0",), requested_steps=3)

            self.assertEqual(report["requested_steps"], 3)
            self.assertEqual(report["attempted_steps"], 1)
            self.assertFalse(report["all_steps_applied"])

    def test_recomputes_realized_action_instead_of_trusting_result_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            result_path = root / "control_sessions" / "session-0" / "result.json"
            result = json.loads(result_path.read_text())
            result["actual_action"][0] = 0.02
            result_path.write_text(json.dumps(result))

            with self.assertRaisesRegex(ValueError, "realization"):
                ControlStepSummary.from_session(
                    ControlSession.at(root / "control_sessions", "session-0")
                )

    def test_rejects_shadow_objective_rebound_to_another_start_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_pose = DroidPose((0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6))
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                target_pose=target_pose,
            )
            session = ControlSession.at(root / "control_sessions", "session-0")
            observation, _ = session.load_capture()
            response = session.load_response()
            calibration = ActionResponseCalibration.fit(
                tuple(
                    ActionResponseTrial(
                        f"trial-{axis}",
                        axis,
                        DroidAction(
                            (
                                *(0.002 if index == axis else 0.0 for index in range(3)),
                                *(0.004 if index == axis else 0.0 for index in range(3)),
                                0.2,
                            )
                        ),
                        DroidAction(
                            (
                                *(0.001 if index == axis else 0.0 for index in range(3)),
                                *(0.001 if index == axis else 0.0 for index in range(3)),
                                0.05,
                            )
                        ),
                    )
                    for axis in range(3)
                )
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(calibration.to_dict()))
            request = ShadowPlanningRequest(
                observation,
                response,
                Path("/tmp/adapter.pth"),
                ShadowSearchConfig().planner,
                CalibrationIdentity.from_calibration(
                    calibration_path, calibration
                ),
            )
            session.shadow_request_path.write_text(json.dumps(request.to_dict()))
            tampered_start = DroidPose(
                (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
            )
            shadow = plan_shadow_candidates(
                observation_id=observation.observation_id,
                direct_actions=response.actions,
                score=lambda candidates: np.square(candidates).sum(axis=(1, 2)),
                proposal=response.proposal,
                adapter=Path("/tmp/adapter.pth"),
                config=ShadowSearchConfig(
                    planner=CEMConfig(iterations=1, samples=4, elites=2)
                ),
                task_progress=TaskProgressObjective(
                    tampered_start, target_pose, calibration
                ),
            )
            session.shadow_path.write_text(json.dumps(shadow.to_dict()))

            with self.assertRaisesRegex(ValueError, "not bound"):
                session.load_shadow()

    def test_rejects_target_or_observation_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                observation_id=123,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                observation_id=123,
                target_frame="recordings/reference/wrist/frame_000008.png",
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            with self.assertRaisesRegex(ValueError, "observation IDs"):
                self._report(
                    root, ("session-0", "session-1"), requested_steps=2
                )

    def test_rejects_a_changed_target_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                target_frame="recordings/reference/wrist/frame_000008.png",
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            with self.assertRaisesRegex(ValueError, "target frame"):
                self._report(
                    root, ("session-0", "session-1"), requested_steps=2
                )

    def test_advances_one_reference_target_per_grasp_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "manifest.json").write_text(
                json.dumps({"metadata": {"task": "reach_and_grasp"}})
            )
            (reference / "steps.jsonl").write_text(
                "\n".join(
                    json.dumps(
                        {
                            "index": index,
                            "end_effector_pose": [
                                position, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                            ],
                        }
                    )
                    for index, position in ((89, 0.43), (90, 0.44))
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                target_frame="recordings/reference/wrist/frame_000089.png",
                warmup_frames=86,
                target_pose=DroidPose(
                    (0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
                ),
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                target_frame="recordings/reference/wrist/frame_000090.png",
                warmup_frames=87,
                captured_at=101.0,
                previous_action_x=0.01,
                target_pose=DroidPose(
                    (0.45, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
                ),
            )

            report = self._report(
                root, ("session-0", "session-1"), requested_steps=2
            )

        self.assertEqual(report["reference_task"], "reach_and_grasp")
        self.assertAlmostEqual(report["final_goal_error"]["translation_meters"], 0.03)

    def test_rejects_a_non_monotonic_warmup_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                warmup_frames=4,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                warmup_frames=4,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            with self.assertRaisesRegex(ValueError, "warm-up"):
                self._report(
                    root, ("session-0", "session-1"), requested_steps=2
                )

    def test_persists_an_incomplete_terminal_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "control_sessions" / "session-0").mkdir(parents=True)

            report = self._report(
                root,
                ("session-0",),
                requested_steps=3,
                orchestration_failure=OrchestrationFailure(
                    OrchestrationOperation.INITIAL_CONTROL_STEP,
                    1,
                ),
            )

            self.assertEqual(report["attempted_steps"], 1)
            self.assertEqual(report["complete_steps"], 0)
            self.assertEqual(report["applied_steps"], 0)
            self.assertEqual(report["steps"][0]["status"], "orchestration_failed")
            self.assertEqual(
                report["orchestration_failure"]["operation"],
                "initial_control_step",
            )

    def test_parses_followup_failure_step_identity_explicitly(self) -> None:
        failure = OrchestrationFailure.parse("followup_capture_03:exit_124")

        self.assertEqual(failure.operation, OrchestrationOperation.FOLLOWUP_CAPTURE)
        self.assertEqual(failure.step_index, 3)
        self.assertEqual(failure.exit_code, 124)
        with self.assertRaisesRegex(ValueError, "malformed"):
            OrchestrationFailure.parse("followup_capture:exit_1")

    def test_rejects_a_stale_persisted_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "manifest.json").write_text("{}")
            (reference / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 7,
                        "end_effector_pose": [
                            0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        ],
                    }
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            result_path = root / "control_sessions" / "session-0" / "result.json"
            payload = json.loads(result_path.read_text())
            payload["observation_age_seconds"] = 3.1
            result_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "freshness"):
                self._report(root, ("session-0",), requested_steps=1)

    def test_reports_a_stale_step_that_the_gate_safely_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            result_path = root / "control_sessions" / "session-0" / "result.json"
            payload = json.loads(result_path.read_text())
            payload["status"] = "blocked"
            payload["gate"]["passed"] = False
            payload["gate"]["reasons"] = ["stale_observation"]
            for attempt in payload["safety_projection_attempts"]:
                attempt["gate"]["passed"] = False
                attempt["gate"]["reasons"] = ["stale_observation"]
            payload["selected_action_scale"] = None
            payload["observation_age_seconds"] = 3.0
            for field in (
                "raw_proposed_action",
                "commanded_action",
                "actual_action",
                "action_tracking",
                "post_action_pose",
                "post_action_joint_positions",
                "maximum_joint_tracking_error_rad",
                "post_action_contact_force_newtons",
                "post_action_collision_detected",
                "post_action_frame",
            ):
                payload.pop(field)
            result_path.write_text(json.dumps(payload))

            report = self._report(root, ("session-0",), requested_steps=2)

            self.assertEqual(report["complete_steps"], 1)
            self.assertEqual(report["applied_steps"], 0)
            self.assertEqual(report["steps"][0]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
