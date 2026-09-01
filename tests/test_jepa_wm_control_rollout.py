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
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_rollout import (
    ControlRolloutReport,
    ControlStepSummary,
    OrchestrationFailure,
    OrchestrationOperation,
    _contact_grasp_retained_direction,
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

    def test_current_contact_grasp_reconstructs_an_overlapping_dynamic_scale(
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
            pose = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.75))
            observation = ControlObservation(
                101,
                100.0,
                Path("control_sessions/session-1/context.png"),
                ControlTarget(
                    Path("recordings/reference/wrist/frame_000116.png"),
                    DroidPose((0.4232, 0.0, 0.5, 0.0, 0.0, 0.0, 0.75)),
                ),
                Path("/tmp/proposal.pth"),
                pose,
                DroidAction((0.0,) * 7),
                113,
            )
            actions = (
                DroidAction((0.0002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
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
                plug_attached=True,
                current_gripper_width_m=0.02,
                active_drive_target=drive_target,
                contact_grasp_target_policy=CONTACT_GRASP_TARGET_POLICY,
            )
            live_state = ControlSafetySnapshot(
                joints,
                0.02,
                (0.0, 0.0, 1.0),
                0.0,
                False,
                True,
            )
            refresh = InsertionEvaluationRefresh(100.3, live_state, pose)
            raw = CONTACT_GRASP_TARGET_POLICY.action_for_execution(
                actions,
                plug_attached=True,
            )
            scale = DroidActionScale(1.0, 0.125, 0.0)
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
                    True,
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
            CONTACT_GRASP_TARGET_POLICY.uses_exact_coarse_translation_projection,
        )
        self.assertEqual(
            scale_policy.call_args.kwargs["coarse_orientation_hold_fallback"],
            CONTACT_GRASP_TARGET_POLICY.uses_coarse_orientation_hold_fallback,
        )
        self.assertEqual(
            scale_policy.call_args.kwargs[
                "minimum_coarse_translation_command_meters"
            ],
            CONTACT_GRASP_TARGET_POLICY.minimum_coarse_translation_command_meters,
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
        )
        rolled_back_result = replace(
            result,
            status=ControlResultStatus.ROLLED_BACK_PROGRESS,
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
            target=ControlTarget(Path(target_frame), target_pose),
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
            insertion_target_policy=insertion_target_policy,
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
            )

            report = self._report(
                root, ("session-0", "session-1"), requested_steps=2
            )

        self.assertEqual(report["reference_task"], "reach_and_grasp")
        self.assertAlmostEqual(report["final_goal_error"]["translation_meters"], 0.02)

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
