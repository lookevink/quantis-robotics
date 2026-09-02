from dataclasses import replace
from pathlib import Path
import unittest

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.control_safety import ControlGateDecision, SafetyProjectionAttempt
from jepa_wm.control_tracking import (
    ActionTrackingDecision,
    evaluate_action_tracking,
    evaluate_command_realization,
)
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
    INSERTION_CONTROL_TARGET_POLICY,
)
from jepa_wm.insertion_rollout import InsertionRolloutPosition
from jepa_wm.insertion_task import InsertionTarget, InsertionTaskStep
from jepa_wm.joint_drive import JointDriveTarget
from sim.control_session import (
    ControlExecutionPolicy,
    ControlResult,
    ControlResultStatus,
    ControlSessionState,
    GraspToInsertionLineage,
    PostActionEvidence,
)


class GraspToInsertionLineageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.joints = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)
        self.pose = DroidPose((0.34, -0.29, 0.48, 0.0, 0.0, 0.0, 0.775))
        self.observation = ControlObservation(
            101,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png"), self.pose),
            Path("/tmp/grasp.pth"),
            self.pose,
            DroidAction((0.0,) * 7),
            8,
        )
        self.state = ControlSessionState(
            "grasp-07",
            "contact-reference",
            12401,
            "control-grasp",
            self.joints,
            False,
            0.0,
            execution_policy=ControlExecutionPolicy.DIRECT,
            plug_position=(0.30, -0.29, 0.48),
            plug_attached=True,
            current_gripper_width_m=0.018,
            active_drive_target=JointDriveTarget(self.joints, 0.0179),
        )
        next_pose = DroidPose((*self.pose.values[:6], 0.775))
        gate = ControlGateDecision(101, next_pose, ())
        attempt = SafetyProjectionAttempt(
            DroidActionScale(1.0, 1.0, 1.0),
            gate,
            0.001,
            self.joints,
        )
        tracking = ActionTrackingDecision(1.0, 1.0, 0.0, 0.0, 0.0, ())
        post_action = PostActionEvidence(
            DroidAction((0.0,) * 7),
            DroidAction((0.0,) * 7),
            DroidAction((0.0,) * 7),
            tracking,
            self.pose,
            self.joints,
            0.0,
            0.0,
            False,
            {"path": "post.png", "shape": [512, 512, 4]},
            plug_position=(0.30, -0.29, 0.48),
            plug_attached=True,
        )
        self.result = ControlResult(
            ControlResultStatus.APPLIED,
            "grasp-07",
            gate,
            (attempt,),
            attempt.scale,
            0.5,
            0.0,
            0.0,
            0.0,
            post_action,
        )

    def transition(self) -> tuple[ControlObservation, ControlSessionState]:
        lineage = GraspToInsertionLineage(
            self.observation,
            self.state,
            self.result,
        )
        observation = replace(
            self.observation,
            observation_id=102,
            expected_proposal=Path("/tmp/insertion.pth"),
            warmup_frames=CONTACT_INSERTION_RECORDING.start_index(
                ContactInsertionSegment.GRASP_ATTACH
            ),
        )
        state = replace(
            self.state,
            session_id="insertion-safety1",
            previous_session_id="grasp-07",
            execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            insertion_target_policy=INSERTION_CONTROL_TARGET_POLICY.for_followup(),
            active_drive_target=lineage.active_drive_target,
            insertion_rollout_position=InsertionRolloutPosition(1, 4),
        )
        return observation, state

    def test_binds_safe_applied_grasp_to_initial_insertion_source(self) -> None:
        lineage = GraspToInsertionLineage(
            self.observation,
            self.state,
            self.result,
        )
        observation, state = self.transition()

        lineage.validate_source(observation, state)
        self.assertEqual(
            lineage.active_drive_target,
            self.result.applied_drive_target(held_gripper_width_m=0.0179),
        )
        self.assertEqual(lineage.active_drive_target.gripper_width_m, 0.0179)

    def test_rejects_drive_target_substitution_at_phase_boundary(self) -> None:
        lineage = GraspToInsertionLineage(
            self.observation,
            self.state,
            self.result,
        )
        observation, state = self.transition()
        tampered = replace(
            state,
            active_drive_target=replace(
                state.active_drive_target,
                gripper_width_m=0.019,
            ),
        )

        with self.assertRaisesRegex(ValueError, "lineage"):
            lineage.validate_source(observation, tampered)

    def test_rejects_unattached_grasp(self) -> None:
        post_action = replace(self.result.post_action, plug_attached=False)
        result = replace(self.result, post_action=post_action)

        with self.assertRaisesRegex(ValueError, "safe applied grasp"):
            GraspToInsertionLineage(self.observation, self.state, result)

    def test_rejects_gripper_frame_claim_not_bound_to_raw_evidence(self) -> None:
        task_step = InsertionTaskStep(
            (0.30, -0.29, 0.48),
            (0.26, -0.29, 0.48),
            True,
            0.0,
            True,
            False,
            0.0,
        )
        post_action = replace(
            self.result.post_action,
            tracking=evaluate_action_tracking(
                self.result.post_action.commanded_action,
                self.result.post_action.actual_action,
            ),
            insertion_task_step=task_step,
            insertion_target=InsertionTarget(
                (0.20, -0.29, 0.48),
                (-1.0, 0.0, 0.0),
            ),
            command_realization=evaluate_command_realization(
                self.result.post_action.commanded_action,
                self.result.post_action.actual_action,
            ),
            plug_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            socket_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            gripper_frame_world_position=(0.26, -0.29, 0.48),
        )
        post_action.validate_derived_evidence(
            post_action.commanded_action,
            post_action.actual_action,
            ControlExecutionPolicy.DIRECT,
        )
        tampered = replace(
            post_action,
            insertion_task_step=replace(
                task_step,
                gripper_frame_position=(0.28, -0.29, 0.48),
            ),
        )

        with self.assertRaisesRegex(ValueError, "insertion evidence"):
            tampered.validate_derived_evidence(
                tampered.commanded_action,
                tampered.actual_action,
                ControlExecutionPolicy.DIRECT,
            )


if __name__ == "__main__":
    unittest.main()
