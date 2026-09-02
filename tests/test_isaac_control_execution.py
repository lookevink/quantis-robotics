from __future__ import annotations

import asyncio
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_tracking import evaluate_command_completion
from jepa_wm.control_safety import (
    ControlInterlockEvidence,
    ControlGateReason,
    INSERTION_TARGET_PROGRESS,
    SimulatorSafetyLimits,
)
from jepa_wm.joint_settlement import (
    GripperSettlementCriterion,
    JointSettlementAttempt,
    TrackedJointSettlementPolicy,
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.insertion_trial import (
    InsertionTrialRollbackFailure,
    InsertionTrialRollbackFailureReason,
)
from sim.isaac_control_execution import (
    CONTACT_GRASP_MAXIMUM_JOINT_TRACKING_ERROR_RADIANS,
    CONTACT_GRASP_ROLLBACK_SETTLEMENT,
    CONTACT_GRASP_SETTLEMENT_MAXIMUM_UPDATES,
    EXPERIMENTAL_CANDIDATE_SETTLEMENT_MAXIMUM_ARM_ERROR_RADIANS,
    EXPERIMENTAL_CANDIDATE_SETTLEMENT_MAXIMUM_GRIPPER_ERROR_METERS,
    UNKNOWN_START_ROLLBACK_SETTLEMENT,
    ExecutionSafetyContext,
    InsertionTrialRollbackFailed,
    UnsettledJointCommand,
    apply_control_response,
    capture_synchronized_post_action,
    select_safe_projection,
    rollback_control_command,
    rollback_insertion_trial_command,
    requires_synchronized_evaluation_refresh,
    is_programming_error,
    settle_contact_grasp_command,
    settle_joint_command,
    settle_tracked_joint_command,
    synchronized_actual_command,
)
from sim.isaac_demo_kinematics import SolvedPose
from sim.isaac_demo_runtime import JointCommand


class ControlExecutionLifecycleTest(unittest.TestCase):
    def test_contact_grasp_settlement_matches_task_and_continuity_gates(
        self,
    ) -> None:
        self.assertEqual(
            CONTACT_GRASP_MAXIMUM_JOINT_TRACKING_ERROR_RADIANS,
            0.01,
        )
        self.assertEqual(
            CONTACT_GRASP_ROLLBACK_SETTLEMENT.maximum_arm_error_radians,
            SimulatorSafetyLimits().maximum_observation_joint_drift_radians,
        )
        self.assertEqual(
            CONTACT_GRASP_ROLLBACK_SETTLEMENT.maximum_gripper_error_meters,
            2.5e-4,
        )
        self.assertEqual(CONTACT_GRASP_SETTLEMENT_MAXIMUM_UPDATES, 192)

    def test_reset_trial_candidate_reuses_synchronized_freshness_authority(
        self,
    ) -> None:
        self.assertTrue(
            requires_synchronized_evaluation_refresh(
                ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
                contact_grasp_execution=False,
            )
        )
        self.assertLessEqual(
            EXPERIMENTAL_CANDIDATE_SETTLEMENT_MAXIMUM_ARM_ERROR_RADIANS,
            1e-3,
        )
        self.assertLessEqual(
            EXPERIMENTAL_CANDIDATE_SETTLEMENT_MAXIMUM_GRIPPER_ERROR_METERS,
            5e-4,
        )
        self.assertEqual(
            UNKNOWN_START_ROLLBACK_SETTLEMENT.maximum_arm_error_radians,
            1e-3,
        )

    def test_rejects_invalid_insertion_binding_before_live_synchronization(
        self,
    ) -> None:
        session = Mock()
        persisted_state = SimpleNamespace(execution_policy=object())
        session.load.return_value = (object(), object(), persisted_state)
        session.load_insertion_trial_binding.side_effect = RuntimeError(
            "invalid insertion binding"
        )
        runtime = SimpleNamespace(
            actuators=object(),
            attachment=object(),
            sensor=object(),
        )
        stage = object()

        omni = ModuleType("omni")
        kit = ModuleType("omni.kit")
        app = ModuleType("omni.kit.app")
        app.get_app = lambda: SimpleNamespace(next_update_async=AsyncMock())
        kit.app = app
        timeline = ModuleType("omni.timeline")
        timeline.get_timeline_interface = lambda: object()
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
        omni.kit = kit
        omni.timeline = timeline
        omni.usd = usd

        prims = ModuleType("isaacsim.core.experimental.prims")
        prims.Articulation = Mock()
        simulation_manager = ModuleType("isaacsim.core.simulation_manager")
        simulation_manager.SimulationManager = SimpleNamespace(
            get_physics_sim_view=lambda: object(),
            initialize_physics=Mock(),
        )

        synchronize = AsyncMock(
            side_effect=RuntimeError("live synchronization should not run")
        )
        with (
            patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.kit": kit,
                    "omni.kit.app": app,
                    "omni.timeline": timeline,
                    "omni.usd": usd,
                    "isaacsim.core.experimental.prims": prims,
                    "isaacsim.core.simulation_manager": simulation_manager,
                },
            ),
            patch(
                "sim.isaac_control_execution.ControlSession.at",
                return_value=session,
            ),
            patch(
                "sim.isaac_control_execution.is_insertion_trial_execution_policy",
                return_value=True,
            ),
            patch(
                "sim.isaac_control_execution.live_runtime_for",
                return_value=runtime,
            ),
            patch(
                "sim.isaac_control_execution.synchronized_insertion_execution_runtime",
                synchronize,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid insertion binding"):
                asyncio.run(apply_control_response("insertion-session"))

        synchronize.assert_not_awaited()


class ControlProjectionTest(unittest.TestCase):
    def test_longer_probe_period_preserves_velocity_limit_without_rejection(
        self,
    ) -> None:
        joints = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0))
        observation = ControlObservation(
            123,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png")),
            Path("/tmp/proposal.pth"),
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            DroidAction((0.0,) * 7),
            43,
        )
        proposal = ProposedControl(
            123,
            100.1,
            (DroidAction((0.0,) * 7),) * 3,
            Path("/tmp/proposal.pth"),
        )
        proposed = joints.copy()
        proposed[0] += 0.2
        fixed_period = ExecutionSafetyContext(
            observation,
            JointCommand(joints, 0.04),
            tuple(joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )
        longer_period = replace(fixed_period, control_period_seconds=0.5)

        fixed = fixed_period.evaluate(proposal, tuple(proposed), now_unix_seconds=100.2)
        longer = longer_period.evaluate(
            proposal, tuple(proposed), now_unix_seconds=100.2
        )

        self.assertIn(ControlGateReason.JOINT_VELOCITY_VIOLATION, fixed.reasons)
        self.assertNotIn(ControlGateReason.JOINT_VELOCITY_VIOLATION, longer.reasons)

    def test_target_progress_rejects_overshoot_until_quarter_scale(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=100.0,
            context_frame=Path("context.png"),
            target=ControlTarget(
                Path("target.png"),
                DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            ),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        proposal = ProposedControl(
            123,
            100.1,
            (
                DroidAction((0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                DroidAction((0.0,) * 7),
                DroidAction((0.0,) * 7),
            ),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )

        def solve(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            self.assertEqual(orientation_tolerance_radians, 0.001)
            return SolvedPose(
                pose, warm_start, np.zeros(3), 0.04, 0.0, 0.0, pose
            )

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            target_progress=INSERTION_TARGET_PROGRESS,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(attempts[-1].scale, DroidActionScale.uniform(0.25))
        self.assertTrue(attempts[-1].gate.passed)
        self.assertTrue(
            all(
                attempt.gate.reasons
                == (ControlGateReason.TARGET_PROGRESS_INSUFFICIENT,)
                for attempt in attempts[:-1]
            )
        )

        missing_target = replace(
            observation,
            target=ControlTarget(Path("target.png")),
        )
        missing_attempts, missing_selection = select_safe_projection(
            replace(context, observation=missing_target),
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            target_progress=INSERTION_TARGET_PROGRESS,
        )
        self.assertIsNone(missing_selection)
        self.assertTrue(
            all(
                attempt.gate.reasons == (ControlGateReason.TARGET_POSE_MISSING,)
                for attempt in missing_attempts
            )
        )

    def test_falls_back_after_quarter_scale_ik_failure(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
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
        proposal = ProposedControl(
            123,
            100.1,
            (
                DroidAction((0.004, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1)),
                DroidAction((0.0,) * 7),
                DroidAction((0.0,) * 7),
            ),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )
        calls = 0

        def solve(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            self.assertEqual(orientation_tolerance_radians, 0.001)
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("quarter-scale IK failed")
            joints = warm_start.copy()
            joints[0] += 0.001
            return SolvedPose(
                pose, joints, np.zeros(3), 0.04, 0.0, 0.0, pose
            )

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
        )

        self.assertEqual(len(attempts), 2)
        self.assertIn(ControlGateReason.IK_SOLUTION_FAILED, attempts[0].gate.reasons)
        self.assertIsNotNone(selected)
        self.assertEqual(
            attempts[1].scale,
            DroidActionScale(0.5, 0.125, 1.0),
        )

        bounded_attempts, bounded = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            action_scales=(DroidActionScale(0.5, 0.125, 1.0),),
        )
        self.assertIsNotNone(bounded)
        self.assertEqual(
            tuple(attempt.scale for attempt in bounded_attempts),
            (DroidActionScale(0.5, 0.125, 1.0),),
        )

    def test_fails_closed_when_ik_solution_cannot_complete_active_axis(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        start = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=100.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png")),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=start,
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        action = DroidAction((0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0))
        proposal = ProposedControl(
            123,
            100.1,
            (action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )

        def solve(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            self.assertIn(
                orientation_tolerance_radians,
                (0.00025, 0.0005, 0.00075, 0.001),
            )
            underrealized = start.applied(
                DroidAction((0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0))
            )
            return SolvedPose(
                pose,
                warm_start,
                np.zeros(3),
                0.04,
                0.0,
                0.0,
                underrealized,
            )

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            action_scales=(DroidActionScale.uniform(1.0),),
        )

        self.assertIsNone(selected)
        self.assertEqual(
            attempts[0].gate.reasons,
            (ControlGateReason.IK_SOLUTION_FAILED,),
        )

    def test_preserves_safety_failure_before_fk_completion_failure(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        start = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        observation = ControlObservation(
            123,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png")),
            Path("/tmp/proposal.pth"),
            start,
            DroidAction((0.0,) * 7),
            4,
        )
        action = DroidAction((0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0))
        proposal = ProposedControl(
            123,
            100.1,
            (action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )

        def solve(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            self.assertIn(
                orientation_tolerance_radians,
                (0.00025, 0.0005, 0.00075, 0.001),
            )
            unsafe_joints = warm_start.copy()
            unsafe_joints[0] = 3.0
            underrealized = start.applied(
                DroidAction((0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0))
            )
            return SolvedPose(
                pose,
                unsafe_joints,
                np.zeros(3),
                0.04,
                0.0,
                0.0,
                underrealized,
            )

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            action_scales=(DroidActionScale.uniform(1.0),),
        )

        self.assertIsNone(selected)
        self.assertIn(
            ControlGateReason.JOINT_LIMIT_VIOLATION,
            attempts[0].gate.reasons,
        )
        self.assertNotIn(
            ControlGateReason.IK_SOLUTION_FAILED,
            attempts[0].gate.reasons,
        )

    def test_active_rotation_uses_looser_ik_only_when_exact_fk_passes(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        start = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        observation = ControlObservation(
            123,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png")),
            Path("/tmp/proposal.pth"),
            start,
            DroidAction((0.0,) * 7),
            4,
        )
        action = DroidAction((0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0))
        proposal = ProposedControl(
            123,
            100.1,
            (action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )
        tolerances = []

        def solve(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            tolerances.append(orientation_tolerance_radians)
            if orientation_tolerance_radians == 0.00025:
                raise RuntimeError("strict branch unavailable")
            if orientation_tolerance_radians == 0.0005:
                return SolvedPose(
                    pose,
                    warm_start,
                    np.zeros(3),
                    0.04,
                    0.0,
                    0.0,
                    start.applied(
                        DroidAction((0.0, 0.0, 0.0, 0.0026, 0.0, 0.0, 0.0))
                    ),
                )
            passing_joints = warm_start.copy()
            passing_joints[0] += 0.003987
            return SolvedPose(
                pose,
                passing_joints,
                np.zeros(3),
                0.04,
                0.0,
                0.0,
                start.applied(
                    DroidAction((0.0, 0.0, 0.0, 0.0034, 0.0, 0.0, 0.0))
                ),
            )

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            action_scales=(DroidActionScale.uniform(1.0),),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(tolerances, [0.00025, 0.0005, 0.00075, 0.001])
        self.assertEqual(
            attempts[0].ik_orientation_tolerance_radians,
            0.00075,
        )
        self.assertAlmostEqual(attempts[0].maximum_joint_delta_rad, 0.003987)

    def test_active_rotation_prefers_the_passing_branch_closest_to_complete(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        start = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        observation = ControlObservation(
            123,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png")),
            Path("/tmp/proposal.pth"),
            start,
            DroidAction((0.0,) * 7),
            4,
        )
        action = DroidAction((0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0))
        proposal = ProposedControl(
            123,
            100.1,
            (action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )

        def solve(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            realized_rotation = {
                0.00025: 0.0026,
                0.0005: 0.00318,
                0.00075: 0.0034,
                0.001: 0.0026,
            }[orientation_tolerance_radians]
            joints = warm_start.copy()
            joints[0] += orientation_tolerance_radians * 5.0
            return SolvedPose(
                pose,
                joints,
                np.zeros(3),
                0.04,
                0.0,
                orientation_tolerance_radians,
                start.applied(
                    DroidAction(
                        (0.0, 0.0, 0.0, realized_rotation, 0.0, 0.0, 0.0)
                    )
                ),
            )

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            action_scales=(DroidActionScale.uniform(1.0),),
        )

        self.assertIsNotNone(selected)
        self.assertEqual(
            attempts[0].ik_orientation_tolerance_radians,
            0.00075,
        )
        self.assertAlmostEqual(
            selected.solved_pose.orientation_error_rad,
            0.00075,
        )

    def test_active_rotation_reserve_falls_back_to_orientation_hold(self) -> None:
        current_joints = np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5))
        start = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        observation = ControlObservation(
            123,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png")),
            Path("/tmp/proposal.pth"),
            start,
            DroidAction((0.0,) * 7),
            4,
        )
        action = DroidAction((0.0005, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0))
        proposal = ProposedControl(
            123,
            100.1,
            (action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        context = ExecutionSafetyContext(
            observation,
            JointCommand(current_joints, 0.04),
            tuple(current_joints),
            0.0,
            False,
            SimulatorSafetyLimits(),
        )

        def solve(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            rotation = pose.values[3]
            achieved = start.applied(
                DroidAction(
                    (
                        pose.values[0] - start.values[0],
                        0.0,
                        0.0,
                        rotation * 0.79,
                        0.0,
                        0.0,
                        0.0,
                    )
                )
            )
            return SolvedPose(
                pose,
                warm_start,
                np.zeros(3),
                0.04,
                0.0,
                orientation_tolerance_radians,
                achieved,
            )

        attempts, selected = select_safe_projection(
            context,
            proposal,
            solve=solve,
            now_unix_seconds=100.2,
            action_scales=(
                DroidActionScale.uniform(1.0),
                DroidActionScale(1.0, 0.0, 1.0),
            ),
            minimum_active_rotation_progress_fraction=0.8,
        )

        self.assertIsNotNone(selected)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            attempts[0].gate.reasons,
            (ControlGateReason.IK_SOLUTION_FAILED,),
        )
        self.assertEqual(attempts[1].scale.rotation, 0.0)
        self.assertEqual(selected.proposal.first_action.values[3:6], (0.0, 0.0, 0.0))

        def translation_lower_than_rotation(
            pose: DroidPose,
            warm_start: np.ndarray,
            orientation_tolerance_radians: float,
        ) -> SolvedPose:
            return SolvedPose(
                pose,
                warm_start,
                np.zeros(3),
                0.04,
                0.0,
                orientation_tolerance_radians,
                start.applied(
                    DroidAction((0.00038, 0.0, 0.0, 0.0036, 0.0, 0.0, 0.0))
                ),
            )

        _, rotation_selected = select_safe_projection(
            context,
            proposal,
            solve=translation_lower_than_rotation,
            now_unix_seconds=100.2,
            action_scales=(DroidActionScale.uniform(1.0),),
            minimum_active_rotation_progress_fraction=0.8,
        )

        self.assertIsNotNone(rotation_selected)


class FakeTimeline:
    def __init__(self, articulation: FakeArticulation | None = None) -> None:
        self.events: list[str] = []
        self.articulation = articulation
        self.auto_update = False

    def is_playing(self) -> bool:
        return False

    def set_auto_update(self, value: bool) -> None:
        self.auto_update = value

    def play(self) -> None:
        self.events.append("play")

    def pause(self) -> None:
        self.events.append("pause")
        if self.articulation is not None:
            self.articulation.valid = False


class FakeArticulation:
    def __init__(self, valid: bool) -> None:
        self.valid = valid

    def is_physics_tensor_entity_valid(self) -> bool:
        return self.valid


class FakeActuators:
    def __init__(self, valid: bool) -> None:
        self.articulation = FakeArticulation(valid)
        self.command = object()

    def actual_command(self) -> object:
        if not self.articulation.valid:
            raise AssertionError("physics tensor is invalid")
        return self.command

    def apply_drive_command(self, command: object) -> None:
        self.command = command


class IsaacControlExecutionTest(unittest.TestCase):
    def test_reads_post_action_state_after_camera_capture_advances_physics(
        self,
    ) -> None:
        before = JointCommand(np.zeros(7), 0.04)
        after = JointCommand(np.ones(7), 0.02)
        actuators = FakeActuators(valid=True)
        actuators.command = before
        snapshot = object()

        observed = 0

        def observe_safety() -> object:
            nonlocal observed
            observed += 1
            return object()

        async def capture(*_args: object, observe_safety=None) -> dict[str, object]:
            self.assertIsNotNone(observe_safety)
            observe_safety()
            actuators.command = after
            return {"path": "/tmp/post.png", "shape": [512, 512, 4]}

        with (
            patch("sim.isaac_control_execution.capture_camera_frame", capture),
            patch(
                "sim.isaac_control_execution.read_control_contact",
                return_value=(False, 0.5),
            ),
            patch(
                "sim.isaac_control_execution.recording_snapshot",
                return_value=snapshot,
            ) as recording,
        ):
            captured = asyncio.run(
                capture_synchronized_post_action(
                    actuators,
                    object(),
                    object(),
                    Path("/tmp/post.png"),
                    observe_safety=observe_safety,
                )
            )

        self.assertIs(captured.command, after)
        self.assertIs(captured.snapshot, snapshot)
        self.assertEqual(captured.contact_force_newtons, 0.5)
        self.assertEqual(observed, 1)
        recording.assert_called_once()
        self.assertIs(recording.call_args.args[2], after)

    def test_refreshes_a_stale_paused_physics_tensor_before_reading(self) -> None:
        actuators = FakeActuators(valid=False)
        timeline = FakeTimeline(actuators.articulation)

        async def advance() -> None:
            actuators.articulation.valid = True

        command = asyncio.run(synchronized_actual_command(actuators, timeline, advance))

        self.assertIs(command, actuators.command)
        self.assertEqual(timeline.events, ["play", "pause"])

    def test_pauses_when_command_tensor_resume_fails(self) -> None:
        actuators = FakeActuators(valid=False)
        timeline = FakeTimeline()
        timeline.play = Mock(side_effect=RuntimeError("resume failed"))

        async def advance() -> None:
            raise AssertionError("failed resume must not advance")

        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            asyncio.run(synchronized_actual_command(actuators, timeline, advance))

        self.assertEqual(timeline.events, ["pause"])

    def test_does_not_advance_an_already_valid_tensor(self) -> None:
        actuators = FakeActuators(valid=True)
        timeline = FakeTimeline()
        advanced = False

        async def advance() -> None:
            nonlocal advanced
            advanced = True

        command = asyncio.run(synchronized_actual_command(actuators, timeline, advance))

        self.assertIs(command, actuators.command)
        self.assertFalse(advanced)
        self.assertEqual(timeline.events, [])

    def test_settling_polls_safety_after_every_physics_update(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)
        updates = 0
        observations = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1

        def observe_safety() -> object:
            nonlocal observations
            observations += 1
            if observations == 2:
                raise RuntimeError("transient force")
            return object()

        with self.assertRaisesRegex(RuntimeError, "transient force"):
            asyncio.run(
                settle_joint_command(
                    actuators,
                    np.ones(7),
                    advance,
                    observe_safety=observe_safety,
                )
            )

        self.assertEqual(updates, 2)
        self.assertEqual(observations, 2)

    def test_contact_grasp_settlement_can_observe_beyond_generic_eight_updates(
        self,
    ) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.ones(7), 0.044)
        updates = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1
            if updates == 12:
                actuators.command = JointCommand(np.zeros(7), 0.04)

        asyncio.run(
            settle_joint_command(
                actuators,
                np.zeros(7),
                advance,
                maximum_updates=96,
                maximum_arm_error_radians=5e-3,
                gripper=GripperSettlementCriterion(0.04, 1e-6),
            )
        )

        self.assertEqual(updates, 12)

    def test_contact_grasp_settlement_fails_closed_at_its_gripper_bound(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.044)
        updates = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1

        with self.assertRaisesRegex(RuntimeError, "error_meters=0.004000000"):
            asyncio.run(
                settle_joint_command(
                    actuators,
                    np.zeros(7),
                    advance,
                    maximum_updates=3,
                    gripper=GripperSettlementCriterion(0.04, 1e-6),
                )
            )

        self.assertEqual(updates, 3)

    def test_contact_grasp_settlement_reports_arm_failure_when_gripper_passed(
        self,
    ) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.full(7, 0.001117), 0.040008149)

        async def advance() -> None:
            return None

        with self.assertRaisesRegex(
            RuntimeError,
            "arm did not settle.*error_radians=0.001117000.*maximum_radians=0.001000000",
        ):
            asyncio.run(
                settle_joint_command(
                    actuators,
                    np.zeros(7),
                    advance,
                    maximum_updates=3,
                    maximum_arm_error_radians=1e-3,
                    gripper=GripperSettlementCriterion(0.04, 2.5e-4),
                )
            )

    def test_contact_grasp_settlement_accepts_the_final_bounded_update(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.ones(7), 0.044)
        updates = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1
            if updates == 3:
                actuators.command = JointCommand(np.zeros(7), 0.04)

        asyncio.run(
            settle_joint_command(
                actuators,
                np.zeros(7),
                advance,
                maximum_updates=3,
                maximum_arm_error_radians=1e-3,
                gripper=GripperSettlementCriterion(0.04, 2.5e-4),
            )
        )
        self.assertEqual(updates, 3)

    def test_contact_grasp_settlement_accepts_task_tracking_above_one_milliradian(
        self,
    ) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.full(7, 0.001117), 0.04)
        target = JointCommand(np.zeros(7), 0.04)
        passed = SimpleNamespace(
            passed=True,
            reasons=(),
            translation_error_meters=4.9e-4,
            rotation_error_radians=2e-3,
        )
        updates = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1

        with (
            patch(
                "sim.isaac_control_execution.recording_snapshot",
                return_value=SimpleNamespace(end_effector_pose=object()),
            ),
            patch(
                "sim.isaac_control_execution.action_between",
                return_value=DroidAction((0.0,) * 7),
            ),
            patch(
                "sim.isaac_control_execution.evaluate_action_tracking",
                return_value=passed,
            ) as evaluate_tracking,
        ):
            asyncio.run(
                settle_contact_grasp_command(
                    actuators,
                    object(),
                    target,
                    DroidPose((0.0,) * 7),
                    DroidAction((0.0,) * 7),
                    advance,
                    maximum_updates=3,
                )
            )

        self.assertEqual(updates, 1)
        tracking_limits = evaluate_tracking.call_args.args[2]
        self.assertEqual(tracking_limits.maximum_translation_error_meters, 5e-4)
        self.assertEqual(tracking_limits.minimum_direction_cosine, 0.5)
        self.assertEqual(tracking_limits.maximum_rotation_error_radians, 3e-3)

    def test_contact_grasp_settlement_waits_for_task_tracking(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)
        failed = SimpleNamespace(
            passed=False,
            reasons=(SimpleNamespace(value="translation_error"),),
            translation_error_meters=5.1e-4,
            rotation_error_radians=2e-3,
        )
        passed = SimpleNamespace(
            passed=True,
            reasons=(),
            translation_error_meters=4.9e-4,
            rotation_error_radians=2e-3,
        )
        updates = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1

        with (
            patch(
                "sim.isaac_control_execution.recording_snapshot",
                return_value=SimpleNamespace(end_effector_pose=object()),
            ),
            patch(
                "sim.isaac_control_execution.action_between",
                return_value=DroidAction((0.0,) * 7),
            ),
            patch(
                "sim.isaac_control_execution.evaluate_action_tracking",
                side_effect=(failed, passed, passed),
            ),
        ):
            asyncio.run(
                settle_contact_grasp_command(
                    actuators,
                    object(),
                    JointCommand(np.zeros(7), 0.04),
                    DroidPose((0.0,) * 7),
                    DroidAction((0.0,) * 7),
                    advance,
                    maximum_updates=3,
                )
            )

        self.assertEqual(updates, 2)

    def test_contact_grasp_settlement_rejects_safe_but_underrealized_motion(
        self,
    ) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)

        async def advance() -> None:
            return None

        commanded = DroidAction((0.0006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        realized = DroidAction((0.00012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        with (
            patch(
                "sim.isaac_control_execution.recording_snapshot",
                return_value=SimpleNamespace(end_effector_pose=object()),
            ),
            patch(
                "sim.isaac_control_execution.action_between",
                return_value=realized,
            ),
            self.assertRaisesRegex(RuntimeError, "translation_underrealized"),
        ):
            asyncio.run(
                settle_contact_grasp_command(
                    actuators,
                    object(),
                    JointCommand(np.zeros(7), 0.04),
                    DroidPose((0.0,) * 7),
                    commanded,
                    advance,
                    maximum_updates=2,
                )
            )

    def test_programming_errors_are_not_normal_rollout_failures(self) -> None:
        self.assertTrue(is_programming_error(TypeError("bad call")))
        self.assertTrue(is_programming_error(AssertionError("broken invariant")))
        self.assertTrue(is_programming_error(ZeroDivisionError("bad arithmetic")))
        self.assertTrue(is_programming_error(ImportError("missing dependency")))
        self.assertFalse(is_programming_error(RuntimeError("physics timeout")))

    def test_contact_grasp_settlement_aborts_a_stable_realization_plateau(
        self,
    ) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)

        async def advance() -> None:
            return None

        with (
            patch(
                "sim.isaac_control_execution.recording_snapshot",
                return_value=SimpleNamespace(end_effector_pose=object()),
            ),
            patch(
                "sim.isaac_control_execution.action_between",
                return_value=DroidAction(
                    (0.00012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                ),
            ),
            self.assertRaisesRegex(
                RuntimeError, "stopped making realizable progress"
            ) as raised,
        ):
            asyncio.run(
                settle_contact_grasp_command(
                    actuators,
                    object(),
                    JointCommand(np.zeros(7), 0.04),
                    DroidPose((0.0,) * 7),
                    DroidAction((0.0006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                    advance,
                    maximum_updates=40,
                )
            )

        self.assertIn("commanded_translation=(0.0006, 0.0, 0.0)", str(raised.exception))
        self.assertIn("realized_translation=(0.00012, 0.0, 0.0)", str(raised.exception))
        self.assertIn("joint_error_radians=0.000000000", str(raised.exception))

    def test_contact_grasp_settlement_polls_safety_and_fails_at_bound(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)
        failed = SimpleNamespace(
            passed=False,
            reasons=(SimpleNamespace(value="translation_error"),),
            translation_error_meters=5.1e-4,
            rotation_error_radians=2e-3,
        )
        updates = 0
        observations = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1

        def observe_safety() -> object:
            nonlocal observations
            observations += 1
            return object()

        with (
            patch(
                "sim.isaac_control_execution.recording_snapshot",
                return_value=SimpleNamespace(end_effector_pose=object()),
            ),
            patch(
                "sim.isaac_control_execution.action_between",
                return_value=DroidAction((0.0,) * 7),
            ),
            patch(
                "sim.isaac_control_execution.evaluate_action_tracking",
                return_value=failed,
            ),
            self.assertRaisesRegex(
                RuntimeError,
                "tracking_reasons=\\['translation_error'\\]",
            ),
        ):
            asyncio.run(
                settle_contact_grasp_command(
                    actuators,
                    object(),
                    JointCommand(np.zeros(7), 0.04),
                    DroidPose((0.0,) * 7),
                    DroidAction((0.0,) * 7),
                    advance,
                    observe_safety=observe_safety,
                    maximum_updates=2,
                )
            )

        self.assertEqual(updates, 2)
        self.assertEqual(observations, 2)

    def test_insertion_settlement_requires_consecutive_command_relative_updates(
        self,
    ) -> None:
        actuators = FakeActuators(valid=True)
        readings = iter((4e-4, 8e-4, 4e-4, 3e-4))
        target = np.zeros(7)
        updates = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1
            error = next(readings)
            actuators.command = JointCommand(
                np.asarray((error, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                0.04,
            )

        evidence = asyncio.run(
            settle_tracked_joint_command(
                actuators,
                np.asarray((0.002,) + (0.0,) * 6),
                target,
                advance,
                TrackedJointSettlementPolicy(),
            )
        )

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence.updates_used, 4)
        self.assertEqual(evidence.required_tracking_error_radians, 5e-4)
        self.assertEqual(evidence.passing_tracking_errors_radians, (4e-4, 3e-4))

    def test_insertion_settlement_timeout_preserves_every_tracking_update(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.ones(7), 0.04)

        async def advance() -> None:
            return None

        with self.assertRaises(UnsettledJointCommand) as raised:
            asyncio.run(
                settle_tracked_joint_command(
                    actuators,
                    np.ones(7),
                    np.zeros(7),
                    advance,
                    TrackedJointSettlementPolicy(
                        required_consecutive_updates=2,
                        maximum_updates=3,
                    ),
                )
            )

        self.assertEqual(raised.exception.attempt.tracking_errors_radians, (1.0,) * 3)
        self.assertEqual(raised.exception.attempt.final_joint_positions, (1.0,) * 7)
        impossible = JointSettlementAttempt(
            raised.exception.attempt.requested_joint_motion_radians,
            raised.exception.attempt.required_tracking_error_radians,
            (4e-4, 3e-4, 1.0),
            raised.exception.attempt.final_joint_positions,
        )
        with self.assertRaisesRegex(ValueError, "does not match"):
            impossible.validate(
                TrackedJointSettlementPolicy(
                    required_consecutive_updates=2,
                    maximum_updates=3,
                ),
                expected_requested_motion_radians=1.0,
                expected_target_joint_positions=(0.0,) * 7,
            )

    def test_insertion_settlement_aborts_an_underrealized_plateau(self) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)
        updates = 0
        completion = evaluate_command_completion(
            DroidAction((0.0006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.00012, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
        )

        async def advance() -> None:
            nonlocal updates
            updates += 1

        with self.assertRaisesRegex(
            RuntimeError, "stopped making realizable progress"
        ) as raised:
            asyncio.run(
                settle_tracked_joint_command(
                    actuators,
                    np.ones(7),
                    np.zeros(7),
                    advance,
                    TrackedJointSettlementPolicy(maximum_updates=40),
                    observe_completion=lambda: completion,
                )
            )

        self.assertLess(updates, 40)
        self.assertIn("commanded_translation=(0.0006, 0.0, 0.0)", str(raised.exception))
        self.assertIn("realized_translation=(0.00012, 0.0, 0.0)", str(raised.exception))
        self.assertIn("tracking_reasons=", str(raised.exception))
        self.assertIn("translation_error_meters=", str(raised.exception))

    def test_insertion_settlement_requires_two_task_space_completion_samples(
        self,
    ) -> None:
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.zeros(7), 0.04)
        updates = 0
        completion = evaluate_command_completion(
            DroidAction((0.0006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.0006, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
        )

        async def advance() -> None:
            nonlocal updates
            updates += 1

        evidence = asyncio.run(
            settle_tracked_joint_command(
                actuators,
                np.ones(7),
                np.zeros(7),
                advance,
                TrackedJointSettlementPolicy(
                    required_consecutive_updates=1,
                    maximum_updates=3,
                ),
                observe_completion=lambda: completion,
            )
        )

        self.assertEqual(updates, 2)
        self.assertEqual(evidence.updates_used, 2)

    def test_rollback_is_observed_and_verified_after_its_physics_update(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.ones(7), 0.02)
        events = []

        async def advance() -> None:
            events.append("advance")

        def observe_safety() -> object:
            events.append("observe")
            return object()

        attachment = type("Attachment", (), {"attached": True})()
        asyncio.run(
            rollback_control_command(
                actuators,
                target,
                attachment,
                advance,
                expected_attachment=True,
                observe_safety=observe_safety,
            )
        )

        self.assertEqual(events, ["advance", "observe", "advance", "observe"])
        self.assertIs(actuators.command, target)

    def test_rollback_waits_for_consecutive_drive_tracking_passes(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        actuators = FakeActuators(valid=True)
        tracking = iter((0.01, 5e-4, 2e-4))
        updates = 0

        def actual_command() -> JointCommand:
            error = next(tracking)
            return JointCommand(np.full(7, error), 0.04)

        actuators.actual_command = actual_command

        async def advance() -> None:
            nonlocal updates
            updates += 1

        asyncio.run(
            rollback_control_command(
                actuators,
                target,
                type("Attachment", (), {"attached": True})(),
                advance,
                expected_attachment=True,
            )
        )

        self.assertEqual(updates, 3)

    def test_insertion_rollback_uses_the_same_tracked_settlement_policy(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(
            np.asarray((0.002,) + (0.0,) * 6),
            0.04,
        )
        updates = 0
        observations = 0

        async def advance() -> None:
            nonlocal updates
            updates += 1

        def observe_safety() -> object:
            nonlocal observations
            observations += 1
            return object()

        attachment = type("Attachment", (), {"attached": True})()
        evidence = asyncio.run(
            rollback_insertion_trial_command(
                actuators,
                target,
                attachment,
                advance,
                TrackedJointSettlementPolicy(),
                settlement_target=target,
                expected_attachment=True,
                observe_safety=observe_safety,
                interlock_evidence=lambda: ControlInterlockEvidence(0.0, False),
                maximum_contact_force_newtons=2.0,
                maximum_gripper_error_meters=1e-3,
            )
        )

        self.assertIsNotNone(evidence)
        self.assertEqual(
            evidence.joint_settlement.requested_joint_motion_radians,
            0.002,
        )
        self.assertEqual(
            evidence.joint_settlement.passing_tracking_errors_radians,
            (0.0, 0.0),
        )
        self.assertEqual(
            evidence.settlement.gripper.trace.errors_meters,
            (0.0, 0.0),
        )
        self.assertEqual(evidence.settlement.gripper.end_width_meters, 0.04)
        self.assertEqual(updates, 2)
        self.assertEqual(observations, 2)

    def test_insertion_rollback_separates_loaded_reset_from_active_drive_target(
        self,
    ) -> None:
        active_drive_target = JointCommand(np.zeros(7), 0.04)
        stable_loaded_reset = JointCommand(np.full(7, 0.001), 0.04)
        actuators = FakeActuators(valid=True)
        actual = JointCommand(np.full(7, 0.003), 0.04)
        applied = []

        def actual_command() -> JointCommand:
            return actual

        def apply_drive_command(command: JointCommand) -> None:
            applied.append(command)

        async def advance() -> None:
            nonlocal actual
            actual = JointCommand(
                applied[-1].arm_positions + 0.001,
                applied[-1].gripper_width_m,
            )

        actuators.actual_command = actual_command
        actuators.apply_drive_command = apply_drive_command
        evidence = asyncio.run(
            rollback_insertion_trial_command(
                actuators,
                active_drive_target,
                type("Attachment", (), {"attached": True})(),
                advance,
                TrackedJointSettlementPolicy(),
                settlement_target=stable_loaded_reset,
                expected_attachment=True,
                observe_safety=lambda: object(),
                interlock_evidence=lambda: ControlInterlockEvidence(0.0, False),
                maximum_contact_force_newtons=2.0,
                maximum_gripper_error_meters=1e-3,
            )
        )

        self.assertEqual(applied, [active_drive_target])
        self.assertEqual(
            evidence.drive_target,
            JointDriveTarget(
                tuple(active_drive_target.arm_positions),
                active_drive_target.gripper_width_m,
            ),
        )
        self.assertEqual(
            evidence.target_joint_positions,
            tuple(stable_loaded_reset.arm_positions),
        )
        self.assertEqual(evidence.end_joint_positions, (0.001,) * 7)

    def test_insertion_rollback_failure_preserves_raw_terminal_state(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        actuators = FakeActuators(valid=True)
        actuators.command = JointCommand(np.ones(7), 0.04)
        actuators.apply_drive_command = Mock()

        async def advance() -> None:
            return None

        with self.assertRaises(InsertionTrialRollbackFailed) as raised:
            asyncio.run(
                rollback_insertion_trial_command(
                    actuators,
                    target,
                    type("Attachment", (), {"attached": True})(),
                    advance,
                    TrackedJointSettlementPolicy(
                        required_consecutive_updates=2,
                        maximum_updates=3,
                    ),
                    settlement_target=target,
                    expected_attachment=True,
                    observe_safety=lambda: object(),
                    interlock_evidence=lambda: ControlInterlockEvidence(0.0, False),
                    maximum_contact_force_newtons=2.0,
                    maximum_gripper_error_meters=1e-3,
                )
            )

        evidence = raised.exception.evidence
        self.assertEqual(evidence.start_joint_positions, (1.0,) * 7)
        self.assertEqual(evidence.target_joint_positions, (0.0,) * 7)
        self.assertEqual(evidence.end_joint_positions, (1.0,) * 7)
        self.assertTrue(evidence.plug_attached)
        self.assertEqual(
            evidence.settlement_attempt.tracking_errors_radians, (1.0,) * 3
        )
        self.assertEqual(
            evidence.settlement_attempt.gripper.trace.errors_meters,
            (0.0,) * 3,
        )
        evidence.validate(
            TrackedJointSettlementPolicy(
                required_consecutive_updates=2,
                maximum_updates=3,
            ),
            expected_target_joint_positions=(0.0,) * 7,
            expected_drive_target=JointDriveTarget((0.0,) * 7, 0.04),
            expected_attachment=True,
            maximum_contact_force_newtons=2.0,
            expected_target_gripper_width_meters=0.04,
            expected_gripper_error_meters=1e-3,
        )
        without_gripper = evidence.to_dict()
        del without_gripper["settlement_attempt"]["gripper_settlement"]
        with self.assertRaisesRegex(ValueError, "incomplete"):
            InsertionTrialRollbackFailure.from_dict(without_gripper)
        lower_priority_reason = replace(
            evidence,
            reason=InsertionTrialRollbackFailureReason.DRIVE_COMMAND_REJECTED,
        )
        with self.assertRaisesRegex(ValueError, "reason"):
            lower_priority_reason.validate(
                TrackedJointSettlementPolicy(
                    required_consecutive_updates=2,
                    maximum_updates=3,
                ),
                expected_target_joint_positions=(0.0,) * 7,
                expected_drive_target=JointDriveTarget((0.0,) * 7, 0.04),
                expected_attachment=True,
                maximum_contact_force_newtons=2.0,
                expected_target_gripper_width_meters=0.04,
                expected_gripper_error_meters=1e-3,
            )
        self.assertEqual(
            InsertionTrialRollbackFailure.from_dict(evidence.to_dict()),
            evidence,
        )


if __name__ == "__main__":
    unittest.main()
