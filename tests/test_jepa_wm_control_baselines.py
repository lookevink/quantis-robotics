from __future__ import annotations

import unittest
from pathlib import Path

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_baselines import (
    NonModelBaselinePolicy,
    RealizedBaselineComparison,
    RealizedBaselineReport,
    build_baseline_response,
)
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_rollout import ControlRolloutReport, ControlStepSummary
from jepa_wm.control_safety import ControlGateDecision, SafetyProjectionAttempt
from jepa_wm.control_tracking import ActionTrackingDecision
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSessionState,
    PostActionEvidence,
)


class RealizedBaselineComparisonTest(unittest.TestCase):
    def _rollout(
        self,
        rollout_id: str,
        proposal: Path,
        initial: DroidPose,
        final: DroidPose,
        target: DroidPose,
        actions: tuple[DroidAction, ...],
        *,
        joints: tuple[float, ...] = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
        collision_detected: bool = False,
        contact_force_newtons: float = 0.0,
    ) -> ControlRolloutReport:
        observation = ControlObservation(
            7,
            100.0,
            Path("context.png"),
            ControlTarget(Path("recordings/reference/wrist/frame_000007.png")),
            proposal,
            initial,
            DroidAction((0.0,) * 7),
            4,
        )
        response = ProposedControl(7, 100.1, actions, proposal)
        scale = DroidActionScale(1.0, 0.25, 0.25)
        commanded = scale.apply(actions[0])
        gate = ControlGateDecision(7, initial.applied(commanded), ())
        tracking = ActionTrackingDecision(1.0, 1.0, 0.0, 0.0, 0.0, ())
        result = ControlResult(
            ControlResultStatus.APPLIED,
            f"{rollout_id}-00",
            gate,
            (SafetyProjectionAttempt(scale, gate, 0.0, joints),),
            scale,
            0.2,
            0.0,
            0.0,
            0.0,
            PostActionEvidence(
                actions[0],
                commanded,
                commanded,
                tracking,
                final,
                joints,
                0.0,
                0.0,
                False,
                {"path": "/tmp/post.png", "shape": [512, 512, 4]},
            ),
        )
        state = ControlSessionState(
            f"{rollout_id}-00",
            "reference",
            11401,
            f"control-{rollout_id}",
            joints,
            collision_detected,
            contact_force_newtons,
        )
        return ControlRolloutReport(
            rollout_id,
            "reference",
            11401,
            proposal,
            1,
            (ControlStepSummary(state, observation, response, result),),
            target,
        )

    def test_builds_policy_bound_zero_and_scripted_responses(self) -> None:
        observation = ControlObservation(
            7,
            100.0,
            Path("context.png"),
            ControlTarget(Path("target.png")),
            Path("/tmp/baseline_zero.pth"),
            DroidPose((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2)),
            DroidAction((0.0,) * 7),
            4,
        )

        zero = build_baseline_response(
            observation,
            NonModelBaselinePolicy.ZERO,
            created_at_unix_seconds=100.1,
        )

        self.assertEqual(zero.proposal, observation.expected_proposal)
        self.assertEqual(zero.actions, (DroidAction((0.0,) * 7),) * 3)

        scripted_observation = ControlObservation(
            observation.observation_id,
            observation.captured_at_unix_seconds,
            observation.context_frame,
            observation.target,
            Path("/tmp/baseline_scripted.pth"),
            observation.pose,
            observation.previous_action,
            observation.warmup_frames,
        )
        scripted_actions = (
            DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        scripted = build_baseline_response(
            scripted_observation,
            NonModelBaselinePolicy.SCRIPTED,
            scripted_actions=scripted_actions,
            created_at_unix_seconds=100.1,
        )
        self.assertEqual(scripted.actions, scripted_actions)

    def test_binds_a_realized_report_to_explicit_trial_roles(self) -> None:
        initial = DroidPose((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2))
        target = DroidPose((0.03, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2))
        zero_actions = (DroidAction((0.0,) * 7),) * 3
        direct_actions = (
            DroidAction((0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            *zero_actions[1:],
        )
        scripted_actions = (
            DroidAction((0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            *zero_actions[1:],
        )
        report = RealizedBaselineReport.from_rollouts(
            "baseline-proof",
            self._rollout(
                "direct", Path("/tmp/proposal.pth"), initial,
                DroidPose((0.02, *initial.values[1:])), target, direct_actions,
            ),
            self._rollout(
                "zero", Path("/tmp/baseline_zero.pth"), initial,
                initial, target, zero_actions,
            ),
            self._rollout(
                "scripted", Path("/tmp/baseline_scripted.pth"), initial,
                target, target, scripted_actions,
            ),
        )

        payload = report.to_dict()

        self.assertEqual(payload["schema"], "quantis.jepa_wm_realized_baselines.v1")
        self.assertAlmostEqual(
            payload["outcomes"]["direct"]["translation_progress_meters"],
            0.02,
        )
        self.assertFalse(payload["candidate_authority_granted"])

        tampered_zero = self._rollout(
            "zero-tampered",
            Path("/tmp/baseline_zero.pth"),
            initial,
            initial,
            target,
            direct_actions,
        )
        with self.assertRaisesRegex(ValueError, "nonzero"):
            RealizedBaselineReport.from_rollouts(
                "baseline-proof", report.direct, tampered_zero, report.scripted
            )

    def test_compares_each_pose_error_dimension_without_scalarizing_units(self) -> None:
        comparison = RealizedBaselineComparison.from_poses(
            initial=DroidPose((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2)),
            target=DroidPose((0.03, 0.0, 0.5, 0.03, 0.0, 0.0, 0.8)),
            zero_final=DroidPose((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2)),
            direct_final=DroidPose((0.02, 0.0, 0.5, 0.02, 0.0, 0.0, 0.7)),
            scripted_final=DroidPose((0.03, 0.0, 0.5, 0.03, 0.0, 0.0, 0.8)),
        )

        payload = comparison.to_dict()

        self.assertAlmostEqual(
            payload["outcomes"]["zero"]["translation_progress_meters"], 0.0
        )
        self.assertAlmostEqual(
            payload["outcomes"]["direct"]["translation_progress_meters"], 0.02
        )
        self.assertAlmostEqual(
            payload["outcomes"]["scripted"]["translation_progress_meters"], 0.03
        )
        self.assertEqual(
            payload["direct_improves_over_zero"],
            {"translation": True, "rotation": True, "gripper": True},
        )
        self.assertFalse(payload["direct_baseline_gate_passed"])
        self.assertFalse(payload["candidate_authority_granted"])

    def test_requires_direct_to_reach_the_scripted_tolerance(self) -> None:
        target = DroidPose((0.03, 0.0, 0.5, 0.03, 0.0, 0.0, 0.8))
        comparison = RealizedBaselineComparison.from_poses(
            initial=DroidPose((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2)),
            target=target,
            zero_final=DroidPose((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2)),
            direct_final=target,
            scripted_final=target,
        )

        self.assertTrue(comparison.direct_baseline_gate_passed)
        self.assertFalse(comparison.candidate_authority_granted)

    def test_rejects_trials_with_a_different_joint_or_contact_reset(self) -> None:
        initial = DroidPose((0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2))
        target = DroidPose((0.03, 0.0, 0.5, 0.0, 0.0, 0.0, 0.2))
        actions = (DroidAction((0.0,) * 7),) * 3
        direct = self._rollout(
            "direct", Path("/tmp/proposal.pth"), initial, initial, target, actions
        )
        scripted = self._rollout(
            "scripted",
            Path("/tmp/baseline_scripted.pth"),
            initial,
            target,
            target,
            actions,
        )
        mismatched_zero = self._rollout(
            "zero",
            Path("/tmp/baseline_zero.pth"),
            initial,
            initial,
            target,
            actions,
            joints=(0.01, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
        )

        with self.assertRaisesRegex(ValueError, "same reset"):
            RealizedBaselineReport.from_rollouts(
                "joint-mismatch", direct, mismatched_zero, scripted
            )

        contacted_zero = self._rollout(
            "zero-contact",
            Path("/tmp/baseline_zero.pth"),
            initial,
            initial,
            target,
            actions,
            contact_force_newtons=0.1,
        )
        with self.assertRaisesRegex(ValueError, "collision or contact"):
            RealizedBaselineReport.from_rollouts(
                "contact-mismatch", direct, contacted_zero, scripted
            )


if __name__ == "__main__":
    unittest.main()
