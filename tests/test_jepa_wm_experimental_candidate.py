from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.experimental_candidate import (
    ExperimentalCandidateAuthority,
    build_experimental_candidate_response,
)
from jepa_wm.candidate_trial import RealizedCandidateComparison
from jepa_wm.control_baselines import ControlPolicy, RealizedPolicyOutcome
from jepa_wm.control_rollout import PoseError
from jepa_wm.shadow_planning import ShadowSearchConfig, plan_shadow_candidates
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
    SafetyProjectionAttempt,
)


class ExperimentalCandidateTest(unittest.TestCase):
    def _shadow(self):
        direct = (
            DroidAction((0.005, 0.0, 0.0, 0.01, 0.0, 0.0, 0.05)),
            DroidAction((0.004, 0.0, 0.0, 0.01, 0.0, 0.0, 0.04)),
            DroidAction((0.003, 0.0, 0.0, 0.01, 0.0, 0.0, 0.03)),
        )

        desired = np.asarray([action.values for action in direct]) * 0.8

        def score(candidates):
            return ((candidates - desired[None, :, :]) ** 2).sum(axis=(1, 2))

        return plan_shadow_candidates(
            observation_id=91,
            direct_actions=direct,
            score=score,
            proposal=Path("/tmp/direct.pth"),
            adapter=Path("/tmp/adapter.pth"),
            config=ShadowSearchConfig(),
        )

    def test_rebinds_only_a_passing_shadow_candidate_to_an_experimental_session(self):
        shadow = self._shadow()
        scale = ACTION_SCALES[0]
        safety = ShadowSafetyEvidence(
            observation_id=91,
            evaluated_at_unix_seconds=102.0,
            counterfactual_as_of_unix_seconds=101.0,
            planned_actions=shadow.planned.actions,
            attempts=(
                SafetyProjectionAttempt(
                    scale,
                    ControlGateDecision(
                        91,
                        DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                        (),
                    ),
                    0.01,
                    (0.0,) * 7,
                ),
            ),
            selected_action_scale=scale,
        )
        observation = ControlObservation(
            observation_id=92,
            captured_at_unix_seconds=103.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png")),
            expected_proposal=Path("/tmp/reset-trial-policy.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )

        binding, response = build_experimental_candidate_response(
            execution_session_id="candidate-00",
            source_session_id="direct-00",
            observation=observation,
            shadow=shadow,
            safety=safety,
            created_at_unix_seconds=103.1,
        )

        self.assertEqual(response.actions, shadow.planned.actions)
        self.assertEqual(response.proposal, observation.expected_proposal)
        self.assertEqual(
            binding.authority,
            ExperimentalCandidateAuthority.RESET_TRIAL_ONLY,
        )
        self.assertFalse(binding.production_authority_granted)
        self.assertEqual(type(binding).from_dict(binding.to_dict()), binding)

    def test_rejects_a_non_experimental_target_or_failed_shadow_gate(self):
        shadow = self._shadow()
        observation = ControlObservation(
            observation_id=92,
            captured_at_unix_seconds=103.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png")),
            expected_proposal=Path("/tmp/direct.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        blocked = ShadowSafetyEvidence(
            observation_id=91,
            evaluated_at_unix_seconds=102.0,
            counterfactual_as_of_unix_seconds=101.0,
            planned_actions=shadow.planned.actions,
            attempts=tuple(
                SafetyProjectionAttempt(
                    scale,
                    ControlGateDecision(91, observation.pose, (reason,)),
                    0.0,
                    (0.0,) * 7,
                )
                for scale, reason in zip(
                    ACTION_SCALES,
                    (
                        ControlGateReason.COLLISION_DETECTED,
                    )
                    * len(ACTION_SCALES),
                )
            ),
            selected_action_scale=None,
        )

        with self.assertRaises(ValueError):
            build_experimental_candidate_response(
                execution_session_id="candidate-00",
                source_session_id="direct-00",
                observation=observation,
                shadow=shadow,
                safety=blocked,
            )

    def test_candidate_gate_is_multidimensional_and_never_grants_authority(self):
        initial = PoseError(0.03, 0.03, 0.3)

        def outcome(policy, translation, rotation, gripper):
            return RealizedPolicyOutcome(
                policy,
                initial,
                PoseError(translation, rotation, gripper),
            )

        comparison = RealizedCandidateComparison(
            zero=outcome(ControlPolicy.ZERO, 0.029, 0.029, 0.29),
            direct=outcome(ControlPolicy.DIRECT, 0.028, 0.028, 0.20),
            scripted=outcome(ControlPolicy.SCRIPTED, 0.005, 0.005, 0.10),
            candidate=outcome(
                ControlPolicy.EXPERIMENTAL_CANDIDATE,
                0.004,
                0.004,
                0.09,
            ),
        )

        self.assertTrue(comparison.candidate_trial_gate_passed)
        self.assertFalse(comparison.production_authority_granted)
        self.assertFalse(comparison.to_dict()["production_authority_granted"])


if __name__ == "__main__":
    unittest.main()
