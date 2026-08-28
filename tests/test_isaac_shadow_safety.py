from __future__ import annotations

import unittest

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_safety import (
    ACTION_SCALES,
    CONTACT_GRASP_ACTION_SCALES,
    CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES,
    CONTACT_GRASP_REDUCED_CLOSING_ACTION_SCALE_POLICIES,
    CONTACT_GRASP_FINE_ACTION_SCALES,
    CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES,
    CONTACT_GRASP_ULTRAFINE_ACTION_SCALES,
    LEGACY_ACTION_SCALES,
    LEGACY_TRACKING_BOUNDED_ACTION_SCALES,
    ORIENTATION_HOLD_ACTION_SCALES,
    TRACKING_BOUNDED_ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
    SafetyProjectionAttempt,
    contact_grasp_action_scales,
    insertion_projection_policy_for_attempts,
)
from jepa_wm.shadow_safety import ShadowSafetyEvidence


class ShadowSafetyEvidenceTest(unittest.TestCase):
    def test_reads_the_historical_positional_tracking_roster(self) -> None:
        self.assertEqual(
            insertion_projection_policy_for_attempts(
                LEGACY_TRACKING_BOUNDED_ACTION_SCALES[:2]
            ),
            LEGACY_TRACKING_BOUNDED_ACTION_SCALES,
        )

    def test_tracking_bounded_insertion_scale_has_a_unique_policy_owner(self) -> None:
        self.assertEqual(
            insertion_projection_policy_for_attempts(
                (TRACKING_BOUNDED_ACTION_SCALES[0],)
            ),
            TRACKING_BOUNDED_ACTION_SCALES,
        )

    def test_contact_grasp_scales_small_proposals_above_the_noise_floor(self) -> None:
        policy = CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES[0]
        scale = policy[0]

        self.assertEqual(
            scale.to_dict(),
            {"translation": 0.25, "rotation": 0.125, "gripper": 0.25},
        )
        self.assertEqual(
            contact_grasp_action_scales(
                DroidAction((0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05))
            ),
            policy,
        )
        self.assertEqual(
            insertion_projection_policy_for_attempts((scale,)),
            policy,
        )

    def test_contact_grasp_bounds_large_translation_commands(self) -> None:
        scales = contact_grasp_action_scales(
            DroidAction((0.007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        )

        self.assertEqual(scales, CONTACT_GRASP_FINE_ACTION_SCALES)
        self.assertLessEqual(0.007 * scales[0].translation, 0.001)
        self.assertEqual(
            insertion_projection_policy_for_attempts((scales[0],)),
            CONTACT_GRASP_FINE_ACTION_SCALES,
        )

    def test_contact_grasp_adds_an_ultrafine_ik_fallback(self) -> None:
        action = DroidAction((0.011, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        scales = contact_grasp_action_scales(action)

        self.assertEqual(scales, CONTACT_GRASP_ULTRAFINE_ACTION_SCALES)
        self.assertLessEqual(0.011 * scales[0].translation, 0.001)
        self.assertEqual(scales[1].translation, 0.03125)

    def test_contact_grasp_does_not_reenlarge_translation_during_reopening(
        self,
    ) -> None:
        action = DroidAction(
            (-0.015083, -0.001079, 0.000624, 0.0, 0.0, 0.0, -0.020897)
        )

        scales = contact_grasp_action_scales(action)

        self.assertEqual(scales[0], DroidActionScale(0.03125, 0.125, 0.125))
        translation_norm = sum(
            value * value for value in action.values[:3]
        ) ** 0.5
        self.assertLess(translation_norm * scales[0].translation, 0.0005)

    def test_attached_contact_grasp_holds_gripper_and_uses_transport_scale(
        self,
    ) -> None:
        action = DroidAction(
            (-0.019864, -0.001952, 0.001272, 0.0, 0.0, 0.0, -0.077043)
        )

        scales = contact_grasp_action_scales(
            action,
            attachment_acquired=True,
        )

        self.assertEqual(scales, CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES[3])
        self.assertEqual(scales[0], DroidActionScale(0.03125, 0.125, 0.0))
        translation_norm = sum(
            value * value for value in action.values[:3]
        ) ** 0.5
        self.assertLessEqual(translation_norm * scales[0].translation, 0.00075)
        self.assertEqual(scales[0].apply(action).values[6], 0.0)
        self.assertEqual(
            insertion_projection_policy_for_attempts((scales[0],)),
            scales,
        )

    def test_contact_grasp_closure_uses_independent_gripper_calibration(
        self,
    ) -> None:
        action = DroidAction(
            (-0.015083, -0.001079, 0.000624, 0.0, 0.0, 0.0, 0.044233)
        )

        scales = contact_grasp_action_scales(action)

        self.assertEqual(scales, CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES[2])
        self.assertEqual(scales[0], DroidActionScale(0.0625, 0.125, 0.25))
        self.assertLess(
            sum(value * value for value in action.values[:3]) ** 0.5
            * scales[0].translation,
            0.001,
        )

    def test_live_closure_calibration_crosses_the_nominal_acquisition_width(
        self,
    ) -> None:
        pose = DroidPose(
            (0.328385, -0.282350, 0.454648, 0.0, 0.0, 0.0, 0.614173)
        )
        action = DroidAction(
            (-0.015083, -0.001079, 0.000624, 0.0, 0.0, 0.0, 0.044233)
        )
        scale = contact_grasp_action_scales(action)[0]

        calibrated = pose.applied(scale.apply(action))
        historical = pose.applied(
            DroidActionScale(scale.translation, scale.rotation, 0.125).apply(
                action
            )
        )

        self.assertLess((1.0 - calibrated.values[6]) * 0.08, 0.03)
        self.assertGreater((1.0 - historical.values[6]) * 0.08, 0.03)

    def test_large_gripper_closure_remains_on_the_exercised_scale(self) -> None:
        action = DroidAction(
            (-0.004619, -0.000686, 0.000495, 0.0, 0.0, 0.0, 0.135577)
        )

        scales = contact_grasp_action_scales(action)

        self.assertEqual(scales, CONTACT_GRASP_FINE_ACTION_SCALES)
        self.assertLess(action.values[6] * scales[0].gripper * 0.08, 0.0015)
        self.assertGreater(
            action.values[6]
            * CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES[1][0].gripper
            * 0.08,
            0.0015,
        )

    def test_very_large_gripper_closure_reduces_scale_to_the_command_bound(
        self,
    ) -> None:
        action = DroidAction(
            (-0.0017157, 0.0001624, -0.0001836, 0.0, 0.0, 0.0, 0.215816)
        )

        scales = contact_grasp_action_scales(action)

        self.assertEqual(
            scales,
            CONTACT_GRASP_REDUCED_CLOSING_ACTION_SCALE_POLICIES[0][1],
        )
        self.assertEqual(scales[0].gripper, 0.0625)
        self.assertLessEqual(
            action.values[6] * scales[0].gripper * 0.08,
            0.0015,
        )

    def test_round_trips_shadow_only_counterfactual_evidence(self) -> None:
        scale = ACTION_SCALES[0]
        gate = ControlGateDecision(
            9,
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            (),
        )
        evidence = ShadowSafetyEvidence(
            observation_id=9,
            evaluated_at_unix_seconds=101.0,
            counterfactual_as_of_unix_seconds=100.2,
            planned_actions=(DroidAction((0.0,) * 7),) * 3,
            attempts=(SafetyProjectionAttempt(scale, gate, 0.01, (0.0,) * 7),),
            selected_action_scale=scale,
        )

        self.assertTrue(evidence.passed)
        self.assertEqual(evidence.to_dict()["authority"], "shadow_only")
        self.assertEqual(ShadowSafetyEvidence.from_dict(evidence.to_dict()), evidence)

    def test_round_trips_persisted_legacy_projection_evidence(self) -> None:
        scale = LEGACY_ACTION_SCALES[0]
        gate = ControlGateDecision(
            9,
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            (),
        )
        evidence = ShadowSafetyEvidence(
            observation_id=9,
            evaluated_at_unix_seconds=101.0,
            counterfactual_as_of_unix_seconds=100.2,
            planned_actions=(DroidAction((0.0,) * 7),) * 3,
            attempts=(SafetyProjectionAttempt(scale, gate, 0.01, (0.0,) * 7),),
            selected_action_scale=scale,
        )

        self.assertEqual(ShadowSafetyEvidence.from_dict(evidence.to_dict()), evidence)

    def test_rejects_a_selected_scale_without_a_passing_attempt(self) -> None:
        scale = ACTION_SCALES[0]
        blocked = ControlGateDecision(
            9,
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            (ControlGateReason.COLLISION_DETECTED,),
        )

        with self.assertRaisesRegex(ValueError, "selection"):
            ShadowSafetyEvidence(
                observation_id=9,
                evaluated_at_unix_seconds=101.0,
                counterfactual_as_of_unix_seconds=100.2,
                planned_actions=(DroidAction((0.0,) * 7),) * 3,
                attempts=(
                    SafetyProjectionAttempt(scale, blocked, 0.01, (0.0,) * 7),
                ),
                selected_action_scale=scale,
            )

    def test_rejects_insertion_only_orientation_hold_scales(self) -> None:
        scale = ORIENTATION_HOLD_ACTION_SCALES[0]
        gate = ControlGateDecision(
            9,
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            (),
        )

        with self.assertRaisesRegex(ValueError, "projection order"):
            ShadowSafetyEvidence(
                observation_id=9,
                evaluated_at_unix_seconds=101.0,
                counterfactual_as_of_unix_seconds=100.2,
                planned_actions=(DroidAction((0.0,) * 7),) * 3,
                attempts=(
                    SafetyProjectionAttempt(scale, gate, 0.01, (0.0,) * 7),
                ),
                selected_action_scale=scale,
            )

    def test_rejects_tampered_command_authority(self) -> None:
        scale = ACTION_SCALES[0]
        gate = ControlGateDecision(
            9,
            DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            (),
        )
        evidence = ShadowSafetyEvidence(
            observation_id=9,
            evaluated_at_unix_seconds=101.0,
            counterfactual_as_of_unix_seconds=100.2,
            planned_actions=(DroidAction((0.0,) * 7),) * 3,
            attempts=(SafetyProjectionAttempt(scale, gate, 0.01, (0.0,) * 7),),
            selected_action_scale=scale,
        )
        payload = evidence.to_dict()
        payload["authority"] = "command"

        with self.assertRaisesRegex(ValueError, "authority"):
            ShadowSafetyEvidence.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
