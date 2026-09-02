from __future__ import annotations

import unittest

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_safety import (
    ACTION_SCALES,
    CONTACT_GRASP_ACTION_SCALES,
    CONTACT_GRASP_CLOSING_ACTION_SCALE_POLICIES,
    CONTACT_GRASP_COARSE_ACTION_SCALE_POLICIES,
    CONTACT_GRASP_REDUCED_CLOSING_ACTION_SCALE_POLICIES,
    CONTACT_GRASP_FINE_ACTION_SCALES,
    CONTACT_GRASP_MICRO_ACTION_SCALES,
    CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES,
    CONTACT_GRASP_ULTRAFINE_ACTION_SCALES,
    DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES,
    LEGACY_ACTION_SCALES,
    LEGACY_TRACKING_BOUNDED_ACTION_SCALES,
    MINIMUM_DIRECTION_OBSERVABLE_ROTATION_RADIANS,
    ORIENTATION_HOLD_ACTION_SCALES,
    TRACKING_BOUNDED_ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
    SafetyProjectionAttempt,
    contact_grasp_action_scales,
    insertion_projection_policy_for_attempts,
)
from jepa_wm.control_tracking import evaluate_command_realization
from jepa_wm.shadow_safety import ShadowSafetyEvidence


class ShadowSafetyEvidenceTest(unittest.TestCase):
    def test_current_grasp_policy_resolves_failed_v19_attached_turn(self) -> None:
        # Exact direct proposal that terminalized the frozen v19 canary because
        # its 12.93 mrad turn was scaled to an unobservable 1.62 mrad command.
        action = DroidAction(
            (
                -0.0009391895728185773,
                0.0006117846933193505,
                -0.001291528227739036,
                -0.003788558766245842,
                0.0007719331188127398,
                0.012341589666903019,
                -0.0009279325604438782,
            )
        )

        scales = contact_grasp_action_scales(
            action,
            attachment_acquired=True,
            require_directional_transport_progress=True,
            require_resolvable_transport=True,
            require_axis_resolvable_transport=True,
            require_resolvable_rotation=True,
        )
        commanded = scales[0].apply(action)

        self.assertEqual(scales[0].rotation, 0.5)
        self.assertGreaterEqual(
            sum(value * value for value in commanded.values[3:6]) ** 0.5,
            MINIMUM_DIRECTION_OBSERVABLE_ROTATION_RADIANS,
        )

    def test_current_grasp_policy_promotes_v15_turn_above_resolution(self) -> None:
        action = DroidAction(
            (
                0.004746975377202034,
                0.0003092704282607883,
                -0.0037363399751484394,
                -0.0037258435040712357,
                -0.00878340657800436,
                -0.010692881420254707,
                0.002727515995502472,
            )
        )

        scales = contact_grasp_action_scales(
            action,
            coarse_acquisition=True,
            require_resolvable_rotation=True,
        )
        commanded = scales[0].apply(action)

        self.assertEqual(scales[0].rotation, 0.25)
        self.assertGreaterEqual(
            sum(value * value for value in commanded.values[3:6]) ** 0.5,
            MINIMUM_DIRECTION_OBSERVABLE_ROTATION_RADIANS,
        )

    def test_current_grasp_policy_holds_unresolvable_turn(self) -> None:
        action = DroidAction((0.004, 0.0, 0.0, 0.001, 0.001, 0.001, 0.0))

        current = contact_grasp_action_scales(
            action,
            coarse_acquisition=True,
            require_resolvable_rotation=True,
        )
        historical = contact_grasp_action_scales(
            action,
            coarse_acquisition=True,
        )

        self.assertTrue(all(scale.rotation == 0.0 for scale in current))
        self.assertEqual(historical[0].rotation, 0.125)

    def test_robust_rotation_policy_retains_ordered_hold_fallbacks(self) -> None:
        action = DroidAction(
            (-0.008, 0.0, 0.0, -0.000869, 0.002821, 0.001887, 0.0)
        )

        scales = contact_grasp_action_scales(
            action,
            attachment_acquired=True,
            require_directional_transport_progress=True,
            require_resolvable_transport=True,
            require_axis_resolvable_transport=True,
            require_resolvable_rotation=True,
            active_rotation_hold_fallback=True,
        )

        active = tuple(scale for scale in scales if scale.rotation > 0.0)
        holds = tuple(scale for scale in scales if scale.rotation == 0.0)
        self.assertTrue(active)
        self.assertEqual(len(active), len(holds))
        self.assertEqual(
            tuple(scale.translation for scale in holds),
            tuple(scale.translation for scale in active),
        )
        self.assertEqual(
            tuple(scale.gripper for scale in holds),
            tuple(scale.gripper for scale in active),
        )

    def test_coarse_acquisition_decouples_opening_from_arm_scale(self) -> None:
        action = DroidAction(
            (0.0015, 0.0002, -0.0005, 0.0, 0.0, 0.0, -0.02)
        )

        scales = contact_grasp_action_scales(action, coarse_acquisition=True)

        self.assertEqual(scales, CONTACT_GRASP_COARSE_ACTION_SCALE_POLICIES[0])
        self.assertEqual(scales[0], DroidActionScale(1.0, 0.125, 0.125))
        self.assertEqual(scales[-1], DroidActionScale(0.03125, 0.0, 0.125))
        self.assertLessEqual(
            sum(value * value for value in scales[0].apply(action).values[:3])
            ** 0.5,
            0.002,
        )

    def test_coarse_acquisition_bounds_large_arm_and_closing_commands(self) -> None:
        action = DroidAction(
            (0.015, 0.001, -0.001, 0.0, 0.0, 0.0, 0.2)
        )

        scales = contact_grasp_action_scales(action, coarse_acquisition=True)
        commanded = scales[0].apply(action)

        self.assertEqual(scales[0].translation, 0.125)
        self.assertLessEqual(
            sum(value * value for value in commanded.values[:3]) ** 0.5,
            0.002,
        )
        self.assertLessEqual(commanded.values[6] * 0.08, 0.0015)

    def test_exact_coarse_acquisition_fills_but_never_exceeds_bound(self) -> None:
        action = DroidAction(
            (0.0045, 0.0008, -0.0022, 0.0, 0.0, 0.0, -0.01)
        )

        scales = contact_grasp_action_scales(
            action,
            coarse_acquisition=True,
            exact_coarse_translation_projection=True,
        )
        norms = [
            sum(value * value for value in scale.apply(action).values[:3]) ** 0.5
            for scale in scales
        ]

        self.assertAlmostEqual(norms[0], 0.002)
        self.assertTrue(all(norm <= 0.002 for norm in norms))
        self.assertGreater(scales[0].translation, 0.25)
        self.assertEqual(scales[-1].rotation, 0.0)

    def test_exact_coarse_rotation_tries_resolvable_translation_hold_before_halving(
        self,
    ) -> None:
        action = DroidAction(
            (
                0.0027388702146708965,
                -0.00015498205902986228,
                0.0011964633595198393,
                -0.0027516658883541822,
                -0.0020346895325928926,
                -0.00192593305837363,
                -0.0006209798157215118,
            )
        )

        scales = contact_grasp_action_scales(
            action,
            coarse_acquisition=True,
            maximum_coarse_translation_command_meters=0.001,
            require_resolvable_rotation=True,
            exact_coarse_translation_projection=True,
            coarse_orientation_hold_fallback=True,
            minimum_coarse_translation_command_meters=0.0005,
        )
        projected_norms = tuple(
            sum(value * value for value in scale.apply(action).values[:3]) ** 0.5
            for scale in scales
        )

        self.assertAlmostEqual(projected_norms[0], 0.001)
        self.assertAlmostEqual(projected_norms[1], 0.001)
        self.assertEqual(scales[0].rotation, 1.0)
        self.assertEqual(scales[1].rotation, 0.0)

    def test_resolution_floored_coarse_policy_rejects_sub_resolution_action(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "below controller resolution"):
            contact_grasp_action_scales(
                DroidAction((0.00049, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0)),
                coarse_acquisition=True,
                maximum_coarse_translation_command_meters=0.001,
                require_resolvable_rotation=True,
                exact_coarse_translation_projection=True,
                coarse_orientation_hold_fallback=True,
                minimum_coarse_translation_command_meters=0.0005,
            )

    def test_resolution_floor_remains_active_during_fine_close(self) -> None:
        action = DroidAction(
            (0.0014771344, 0.0003514159, 0.0007743249, 0.0, 0.0, 0.0, 0.01)
        )

        scales = contact_grasp_action_scales(
            action,
            coarse_acquisition=False,
            resolution_floored_acquisition=True,
            maximum_coarse_translation_command_meters=0.001,
            exact_coarse_translation_projection=True,
            minimum_coarse_translation_command_meters=0.0005,
        )
        projected_norms = tuple(
            sum(value * value for value in scale.apply(action).values[:3]) ** 0.5
            for scale in scales
        )

        self.assertAlmostEqual(projected_norms[0], 0.001)
        self.assertTrue(all(norm >= 0.0005 for norm in projected_norms))

    def test_tracking_overlap_close_selects_the_interior_command(self) -> None:
        action = DroidAction(
            (0.0013759080, 0.0005052318, 0.0021299466, 0.0, 0.0, 0.0, 0.18)
        )

        scales = contact_grasp_action_scales(
            action,
            resolution_floored_acquisition=True,
            maximum_coarse_translation_command_meters=0.001,
            maximum_resolution_floored_translation_command_meters=0.00075,
            exact_coarse_translation_projection=True,
            minimum_coarse_translation_command_meters=0.0005,
        )
        projected_norms = tuple(
            sum(value * value for value in scale.apply(action).values[:3]) ** 0.5
            for scale in scales
        )

        self.assertAlmostEqual(projected_norms[0], 0.00075)
        self.assertAlmostEqual(projected_norms[-1], 0.0005)

    def test_resolution_floor_does_not_reenlarge_reopening_drift(self) -> None:
        action = DroidAction(
            (0.0147, 0.0032, -0.0017, 0.0, 0.0, 0.0, -0.01)
        )

        historical = contact_grasp_action_scales(action)
        current = contact_grasp_action_scales(
            action,
            resolution_floored_acquisition=True,
            maximum_coarse_translation_command_meters=0.001,
            exact_coarse_translation_projection=True,
            minimum_coarse_translation_command_meters=0.0005,
        )

        self.assertEqual(current, historical)
        self.assertLess(
            sum(value * value for value in current[0].apply(action).values[:3])
            ** 0.5,
            0.0005,
        )

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

    def test_contact_grasp_micro_policy_falls_back_to_orientation_hold(self) -> None:
        action = DroidAction(
            (0.001072, 0.001511, -0.000810, 0.003429, -0.017439, -0.003272, -0.000746)
        )

        scales = contact_grasp_action_scales(action)

        self.assertEqual(scales, CONTACT_GRASP_MICRO_ACTION_SCALES)
        self.assertTrue(
            all(scale.translation == 0.03125 for scale in scales)
        )
        self.assertTrue(all(scale.gripper == 0.125 for scale in scales))
        self.assertEqual(
            tuple(scale.rotation for scale in scales),
            (0.125, 0.0),
        )
        self.assertEqual(
            insertion_projection_policy_for_attempts(scales[:2]),
            CONTACT_GRASP_MICRO_ACTION_SCALES,
        )

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

    def test_directional_v2_keeps_small_actions_direction_active(
        self,
    ) -> None:
        action = DroidAction(
            (-0.000236, -0.000198, 0.000125, 0.0, 0.0, 0.0, 0.02)
        )

        scales = contact_grasp_action_scales(
            action,
            attachment_acquired=True,
            require_directional_transport_progress=True,
        )

        self.assertEqual(
            scales,
            DIRECTIONAL_CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES[0],
        )
        self.assertEqual(scales[0], DroidActionScale(1.0, 0.125, 0.0))
        translation_norm = sum(
            value * value for value in scales[0].apply(action).values[:3]
        ) ** 0.5
        self.assertGreaterEqual(translation_norm, 1e-4)
        self.assertLessEqual(translation_norm, 0.00075)
        self.assertEqual(
            insertion_projection_policy_for_attempts((scales[0],)), scales
        )

    def test_directional_v2_rejects_sub_tracking_action(self) -> None:
        action = DroidAction(
            (9e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02)
        )

        with self.assertRaisesRegex(ValueError, "below tracking activity"):
            contact_grasp_action_scales(
                action,
                attachment_acquired=True,
                require_directional_transport_progress=True,
            )

        self.assertEqual(
            contact_grasp_action_scales(
                action,
                attachment_acquired=True,
            ),
            CONTACT_GRASP_TRANSPORT_ACTION_SCALE_POLICIES[0],
        )

    def test_current_transport_projects_horizon_inside_controller_band(self) -> None:
        action = DroidAction(
            (-0.000752, -0.000459, 0.000449, 0.0, 0.0, 0.0, 0.03)
        )

        scales = contact_grasp_action_scales(
            action,
            attachment_acquired=True,
            require_directional_transport_progress=True,
            require_resolvable_transport=True,
        )

        projected_norms = tuple(
            sum(value * value for value in scale.apply(action).values[:3]) ** 0.5
            for scale in scales
        )
        self.assertEqual(len(scales), 3)
        self.assertTrue(
            all(
                0.0005 - 1e-12 <= norm <= 0.00075 + 1e-12
                for norm in projected_norms
            )
        )
        self.assertEqual(scales[0].apply(action).values[-1], 0.0)

    def test_axis_resolvable_transport_keeps_one_component_above_the_floor(
        self,
    ) -> None:
        action = DroidAction(
            (-0.000752, -0.000459, 0.000449, 0.0, 0.0, 0.0, 0.03)
        )

        scales = contact_grasp_action_scales(
            action,
            attachment_acquired=True,
            require_directional_transport_progress=True,
            require_resolvable_transport=True,
            require_axis_resolvable_transport=True,
        )

        self.assertTrue(
            all(
                max(abs(value) for value in scale.apply(action).values[:3])
                >= 0.0005 - 1e-12
                for scale in scales
            )
        )
        minimum_command = scales[-1].apply(action)
        self.assertTrue(
            evaluate_command_realization(
                minimum_command,
                minimum_command,
            ).passed
        )

    def test_axis_resolvable_transport_rejects_a_diagonal_below_the_axis_floor(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "below controller axis resolution"):
            contact_grasp_action_scales(
                DroidAction(
                    (0.0004, 0.0004, 0.0004, 0.0, 0.0, 0.0, 0.0)
                ),
                attachment_acquired=True,
                require_directional_transport_progress=True,
                require_resolvable_transport=True,
                require_axis_resolvable_transport=True,
            )

    def test_axis_resolvable_transport_requires_the_versioned_resolvable_policy(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "requires a resolvable policy"):
            contact_grasp_action_scales(
                DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                attachment_acquired=True,
                require_axis_resolvable_transport=True,
            )

    def test_current_transport_derives_a_valid_scale_across_old_roster_gap(self) -> None:
        action = DroidAction((0.0061, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02))

        scales = contact_grasp_action_scales(
            action,
            attachment_acquired=True,
            require_directional_transport_progress=True,
            require_resolvable_transport=True,
        )

        self.assertAlmostEqual(scales[0].translation * 0.0061, 0.00075)
        self.assertAlmostEqual(scales[-1].translation * 0.0061, 0.0005)

    def test_current_transport_rejects_a_sub_resolution_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "below controller resolution"):
            contact_grasp_action_scales(
                DroidAction((0.00049, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02)),
                attachment_acquired=True,
                require_directional_transport_progress=True,
                require_resolvable_transport=True,
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
