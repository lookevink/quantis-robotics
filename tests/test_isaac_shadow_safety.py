from __future__ import annotations

import unittest

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_safety import (
    ACTION_SCALES,
    LEGACY_ACTION_SCALES,
    ControlGateDecision,
    ControlGateReason,
    SafetyProjectionAttempt,
)
from jepa_wm.shadow_safety import ShadowSafetyEvidence


class ShadowSafetyEvidenceTest(unittest.TestCase):
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
            attempts=(SafetyProjectionAttempt(scale, blocked, 0.01, (0.0,) * 7),),
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
