from pathlib import Path
import unittest

import numpy as np

from jepa_wm.action import DroidAction
from jepa_wm.action_prior import ActionPriorConfig
from jepa_wm.planner import CEMConfig, PlannerActionBounds
from jepa_wm.shadow_planning import (
    CandidateTrustRegion,
    ProposalCenteredBounds,
    ShadowSearchConfig,
    ShadowSearchEvidence,
    ShadowPlanningRequest,
    plan_shadow_candidates,
)
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.action import DroidPose
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
    ActionResponseTrial,
    TaskProgressObjective,
)


class ShadowCandidatePlanningTest(unittest.TestCase):
    def test_round_trips_a_request_bound_to_the_direct_proposal(self) -> None:
        actions = (DroidAction((0.0,) * 7),) * 3
        request = ShadowPlanningRequest(
            observation=ControlObservation(
                observation_id=12,
                captured_at_unix_seconds=100.0,
                context_frame=Path("context.png"),
                target=ControlTarget(Path("target.png")),
                expected_proposal=Path("/tmp/proposal.pth"),
                pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                previous_action=DroidAction((0.0,) * 7),
                warmup_frames=4,
            ),
            direct_control=ProposedControl(
                observation_id=12,
                created_at_unix_seconds=100.2,
                actions=actions,
                proposal=Path("/tmp/proposal.pth"),
            ),
            expected_adapter=Path("/tmp/adapter.pth"),
            expected_planner=CEMConfig(
                iterations=5, samples=128, elites=12, seed=235
            ),
        )

        self.assertEqual(ShadowPlanningRequest.from_dict(request.to_dict()), request)

    def test_rejects_a_request_for_another_observation(self) -> None:
        observation = ControlObservation(
            observation_id=12,
            captured_at_unix_seconds=100.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png")),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        direct = ProposedControl(
            observation_id=13,
            created_at_unix_seconds=100.2,
            actions=(DroidAction((0.0,) * 7),) * 3,
            proposal=Path("/tmp/proposal.pth"),
        )

        with self.assertRaisesRegex(ValueError, "observation identity"):
            ShadowPlanningRequest(
                observation,
                direct,
                Path("/tmp/adapter.pth"),
                ShadowSearchConfig().planner,
            )

    def test_clips_every_candidate_to_global_and_proposal_centered_bounds(self) -> None:
        center = np.asarray(
            [
                [0.019, 0.0, 0.0, 0.07, 0.0, 0.0, 0.20],
                [0.0] * 7,
                [0.0] * 7,
            ]
        )
        bounds = ProposalCenteredBounds(
            center,
            PlannerActionBounds(),
            CandidateTrustRegion(
                maximum_translation_deviation=0.001,
                maximum_rotation_deviation=0.004,
                maximum_gripper_deviation=0.02,
            ),
        )
        candidates = np.full((5, 3, 7), 1.0)

        clipped = bounds.clip(candidates)

        self.assertTrue(
            np.all(np.linalg.norm(clipped[:, :, :3], axis=2) <= 0.0200001)
        )
        self.assertTrue(
            np.all(
                np.linalg.norm(clipped[:, :, :3] - center[None, :, :3], axis=2)
                <= 0.0010001
            )
        )
        self.assertTrue(
            np.all(
                np.abs(clipped[:, :, 6] - center[None, :, 6]) <= 0.0200001
            )
        )

    def test_search_improves_energy_without_changing_command_authority(self) -> None:
        direct = (
            DroidAction((0.005, 0.0, 0.0, 0.01, 0.0, 0.0, 0.05)),
            DroidAction((0.004, 0.0, 0.0, 0.01, 0.0, 0.0, 0.04)),
            DroidAction((0.003, 0.0, 0.0, 0.01, 0.0, 0.0, 0.03)),
        )
        desired = np.asarray([action.values for action in direct]) * 0.8

        def score(candidates: np.ndarray) -> np.ndarray:
            return np.square(candidates - desired[None, :, :]).sum(axis=(1, 2))

        evidence = plan_shadow_candidates(
            observation_id=91,
            direct_actions=direct,
            score=score,
            proposal=Path("/tmp/proposal.pth"),
            adapter=Path("/tmp/adapter.pth"),
            config=ShadowSearchConfig(
                planner=CEMConfig(
                    iterations=5,
                    samples=160,
                    elites=12,
                    seed=7,
                ),
                prior=ActionPriorConfig(penalty_weight=1e-5),
            ),
        )

        self.assertGreater(evidence.energy_improvement, 0.0)
        self.assertTrue(evidence.first_action_gate.passed)
        self.assertTrue(evidence.passes_shadow_gate)
        self.assertEqual(evidence.authority.value, "shadow_only")
        self.assertEqual(evidence.candidates_scored, 800)
        self.assertGreaterEqual(evidence.planning_seconds, 0.0)
        self.assertEqual(
            ShadowSearchEvidence.from_dict(evidence.to_dict()),
            evidence,
        )
        runtime_rounding = evidence.to_dict()
        runtime_rounding["first_action_gate"]["cosine"] += 5e-16
        self.assertEqual(
            ShadowSearchEvidence.from_dict(runtime_rounding).first_action_gate.reasons,
            evidence.first_action_gate.reasons,
        )
        objective_tamper = evidence.to_dict()
        objective_tamper["planned"]["objective"] += 1.0
        with self.assertRaisesRegex(ValueError, "objective"):
            ShadowSearchEvidence.from_dict(objective_tamper)

        bounds_tamper = evidence.to_dict()
        bounds_tamper["planned"]["actions"][0][0] = 0.1
        with self.assertRaisesRegex(ValueError, "bounds"):
            ShadowSearchEvidence.from_dict(bounds_tamper)

        gate_tamper = evidence.to_dict()
        gate_tamper["first_action_gate"]["cosine"] = -1.0
        with self.assertRaisesRegex(ValueError, "first-action"):
            ShadowSearchEvidence.from_dict(gate_tamper)

    def test_task_progress_reranking_rejects_the_latent_winner(self) -> None:
        direct = (DroidAction((0.0,) * 7),) * 3
        latent_winner = np.zeros((3, 7), dtype=np.float64)
        latent_winner[0, 0] = -0.002
        latent_winner[0, 6] = -0.1

        def score(candidates: np.ndarray) -> np.ndarray:
            return np.square(candidates - latent_winner[None, :, :]).sum(
                axis=(1, 2)
            )

        calibration = ActionResponseCalibration.fit(
            tuple(
                ActionResponseTrial(
                    f"trial-{index}",
                    index + 1,
                    DroidAction(
                        (
                            *(0.002 if axis == index else 0.0 for axis in range(3)),
                            *(0.004 if axis == index else 0.0 for axis in range(3)),
                            0.2,
                        )
                    ),
                    DroidAction(
                        (
                            *(0.001 if axis == index else 0.0 for axis in range(3)),
                            *(0.001 if axis == index else 0.0 for axis in range(3)),
                            0.05,
                        )
                    ),
                )
                for index in range(3)
            )
        )
        task_progress = TaskProgressObjective(
            DroidPose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25)),
            DroidPose((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5)),
            calibration,
        )

        evidence = plan_shadow_candidates(
            observation_id=92,
            direct_actions=direct,
            score=score,
            proposal=Path("/tmp/proposal.pth"),
            adapter=Path("/tmp/adapter.pth"),
            config=ShadowSearchConfig(
                planner=CEMConfig(
                    iterations=5,
                    samples=300,
                    elites=20,
                    seed=9,
                ),
            ),
            task_progress=task_progress,
        )

        self.assertEqual(evidence.planned.task_penalty, 0.0)
        self.assertGreater(evidence.planned.actions[0].values[0], 0.0)
        self.assertGreater(evidence.planned.actions[0].values[6], 0.0)
        payload = evidence.to_dict()
        self.assertTrue(payload["task_progress_assessments"]["planned"]["passed"])
        self.assertEqual(ShadowSearchEvidence.from_dict(payload), evidence)
        claims_tamper = evidence.to_dict()
        claims_tamper["passes_task_progress_gate"] = False
        with self.assertRaisesRegex(ValueError, "claims"):
            ShadowSearchEvidence.from_dict(claims_tamper)
        assessment_tamper = evidence.to_dict()
        assessment_tamper["task_progress_assessments"]["planned"][
            "predicted_reduction"
        ]["translation_meters"] += 1.0
        with self.assertRaisesRegex(ValueError, "assessment"):
            ShadowSearchEvidence.from_dict(assessment_tamper)


if __name__ == "__main__":
    unittest.main()
