from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

from jepa_wm.action import DroidAction
from jepa_wm.action_prior import ActionPriorConfig, EmpiricalActionPrior
from jepa_wm.insertion_planner import (
    INSERTION_DENSE_PLANNER_PROFILE,
    INSERTION_SAMPLED_READINESS_PLANNER_PROFILE,
    insertion_planner_profile,
)
from jepa_wm.insertion_planner_profile import InsertionPlannerProfileName
from jepa_wm.planner import PlannerActionBounds, ProposalCenteredBounds
from jepa_wm.planner_readiness import FirstActionGate, FirstActionReason
from jepa_wm.planner_policy import (
    GoalActionAlignment,
    RefinementRejectionReason,
)
from jepa_wm.planner_objective import (
    CandidateObjective,
    evaluate_planner_objective,
)
from jepa_wm.planner_report import (
    CandidateEvaluation,
    PlannerInitialization,
    PlannerRolloutEvaluation,
)
from jepa_wm.task_proposal_readiness import TaskProposalArtifactEvidence
from jepa_wm.training_artifact import ArtifactIdentity, TrainingArtifactMetadata

if torch is not None:
    from jepa_wm.adapter import ActionAdapterContract
    from jepa_wm.candidate_negatives import CandidateMiningConfig
    from jepa_wm.insertion_planner_benchmark import (
        validate_insertion_benchmark_inputs,
    )
    from jepa_wm.insertion_wm_readiness import InsertionAdapterEvidence


class InsertionPlannerProfileTest(unittest.TestCase):
    @staticmethod
    def _scores(energy: float) -> CandidateObjective:
        return CandidateObjective(energy, 0.0, 0.0)

    @staticmethod
    def _metadata(
        *,
        revision: str = "revision",
        camera: str = "wrist",
    ) -> TrainingArtifactMetadata:
        return TrainingArtifactMetadata(
            base_model="jepa_wm_droid",
            source_revision=revision,
            camera=camera,
            training_recordings=("insertion-train-00",),
            training_steps=1,
        )

    @classmethod
    def _adapter_evidence(
        cls,
        *,
        revision: str = "revision",
        camera: str = "wrist",
    ):
        metadata = cls._metadata(revision=revision, camera=camera)
        return InsertionAdapterEvidence(
            ArtifactIdentity(Path("/adapter.pth"), "b" * 64),
            ActionAdapterContract.current(
                metadata,
                training_selection_fingerprint=None,
                training_config_fingerprint="c" * 64,
            ),
            CandidateMiningConfig(),
        )

    def test_samples_the_insertion_stroke_with_a_pinned_search_identity(self) -> None:
        profile = INSERTION_SAMPLED_READINESS_PLANNER_PROFILE

        self.assertEqual(profile.window.context_indices, tuple(range(44, 108, 8)))
        self.assertEqual(profile.planner.iterations, 4)
        self.assertEqual(profile.planner.samples, 64)
        self.assertEqual(profile.planner.elites, 8)
        self.assertEqual(profile.prior.penalty_weight, 1e-5)
        self.assertEqual(
            profile.task_policy.context_matched_candidates.candidates_per_context,
            12,
        )

    def test_dense_profile_covers_every_insertion_command_context(self) -> None:
        self.assertEqual(
            INSERTION_DENSE_PLANNER_PROFILE.window.context_indices,
            tuple(range(43, 107)),
        )
        self.assertEqual(
            INSERTION_DENSE_PLANNER_PROFILE.task_policy,
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy,
        )
        self.assertEqual(
            INSERTION_DENSE_PLANNER_PROFILE.planner,
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.planner,
        )
        self.assertEqual(
            INSERTION_DENSE_PLANNER_PROFILE.name.descriptor.report_suffix,
            "insertion_dense_planner_readiness",
        )
        self.assertIs(
            insertion_planner_profile(
                InsertionPlannerProfileName.DENSE_EXECUTION.value
            ),
            INSERTION_DENSE_PLANNER_PROFILE,
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
    def test_context_candidates_are_exact_context_and_proposal_bounded(self) -> None:
        from types import SimpleNamespace

        from jepa_wm.benchmark_planner import context_matched_candidates

        center = np.zeros((3, 7), dtype=np.float64)
        bounds = ProposalCenteredBounds(
            center,
            PlannerActionBounds(),
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy.proposal_trust_region,
        )
        matching = SimpleNamespace(
            context=(SimpleNamespace(index=44),),
            actions=(
                DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                DroidAction((0.0,) * 7),
                DroidAction((0.0,) * 7),
            ),
        )
        other = SimpleNamespace(
            context=(SimpleNamespace(index=52),),
            actions=(DroidAction((0.0,) * 7),) * 3,
        )

        candidates = context_matched_candidates(
            (matching, other),
            44,
            bounds,
            expected_count=1,
        )

        self.assertEqual(candidates.shape, (1, 3, 7))
        self.assertAlmostEqual(np.linalg.norm(candidates[0, 0, :3]), 0.001)
        with self.assertRaisesRegex(ValueError, "cover the training corpus"):
            context_matched_candidates(
                (matching, other),
                44,
                bounds,
                expected_count=2,
            )

    def test_treats_submillimeter_insertion_as_active_motion(self) -> None:
        gate = FirstActionGate(
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy.first_action_thresholds
        )
        recorded = DroidAction((6.3e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

        aligned = gate.evaluate(
            recorded,
            DroidAction((5.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        opposite = gate.evaluate(
            recorded,
            DroidAction((-5.0e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )

        self.assertTrue(aligned.recorded_action_is_active)
        self.assertTrue(aligned.passed)
        self.assertFalse(opposite.passed)
        self.assertEqual(opposite.reasons, (FirstActionReason.DIRECTION_MISMATCH,))

    def test_goal_alignment_penalizes_latent_shortcuts(self) -> None:
        policy = GoalActionAlignment(
            minimum_cosine=0.95,
            failure_penalty=0.01,
        )
        goal = DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        candidates = np.zeros((2, 3, 7), dtype=np.float64)
        candidates[0, 0, 0] = 0.0005
        candidates[1, 0, 1] = 0.0005

        penalties = policy.penalty(candidates, goal)

        self.assertEqual(penalties[0], 0.0)
        self.assertGreaterEqual(penalties[1], 0.01)
        self.assertTrue(policy.evaluate(DroidAction(tuple(candidates[0, 0])), goal).passed)
        self.assertFalse(policy.evaluate(DroidAction(tuple(candidates[1, 0])), goal).passed)

    def test_objective_components_have_one_total_owner(self) -> None:
        candidates = np.zeros((2, 3, 7), dtype=np.float64)
        prior = EmpiricalActionPrior.fit(
            candidates[:1],
            ActionPriorConfig(),
        )

        components = evaluate_planner_objective(
            candidates,
            lambda _: np.asarray((0.1, 0.2)),
            prior,
            lambda _: np.asarray((0.01, 0.0)),
        )

        self.assertAlmostEqual(components.candidate(0).total, 0.11)
        self.assertEqual(components.candidate(1), CandidateObjective(0.2, 0.0, 0.0))

    def test_persisted_rollout_decision_uses_the_insertion_gate(self) -> None:
        recorded = np.zeros((3, 7), dtype=np.float64)
        planned = np.zeros((3, 7), dtype=np.float64)
        recorded[0, 0] = 6.3e-5
        planned[0, 0] = -5.0e-5
        evaluation = PlannerRolloutEvaluation(
            context_index=44,
            target_index=47,
            recorded_actions=recorded,
            recorded_energy=0.01,
            zero_energy=0.011,
            initialization=PlannerInitialization.PROPOSAL,
            initial_candidate=CandidateEvaluation(recorded, self._scores(0.01)),
            searched_candidate=CandidateEvaluation(planned, self._scores(0.009)),
            goal_action=DroidAction(tuple(recorded[0])),
        )

        payload = evaluation.to_dict(
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy
        )

        self.assertFalse(payload["searched_first_action_gate"]["passed"])
        self.assertEqual(
            payload["searched_first_action_gate"]["reasons"],
            ["direction_mismatch"],
        )

    def test_persists_reconstructible_goal_alignment(self) -> None:
        recorded = np.zeros((3, 7), dtype=np.float64)
        planned = np.zeros((3, 7), dtype=np.float64)
        planned[0, 0] = 0.0005
        goal = DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        evaluation = PlannerRolloutEvaluation(
            context_index=44,
            target_index=47,
            recorded_actions=recorded,
            recorded_energy=0.01,
            zero_energy=0.011,
            initialization=PlannerInitialization.PROPOSAL,
            initial_candidate=CandidateEvaluation(recorded, self._scores(0.01)),
            searched_candidate=CandidateEvaluation(planned, self._scores(0.009)),
            goal_action=goal,
        )

        payload = evaluation.to_dict(
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy
        )

        self.assertEqual(payload["goal_action"], list(goal.values))
        self.assertEqual(
            payload["searched_goal_action_alignment"],
            {"cosine": 1.0, "passed": True},
        )

    def test_refinement_requires_goal_alignment_and_latent_improvement(self) -> None:
        policy = INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy
        acceptance = policy.refinement_acceptance
        self.assertIsNotNone(acceptance)
        self.assertEqual(acceptance.to_dict()["unaligned_initial"], "blocked")

        aligned = policy.goal_action_alignment.evaluate(
            DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )
        misaligned = policy.goal_action_alignment.evaluate(
            DroidAction((0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )

        self.assertTrue(acceptance.evaluate(0.01, 0.009, aligned).accepted)
        self.assertEqual(
            acceptance.evaluate(0.01, 0.011, aligned).reasons,
            (RefinementRejectionReason.INSUFFICIENT_LATENT_IMPROVEMENT,),
        )
        self.assertEqual(
            acceptance.evaluate(0.01, 0.009, misaligned).reasons,
            (RefinementRejectionReason.GOAL_MISALIGNED,),
        )

    def test_rejected_refinement_falls_back_to_initial_candidate(self) -> None:
        initial = np.zeros((3, 7), dtype=np.float64)
        searched = np.zeros((3, 7), dtype=np.float64)
        initial[0, 0] = 0.0005
        searched[0, 1] = 0.0005
        goal = DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        evaluation = PlannerRolloutEvaluation(
            context_index=44,
            target_index=47,
            recorded_actions=initial,
            recorded_energy=0.01,
            zero_energy=0.011,
            initialization=PlannerInitialization.PROPOSAL,
            initial_candidate=CandidateEvaluation(initial, self._scores(0.01)),
            searched_candidate=CandidateEvaluation(searched, self._scores(0.009)),
            goal_action=goal,
        )

        payload = evaluation.to_dict(
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy
        )

        self.assertFalse(payload["refinement_acceptance"]["accepted"])
        self.assertEqual(
            payload["refinement_acceptance"]["reasons"], ["goal_misaligned"]
        )
        self.assertEqual(payload["selected_source"], "initial")
        self.assertEqual(payload["selected_actions"], initial.tolist())

    def test_blocks_when_search_and_initial_candidate_are_goal_misaligned(self) -> None:
        initial = np.zeros((3, 7), dtype=np.float64)
        searched = np.zeros((3, 7), dtype=np.float64)
        initial[0, 1] = 0.0005
        searched[0, 1] = 0.0004
        evaluation = PlannerRolloutEvaluation(
            context_index=44,
            target_index=47,
            recorded_actions=np.array(initial, copy=True),
            recorded_energy=0.01,
            zero_energy=0.011,
            initialization=PlannerInitialization.PROPOSAL,
            initial_candidate=CandidateEvaluation(initial, self._scores(0.01)),
            searched_candidate=CandidateEvaluation(searched, self._scores(0.009)),
            goal_action=DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        )

        payload = evaluation.to_dict(
            INSERTION_SAMPLED_READINESS_PLANNER_PROFILE.task_policy
        )

        self.assertEqual(payload["selection_status"], "blocked")
        self.assertIsNone(payload["selected_source"])
        self.assertIsNone(payload["selected_actions"])
        self.assertIsNone(payload["selected_energy"])
        self.assertFalse(payload["initial_goal_action_alignment"]["passed"])

    @unittest.skipIf(torch is None, "PyTorch is not installed locally")
    def test_rejects_adapter_and_proposal_from_different_corpora(self) -> None:
        with (
            patch(
                "jepa_wm.insertion_planner_benchmark.ContactInsertionEvidence.from_recording"
            ),
            patch(
                "jepa_wm.insertion_planner_benchmark.validate_insertion_adapter",
                return_value=self._adapter_evidence(),
            ),
            patch(
                "jepa_wm.insertion_planner_benchmark.validate_insertion_proposal",
                return_value=TaskProposalArtifactEvidence(
                    ArtifactIdentity(Path("/proposal.pth"), "a" * 64),
                    self._metadata(revision="other-revision"),
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "one wrist-camera training corpus"):
                validate_insertion_benchmark_inputs(
                    Path("held-out"), Path("adapter.pth"), Path("proposal.pth")
                )

    @unittest.skipIf(torch is None, "PyTorch is not installed locally")
    def test_rejects_matching_non_wrist_artifacts(self) -> None:
        metadata = self._metadata(camera="presentation")
        with (
            patch(
                "jepa_wm.insertion_planner_benchmark.ContactInsertionEvidence.from_recording"
            ),
            patch(
                "jepa_wm.insertion_planner_benchmark.validate_insertion_adapter",
                return_value=self._adapter_evidence(camera="presentation"),
            ),
            patch(
                "jepa_wm.insertion_planner_benchmark.validate_insertion_proposal",
                return_value=TaskProposalArtifactEvidence(
                    ArtifactIdentity(Path("/proposal.pth"), "a" * 64), metadata
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "wrist-camera"):
                validate_insertion_benchmark_inputs(
                    Path("held-out"), Path("adapter.pth"), Path("proposal.pth")
                )

    @unittest.skipIf(torch is None, "PyTorch is not installed locally")
    def test_accepts_matching_typed_adapter_and_proposal_evidence(self) -> None:
        metadata = self._metadata()
        with (
            patch(
                "jepa_wm.insertion_planner_benchmark.ContactInsertionEvidence.from_recording"
            ),
            patch(
                "jepa_wm.insertion_planner_benchmark.validate_insertion_adapter",
                return_value=self._adapter_evidence(),
            ),
            patch(
                "jepa_wm.insertion_planner_benchmark.validate_insertion_proposal",
                return_value=TaskProposalArtifactEvidence(
                    ArtifactIdentity(Path("/proposal.pth"), "a" * 64), metadata
                ),
            ),
        ):
            validate_insertion_benchmark_inputs(
                Path("held-out"), Path("adapter.pth"), Path("proposal.pth")
            )


if __name__ == "__main__":
    unittest.main()
