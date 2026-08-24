from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

from jepa_wm.action import DroidAction
from jepa_wm.insertion_planner import INSERTION_PLANNER_PROFILE
from jepa_wm.planner_readiness import FirstActionGate, FirstActionReason
from jepa_wm.planner_report import (
    CandidateEvaluation,
    PlannerInitialization,
    PlannerRolloutEvaluation,
)
from jepa_wm.task_proposal_readiness import TaskProposalArtifactEvidence
from jepa_wm.training_artifact import ArtifactIdentity, TrainingArtifactMetadata

if torch is not None:
    from jepa_wm.insertion_planner_benchmark import (
        validate_insertion_benchmark_inputs,
    )


class InsertionPlannerProfileTest(unittest.TestCase):
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

    def test_samples_the_insertion_stroke_with_a_pinned_search_identity(self) -> None:
        profile = INSERTION_PLANNER_PROFILE

        self.assertEqual(profile.window.context_indices, tuple(range(44, 108, 8)))
        self.assertEqual(profile.planner.iterations, 4)
        self.assertEqual(profile.planner.samples, 64)
        self.assertEqual(profile.planner.elites, 8)
        self.assertEqual(profile.prior.penalty_weight, 1e-5)

    def test_treats_submillimeter_insertion_as_active_motion(self) -> None:
        gate = FirstActionGate(
            INSERTION_PLANNER_PROFILE.task_policy.first_action_thresholds
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
            initial_candidate=CandidateEvaluation(recorded, 0.01, 0.01),
            planned_candidate=CandidateEvaluation(planned, 0.009, 0.009),
        )

        payload = evaluation.to_dict(
            FirstActionGate(
                INSERTION_PLANNER_PROFILE.task_policy.first_action_thresholds
            )
        )

        self.assertFalse(payload["first_action_gate"]["passed"])
        self.assertEqual(
            payload["first_action_gate"]["reasons"], ["direction_mismatch"]
        )

    @unittest.skipIf(torch is None, "PyTorch is not installed locally")
    def test_rejects_adapter_and_proposal_from_different_corpora(self) -> None:
        with (
            patch(
                "jepa_wm.insertion_planner_benchmark.ContactInsertionEvidence.from_recording"
            ),
            patch(
                "jepa_wm.insertion_planner_benchmark.validate_insertion_adapter",
                return_value=SimpleNamespace(metadata=self._metadata()),
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
                return_value=SimpleNamespace(metadata=metadata),
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


if __name__ == "__main__":
    unittest.main()
