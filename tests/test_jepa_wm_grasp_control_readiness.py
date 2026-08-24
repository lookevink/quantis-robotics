from __future__ import annotations

from hashlib import sha256
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jepa_wm.control_baselines import RealizedBaselineReport
from jepa_wm.grasp_control_readiness import (
    GraspControlReadinessEvidence,
    GraspControlReadinessSummary,
)
from jepa_wm.grasp_task import ReachAndGraspDecision, ReachAndGraspFailure
from jepa_wm.training_artifact import ArtifactIdentity


class GraspControlReadinessSummaryTest(unittest.TestCase):
    @staticmethod
    def _decision(*, passed: bool, displacement: float = 0.05) -> ReachAndGraspDecision:
        return ReachAndGraspDecision(
            1 if passed else None,
            8 if passed else 0,
            displacement if passed else 0.0,
            () if passed else (ReachAndGraspFailure.NO_ATTACHMENT_TRANSITION,),
        )

    @classmethod
    def _evidence(
        cls,
        seed: int,
        *,
        direct_passed: bool = True,
        zero_passed: bool = False,
        scripted_passed: bool = True,
        fingerprint: str = "a" * 64,
        pose_gate_passed: bool = False,
    ) -> GraspControlReadinessEvidence:
        def rollout(role: str, decision: ReachAndGraspDecision):
            return SimpleNamespace(
                rollout_id=f"{role}-{seed}",
                reach_and_grasp=decision,
                reference_task="reach_and_grasp",
                complete_steps=(
                    SimpleNamespace(
                        response=SimpleNamespace(
                            proposal_fingerprint=(
                                fingerprint if role == "direct" else None
                            )
                        )
                    ),
                ),
            )

        report = SimpleNamespace(
            experiment_id=f"grasp-baseline-{seed}",
            reference_recording=f"grasp-held-{seed}",
            seed=seed,
            direct=rollout("direct", cls._decision(passed=direct_passed)),
            zero=rollout("zero", cls._decision(passed=zero_passed)),
            scripted=rollout(
                "scripted",
                cls._decision(passed=scripted_passed, displacement=0.07),
            ),
            comparison=SimpleNamespace(
                direct_baseline_gate_passed=pose_gate_passed
            ),
        )
        return GraspControlReadinessEvidence(
            report,
            ArtifactIdentity(Path("/tmp/proposal.pth"), fingerprint),
        )

    def test_two_task_successes_authorize_filming_without_pose_gate_promotion(self) -> None:
        summary = GraspControlReadinessSummary.from_evidence(
            (self._evidence(12400), self._evidence(12401))
        )

        self.assertEqual(summary.whole_seed_count, 2)
        self.assertEqual(summary.task_pass_count, 2)
        self.assertTrue(summary.filming_readiness_passed)
        self.assertFalse(summary.production_authority_granted)
        self.assertFalse(
            summary.to_dict()["trials"][0]["generic_pose_baseline_gate_passed"]
        )

    def test_zero_or_scripted_task_outcome_must_validate_the_comparison(self) -> None:
        zero_succeeds = GraspControlReadinessSummary.from_evidence(
            (self._evidence(12400, zero_passed=True), self._evidence(12401))
        )
        scripted_fails = GraspControlReadinessSummary.from_evidence(
            (self._evidence(12400, scripted_passed=False), self._evidence(12401))
        )

        self.assertFalse(zero_succeeds.filming_readiness_passed)
        self.assertFalse(scripted_fails.filming_readiness_passed)

    def test_requires_unique_whole_seeds(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique evaluation seeds"):
            GraspControlReadinessSummary.from_evidence(
                (self._evidence(12400), self._evidence(12400))
            )

    def test_requires_one_exact_proposal_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical proposal artifact"):
            GraspControlReadinessSummary.from_evidence(
                (
                    self._evidence(12400),
                    self._evidence(12401, fingerprint="b" * 64),
                )
            )

    def test_rejects_non_grasp_task_or_unbound_proposal(self) -> None:
        evidence = self._evidence(12400)
        evidence.report.direct.reference_task = None
        with self.assertRaisesRegex(ValueError, "canonical grasp task"):
            GraspControlReadinessEvidence(evidence.report, evidence.proposal)

        evidence = self._evidence(12400)
        evidence.report.direct.complete_steps[0].response.proposal_fingerprint = None
        with self.assertRaisesRegex(ValueError, "proposal fingerprint"):
            GraspControlReadinessEvidence(evidence.report, evidence.proposal)

    def test_persisted_evidence_rejects_checkpoint_replaced_at_same_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            proposal = root / "proposal.pth"
            proposal.write_bytes(b"proposal-v1")
            fingerprint = sha256(proposal.read_bytes()).hexdigest()
            proposal.with_suffix(".pth.json").write_text(
                json.dumps(
                    {
                        "proposal_fingerprint": fingerprint,
                        "metadata": {
                            "base_model": "jepa_wm_droid",
                            "source_revision": "test-revision",
                            "camera": "wrist",
                            "training_recordings": ["train-1"],
                            "training_steps": 1,
                        },
                    }
                )
            )
            evidence = self._evidence(12400, fingerprint=fingerprint)
            evidence.report.direct.proposal = proposal

            with patch.object(
                RealizedBaselineReport,
                "load_persisted",
                return_value=evidence.report,
            ):
                loaded = GraspControlReadinessEvidence.from_persisted(
                    root,
                    evidence.report.experiment_id,
                )
                self.assertEqual(loaded.proposal.fingerprint, fingerprint)

                proposal.write_bytes(b"proposal-v2")
                with self.assertRaisesRegex(
                    ValueError,
                    "proposal_fingerprint does not match",
                ):
                    GraspControlReadinessEvidence.from_persisted(
                        root,
                        evidence.report.experiment_id,
                    )


if __name__ == "__main__":
    unittest.main()
