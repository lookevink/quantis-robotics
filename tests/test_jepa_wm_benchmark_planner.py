from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.benchmark_planner import (
        AdapterBenchmarkArtifact,
        BenchmarkInitialization,
        EffectiveBenchmarkIdentity,
        PROPOSAL_REFINEMENT_PRIOR,
        ProposalBenchmarkArtifact,
        _prior_output_token,
        validate_benchmark_recording,
    )
    from jepa_wm.action import ActionSelectionBounds
    from jepa_wm.action_prior import ActionPriorConfig
    from jepa_wm.domain_recording import DomainRecording
    from jepa_wm.planner import CEMConfig, PlannerActionBounds
    from jepa_wm.planner_report import PlannerInitialization
    from jepa_wm.trajectory import RolloutWindow
    from jepa_wm.training_artifact import (
        ArtifactIdentity,
        TrainingArtifactIdentity,
        TrainingArtifactMetadata,
        artifact_fingerprint,
    )
    from sim.exploration import DatasetSplit


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class PlannerBenchmarkProvenanceTest(unittest.TestCase):
    @staticmethod
    def _recording(root: Path, name: str, split: str, seed: int) -> Path:
        recording = root / name
        recording.mkdir()
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                    "recording_id": name,
                    "metadata": {
                        "dataset": "jepa_wm_domain_v1",
                        "split": split,
                        "seed": seed,
                    },
                }
            )
        )
        return recording

    @staticmethod
    def _metadata(*recordings: str) -> TrainingArtifactMetadata:
        return TrainingArtifactMetadata(
            base_model="jepa_wm_droid",
            source_revision="revision",
            camera="wrist",
            training_recordings=recordings,
            training_steps=100,
        )

    def _identity(self, name: str, marker: str = "a") -> TrainingArtifactIdentity:
        return TrainingArtifactIdentity(
            Path("/tmp") / name,
            marker * 64,
            self._metadata("train-a"),
        )

    def test_training_calibration_requires_membership_in_both_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(Path(temp_dir), "train-a", "train", 10)

            validated = validate_benchmark_recording(
                recording,
                expected_split=DatasetSplit.TRAIN,
                adapter_metadata=self._metadata("train-a"),
                proposal_metadata=self._metadata("train-a"),
            )

            self.assertEqual(validated.seed, 10)
            with self.assertRaisesRegex(ValueError, "every model artifact"):
                validate_benchmark_recording(
                    recording,
                    expected_split=DatasetSplit.TRAIN,
                    adapter_metadata=self._metadata("train-a"),
                    proposal_metadata=self._metadata("train-b"),
                )
            with self.assertRaisesRegex(ValueError, "requires a proposal"):
                validate_benchmark_recording(
                    recording,
                    expected_split=DatasetSplit.TRAIN,
                    adapter_metadata=self._metadata("train-a"),
                    proposal_metadata=None,
                )

    def test_held_out_evaluation_rejects_training_membership(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = self._recording(
                Path(temp_dir), "held-a", "held_out", 20
            )

            with self.assertRaisesRegex(ValueError, "used for model training"):
                validate_benchmark_recording(
                    recording,
                    expected_split=DatasetSplit.HELD_OUT,
                    adapter_metadata=self._metadata("held-a"),
                    proposal_metadata=self._metadata("train-a"),
                )

    def test_prior_weight_has_a_safe_filename_token(self) -> None:
        self.assertEqual(_prior_output_token(0.001), "0d001")
        self.assertEqual(PROPOSAL_REFINEMENT_PRIOR.penalty_weight, 0.01)

    def test_effective_identity_changes_with_every_execution_contract(self) -> None:
        adapter = AdapterBenchmarkArtifact(self._identity("adapter.pth"))
        initialization = BenchmarkInitialization(
            PlannerInitialization.PROPOSAL,
            ActionPriorConfig(penalty_weight=0.01),
            adapter,
            ProposalBenchmarkArtifact(self._identity("proposal.pth", "b")),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            recording_path = self._recording(
                Path(temp_dir), "held-a", "held_out", 20
            )
            identity = EffectiveBenchmarkIdentity(
                source_revision="revision",
                base_checkpoint=ArtifactIdentity(Path("/tmp/base.pth"), "d" * 64),
                recording=DomainRecording.from_path(
                    recording_path, expected_split=DatasetSplit.HELD_OUT
                ),
                camera="wrist",
                window=RolloutWindow(76, 8, 1),
                selection_bounds=ActionSelectionBounds(),
                planner_bounds=PlannerActionBounds(),
                planner_config=CEMConfig(),
                scoring_batch_size=64,
                initialization=initialization,
            )

            variants = (
                replace(identity, window=RolloutWindow(76, 8, 2)),
                replace(identity, planner_config=CEMConfig(seed=235)),
                replace(
                    identity,
                    base_checkpoint=ArtifactIdentity(
                        Path("/tmp/base.pth"), "e" * 64
                    ),
                ),
                replace(
                    identity,
                    initialization=replace(
                        initialization,
                        adapter=AdapterBenchmarkArtifact(
                            self._identity("adapter.pth", "c")
                        ),
                    ),
                ),
                replace(
                    identity,
                    initialization=replace(
                        initialization,
                        prior=ActionPriorConfig(
                            minimum_translation_std=0.002,
                            penalty_weight=0.01,
                        ),
                    ),
                ),
            )
            self.assertEqual(len(identity.fingerprint), 64)
            self.assertTrue(
                all(variant.fingerprint != identity.fingerprint for variant in variants)
            )

    def test_initialization_rejects_contradictory_variant(self) -> None:
        with self.assertRaisesRegex(ValueError, "kind and proposal disagree"):
            BenchmarkInitialization(
                kind=PlannerInitialization.PROPOSAL,
                prior=ActionPriorConfig(),
                adapter=AdapterBenchmarkArtifact(self._identity("adapter.pth")),
            )

    def test_training_identity_rejects_sidecar_fingerprint_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact = Path(temp_dir) / "adapter.pth"
            artifact.write_bytes(b"weights")
            artifact.with_suffix(".pth.json").write_text(
                json.dumps(
                    {
                        "adapter_fingerprint": "0" * 64,
                        "metadata": self._metadata("train-a").to_dict(),
                    }
                )
            )
            self.assertNotEqual(artifact_fingerprint(artifact), "0" * 64)
            with self.assertRaisesRegex(ValueError, "does not match"):
                TrainingArtifactIdentity.from_artifact(
                    artifact, fingerprint_field="adapter_fingerprint"
                )


if __name__ == "__main__":
    unittest.main()
