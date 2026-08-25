from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action import DroidAction
    from jepa_wm.adapt_recording import (
        AdaptationConfig,
        ContrastiveTermConfig,
        ShuffledEpochSampler,
        TRAINING_BOUNDS,
        mismatched_negative_candidates,
        validated_training_recordings,
    )
    from jepa_wm.control_protocol import TaskContextIndex
    from jepa_wm.rollout_training import RolloutTrainingSelection
    from jepa_wm.trajectory import RolloutWindow
    from jepa_wm.training_artifact import ArtifactIdentity


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class MismatchedNegativeCandidatesTest(unittest.TestCase):
    def test_serializes_exact_initial_adapter_identity(self) -> None:
        identity = ArtifactIdentity(Path("/tmp/generic.pth"), "a" * 64)

        payload = AdaptationConfig(initial_adapter=identity).to_dict()

        self.assertEqual(payload["initial_adapter"], identity.to_dict())

    def test_shuffled_epoch_sampler_covers_every_rollout_before_reuse(self) -> None:
        first = ShuffledEpochSampler(5, 1, 17)
        second = ShuffledEpochSampler(5, 1, 17)

        first_epoch = [int(first.next_indices().item()) for _ in range(5)]
        replay = [int(second.next_indices().item()) for _ in range(5)]

        self.assertEqual(set(first_epoch), set(range(5)))
        self.assertEqual(replay, first_epoch)
        self.assertEqual(
            first.to_dict(),
            {
                "strategy": "seeded_shuffled_epochs",
                "rollouts_per_epoch": 5,
                "batch_size": 1,
                "seed": 17,
                "samples_drawn": 5,
                "complete_epochs": 1,
            },
        )

    @staticmethod
    def _rollout(context_index: int, translation: float):
        action = DroidAction((translation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        return SimpleNamespace(
            task_context_index=TaskContextIndex(context_index),
            actions=(action, action, action),
        )

    def test_excludes_same_context_and_identical_action_sequences(self) -> None:
        rollouts = (
            self._rollout(69, 0.01),
            self._rollout(69, 0.02),
            self._rollout(70, 0.01),
            self._rollout(71, 0.03),
        )

        candidates = mismatched_negative_candidates(rollouts)

        self.assertEqual(candidates[0], (3,))
        self.assertEqual(candidates[1], (2, 3))

    def test_rejects_a_rollout_without_a_meaningful_negative(self) -> None:
        rollouts = (
            self._rollout(69, 0.01),
            self._rollout(70, 0.01),
        )

        with self.assertRaisesRegex(ValueError, "different-context"):
            mismatched_negative_candidates(rollouts)

    def test_contrastive_term_owns_weight_and_margin(self) -> None:
        term = ContrastiveTermConfig(weight=2.0, margin=0.5)

        loss = term.loss(torch.tensor((1.0,)), torch.tensor((1.25,)))

        torch.testing.assert_close(loss, torch.tensor(0.5))
        self.assertEqual(term.to_dict(), {"weight": 2.0, "margin": 0.5})

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

    def test_accepts_unique_training_domain_recordings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = (
                self._recording(root, "train-a", "train", 10),
                self._recording(root, "train-b", "train", 11),
            )

            recordings = validated_training_recordings(paths)

            self.assertEqual(tuple(item.seed for item in recordings), (10, 11))

    def test_rejects_held_out_or_duplicate_seed_training_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            held_out = self._recording(root, "held", "held_out", 20)
            with self.assertRaisesRegex(ValueError, "expected 'train'"):
                validated_training_recordings((held_out,))

            first = self._recording(root, "train-a", "train", 30)
            second = self._recording(root, "train-b", "train", 30)
            with self.assertRaisesRegex(ValueError, "unique identities and seeds"):
                validated_training_recordings((first, second))

    def test_selects_and_reports_the_exact_training_window_per_recording(self) -> None:
        recordings = (
            SimpleNamespace(name="train-a", path=Path("/tmp/train-a")),
            SimpleNamespace(name="train-b", path=Path("/tmp/train-b")),
        )

        def rollouts_for(recording, **kwargs):
            return tuple(
                SimpleNamespace(context=(SimpleNamespace(index=index),))
                for index in range(109)
            )

        with patch(
            "jepa_wm.rollout_training.load_rollouts",
            side_effect=rollouts_for,
        ):
            selection = RolloutTrainingSelection.load(
                tuple(recording.path for recording in recordings),
                camera="wrist",
                bounds=TRAINING_BOUNDS,
                window=RolloutWindow(21, 88, 1),
            )

        self.assertEqual(len(selection.rollouts), 176)
        self.assertEqual(selection.recordings[0].recording, "train-a")
        self.assertEqual(
            selection.recordings[0].context_indices,
            tuple(range(21, 109)),
        )
        self.assertEqual(
            selection.recordings[1].context_indices,
            tuple(range(21, 109)),
        )


if __name__ == "__main__":
    unittest.main()
