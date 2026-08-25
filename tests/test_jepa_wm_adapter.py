from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from jepa_wm.adapter import (
        ActionAdapterContract,
        LoadedActionAdapter,
        action_adapter_parameters,
        apply_action_adapter,
        save_action_adapter,
    )
    from jepa_wm.training_artifact import ArtifactIdentity, TrainingArtifactMetadata
    from jepa_wm.contract import MODEL_ID


def _model() -> SimpleNamespace:
    predictor = SimpleNamespace(action_encoder=torch.nn.Linear(7, 4))
    return SimpleNamespace(model=SimpleNamespace(predictor=predictor))


@unittest.skipUnless(torch is not None, "PyTorch is optional in the local test runtime")
class ActionAdapterTest(unittest.TestCase):
    def test_trains_only_action_dependent_weights(self) -> None:
        model = _model()

        parameters = action_adapter_parameters(model)

        self.assertEqual(parameters, (model.model.predictor.action_encoder.weight,))

    def test_saves_and_applies_only_the_action_encoder(self) -> None:
        source_model = _model()
        metadata = TrainingArtifactMetadata(
            base_model=MODEL_ID,
            source_revision="revision",
            camera="wrist",
            training_recordings=("trajectory-train",),
            training_steps=10,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.pth"
            save_action_adapter(
                source_model,
                path,
                ActionAdapterContract.current(
                    metadata,
                    training_selection_fingerprint=None,
                    training_config_fingerprint="a" * 64,
                ),
            )
            target_model = _model()

            loaded = apply_action_adapter(target_model, path)

        self.assertEqual(loaded, metadata)
        for source, target in zip(
            action_adapter_parameters(source_model),
            action_adapter_parameters(target_model),
        ):
            torch.testing.assert_close(source, target)

    def test_rejects_an_adapter_for_another_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "metadata"):
            TrainingArtifactMetadata(
                base_model="another_model",
                source_revision="revision",
                camera="wrist",
                training_recordings=("trajectory-train",),
                training_steps=10,
            )

    def test_loaded_adapter_applies_the_exact_fingerprinted_bytes(self) -> None:
        source_model = _model()
        metadata = TrainingArtifactMetadata(
            base_model=MODEL_ID,
            source_revision="revision",
            camera="wrist",
            training_recordings=("trajectory-train",),
            training_steps=10,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter.pth"
            save_action_adapter(
                source_model,
                path,
                ActionAdapterContract.current(
                    metadata,
                    training_selection_fingerprint=None,
                    training_config_fingerprint="a" * 64,
                ),
            )
            identity = ArtifactIdentity.from_artifact(path)
            loaded = LoadedActionAdapter.load(path, expected_identity=identity)
            path.write_bytes(b"replaced after loading")
            target_model = _model()

            loaded.apply(target_model)

        for source, target in zip(
            action_adapter_parameters(source_model),
            action_adapter_parameters(target_model),
        ):
            torch.testing.assert_close(source, target)


if __name__ == "__main__":
    unittest.main()
