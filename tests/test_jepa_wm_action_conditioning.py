from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_conditioning import (
        ACTION_CONDITIONING_SCHEMA,
        ActionConditioningContract,
        ActionConditioningKind,
        ActionConditioningSpec,
        LoadedActionConditioning,
        action_conditioning_parameters,
        action_regime_context,
        install_action_conditioning,
        save_action_conditioning,
    )
    from jepa_wm.contract import MODEL_ID
    from jepa_wm.training_artifact import (
        ArtifactIdentity,
        TrainingArtifactMetadata,
    )


def _model() -> SimpleNamespace:
    encoder = torch.nn.Linear(7, 4)
    return SimpleNamespace(model=SimpleNamespace(predictor=SimpleNamespace(action_encoder=encoder)))


def _metadata() -> TrainingArtifactMetadata:
    return TrainingArtifactMetadata(
        base_model=MODEL_ID,
        source_revision="revision",
        camera="wrist",
        training_recordings=("train-00",),
        training_steps=10,
    )


@unittest.skipIf(torch is None, "PyTorch is required for action conditioning")
class ActionConditioningTest(unittest.TestCase):
    def test_nonlinear_residual_is_exactly_zero_at_installation(self) -> None:
        torch.manual_seed(3)
        model = _model()
        actions = torch.randn(2, 3, 7)
        expected = model.model.predictor.action_encoder(actions)

        installed = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.NONLINEAR_RESIDUAL,
                hidden_dimension=3,
            ),
        )

        torch.testing.assert_close(installed(actions), expected, rtol=0.0, atol=0.0)
        parameters = action_conditioning_parameters(model)
        self.assertTrue(any(parameter is installed.base.weight for parameter in parameters))
        self.assertFalse(any(parameter is installed.base.bias for parameter in parameters))
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 64)

    def test_oracle_routes_are_explicit_and_scoped(self) -> None:
        model = _model()
        installed = install_action_conditioning(
            model,
            ActionConditioningSpec(ActionConditioningKind.ORACLE_REGIME_RESIDUAL),
        )
        with torch.no_grad():
            installed.residuals[0].weight.fill_(1.0)
            installed.residuals[1].weight.fill_(-1.0)
        actions = torch.ones(2, 1, 7)

        with self.assertRaisesRegex(ValueError, "regime routes"):
            installed(actions)
        with action_regime_context(model, torch.tensor((0, 1))):
            routed = installed(actions)

        torch.testing.assert_close(
            routed[0] - installed.base(actions)[0],
            torch.full((1, 4), 7.0),
        )
        torch.testing.assert_close(
            routed[1] - installed.base(actions)[1],
            torch.full((1, 4), -7.0),
        )
        with self.assertRaisesRegex(ValueError, "regime routes"):
            installed(actions)

    def test_versioned_artifact_round_trips_exact_family(self) -> None:
        source = _model()
        spec = ActionConditioningSpec(
            ActionConditioningKind.NONLINEAR_RESIDUAL,
            hidden_dimension=3,
        )
        installed = install_action_conditioning(source, spec)
        with torch.no_grad():
            installed.residual_out.weight.fill_(0.25)
        contract = ActionConditioningContract(
            ACTION_CONDITIONING_SCHEMA,
            _metadata(),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            spec,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "conditioning.pth"
            save_action_conditioning(source, path, contract)
            identity = ArtifactIdentity.from_artifact(path)
            loaded = LoadedActionConditioning.load(path, expected_identity=identity)
            target = _model()

            metadata = loaded.apply(target, expected_source_revision="revision")

        self.assertEqual(metadata, _metadata())
        self.assertEqual(loaded.contract, contract)
        self.assertEqual(
            target.model.predictor.action_encoder.spec,
            spec,
        )
        for expected, actual in zip(
            source.model.predictor.action_encoder.state_dict().values(),
            target.model.predictor.action_encoder.state_dict().values(),
        ):
            torch.testing.assert_close(actual, expected)

    def test_rejects_changed_artifact_identity_before_loading(self) -> None:
        model = _model()
        spec = ActionConditioningSpec(ActionConditioningKind.GLOBAL_LINEAR)
        install_action_conditioning(model, spec)
        contract = ActionConditioningContract(
            ACTION_CONDITIONING_SCHEMA,
            _metadata(),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            spec,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "conditioning.pth"
            save_action_conditioning(model, path, contract)
            wrong = ArtifactIdentity(path.resolve(), "d" * 64)

            with self.assertRaisesRegex(ValueError, "identity changed"):
                LoadedActionConditioning.load(path, expected_identity=wrong)


if __name__ == "__main__":
    unittest.main()
