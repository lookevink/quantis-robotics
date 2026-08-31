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
        BASE_COMMAND_ROUTE,
        NEGATIVE_X_COMMAND_ROUTE,
        POSITIVE_X_COMMAND_ROUTE,
        ActionConditioningContract,
        ActionConditioningKind,
        ActionConditioningSpec,
        LoadedActionConditioning,
        ObservedContextResidualActionEncoder,
        ObservedContextRoutingSpec,
        RuntimeCommandResidualActionEncoder,
        RuntimeCommandRoutingSpec,
        action_conditioning_parameters,
        action_regime_context,
        install_action_conditioning,
        observed_action_context,
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
    @staticmethod
    def _runtime_routing() -> RuntimeCommandRoutingSpec:
        return RuntimeCommandRoutingSpec(
            signed_x_deadband=1e-4,
            translation_activity_deadband=1e-4,
            rotation_activity_deadband=1e-3,
            gripper_activity_deadband=5e-3,
        )

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

    def test_runtime_routes_complete_command_horizons_without_phase_input(self) -> None:
        routing = self._runtime_routing()
        actions = torch.tensor(
            [
                [[-2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                [[2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                [[2e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                [[0.0, 2e-4, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
            ]
        )

        routes, active = routing.classify(actions)

        torch.testing.assert_close(
            routes,
            torch.tensor(
                (
                    NEGATIVE_X_COMMAND_ROUTE,
                    POSITIVE_X_COMMAND_ROUTE,
                    BASE_COMMAND_ROUTE,
                    BASE_COMMAND_ROUTE,
                )
            ),
        )
        torch.testing.assert_close(
            active,
            torch.tensor((True, True, False, True)),
        )

    def test_runtime_residual_preserves_base_for_neutral_and_non_x_motion(self) -> None:
        torch.manual_seed(8)
        model = _model()
        installed = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.RUNTIME_COMMAND_RESIDUAL,
                runtime_routing=self._runtime_routing(),
            ),
        )
        self.assertIsInstance(installed, RuntimeCommandResidualActionEncoder)
        with torch.no_grad():
            installed.residuals[0].weight.fill_(1.0)
            installed.residuals[1].weight.fill_(-1.0)
        actions = torch.tensor(
            [
                [[-2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                [[2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
                [[0.0, 2e-4, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3,
            ]
        )
        base = installed.base(actions)

        routed = installed(actions)

        self.assertTrue(torch.all(routed[0] < base[0]))
        self.assertTrue(torch.all(routed[1] < base[1]))
        torch.testing.assert_close(routed[2], base[2], rtol=0.0, atol=0.0)
        torch.testing.assert_close(routed[3], base[3], rtol=0.0, atol=0.0)

    def test_runtime_router_trains_only_residuals_and_starts_at_exact_base(self) -> None:
        torch.manual_seed(9)
        model = _model()
        actions = torch.randn(2, 3, 7)
        expected = model.model.predictor.action_encoder(actions)
        installed = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.RUNTIME_COMMAND_RESIDUAL,
                runtime_routing=self._runtime_routing(),
            ),
        )

        torch.testing.assert_close(installed(actions), expected, rtol=0.0, atol=0.0)
        parameters = action_conditioning_parameters(model)
        self.assertFalse(any(parameter is installed.base.weight for parameter in parameters))
        self.assertFalse(any(parameter is installed.base.bias for parameter in parameters))
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 56)

    def test_runtime_router_rejects_non_horizon_action_shapes(self) -> None:
        routing = self._runtime_routing()

        with self.assertRaisesRegex(ValueError, r"\[batch, horizon, 7\]"):
            routing.classify(torch.zeros(3, 7))

    def test_observed_context_weights_are_continuous_at_the_deadband(self) -> None:
        routing = ObservedContextRoutingSpec(
            signed_x_deadband=1e-4,
            signed_x_transition_width=1e-4,
        )
        observed = torch.zeros((7, 7))
        observed[:, 0] = torch.tensor(
            (-0.00005, -0.00010, -0.00015, -0.00030, 0.00010, 0.00015, 0.00030)
        )

        weights = routing.route_weights(observed)

        torch.testing.assert_close(
            weights,
            torch.tensor(
                (
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.5, 0.0),
                    (1.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.5),
                    (0.0, 1.0),
                )
            ),
        )

    def test_observed_context_router_is_candidate_invariant_and_scoped(self) -> None:
        torch.manual_seed(10)
        model = _model()
        actions = torch.stack((torch.ones((3, 7)), -torch.ones((3, 7))))
        expected = model.model.predictor.action_encoder(actions)
        installed = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.OBSERVED_CONTEXT_RESIDUAL,
                observed_context_routing=ObservedContextRoutingSpec(
                    signed_x_deadband=1e-4,
                    signed_x_transition_width=1e-4,
                ),
            ),
        )
        self.assertIsInstance(installed, ObservedContextResidualActionEncoder)
        self.assertIs(
            installed.residual_for_route(NEGATIVE_X_COMMAND_ROUTE),
            installed.residuals[0],
        )
        self.assertIs(
            installed.residual_for_route(POSITIVE_X_COMMAND_ROUTE),
            installed.residuals[1],
        )
        with self.assertRaisesRegex(ValueError, "negative or positive"):
            installed.residual_for_route(BASE_COMMAND_ROUTE)
        with torch.no_grad():
            installed.residuals[0].weight.fill_(1.0)
            installed.residuals[1].weight.fill_(-1.0)

        with self.assertRaisesRegex(ValueError, "observed action context"):
            installed(actions)
        with observed_action_context(
            model,
            torch.tensor(((-0.00030, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),) * 2),
        ):
            routed = installed(actions)

        base = installed.base(actions)
        torch.testing.assert_close(routed[0] - base[0], torch.full((3, 4), 7.0))
        torch.testing.assert_close(routed[1] - base[1], torch.full((3, 4), -7.0))
        with self.assertRaisesRegex(ValueError, "observed action context"):
            installed(actions)

        with observed_action_context(model, torch.zeros((2, 7))):
            neutral = installed(actions)
        torch.testing.assert_close(neutral, base, rtol=0.0, atol=0.0)

        parameters = action_conditioning_parameters(model)
        self.assertFalse(any(parameter is installed.base.weight for parameter in parameters))
        self.assertFalse(any(parameter is installed.base.bias for parameter in parameters))
        self.assertEqual(sum(parameter.numel() for parameter in parameters), 56)
        self.assertFalse(torch.equal(expected, routed))

    def test_observed_context_artifact_round_trips_exact_family(self) -> None:
        source = _model()
        spec = ActionConditioningSpec(
            ActionConditioningKind.OBSERVED_CONTEXT_RESIDUAL,
            observed_context_routing=ObservedContextRoutingSpec(1e-4, 1e-4),
        )
        installed = install_action_conditioning(source, spec)
        with torch.no_grad():
            installed.residuals[0].weight.fill_(0.25)
            installed.residuals[1].weight.fill_(-0.25)
        contract = ActionConditioningContract(
            ACTION_CONDITIONING_SCHEMA,
            _metadata(),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            spec,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "observed-context.pth"
            save_action_conditioning(source, path, contract)
            loaded = LoadedActionConditioning.load(path)
            target = _model()

            loaded.apply(target, expected_source_revision="revision")

        self.assertEqual(loaded.contract.spec, spec)
        self.assertIsInstance(
            target.model.predictor.action_encoder,
            ObservedContextResidualActionEncoder,
        )
        for expected_value, actual_value in zip(
            installed.state_dict().values(),
            target.model.predictor.action_encoder.state_dict().values(),
        ):
            torch.testing.assert_close(actual_value, expected_value)

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
