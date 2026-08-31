from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_conditioning import (
        ActionConditioningKind,
        ActionConditioningSpec,
        CausalContextResidualActionEncoder,
        PhysicalStateResidualActionEncoder,
        ActionConditioningContract,
        LoadedActionConditioning,
        action_conditioning_parameters,
        causal_residual_parameters,
        causal_router_parameters,
        install_action_conditioning,
        physical_residual_parameters,
        physical_router_parameters,
        save_action_conditioning,
    )
    from jepa_wm.causal_route_probe import (
        CausalRouteProbeConfig,
        CausalRouteProbeDataset,
        run_grouped_causal_route_probe,
    )
    from jepa_wm.causal_context_routing_experiment import (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        _load_experiment_config,
        _passthrough_owned_route_activations,
        _selected_corpus_input_fingerprint,
    )
    from jepa_wm.causal_routing import (
        CausalContextRoutingSpec,
        CausalMotionRoute,
        CausalMotionRouter,
    )
    from jepa_wm.causal_scoring import CausalCandidateScorer
    from jepa_wm.contract import MODEL_ID
    from jepa_wm.insertion_layout import ContactInsertionSegment
    from jepa_wm.physical_observation import (
        PHYSICAL_ROUTING_FEATURE_NAMES,
        PHYSICAL_ROUTING_OBSERVATION_SCHEMA,
        PhysicalRoutingObservation,
    )
    from jepa_wm.physical_routing import (
        PhysicalMotionRouter,
        PhysicalStateRoutingSpec,
    )
    from jepa_wm.physical_route_probe import (
        PhysicalRouteProbeConfig,
        PhysicalRouteProbeDataset,
        run_grouped_physical_route_probe,
    )
    from jepa_wm.physical_state_routing_experiment import (
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT as PHYSICAL_CONFIG_FINGERPRINT,
        _labeling_spec as _physical_labeling_spec,
        _load_experiment_config as _load_physical_experiment_config,
        _routing_spec as _physical_experiment_routing_spec,
    )
    from jepa_wm.physical_scoring import PhysicalCandidateScorer
    from jepa_wm.planning_scoring import LatentGoalScorer
    from jepa_wm.training_artifact import TrainingArtifactMetadata


def _routing_spec() -> CausalContextRoutingSpec:
    return CausalContextRoutingSpec(
        context_dimension=4,
        router_hidden_dimension=8,
        signed_x_deadband=1e-4,
        translation_activity_deadband=1e-4,
        rotation_activity_deadband=1e-3,
        gripper_activity_deadband=1e-3,
        minimum_route_confidence=0.75,
        maximum_residual_ratio=0.15,
    )


def _conditioning_model() -> SimpleNamespace:
    encoder = torch.nn.Linear(7, 4, bias=False)
    return SimpleNamespace(
        model=SimpleNamespace(predictor=SimpleNamespace(action_encoder=encoder))
    )


def _physical_routing_spec() -> PhysicalStateRoutingSpec:
    return PhysicalStateRoutingSpec(
        hidden_dimensions=(8, 8),
        minimum_route_confidence=0.75,
        maximum_residual_ratio=0.15,
    )


class _ScoringModel:
    def __init__(self, encoder: torch.nn.Module) -> None:
        self.model = SimpleNamespace(predictor=SimpleNamespace(action_encoder=encoder))

    def unroll(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        encoded = self.model.predictor.action_encoder(actions.transpose(0, 1))
        return encoded.sum(dim=1).unsqueeze(0)


@unittest.skipIf(torch is None, "PyTorch is required for causal routing")
class CausalRoutingTest(unittest.TestCase):
    def test_frozen_probe_manifest_authenticates(self) -> None:
        path = Path(".scratch/jepa-causal-context-routing-v1/experiment-config.json")

        experiment = _load_experiment_config(path)

        self.assertEqual(
            experiment["schema"],
            "quantis.jepa_wm_causal_context_routing_probe.v1",
        )
        self.assertEqual(len(FROZEN_EXPERIMENT_CONFIG_FINGERPRINT), 64)

    def test_future_route_labels_distinguish_hold_and_active_non_x(self) -> None:
        actions = torch.zeros((3, 4, 7))
        actions[:, 1, 0] = -0.001
        actions[:, 2, 0] = 0.001
        actions[:, 3, 6] = 0.0035

        routes = _routing_spec().classify_action_horizons(actions)

        self.assertEqual(
            routes.tolist(),
            [
                CausalMotionRoute.HOLD,
                CausalMotionRoute.RETREAT,
                CausalMotionRoute.ADVANCE,
                CausalMotionRoute.ACTIVE_OTHER,
            ],
        )

    def test_semantic_hold_labels_override_realized_drift(self) -> None:
        actions = torch.zeros((3, 3, 7))
        actions[:, :, 0] = 0.001

        routes = _routing_spec().classify_recorded_horizons(
            actions,
            (
                ContactInsertionSegment.ALIGN,
                ContactInsertionSegment.RETREAT_HOLD,
                ContactInsertionSegment.SEATED_HOLD,
            ),
        )

        self.assertEqual(
            routes.tolist(),
            [
                CausalMotionRoute.ADVANCE,
                CausalMotionRoute.HOLD,
                CausalMotionRoute.HOLD,
            ],
        )

    def test_router_uses_context_and_fails_closed_below_confidence(self) -> None:
        router = CausalMotionRouter(_routing_spec())
        context = torch.zeros((2, 2, 1, 1, 1, 4))
        poses = torch.zeros((2, 7))
        previous = torch.zeros((2, 7))
        with torch.no_grad():
            router.output.weight.zero_()
            router.output.bias.zero_()

        uncertain = router.decide(context, poses, previous)

        self.assertTrue(torch.all(uncertain.failed_closed))
        self.assertEqual(
            uncertain.routes.tolist(),
            [CausalMotionRoute.ACTIVE_OTHER, CausalMotionRoute.ACTIVE_OTHER],
        )
        torch.testing.assert_close(
            uncertain.residual_weights,
            torch.zeros((2, 2)),
            rtol=0.0,
            atol=0.0,
        )

        with torch.no_grad():
            router.output.bias[CausalMotionRoute.RETREAT] = 12.0
        retreat = router.decide(context, poses, previous)

        self.assertFalse(torch.any(retreat.failed_closed))
        self.assertEqual(
            retreat.routes.tolist(),
            [CausalMotionRoute.RETREAT, CausalMotionRoute.RETREAT],
        )
        torch.testing.assert_close(
            retreat.residual_weights,
            torch.tensor(((1.0, 0.0), (1.0, 0.0))),
            rtol=0.0,
            atol=0.0,
        )

    def test_physical_router_normalizes_features_and_fails_closed(self) -> None:
        spec = _physical_routing_spec()
        router = PhysicalMotionRouter(spec)
        features = torch.stack(
            (
                torch.arange(len(PHYSICAL_ROUTING_FEATURE_NAMES)).float(),
                torch.arange(len(PHYSICAL_ROUTING_FEATURE_NAMES)).float() + 2.0,
            )
        )

        router.fit_normalization(features)
        normalized = router.normalized_features(features)
        decision = router.decide(features)

        torch.testing.assert_close(
            normalized.mean(dim=0),
            torch.zeros(len(PHYSICAL_ROUTING_FEATURE_NAMES)),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertTrue(torch.all(decision.failed_closed))
        self.assertEqual(
            decision.routes.tolist(),
            [CausalMotionRoute.ACTIVE_OTHER, CausalMotionRoute.ACTIVE_OTHER],
        )

    def test_physical_router_refuses_unfitted_normalization(self) -> None:
        router = PhysicalMotionRouter(_physical_routing_spec())
        features = torch.zeros((1, len(PHYSICAL_ROUTING_FEATURE_NAMES)))

        with self.assertRaisesRegex(ValueError, "normalization has not been fitted"):
            router.decide(features)

    def test_physical_routing_spec_authenticates_feature_schema(self) -> None:
        payload = _physical_routing_spec().to_dict()

        self.assertEqual(
            payload["feature_names"],
            list(PHYSICAL_ROUTING_FEATURE_NAMES),
        )
        self.assertEqual(
            payload["observation_schema"],
            PHYSICAL_ROUTING_OBSERVATION_SCHEMA,
        )
        self.assertEqual(
            PhysicalStateRoutingSpec.from_dict(payload),
            _physical_routing_spec(),
        )

        payload["feature_names"] = list(reversed(payload["feature_names"]))
        with self.assertRaisesRegex(ValueError, "physical routing specification"):
            PhysicalStateRoutingSpec.from_dict(payload)

        payload = _physical_routing_spec().to_dict()
        payload["observation_schema"] = "quantis.invalid.v1"
        with self.assertRaisesRegex(ValueError, "physical routing specification"):
            PhysicalStateRoutingSpec.from_dict(payload)

    def test_frozen_physical_probe_manifest_authenticates(self) -> None:
        experiment = _load_physical_experiment_config(
            Path(".scratch/jepa-physical-state-routing-v2/experiment-config.json")
        )

        self.assertEqual(
            experiment["schema"],
            "quantis.jepa_wm_physical_state_routing_probe.v1",
        )
        self.assertEqual(len(PHYSICAL_CONFIG_FINGERPRINT), 64)
        self.assertEqual(
            _physical_experiment_routing_spec(experiment),
            PhysicalStateRoutingSpec((64, 64), 0.75, 0.15),
        )
        self.assertEqual(
            _physical_labeling_spec(experiment).gripper_activity_deadband,
            0.005,
        )

    def test_physical_conditioner_enforces_bound_and_exact_passthrough(self) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.PHYSICAL_STATE_RESIDUAL,
                physical_state_routing=_physical_routing_spec(),
            ),
        )
        self.assertIsInstance(encoder, PhysicalStateResidualActionEncoder)
        features = torch.zeros((2, len(PHYSICAL_ROUTING_FEATURE_NAMES)))
        actions = torch.ones((2, 3, 7))
        encoder.router.fit_normalization(features)
        with torch.no_grad():
            encoder.base.weight.fill_(1.0)
            encoder.residuals[0].weight.fill_(20.0)
            encoder.router.output.weight.zero_()
            encoder.router.output.bias.zero_()
            encoder.router.output.bias[CausalMotionRoute.RETREAT] = 12.0

        with encoder.use_physical_observations(features) as decision:
            output = encoder(actions)

        base = encoder.base(actions)
        ratios = torch.linalg.vector_norm(
            output - base, dim=-1
        ) / torch.linalg.vector_norm(base, dim=-1)
        self.assertEqual(
            decision.routes.tolist(),
            [CausalMotionRoute.RETREAT, CausalMotionRoute.RETREAT],
        )
        self.assertLessEqual(float(ratios.max()), 0.150001)

        with torch.no_grad():
            encoder.router.output.bias.zero_()
            encoder.router.output.bias[CausalMotionRoute.HOLD] = 12.0
        with encoder.use_physical_observations(features):
            passthrough = encoder(actions)
        self.assertTrue(torch.equal(passthrough, base))

    def test_physical_training_seams_freeze_base_and_split_router_residuals(
        self,
    ) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.PHYSICAL_STATE_RESIDUAL,
                physical_state_routing=_physical_routing_spec(),
            ),
        )

        router = physical_router_parameters(model)
        residuals = physical_residual_parameters(model)
        combined = action_conditioning_parameters(model)

        self.assertTrue(router)
        self.assertTrue(residuals)
        self.assertEqual(
            {id(parameter) for parameter in router + residuals},
            {id(parameter) for parameter in combined},
        )
        self.assertNotIn(
            id(encoder.base.weight),
            {id(parameter) for parameter in combined},
        )

    def test_physical_scoring_holds_one_route_fixed_for_every_candidate(self) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.PHYSICAL_STATE_RESIDUAL,
                physical_state_routing=_physical_routing_spec(),
            ),
        )
        with torch.no_grad():
            encoder.router.output.weight.zero_()
            encoder.router.output.bias.zero_()
            encoder.router.output.bias[CausalMotionRoute.ADVANCE] = 12.0
        scoring_model = _ScoringModel(encoder)
        scorer = PhysicalCandidateScorer(scoring_model)
        context = torch.zeros((2, 2, 1, 1, 1, 4))
        target = torch.zeros((2, 1, 4))
        features = torch.zeros((2, len(PHYSICAL_ROUTING_FEATURE_NAMES)))
        encoder.router.fit_normalization(features)
        recorded = torch.ones((3, 2, 7))

        scored = scorer.score(
            context,
            target,
            {"recorded": recorded, "zero": torch.zeros_like(recorded)},
            physical_observations=features,
        )

        self.assertEqual(set(scored.energies), {"recorded", "zero"})
        self.assertEqual(
            scored.decision.routes.tolist(),
            [CausalMotionRoute.ADVANCE, CausalMotionRoute.ADVANCE],
        )
        self.assertIsNone(encoder.active_decision)

    def test_live_planner_scopes_one_observed_route_across_candidate_batches(self) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.PHYSICAL_STATE_RESIDUAL,
                physical_state_routing=_physical_routing_spec(),
            ),
        )
        features = torch.zeros((2, len(PHYSICAL_ROUTING_FEATURE_NAMES)))
        encoder.router.fit_normalization(features)
        scoring_model = _ScoringModel(encoder)
        scorer = LatentGoalScorer(
            scoring_model,
            torch.zeros((1, 2, 1, 1, 1, 4)),
            torch.zeros((1, 1, 4)),
            device=torch.device("cpu"),
            batch_size=1,
            physical_routing=PhysicalRoutingObservation(
                tuple(0.0 for _ in PHYSICAL_ROUTING_FEATURE_NAMES)
            ),
        )

        energies = scorer(torch.zeros((2, 3, 7)))

        self.assertEqual(energies.shape, (2,))
        self.assertIsNone(encoder.active_decision)

    def test_physical_artifact_round_trips_router_and_bounded_residuals(self) -> None:
        source = _conditioning_model()
        spec = ActionConditioningSpec(
            ActionConditioningKind.PHYSICAL_STATE_RESIDUAL,
            physical_state_routing=_physical_routing_spec(),
        )
        installed = install_action_conditioning(source, spec)
        installed.router.fit_normalization(
            torch.zeros((2, len(PHYSICAL_ROUTING_FEATURE_NAMES)))
        )
        with torch.no_grad():
            installed.router.feature_mean.fill_(0.5)
            installed.router.output.bias[CausalMotionRoute.RETREAT] = 3.0
            installed.residuals[0].weight.fill_(0.25)
        contract = ActionConditioningContract(
            "quantis.jepa_wm_action_conditioning.v1",
            TrainingArtifactMetadata(
                base_model=MODEL_ID,
                source_revision="revision",
                camera="wrist",
                training_recordings=("train-00",),
                training_steps=10,
            ),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            spec,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "physical.pth"
            save_action_conditioning(source, path, contract)
            loaded = LoadedActionConditioning.load(path)
            target = _conditioning_model()

            loaded.apply(target, expected_source_revision="revision")

        self.assertEqual(loaded.contract, contract)
        for expected, actual in zip(
            installed.state_dict().values(),
            target.model.predictor.action_encoder.state_dict().values(),
        ):
            torch.testing.assert_close(actual, expected)

    def test_conditioner_enforces_hard_residual_ratio(self) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.CAUSAL_CONTEXT_RESIDUAL,
                causal_context_routing=_routing_spec(),
            ),
        )
        self.assertIsInstance(encoder, CausalContextResidualActionEncoder)
        with torch.no_grad():
            encoder.base.weight.fill_(1.0)
            encoder.residuals[0].weight.fill_(20.0)
            encoder.router.output.weight.zero_()
            encoder.router.output.bias.zero_()
            encoder.router.output.bias[CausalMotionRoute.RETREAT] = 12.0
        actions = torch.ones((2, 3, 7))
        context = torch.zeros((2, 2, 1, 1, 1, 4))
        poses = torch.zeros((2, 7))
        previous = torch.zeros((2, 7))

        with encoder.use_causal_context(context, poses, previous) as decision:
            output = encoder(actions)

        base = encoder.base(actions)
        ratios = torch.linalg.vector_norm(
            output - base, dim=-1
        ) / torch.linalg.vector_norm(base, dim=-1)
        self.assertEqual(
            decision.routes.tolist(),
            [CausalMotionRoute.RETREAT, CausalMotionRoute.RETREAT],
        )
        self.assertLessEqual(float(ratios.max()), 0.150001)

    def test_passthrough_routes_preserve_the_base_embedding_exactly(self) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.CAUSAL_CONTEXT_RESIDUAL,
                causal_context_routing=_routing_spec(),
            ),
        )
        actions = torch.ones((2, 3, 7))
        context = torch.zeros((2, 2, 1, 1, 1, 4))
        poses = torch.zeros((2, 7))
        previous = torch.zeros((2, 7))
        with torch.no_grad():
            for residual in encoder.residuals:
                residual.weight.fill_(20.0)
            for route in (CausalMotionRoute.HOLD, CausalMotionRoute.ACTIVE_OTHER):
                encoder.router.output.weight.zero_()
                encoder.router.output.bias.zero_()
                encoder.router.output.bias[route] = 12.0
                with encoder.use_causal_context(context, poses, previous):
                    output = encoder(actions)
                self.assertTrue(torch.equal(output, encoder.base(actions)))

    def test_passthrough_gate_counts_only_residual_activations(self) -> None:
        activations = _passthrough_owned_route_activations(
            {
                "hold": {
                    "predictions": {
                        "hold": 2,
                        "retreat": 1,
                        "advance": 2,
                        "active_other": 3,
                    }
                }
            },
            ("hold",),
        )

        self.assertEqual(activations, {"hold": 3})

    def test_selected_corpus_fingerprint_authenticates_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = Path(temporary_directory) / "train-00"
            recording.mkdir()
            manifest = recording / "manifest.json"
            steps = recording / "steps.jsonl"
            context = recording / "context.png"
            target = recording / "target.png"
            manifest.write_text("{}")
            steps.write_text('{"index": 0}\n')
            context.write_bytes(b"context")
            target.write_bytes(b"target")
            rollout = SimpleNamespace(
                context_paths=(context,),
                target_clip=(target,),
            )
            selection = SimpleNamespace(
                recordings=(
                    SimpleNamespace(recording="train-00", context_indices=(0,)),
                ),
                rollouts=(rollout,),
            )
            before = _selected_corpus_input_fingerprint(
                (recording,),
                selection,
            )

            steps.write_text('{"index": 0, "changed": true}\n')
            after = _selected_corpus_input_fingerprint(
                (recording,),
                selection,
            )

        self.assertNotEqual(before, after)

    def test_training_parameter_seams_freeze_the_base_encoder(self) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.CAUSAL_CONTEXT_RESIDUAL,
                causal_context_routing=_routing_spec(),
            ),
        )

        router = causal_router_parameters(model)
        residuals = causal_residual_parameters(model)
        combined = action_conditioning_parameters(model)

        self.assertTrue(router)
        self.assertTrue(residuals)
        self.assertEqual(
            {id(parameter) for parameter in router + residuals},
            {id(parameter) for parameter in combined},
        )
        self.assertNotIn(
            id(encoder.base.weight),
            {id(parameter) for parameter in combined},
        )

    def test_scoring_seam_holds_one_route_fixed_for_every_candidate(self) -> None:
        model = _conditioning_model()
        encoder = install_action_conditioning(
            model,
            ActionConditioningSpec(
                ActionConditioningKind.CAUSAL_CONTEXT_RESIDUAL,
                causal_context_routing=_routing_spec(),
            ),
        )
        with torch.no_grad():
            encoder.router.output.weight.zero_()
            encoder.router.output.bias.zero_()
            encoder.router.output.bias[CausalMotionRoute.ADVANCE] = 12.0
        scoring_model = _ScoringModel(encoder)
        scorer = CausalCandidateScorer(scoring_model)
        context = torch.zeros((2, 2, 1, 1, 1, 4))
        target = torch.zeros((2, 1, 4))
        poses = torch.zeros((2, 7))
        previous = torch.zeros((2, 7))
        recorded = torch.ones((3, 2, 7))

        scored = scorer.score(
            context,
            target,
            {"recorded": recorded, "zero": torch.zeros_like(recorded)},
            context_poses=poses,
            previous_actions=previous,
        )

        self.assertEqual(set(scored.energies), {"recorded", "zero"})
        self.assertEqual(
            scored.decision.routes.tolist(),
            [CausalMotionRoute.ADVANCE, CausalMotionRoute.ADVANCE],
        )
        self.assertIsNone(encoder.active_decision)

    def test_causal_artifact_round_trips_router_and_bounded_residuals(self) -> None:
        source = _conditioning_model()
        spec = ActionConditioningSpec(
            ActionConditioningKind.CAUSAL_CONTEXT_RESIDUAL,
            causal_context_routing=_routing_spec(),
        )
        installed = install_action_conditioning(source, spec)
        with torch.no_grad():
            installed.router.output.bias[CausalMotionRoute.RETREAT] = 3.0
            installed.residuals[0].weight.fill_(0.25)
        contract = ActionConditioningContract(
            "quantis.jepa_wm_action_conditioning.v1",
            TrainingArtifactMetadata(
                base_model=MODEL_ID,
                source_revision="revision",
                camera="wrist",
                training_recordings=("train-00",),
                training_steps=10,
            ),
            "a" * 64,
            "b" * 64,
            "c" * 64,
            spec,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "causal.pth"
            save_action_conditioning(source, path, contract)
            loaded = LoadedActionConditioning.load(path)
            target = _conditioning_model()

            loaded.apply(target, expected_source_revision="revision")

        self.assertEqual(loaded.contract, contract)
        for expected, actual in zip(
            installed.state_dict().values(),
            target.model.predictor.action_encoder.state_dict().values(),
        ):
            torch.testing.assert_close(actual, expected)

    def test_grouped_probe_predicts_routes_without_candidate_actions(self) -> None:
        spec = CausalContextRoutingSpec(
            context_dimension=4,
            router_hidden_dimension=8,
            signed_x_deadband=1e-4,
            translation_activity_deadband=1e-4,
            rotation_activity_deadband=1e-3,
            gripper_activity_deadband=1e-3,
            minimum_route_confidence=0.55,
            maximum_residual_ratio=0.15,
        )
        labels = torch.tensor((0, 1, 2, 3) * 3)
        contexts = torch.nn.functional.one_hot(labels, num_classes=4).float()
        poses = torch.zeros((12, 7))
        previous = torch.zeros((12, 7))
        future = torch.zeros((3, 12, 7))
        future[:, labels == CausalMotionRoute.RETREAT, 0] = -0.01
        future[:, labels == CausalMotionRoute.ADVANCE, 0] = 0.01
        future[:, labels == CausalMotionRoute.ACTIVE_OTHER, 1] = 0.01
        dataset = CausalRouteProbeDataset.build(
            contexts,
            poses,
            previous,
            future,
            torch.tensor((0,) * 4 + (1,) * 4 + (2,) * 4),
            ("hold", "retreat", "advance", "active_other") * 3,
            routing=spec,
        )

        report = run_grouped_causal_route_probe(
            dataset,
            spec,
            CausalRouteProbeConfig(
                steps=200,
                learning_rate=0.05,
                weight_decay=0.0,
                seed=7,
            ),
            device=torch.device("cpu"),
        )

        self.assertEqual(report["overall"]["accuracy"], 1.0)
        self.assertFalse(report["candidate_actions_used_as_router_inputs"])

    def test_grouped_physical_probe_uses_only_observed_state(self) -> None:
        labels = torch.tensor((0, 1, 2, 3) * 3)
        features = torch.zeros((12, len(PHYSICAL_ROUTING_FEATURE_NAMES)))
        features[:, :4] = torch.nn.functional.one_hot(labels, num_classes=4).float()
        dataset = PhysicalRouteProbeDataset(
            features=features,
            labels=labels,
            groups=torch.tensor((0,) * 4 + (1,) * 4 + (2,) * 4),
            slices=("hold", "retreat", "advance", "active_other") * 3,
        )

        report = run_grouped_physical_route_probe(
            dataset,
            PhysicalStateRoutingSpec(
                hidden_dimensions=(8, 8),
                minimum_route_confidence=0.55,
                maximum_residual_ratio=0.15,
            ),
            PhysicalRouteProbeConfig(
                steps=300,
                learning_rate=0.05,
                weight_decay=0.0,
                seed=17,
            ),
            device=torch.device("cpu"),
        )

        self.assertEqual(report["overall"]["accuracy"], 1.0)
        self.assertEqual(report["runtime_inputs"], ["physical_observation"])
        self.assertFalse(report["candidate_actions_used_as_router_inputs"])
        self.assertFalse(report["visual_latents_used_as_router_inputs"])

    def test_physical_probe_rejects_malformed_feature_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "physical route probe dataset"):
            PhysicalRouteProbeDataset(
                features=torch.tensor(1.0),
                labels=torch.tensor((0,)),
                groups=torch.tensor((0,)),
                slices=("hold",),
            )


if __name__ == "__main__":
    unittest.main()
