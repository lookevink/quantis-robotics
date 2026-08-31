"""Read-only TRAIN probes for the failed runtime-command router."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
from math import isfinite
import os
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch

from jepa_wm.action_conditioning import (
    BASE_COMMAND_ROUTE,
    COMMAND_ROUTE_NAMES,
    NEGATIVE_X_COMMAND_ROUTE,
    POSITIVE_X_COMMAND_ROUTE,
    LoadedActionConditioning,
    RuntimeCommandResidualActionEncoder,
)
from jepa_wm.action_conditioning_training import signed_x_negatives
from jepa_wm.action_routing_experiment import (
    FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
    ROUTING_SPEC,
    TREATMENT_SPEC,
    TRAINING_BOUNDS,
    _authenticate_base_model,
    _load_experiment_config,
    _segment_for_context,
    _validated_selection,
)
from jepa_wm.frames import encode_clips
from jepa_wm.model import load_headless_model
from jepa_wm.objective import terminal_l2_energy
from jepa_wm.persistence import write_json_atomic
from jepa_wm.rollout_scoring import rollout_action_tensor
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.trajectory import RecordedRollout


ROUTER_FINGERPRINT = (
    "45326210f5a47f74a9008670e9bf0be03b3ef40955b3c9af79017588d9b79c30"
)
CONTEXTS_BY_SEGMENT = {
    "retreat": (114, 120, 126, 132, 138, 144, 150, 156),
    "align": (166, 172, 178, 184, 190, 196, 202, 208),
    "insert": (216, 224, 232, 240, 248, 256, 264, 272),
}
GRADIENT_CONTEXTS_BY_ROUTE = {
    NEGATIVE_X_COMMAND_ROUTE: (132, 150),
    POSITIVE_X_COMMAND_ROUTE: (184, 240),
}
PAIRWISE_MARGIN = 0.001


class DiagnosticRuntimeEncoder(torch.nn.Module):
    """The frozen router with an optional read-only route override."""

    def __init__(self, source: RuntimeCommandResidualActionEncoder) -> None:
        super().__init__()
        self.base = source.base
        self.residuals = source.residuals
        self.spec = source.spec
        self._forced_routes: torch.Tensor | None = None

    @contextmanager
    def use_routes(self, routes: torch.Tensor | None) -> Iterator[None]:
        if self._forced_routes is not None:
            raise ValueError("diagnostic routes are already active")
        self._forced_routes = (
            routes.to(device=self.base.weight.device, dtype=torch.long)
            if routes is not None
            else None
        )
        try:
            yield
        finally:
            self._forced_routes = None

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        routes = self._forced_routes
        if routes is None:
            routes, _ = ROUTING_SPEC.classify(actions)
        if actions.shape[0] != routes.shape[0]:
            raise ValueError("diagnostic route batch does not match actions")
        output = self.base(actions)
        mask_shape = (routes.shape[0],) + (1,) * (output.ndim - 1)
        for residual_index, route in enumerate(
            (NEGATIVE_X_COMMAND_ROUTE, POSITIVE_X_COMMAND_ROUTE)
        ):
            mask = (routes == route).reshape(mask_shape)
            output = output + torch.where(
                mask,
                self.residuals[residual_index](actions),
                0.0,
            )
        return output


@dataclass(frozen=True)
class ScoredCandidates:
    recorded: torch.Tensor
    zero: torch.Tensor
    x_zero: torch.Tensor
    x_opposed: torch.Tensor


def _selected_rollouts(recordings: Sequence[Path]) -> tuple[RecordedRollout, ...]:
    selection = _validated_selection(recordings)
    selected_contexts = {
        context for contexts in CONTEXTS_BY_SEGMENT.values() for context in contexts
    }
    selected = tuple(
        rollout
        for rollout in selection.rollouts
        if rollout.context[0].index in selected_contexts
    )
    expected = len(recordings) * len(selected_contexts)
    if len(selected) != expected:
        raise ValueError("diagnostic TRAIN selection is incomplete")
    for recording in recordings:
        recording_root = recording.resolve()
        recording_rollouts = tuple(
            rollout
            for rollout in selected
            if recording_root in rollout.context[0].path.resolve().parents
        )
        if len(recording_rollouts) != len(selected_contexts):
            raise ValueError(
                f"diagnostic selection is incomplete for {recording.name}"
            )
        if {
            rollout.context[0].index for rollout in recording_rollouts
        } != selected_contexts:
            raise ValueError(
                f"diagnostic contexts changed for {recording.name}"
            )
    return selected


def _score_one(
    model: Any,
    encoder: DiagnosticRuntimeEncoder,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    routes: torch.Tensor | None,
    *,
    device: torch.device,
    batch_size: int = 2,
) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, contexts.shape[0], batch_size):
            stop = min(start + batch_size, contexts.shape[0])
            route_batch = routes[start:stop] if routes is not None else None
            with encoder.use_routes(route_batch):
                chunks.append(
                    terminal_l2_energy(
                        model.unroll(
                            contexts[start:stop].to(device),
                            actions[:, start:stop].to(device),
                        ),
                        targets[start:stop].to(device),
                    ).cpu()
                )
    return torch.cat(chunks)


def _score_mode(
    model: Any,
    encoder: DiagnosticRuntimeEncoder,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    recorded_routes: torch.Tensor,
    mode: str,
    *,
    device: torch.device,
) -> ScoredCandidates:
    x_zero, x_opposed = signed_x_negatives(actions)
    forced = {
        "candidate_routed": None,
        "recorded_route_locked": recorded_routes,
        "base_only": torch.full_like(recorded_routes, BASE_COMMAND_ROUTE),
    }[mode]
    return ScoredCandidates(
        **{
            name: _score_one(
                model,
                encoder,
                contexts,
                targets,
                candidate,
                forced,
                device=device,
            )
            for name, candidate in {
                "recorded": actions,
                "zero": torch.zeros_like(actions),
                "x_zero": x_zero,
                "x_opposed": x_opposed,
            }.items()
        }
    )


def _fraction(value: torch.Tensor) -> float:
    return float(value.float().mean())


def _metrics(
    scored: ScoredCandidates,
    indices: torch.Tensor,
) -> dict[str, float | int]:
    recorded = scored.recorded[indices]
    zero = scored.zero[indices]
    x_zero = scored.x_zero[indices]
    x_opposed = scored.x_opposed[indices]
    return {
        "rollouts": int(indices.numel()),
        "recorded_below_zero": _fraction(recorded < zero),
        "recorded_below_x_zero": _fraction(recorded < x_zero),
        "recorded_below_x_opposed": _fraction(recorded < x_opposed),
        "x_zero_below_x_opposed": _fraction(x_zero < x_opposed),
        "signed_triple_order": _fraction(
            (recorded < x_zero) & (x_zero < x_opposed)
        ),
        "mean_zero_minus_recorded": float((zero - recorded).mean()),
        "mean_x_zero_minus_recorded": float((x_zero - recorded).mean()),
        "mean_x_opposed_minus_recorded": float((x_opposed - recorded).mean()),
        "mean_x_opposed_minus_x_zero": float((x_opposed - x_zero).mean()),
    }


def _mode_report(
    scored: ScoredCandidates,
    rollouts: Sequence[RecordedRollout],
) -> dict[str, Any]:
    segments = tuple(_segment_for_context(item.context[0].index) for item in rollouts)
    report = {
        "all": _metrics(scored, torch.arange(len(rollouts), dtype=torch.long))
    }
    for segment in CONTEXTS_BY_SEGMENT:
        indices = torch.tensor(
            [index for index, value in enumerate(segments) if value == segment],
            dtype=torch.long,
        )
        report[segment] = _metrics(scored, indices)
    return report


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left.flatten(), right.flatten()) / denominator)


def _residual_report(
    encoder: DiagnosticRuntimeEncoder,
    actions: torch.Tensor,
    routes: torch.Tensor,
) -> dict[str, Any]:
    base_weight = encoder.base.weight.detach().cpu()
    weights = [residual.weight.detach().cpu() for residual in encoder.residuals]
    by_expert = {}
    batch_actions = actions.transpose(0, 1).to(encoder.base.weight.device)
    device_routes = routes.to(batch_actions.device)
    for index, (name, route) in enumerate(
        (
            ("negative_x", NEGATIVE_X_COMMAND_ROUTE),
            ("positive_x", POSITIVE_X_COMMAND_ROUTE),
        )
    ):
        selected = batch_actions[device_routes == route]
        if selected.shape[0] == 0:
            raise ValueError(f"diagnostic selection has no {name} actions")
        base_embedding = encoder.base(selected)
        residual_embedding = encoder.residuals[index](selected)
        ratios = torch.linalg.vector_norm(residual_embedding, dim=-1) / torch.clamp(
            torch.linalg.vector_norm(base_embedding, dim=-1),
            min=1e-12,
        )
        weight = weights[index]
        by_expert[name] = {
            "frobenius_norm": float(torch.linalg.vector_norm(weight)),
            "column_norms_xyz_rot_gripper": [
                float(value) for value in torch.linalg.vector_norm(weight, dim=0)
            ],
            "column_to_base_ratios": [
                float(value)
                for value in (
                    torch.linalg.vector_norm(weight, dim=0)
                    / torch.clamp(
                        torch.linalg.vector_norm(base_weight, dim=0),
                        min=1e-12,
                    )
                )
            ],
            "mean_residual_to_base_embedding_norm": float(ratios.mean()),
            "maximum_residual_to_base_embedding_norm": float(ratios.max()),
        }
    by_expert["weight_cosine_negative_vs_positive"] = _cosine(
        weights[0], weights[1]
    )
    return by_expert


def _deadband_report(
    encoder: DiagnosticRuntimeEncoder,
    actions: torch.Tensor,
) -> dict[str, float]:
    batch = actions.transpose(0, 1).to(encoder.base.weight.device)
    outside_negative = batch.clone()
    outside_positive = batch.clone()
    inside_negative = batch.clone()
    inside_positive = batch.clone()
    outside_negative[..., 0] = -1.01 * ROUTING_SPEC.signed_x_deadband
    outside_positive[..., 0] = 1.01 * ROUTING_SPEC.signed_x_deadband
    inside_negative[..., 0] = -0.99 * ROUTING_SPEC.signed_x_deadband
    inside_positive[..., 0] = 0.99 * ROUTING_SPEC.signed_x_deadband
    with torch.inference_mode():
        outside_jump = torch.linalg.vector_norm(
            encoder(outside_positive) - encoder(outside_negative),
            dim=-1,
        )
        inside_jump = torch.linalg.vector_norm(
            encoder(inside_positive) - encoder(inside_negative),
            dim=-1,
        )
        base_jump = torch.linalg.vector_norm(
            encoder.base(outside_positive) - encoder.base(outside_negative),
            dim=-1,
        )
    return {
        "mean_embedding_jump_just_outside_deadband": float(outside_jump.mean()),
        "mean_embedding_jump_just_inside_deadband": float(inside_jump.mean()),
        "mean_frozen_base_jump_for_same_x_change": float(base_jump.mean()),
        "outside_to_base_jump_ratio": float(
            outside_jump.mean() / torch.clamp(base_jump.mean(), min=1e-12)
        ),
    }


def _gradient(
    model: Any,
    encoder: DiagnosticRuntimeEncoder,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    indices: Sequence[int],
    parameter: torch.nn.Parameter,
    term: str,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    accumulated = torch.zeros_like(parameter)
    active_count = 0
    total = 0
    for start in range(0, len(indices), 2):
        selected = torch.tensor(indices[start : start + 2], dtype=torch.long)
        action = actions[:, selected].to(device)
        context = contexts[selected].to(device)
        target = targets[selected].to(device)
        recorded = terminal_l2_energy(model.unroll(context, action), target)
        if term == "own_recorded":
            loss = recorded.mean()
            active_count += selected.numel()
        elif term == "opposite_rejection":
            _, opposed = signed_x_negatives(action)
            opposed_energy = terminal_l2_energy(model.unroll(context, opposed), target)
            per_item = torch.relu(PAIRWISE_MARGIN + recorded - opposed_energy)
            active_count += int((per_item > 0).sum())
            loss = per_item.mean()
        else:
            raise ValueError("unknown gradient diagnostic term")
        gradient = torch.autograd.grad(
            loss,
            parameter,
            retain_graph=False,
            allow_unused=True,
        )[0]
        if gradient is not None:
            accumulated += gradient.detach() * selected.numel()
        total += selected.numel()
    return accumulated / total, active_count / total


def _gradient_report(
    model: Any,
    encoder: DiagnosticRuntimeEncoder,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    routes: torch.Tensor,
    rollouts: Sequence[RecordedRollout],
    *,
    device: torch.device,
) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for residual in encoder.residuals:
        residual.weight.requires_grad_(True)
    indices_by_route = {
        route: [
            index
            for index, item in enumerate(rollouts)
            if int(routes[index]) == route
            and item.context[0].index in contexts
        ]
        for route, contexts in GRADIENT_CONTEXTS_BY_ROUTE.items()
    }
    if any(len(indices) != 24 for indices in indices_by_route.values()):
        raise ValueError("gradient probe roster changed")
    negative_indices = indices_by_route[NEGATIVE_X_COMMAND_ROUTE]
    positive_indices = indices_by_route[POSITIVE_X_COMMAND_ROUTE]
    report = {}
    for name, expert_index, own_indices, opposite_indices in (
        ("negative_x", 0, negative_indices, positive_indices),
        ("positive_x", 1, positive_indices, negative_indices),
    ):
        parameter = encoder.residuals[expert_index].weight
        own_gradient, _ = _gradient(
            model,
            encoder,
            contexts,
            targets,
            actions,
            own_indices,
            parameter,
            "own_recorded",
            device=device,
        )
        cross_gradient, active_fraction = _gradient(
            model,
            encoder,
            contexts,
            targets,
            actions,
            opposite_indices,
            parameter,
            "opposite_rejection",
            device=device,
        )
        report[name] = {
            "own_route_examples": len(own_indices),
            "opposite_route_examples": len(opposite_indices),
            "opposite_margin_active_fraction": active_fraction,
            "own_fit_gradient_norm": float(torch.linalg.vector_norm(own_gradient)),
            "cross_rejection_gradient_norm": float(
                torch.linalg.vector_norm(cross_gradient)
            ),
            "own_fit_vs_cross_rejection_gradient_cosine": _cosine(
                own_gradient,
                cross_gradient,
            ),
        }
    return report


def diagnose(
    source: Path,
    checkpoint: Path,
    artifact: Path,
    recordings: Sequence[Path],
    experiment_config: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"counterfactual diagnosis already exists: {output}")
    if output.name != "counterfactual-diagnosis-v1.json":
        raise ValueError("counterfactual diagnosis output name changed")
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual diagnosis requires CUDA")
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    artifact_identity = ArtifactIdentity.from_artifact(artifact)
    if artifact_identity.fingerprint != ROUTER_FINGERPRINT:
        raise ValueError("counterfactual diagnosis router identity changed")
    rollouts = _selected_rollouts(recordings)
    actions = rollout_action_tensor(rollouts)
    routes, _ = ROUTING_SPEC.classify(actions.transpose(0, 1))
    expected_route_roster = {
        NEGATIVE_X_COMMAND_ROUTE: 96,
        POSITIVE_X_COMMAND_ROUTE: 180,
        BASE_COMMAND_ROUTE: 12,
    }
    if any(
        int((routes == route).sum()) != expected
        for route, expected in expected_route_roster.items()
    ):
        raise ValueError("diagnostic command-route roster changed")
    device = torch.device("cuda", torch.cuda.current_device())
    model = load_headless_model(source, checkpoint, device=device)
    loaded = LoadedActionConditioning.load(
        artifact,
        expected_identity=artifact_identity,
    )
    if (
        loaded.contract.spec != TREATMENT_SPEC
        or loaded.contract.experiment_config_fingerprint
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
    ):
        raise ValueError("counterfactual diagnosis router contract changed")
    loaded.apply(
        model,
        expected_source_revision=os.environ["JEPA_WM_REVISION"],
    )
    installed = model.model.predictor.action_encoder
    if not isinstance(installed, RuntimeCommandResidualActionEncoder):
        raise ValueError("counterfactual diagnosis requires the runtime router")
    encoder = DiagnosticRuntimeEncoder(installed)
    model.model.predictor.action_encoder = encoder
    model.eval()

    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=4,
    )
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=4,
    )
    mode_reports = {}
    for mode in ("candidate_routed", "recorded_route_locked", "base_only"):
        scored = _score_mode(
            model,
            encoder,
            contexts,
            targets,
            actions,
            routes,
            mode,
            device=device,
        )
        mode_reports[mode] = _mode_report(scored, rollouts)

    residuals = _residual_report(encoder, actions, routes)
    deadband = _deadband_report(encoder, actions)
    gradients = _gradient_report(
        model,
        encoder,
        contexts,
        targets,
        actions,
        routes,
        rollouts,
        device=device,
    )
    if not all(
        isfinite(value)
        for expert in gradients.values()
        for key, value in expert.items()
        if key.endswith("_norm") and isinstance(value, float)
    ):
        raise ValueError("counterfactual gradient probe is non-finite")
    report = {
        "schema": "quantis.jepa_wm_runtime_router_counterfactual_diagnosis.v1",
        "status": "complete",
        "scope": "read-only TRAIN diagnosis; no training or held-out evaluation",
        "base_model": base_identity,
        "router": artifact_identity.to_dict(),
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "training_recordings": list(loaded.contract.metadata.training_recordings),
        "contexts_by_segment": {
            key: list(value) for key, value in CONTEXTS_BY_SEGMENT.items()
        },
        "rollouts": len(rollouts),
        "route_roster": {
            name: int((routes == route).sum())
            for name, route in (
                ("negative_x", NEGATIVE_X_COMMAND_ROUTE),
                ("positive_x", POSITIVE_X_COMMAND_ROUTE),
                ("base", BASE_COMMAND_ROUTE),
            )
        },
        "modes": mode_reports,
        "residuals": residuals,
        "deadband": deadband,
        "gradients": gradients,
        "model_parameters_updated": False,
        "artifacts_written": 1,
        "held_out_accessed": False,
        "canonical_accessed": False,
        "live_action_authorized": False,
    }
    write_json_atomic(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--recording", type=Path, action="append", required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = diagnose(
        arguments.source,
        arguments.checkpoint,
        arguments.artifact,
        arguments.recording,
        arguments.experiment_config,
        arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
