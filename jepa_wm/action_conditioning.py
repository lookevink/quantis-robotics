"""Versioned action-conditioning families for bounded offline experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO
from math import isfinite
from pathlib import Path
from typing import Any, Iterator

import torch

from jepa_wm.causal_routing import (
    CausalContextRoutingSpec,
    CausalMotionRouter,
    CausalRouteDecision,
)
from jepa_wm.contract import MODEL_ID
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    validate_artifact_fingerprint,
)


ACTION_CONDITIONING_SCHEMA = "quantis.jepa_wm_action_conditioning.v1"
RETAINED_REGIME = 0
POST_REGIME = 1
REGIME_NAMES = ("retained", "post")
BASE_COMMAND_ROUTE = 0
NEGATIVE_X_COMMAND_ROUTE = 1
POSITIVE_X_COMMAND_ROUTE = 2
COMMAND_ROUTE_NAMES = ("base", "negative_x", "positive_x")


class ActionConditioningKind(str, Enum):
    GLOBAL_LINEAR = "global_linear"
    NONLINEAR_RESIDUAL = "nonlinear_residual"
    ORACLE_REGIME_RESIDUAL = "oracle_regime_residual"
    RUNTIME_COMMAND_RESIDUAL = "runtime_command_residual"
    OBSERVED_CONTEXT_RESIDUAL = "observed_context_residual"
    CAUSAL_CONTEXT_RESIDUAL = "causal_context_residual"


@dataclass(frozen=True)
class RuntimeCommandRoutingSpec:
    """Route complete candidate horizons using only their runtime commands."""

    signed_x_deadband: float
    translation_activity_deadband: float
    rotation_activity_deadband: float
    gripper_activity_deadband: float
    horizon_x_statistic: str = "mean"

    def __post_init__(self) -> None:
        values = (
            self.signed_x_deadband,
            self.translation_activity_deadband,
            self.rotation_activity_deadband,
            self.gripper_activity_deadband,
        )
        if not all(isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("runtime command deadbands must be finite and non-negative")
        if self.horizon_x_statistic != "mean":
            raise ValueError("runtime command routing requires mean horizon X")

    @classmethod
    def from_dict(cls, payload: Any) -> RuntimeCommandRoutingSpec:
        if not isinstance(payload, dict) or set(payload) != {
            "signed_x_deadband",
            "translation_activity_deadband",
            "rotation_activity_deadband",
            "gripper_activity_deadband",
            "horizon_x_statistic",
        }:
            raise ValueError("runtime command routing specification is invalid")
        try:
            return cls(
                signed_x_deadband=float(payload["signed_x_deadband"]),
                translation_activity_deadband=float(
                    payload["translation_activity_deadband"]
                ),
                rotation_activity_deadband=float(
                    payload["rotation_activity_deadband"]
                ),
                gripper_activity_deadband=float(payload["gripper_activity_deadband"]),
                horizon_x_statistic=str(payload["horizon_x_statistic"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("runtime command routing specification is invalid") from error

    def to_dict(self) -> dict[str, float | str]:
        return {
            "signed_x_deadband": self.signed_x_deadband,
            "translation_activity_deadband": self.translation_activity_deadband,
            "rotation_activity_deadband": self.rotation_activity_deadband,
            "gripper_activity_deadband": self.gripper_activity_deadband,
            "horizon_x_statistic": self.horizon_x_statistic,
        }

    def classify(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return functional routes and whether each horizon contains activity."""

        if actions.ndim != 3 or actions.shape[-1] != 7 or actions.shape[1] == 0:
            raise ValueError("runtime command routing requires [batch, horizon, 7] actions")
        translation_active = (
            torch.linalg.vector_norm(actions[..., :3], dim=-1)
            > self.translation_activity_deadband
        )
        rotation_active = (
            torch.linalg.vector_norm(actions[..., 3:6], dim=-1)
            > self.rotation_activity_deadband
        )
        gripper_active = actions[..., 6].abs() > self.gripper_activity_deadband
        active = (translation_active | rotation_active | gripper_active).any(dim=1)
        mean_x = actions[..., 0].mean(dim=1)
        routes = torch.full_like(mean_x, BASE_COMMAND_ROUTE, dtype=torch.long)
        routes = torch.where(
            active & (mean_x < -self.signed_x_deadband),
            NEGATIVE_X_COMMAND_ROUTE,
            routes,
        )
        routes = torch.where(
            active & (mean_x > self.signed_x_deadband),
            POSITIVE_X_COMMAND_ROUTE,
            routes,
        )
        return routes, active


@dataclass(frozen=True)
class ObservedContextRoutingSpec:
    """Continuous residual weights derived from one previous realized action."""

    signed_x_deadband: float
    signed_x_transition_width: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.signed_x_deadband)
            or self.signed_x_deadband < 0.0
            or not isfinite(self.signed_x_transition_width)
            or self.signed_x_transition_width <= 0.0
        ):
            raise ValueError("observed-context routing thresholds are invalid")

    @classmethod
    def from_dict(cls, payload: Any) -> ObservedContextRoutingSpec:
        if not isinstance(payload, dict) or set(payload) != {
            "signed_x_deadband",
            "signed_x_transition_width",
        }:
            raise ValueError("observed-context routing specification is invalid")
        try:
            return cls(
                signed_x_deadband=float(payload["signed_x_deadband"]),
                signed_x_transition_width=float(
                    payload["signed_x_transition_width"]
                ),
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "observed-context routing specification is invalid"
            ) from error

    def to_dict(self) -> dict[str, float]:
        return {
            "signed_x_deadband": self.signed_x_deadband,
            "signed_x_transition_width": self.signed_x_transition_width,
        }

    def route_weights(self, previous_actions: torch.Tensor) -> torch.Tensor:
        if previous_actions.ndim != 2 or previous_actions.shape[-1] != 7:
            raise ValueError("observed-context routing requires [batch, 7] actions")
        x = previous_actions[..., 0]
        negative = torch.clamp(
            (-x - self.signed_x_deadband) / self.signed_x_transition_width,
            min=0.0,
            max=1.0,
        )
        positive = torch.clamp(
            (x - self.signed_x_deadband) / self.signed_x_transition_width,
            min=0.0,
            max=1.0,
        )
        return torch.stack((negative, positive), dim=-1)

    def classify(self, previous_actions: torch.Tensor) -> torch.Tensor:
        weights = self.route_weights(previous_actions)
        routes = torch.full(
            (previous_actions.shape[0],),
            BASE_COMMAND_ROUTE,
            dtype=torch.long,
            device=previous_actions.device,
        )
        routes = torch.where(
            weights[:, 0] > 0.0,
            NEGATIVE_X_COMMAND_ROUTE,
            routes,
        )
        return torch.where(
            weights[:, 1] > 0.0,
            POSITIVE_X_COMMAND_ROUTE,
            routes,
        )


@dataclass(frozen=True)
class ActionConditioningSpec:
    kind: ActionConditioningKind
    hidden_dimension: int | None = None
    runtime_routing: RuntimeCommandRoutingSpec | None = None
    observed_context_routing: ObservedContextRoutingSpec | None = None
    causal_context_routing: CausalContextRoutingSpec | None = None

    def __post_init__(self) -> None:
        if self.kind is ActionConditioningKind.NONLINEAR_RESIDUAL:
            if (
                isinstance(self.hidden_dimension, bool)
                or not isinstance(self.hidden_dimension, int)
                or self.hidden_dimension <= 0
            ):
                raise ValueError("nonlinear action conditioning requires a hidden dimension")
            if (
                self.runtime_routing is not None
                or self.observed_context_routing is not None
                or self.causal_context_routing is not None
            ):
                raise ValueError("nonlinear action conditioning cannot use runtime routing")
        elif self.kind is ActionConditioningKind.RUNTIME_COMMAND_RESIDUAL:
            if (
                self.hidden_dimension is not None
                or self.runtime_routing is None
                or self.observed_context_routing is not None
                or self.causal_context_routing is not None
            ):
                raise ValueError("runtime command conditioning requires only routing")
        elif self.kind is ActionConditioningKind.OBSERVED_CONTEXT_RESIDUAL:
            if (
                self.hidden_dimension is not None
                or self.runtime_routing is not None
                or self.observed_context_routing is None
                or self.causal_context_routing is not None
            ):
                raise ValueError(
                    "observed-context conditioning requires only context routing"
                )
        elif self.kind is ActionConditioningKind.CAUSAL_CONTEXT_RESIDUAL:
            if (
                self.hidden_dimension is not None
                or self.runtime_routing is not None
                or self.observed_context_routing is not None
                or self.causal_context_routing is None
            ):
                raise ValueError(
                    "causal-context conditioning requires only causal routing"
                )
        elif (
            self.hidden_dimension is not None
            or self.runtime_routing is not None
            or self.observed_context_routing is not None
            or self.causal_context_routing is not None
        ):
            raise ValueError("only nonlinear action conditioning uses a hidden dimension")

    @classmethod
    def from_dict(cls, payload: Any) -> ActionConditioningSpec:
        if not isinstance(payload, dict) or set(payload) not in (
            {"kind", "hidden_dimension"},
            {"kind", "hidden_dimension", "runtime_routing"},
            {"kind", "hidden_dimension", "observed_context_routing"},
            {"kind", "hidden_dimension", "causal_context_routing"},
        ):
            raise ValueError("action-conditioning specification is invalid")
        try:
            kind = ActionConditioningKind(payload["kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("action-conditioning kind is invalid") from error
        routing_payload = payload.get("runtime_routing")
        routing = (
            RuntimeCommandRoutingSpec.from_dict(routing_payload)
            if routing_payload is not None
            else None
        )
        observed_payload = payload.get("observed_context_routing")
        observed_routing = (
            ObservedContextRoutingSpec.from_dict(observed_payload)
            if observed_payload is not None
            else None
        )
        causal_payload = payload.get("causal_context_routing")
        causal_routing = (
            CausalContextRoutingSpec.from_dict(causal_payload)
            if causal_payload is not None
            else None
        )
        return cls(
            kind,
            payload["hidden_dimension"],
            routing,
            observed_routing,
            causal_routing,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "hidden_dimension": self.hidden_dimension,
        }
        if self.runtime_routing is not None:
            payload["runtime_routing"] = self.runtime_routing.to_dict()
        if self.observed_context_routing is not None:
            payload["observed_context_routing"] = (
                self.observed_context_routing.to_dict()
            )
        if self.causal_context_routing is not None:
            payload["causal_context_routing"] = self.causal_context_routing.to_dict()
        return payload


class GlobalLinearActionEncoder(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.spec = ActionConditioningSpec(ActionConditioningKind.GLOBAL_LINEAR)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.base(actions)


class NonlinearResidualActionEncoder(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, hidden_dimension: int) -> None:
        super().__init__()
        self.base = base
        self.residual_in = torch.nn.Linear(
            base.in_features,
            hidden_dimension,
            bias=True,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.residual_out = torch.nn.Linear(
            hidden_dimension,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        torch.nn.init.zeros_(self.residual_out.weight)
        self.spec = ActionConditioningSpec(
            ActionConditioningKind.NONLINEAR_RESIDUAL,
            hidden_dimension,
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        residual = self.residual_out(torch.nn.functional.silu(self.residual_in(actions)))
        return self.base(actions) + residual


class OracleRegimeResidualActionEncoder(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.residuals = torch.nn.ModuleList(
            torch.nn.Linear(
                base.in_features,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            for _ in REGIME_NAMES
        )
        for residual in self.residuals:
            torch.nn.init.zeros_(residual.weight)
        self.spec = ActionConditioningSpec(
            ActionConditioningKind.ORACLE_REGIME_RESIDUAL
        )
        self._active_routes: torch.Tensor | None = None

    @contextmanager
    def use_routes(self, routes: torch.Tensor) -> Iterator[None]:
        if self._active_routes is not None:
            raise ValueError("action regime routes are already active")
        if (
            routes.ndim != 1
            or routes.numel() == 0
            or routes.dtype == torch.bool
            or torch.any((routes != RETAINED_REGIME) & (routes != POST_REGIME))
        ):
            raise ValueError("action regime routes must contain only zero or one")
        self._active_routes = routes.to(device=self.base.weight.device, dtype=torch.long)
        try:
            yield
        finally:
            self._active_routes = None

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        routes = self._active_routes
        if routes is None or actions.ndim < 2 or actions.shape[0] != routes.shape[0]:
            raise ValueError("oracle action conditioning requires scoped regime routes")
        output = self.base(actions)
        mask_shape = (routes.shape[0],) + (1,) * (output.ndim - 1)
        for route, residual in enumerate(self.residuals):
            mask = (routes == route).reshape(mask_shape)
            output = output + torch.where(mask, residual(actions), 0.0)
        return output


class RuntimeCommandResidualActionEncoder(torch.nn.Module):
    """Apply signed-X residuals while preserving the base map for holds."""

    def __init__(self, base: torch.nn.Linear, routing: RuntimeCommandRoutingSpec) -> None:
        super().__init__()
        self.base = base
        self.residuals = torch.nn.ModuleList(
            torch.nn.Linear(
                base.in_features,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            for _ in (NEGATIVE_X_COMMAND_ROUTE, POSITIVE_X_COMMAND_ROUTE)
        )
        for residual in self.residuals:
            torch.nn.init.zeros_(residual.weight)
        self.spec = ActionConditioningSpec(
            ActionConditioningKind.RUNTIME_COMMAND_RESIDUAL,
            runtime_routing=routing,
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        routing = self.spec.runtime_routing
        assert routing is not None
        routes, _ = routing.classify(actions)
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


class ObservedContextResidualActionEncoder(torch.nn.Module):
    """Apply one candidate-invariant residual blend per observed context."""

    def __init__(
        self,
        base: torch.nn.Linear,
        routing: ObservedContextRoutingSpec,
    ) -> None:
        super().__init__()
        self.base = base
        self.residuals = torch.nn.ModuleList(
            torch.nn.Linear(
                base.in_features,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            for _ in (NEGATIVE_X_COMMAND_ROUTE, POSITIVE_X_COMMAND_ROUTE)
        )
        for residual in self.residuals:
            torch.nn.init.zeros_(residual.weight)
        self.spec = ActionConditioningSpec(
            ActionConditioningKind.OBSERVED_CONTEXT_RESIDUAL,
            observed_context_routing=routing,
        )
        self._active_weights: torch.Tensor | None = None

    def residual_for_route(self, route: int) -> torch.nn.Linear:
        if route == NEGATIVE_X_COMMAND_ROUTE:
            return self.residuals[0]
        if route == POSITIVE_X_COMMAND_ROUTE:
            return self.residuals[1]
        raise ValueError("observed-context residual route must be negative or positive")

    @contextmanager
    def use_observed_actions(self, previous_actions: torch.Tensor) -> Iterator[None]:
        if self._active_weights is not None:
            raise ValueError("observed action context is already active")
        routing = self.spec.observed_context_routing
        assert routing is not None
        self._active_weights = routing.route_weights(previous_actions).to(
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        ).detach()
        try:
            yield
        finally:
            self._active_weights = None

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        weights = self._active_weights
        if weights is None or actions.ndim < 2 or actions.shape[0] != weights.shape[0]:
            raise ValueError(
                "observed action context must be scoped to the candidate batch"
            )
        output = self.base(actions)
        weight_shape = (weights.shape[0],) + (1,) * (output.ndim - 1)
        for residual_index, route in enumerate(
            (NEGATIVE_X_COMMAND_ROUTE, POSITIVE_X_COMMAND_ROUTE)
        ):
            output = output + weights[:, residual_index].reshape(
                weight_shape
            ) * self.residual_for_route(route)(actions)
        return output


class CausalContextResidualActionEncoder(torch.nn.Module):
    """Apply hard-bounded residuals selected once from causal observations."""

    def __init__(
        self,
        base: torch.nn.Linear,
        routing: CausalContextRoutingSpec,
    ) -> None:
        super().__init__()
        if base.out_features != routing.context_dimension:
            raise ValueError(
                "causal routing context dimension must match the action embedding"
            )
        self.base = base
        self.router = CausalMotionRouter(routing).to(
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.residuals = torch.nn.ModuleList(
            torch.nn.Linear(
                base.in_features,
                base.out_features,
                bias=False,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
            for _ in (NEGATIVE_X_COMMAND_ROUTE, POSITIVE_X_COMMAND_ROUTE)
        )
        for residual in self.residuals:
            torch.nn.init.zeros_(residual.weight)
        self.spec = ActionConditioningSpec(
            ActionConditioningKind.CAUSAL_CONTEXT_RESIDUAL,
            causal_context_routing=routing,
        )
        self._active_decision: CausalRouteDecision | None = None

    @property
    def active_decision(self) -> CausalRouteDecision | None:
        return self._active_decision

    def residual_for_route(self, route: int) -> torch.nn.Linear:
        if route == NEGATIVE_X_COMMAND_ROUTE:
            return self.residuals[0]
        if route == POSITIVE_X_COMMAND_ROUTE:
            return self.residuals[1]
        raise ValueError("causal residual route must be retreat or advance")

    def route(
        self,
        context_latents: torch.Tensor,
        context_poses: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> CausalRouteDecision:
        return self.router.decide(context_latents, context_poses, previous_actions)

    @contextmanager
    def use_causal_context(
        self,
        context_latents: torch.Tensor,
        context_poses: torch.Tensor,
        previous_actions: torch.Tensor,
    ) -> Iterator[CausalRouteDecision]:
        if self._active_decision is not None:
            raise ValueError("causal action context is already active")
        decision = self.route(context_latents, context_poses, previous_actions)
        self._active_decision = CausalRouteDecision(
            decision.routes.to(device=self.base.weight.device),
            decision.confidence.to(device=self.base.weight.device),
            decision.residual_weights.to(
                device=self.base.weight.device,
                dtype=self.base.weight.dtype,
            ),
            decision.failed_closed.to(device=self.base.weight.device),
        )
        try:
            yield self._active_decision
        finally:
            self._active_decision = None

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        decision = self._active_decision
        if (
            decision is None
            or actions.ndim < 2
            or actions.shape[0] != decision.routes.shape[0]
        ):
            raise ValueError("causal action context must be scoped to the candidate batch")
        base = self.base(actions)
        weight_shape = (decision.routes.shape[0],) + (1,) * (base.ndim - 1)
        residual = torch.zeros_like(base)
        for index, route in enumerate(
            (NEGATIVE_X_COMMAND_ROUTE, POSITIVE_X_COMMAND_ROUTE)
        ):
            residual = residual + decision.residual_weights[:, index].reshape(
                weight_shape
            ) * self.residual_for_route(route)(actions)
        residual_norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        base_norm = torch.linalg.vector_norm(base, dim=-1, keepdim=True)
        routing = self.spec.causal_context_routing
        assert routing is not None
        maximum = routing.maximum_residual_ratio * base_norm
        denominator = torch.clamp(
            residual_norm,
            min=torch.finfo(residual.dtype).eps,
        )
        scale = torch.clamp(maximum / denominator, max=1.0)
        return base + residual * scale


ActionConditioningEncoder = (
    GlobalLinearActionEncoder
    | NonlinearResidualActionEncoder
    | OracleRegimeResidualActionEncoder
    | RuntimeCommandResidualActionEncoder
    | ObservedContextResidualActionEncoder
    | CausalContextResidualActionEncoder
)


def _predictor(model: Any) -> Any:
    try:
        return model.model.predictor
    except AttributeError as error:
        raise ValueError("model has no predictor") from error


def _installed_encoder(model: Any) -> ActionConditioningEncoder:
    encoder = getattr(_predictor(model), "action_encoder", None)
    if not isinstance(
        encoder,
        (
            GlobalLinearActionEncoder,
            NonlinearResidualActionEncoder,
            OracleRegimeResidualActionEncoder,
            RuntimeCommandResidualActionEncoder,
            ObservedContextResidualActionEncoder,
            CausalContextResidualActionEncoder,
        ),
    ):
        raise ValueError("model has no installed action-conditioning family")
    return encoder


def install_action_conditioning(
    model: Any,
    spec: ActionConditioningSpec,
) -> ActionConditioningEncoder:
    predictor = _predictor(model)
    base = getattr(predictor, "action_encoder", None)
    if not isinstance(base, torch.nn.Linear):
        raise ValueError("action conditioning requires the official linear encoder")
    if spec.kind is ActionConditioningKind.GLOBAL_LINEAR:
        installed: ActionConditioningEncoder = GlobalLinearActionEncoder(base)
    elif spec.kind is ActionConditioningKind.NONLINEAR_RESIDUAL:
        assert spec.hidden_dimension is not None
        installed = NonlinearResidualActionEncoder(base, spec.hidden_dimension)
    elif spec.kind is ActionConditioningKind.ORACLE_REGIME_RESIDUAL:
        installed = OracleRegimeResidualActionEncoder(base)
    elif spec.kind is ActionConditioningKind.RUNTIME_COMMAND_RESIDUAL:
        assert spec.runtime_routing is not None
        installed = RuntimeCommandResidualActionEncoder(base, spec.runtime_routing)
    elif spec.kind is ActionConditioningKind.OBSERVED_CONTEXT_RESIDUAL:
        assert spec.observed_context_routing is not None
        installed = ObservedContextResidualActionEncoder(
            base,
            spec.observed_context_routing,
        )
    else:
        assert spec.causal_context_routing is not None
        installed = CausalContextResidualActionEncoder(
            base,
            spec.causal_context_routing,
        )
    predictor.action_encoder = installed
    return installed


def action_conditioning_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    encoder = _installed_encoder(model)
    parameters: list[torch.nn.Parameter] = []
    if not isinstance(
        encoder,
        (
            RuntimeCommandResidualActionEncoder,
            ObservedContextResidualActionEncoder,
            CausalContextResidualActionEncoder,
        ),
    ):
        parameters.append(encoder.base.weight)
    if isinstance(encoder, NonlinearResidualActionEncoder):
        parameters.extend(encoder.residual_in.parameters())
        parameters.extend(encoder.residual_out.parameters())
    elif isinstance(encoder, OracleRegimeResidualActionEncoder):
        for residual in encoder.residuals:
            parameters.extend(residual.parameters())
    elif isinstance(encoder, RuntimeCommandResidualActionEncoder):
        for residual in encoder.residuals:
            parameters.extend(residual.parameters())
    elif isinstance(encoder, ObservedContextResidualActionEncoder):
        for residual in encoder.residuals:
            parameters.extend(residual.parameters())
    elif isinstance(encoder, CausalContextResidualActionEncoder):
        parameters.extend(encoder.router.parameters())
        for residual in encoder.residuals:
            parameters.extend(residual.parameters())
    return tuple(parameters)


def causal_context_encoder(model: Any) -> CausalContextResidualActionEncoder:
    encoder = _installed_encoder(model)
    if not isinstance(encoder, CausalContextResidualActionEncoder):
        raise ValueError("model has no causal-context action conditioning")
    return encoder


def causal_router_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    return tuple(causal_context_encoder(model).router.parameters())


def causal_residual_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    encoder = causal_context_encoder(model)
    return tuple(
        parameter
        for residual in encoder.residuals
        for parameter in residual.parameters()
    )


@contextmanager
def action_regime_context(model: Any, routes: torch.Tensor) -> Iterator[None]:
    encoder = _installed_encoder(model)
    if isinstance(encoder, OracleRegimeResidualActionEncoder):
        with encoder.use_routes(routes):
            yield
    else:
        yield


@contextmanager
def observed_action_context(
    model: Any,
    previous_actions: torch.Tensor,
) -> Iterator[None]:
    encoder = _installed_encoder(model)
    if not isinstance(encoder, ObservedContextResidualActionEncoder):
        raise ValueError("model has no observed-context action conditioning")
    with encoder.use_observed_actions(previous_actions):
        yield


@dataclass(frozen=True)
class ActionConditioningContract:
    schema: str
    metadata: TrainingArtifactMetadata
    training_selection_fingerprint: str
    training_config_fingerprint: str
    experiment_config_fingerprint: str
    spec: ActionConditioningSpec

    def __post_init__(self) -> None:
        if self.schema != ACTION_CONDITIONING_SCHEMA:
            raise ValueError("unsupported action-conditioning artifact")
        for fingerprint in (
            self.training_selection_fingerprint,
            self.training_config_fingerprint,
            self.experiment_config_fingerprint,
        ):
            validate_artifact_fingerprint(fingerprint)

    @classmethod
    def from_dict(cls, payload: Any) -> ActionConditioningContract:
        if not isinstance(payload, dict):
            raise ValueError("unsupported action-conditioning artifact")
        try:
            return cls(
                str(payload["schema"]),
                TrainingArtifactMetadata.from_dict(payload["metadata"]),
                str(payload["training_selection_fingerprint"]),
                str(payload["training_config_fingerprint"]),
                str(payload["experiment_config_fingerprint"]),
                ActionConditioningSpec.from_dict(payload["spec"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("action-conditioning artifact contract is invalid") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "metadata": self.metadata.to_dict(),
            "training_selection_fingerprint": self.training_selection_fingerprint,
            "training_config_fingerprint": self.training_config_fingerprint,
            "experiment_config_fingerprint": self.experiment_config_fingerprint,
            "spec": self.spec.to_dict(),
        }


def save_action_conditioning(
    model: Any,
    path: Path,
    contract: ActionConditioningContract,
) -> None:
    encoder = _installed_encoder(model)
    if encoder.spec != contract.spec:
        raise ValueError("installed action conditioning disagrees with its contract")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {**contract.to_dict(), "action_encoder": encoder.state_dict()},
        temporary,
    )
    temporary.replace(path)


@dataclass(frozen=True)
class LoadedActionConditioning:
    identity: ArtifactIdentity
    contract: ActionConditioningContract
    state: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        expected_identity: ArtifactIdentity | None = None,
    ) -> LoadedActionConditioning:
        resolved = path.resolve()
        encoded = resolved.read_bytes()
        identity = ArtifactIdentity(resolved, sha256(encoded).hexdigest())
        if expected_identity is not None and identity != expected_identity:
            raise ValueError("action-conditioning artifact identity changed before loading")
        payload = torch.load(BytesIO(encoded), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("action_encoder"), dict
        ):
            raise ValueError("action-conditioning artifact has no encoder state")
        return cls(
            identity,
            ActionConditioningContract.from_dict(payload),
            payload["action_encoder"],
        )

    def apply(
        self,
        model: Any,
        *,
        expected_base_model: str = MODEL_ID,
        expected_source_revision: str | None = None,
    ) -> TrainingArtifactMetadata:
        metadata = self.contract.metadata
        if metadata.base_model != expected_base_model:
            raise ValueError("action conditioning targets another base model")
        if (
            expected_source_revision is not None
            and metadata.source_revision != expected_source_revision
        ):
            raise ValueError("action conditioning source revision does not match")
        encoder = install_action_conditioning(model, self.contract.spec)
        encoder.load_state_dict(self.state, strict=True)
        return metadata


def apply_action_conditioning(
    model: Any,
    path: Path,
    *,
    expected_source_revision: str | None = None,
) -> TrainingArtifactMetadata:
    return LoadedActionConditioning.load(path).apply(
        model,
        expected_source_revision=expected_source_revision,
    )
