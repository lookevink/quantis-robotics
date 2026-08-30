"""Versioned action-conditioning families for bounded offline experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator

import torch

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


class ActionConditioningKind(str, Enum):
    GLOBAL_LINEAR = "global_linear"
    NONLINEAR_RESIDUAL = "nonlinear_residual"
    ORACLE_REGIME_RESIDUAL = "oracle_regime_residual"


@dataclass(frozen=True)
class ActionConditioningSpec:
    kind: ActionConditioningKind
    hidden_dimension: int | None = None

    def __post_init__(self) -> None:
        if self.kind is ActionConditioningKind.NONLINEAR_RESIDUAL:
            if (
                isinstance(self.hidden_dimension, bool)
                or not isinstance(self.hidden_dimension, int)
                or self.hidden_dimension <= 0
            ):
                raise ValueError("nonlinear action conditioning requires a hidden dimension")
        elif self.hidden_dimension is not None:
            raise ValueError("only nonlinear action conditioning uses a hidden dimension")

    @classmethod
    def from_dict(cls, payload: Any) -> ActionConditioningSpec:
        if not isinstance(payload, dict) or set(payload) != {
            "kind",
            "hidden_dimension",
        }:
            raise ValueError("action-conditioning specification is invalid")
        try:
            kind = ActionConditioningKind(payload["kind"])
        except (TypeError, ValueError) as error:
            raise ValueError("action-conditioning kind is invalid") from error
        return cls(kind, payload["hidden_dimension"])

    def to_dict(self) -> dict[str, str | int | None]:
        return {"kind": self.kind.value, "hidden_dimension": self.hidden_dimension}


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


ActionConditioningEncoder = (
    GlobalLinearActionEncoder
    | NonlinearResidualActionEncoder
    | OracleRegimeResidualActionEncoder
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
    else:
        installed = OracleRegimeResidualActionEncoder(base)
    predictor.action_encoder = installed
    return installed


def action_conditioning_parameters(model: Any) -> tuple[torch.nn.Parameter, ...]:
    encoder = _installed_encoder(model)
    parameters: list[torch.nn.Parameter] = [encoder.base.weight]
    if isinstance(encoder, NonlinearResidualActionEncoder):
        parameters.extend(encoder.residual_in.parameters())
        parameters.extend(encoder.residual_out.parameters())
    elif isinstance(encoder, OracleRegimeResidualActionEncoder):
        for residual in encoder.residuals:
            parameters.extend(residual.parameters())
    return tuple(parameters)


@contextmanager
def action_regime_context(model: Any, routes: torch.Tensor) -> Iterator[None]:
    encoder = _installed_encoder(model)
    if isinstance(encoder, OracleRegimeResidualActionEncoder):
        with encoder.use_routes(routes):
            yield
    else:
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
