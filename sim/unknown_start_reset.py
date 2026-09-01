"""Deterministic, simulator-independent reset contract for milestone 20."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, AbstractSet

from sim.exploration import DatasetSplit, build_exploration_plan


UNKNOWN_START_RESET_SCHEMA = "quantis.unknown_start_reset_contract.v1"
UNKNOWN_START_RESET_SAMPLE_SCHEMA = "quantis.unknown_start_reset_sample.v1"


def _fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _valid_fingerprint(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class UnknownStartResetSample:
    seed: int
    split: DatasetSplit
    initial_arm_offset_radians: tuple[float, ...]
    camera_offset_m: tuple[float, ...]
    scene_offset_m: tuple[float, ...]
    socket_scale: float
    light_exposure_delta: float

    def __post_init__(self) -> None:
        vectors = (
            (self.initial_arm_offset_radians, 7),
            (self.camera_offset_m, 3),
            (self.scene_offset_m, 3),
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or self.split is not DatasetSplit.HELD_OUT
            or any(
                len(vector) != length or not all(isfinite(value) for value in vector)
                for vector, length in vectors
            )
            or not isfinite(self.socket_scale)
            or not isfinite(self.light_exposure_delta)
        ):
            raise ValueError("unknown-start reset sample is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": UNKNOWN_START_RESET_SAMPLE_SCHEMA,
            "seed": self.seed,
            "split": self.split.value,
            "initial_arm_offset_radians": list(self.initial_arm_offset_radians),
            "camera_offset_m": list(self.camera_offset_m),
            "scene_offset_m": list(self.scene_offset_m),
            "socket_scale": self.socket_scale,
            "light_exposure_delta": self.light_exposure_delta,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class UnknownStartResetBounds:
    initial_arm_offset_radians: tuple[float, float] = (-0.01, 0.01)
    camera_offset_m: tuple[float, float] = (-0.012, 0.012)
    scene_x_m: tuple[float, float] = (0.0, 0.0)
    scene_y_m: tuple[float, float] = (-0.025, 0.025)
    scene_z_m: tuple[float, float] = (-0.015, 0.015)
    socket_scale: tuple[float, float] = (1.05, 1.05)
    light_exposure_delta: tuple[float, float] = (-0.4, 0.4)

    def __post_init__(self) -> None:
        intervals = (
            self.initial_arm_offset_radians,
            self.camera_offset_m,
            self.scene_x_m,
            self.scene_y_m,
            self.scene_z_m,
            self.socket_scale,
            self.light_exposure_delta,
        )
        if any(
            len(interval) != 2
            or not all(isfinite(value) for value in interval)
            or interval[0] > interval[1]
            for interval in intervals
        ):
            raise ValueError("unknown-start reset bounds are invalid")

    @staticmethod
    def _contains(interval: tuple[float, float], value: float) -> bool:
        return interval[0] <= value <= interval[1]

    def contains(self, sample: UnknownStartResetSample) -> bool:
        return (
            all(
                self._contains(self.initial_arm_offset_radians, value)
                for value in sample.initial_arm_offset_radians
            )
            and all(
                self._contains(self.camera_offset_m, value)
                for value in sample.camera_offset_m
            )
            and self._contains(self.scene_x_m, sample.scene_offset_m[0])
            and self._contains(self.scene_y_m, sample.scene_offset_m[1])
            and self._contains(self.scene_z_m, sample.scene_offset_m[2])
            and self._contains(self.socket_scale, sample.socket_scale)
            and self._contains(
                self.light_exposure_delta,
                sample.light_exposure_delta,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_arm_offset_radians": list(self.initial_arm_offset_radians),
            "camera_offset_m": list(self.camera_offset_m),
            "scene_offset_m": {
                "x": list(self.scene_x_m),
                "y": list(self.scene_y_m),
                "z": list(self.scene_z_m),
            },
            "socket_scale": list(self.socket_scale),
            "light_exposure_delta": list(self.light_exposure_delta),
        }


@dataclass(frozen=True)
class UnknownStartWorkspaceState:
    connector_position_m: tuple[float, ...]
    socket_position_m: tuple[float, ...]
    end_effector_position_m: tuple[float, ...]

    def __post_init__(self) -> None:
        if any(
            len(position) != 3 or not all(isfinite(value) for value in position)
            for position in (
                self.connector_position_m,
                self.socket_position_m,
                self.end_effector_position_m,
            )
        ):
            raise ValueError("unknown-start workspace state is invalid")


@dataclass(frozen=True)
class UnknownStartWorkspaceBounds:
    connector: tuple[tuple[float, float], ...] = (
        (-0.0256, -0.0256),
        (-0.27525, -0.22525),
        (1.305, 1.335),
    )
    socket: tuple[tuple[float, float], ...] = (
        (-0.071, -0.071),
        (-0.275, -0.225),
        (1.305, 1.335),
    )
    initial_end_effector: tuple[tuple[float, float], ...] = (
        (0.22, 0.28),
        (-0.30, -0.20),
        (1.43, 1.53),
    )

    def __post_init__(self) -> None:
        groups = (self.connector, self.socket, self.initial_end_effector)
        if any(
            len(group) != 3
            or any(
                len(interval) != 2
                or not all(isfinite(value) for value in interval)
                or interval[0] > interval[1]
                for interval in group
            )
            for group in groups
        ):
            raise ValueError("unknown-start workspace bounds are invalid")

    @staticmethod
    def _position_in_bounds(
        position: tuple[float, ...],
        bounds: tuple[tuple[float, float], ...],
    ) -> bool:
        return all(
            lower <= value <= upper
            for value, (lower, upper) in zip(position, bounds)
        )

    def contains(self, state: UnknownStartWorkspaceState) -> bool:
        return (
            self._position_in_bounds(state.connector_position_m, self.connector)
            and self._position_in_bounds(state.socket_position_m, self.socket)
            and self._position_in_bounds(
                state.end_effector_position_m,
                self.initial_end_effector,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        def axes(bounds: tuple[tuple[float, float], ...]) -> dict[str, list[float]]:
            return {
                axis: list(interval)
                for axis, interval in zip(("x", "y", "z"), bounds)
            }

        return {
            "connector": axes(self.connector),
            "socket": axes(self.socket),
            "initial_end_effector": axes(self.initial_end_effector),
        }


@dataclass(frozen=True)
class UnknownStartResetContract:
    minimum_seed: int = 62600
    maximum_seed: int = 62699
    bounds: UnknownStartResetBounds = UnknownStartResetBounds()
    workspace: UnknownStartWorkspaceBounds = UnknownStartWorkspaceBounds()

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_seed, bool)
            or not isinstance(self.minimum_seed, int)
            or isinstance(self.maximum_seed, bool)
            or not isinstance(self.maximum_seed, int)
            or self.minimum_seed < 0
            or self.maximum_seed < self.minimum_seed
            or not isinstance(self.bounds, UnknownStartResetBounds)
            or not isinstance(self.workspace, UnknownStartWorkspaceBounds)
        ):
            raise ValueError("unknown-start reset contract is invalid")

    def draw(
        self,
        seed: int,
        *,
        forbidden_seeds: AbstractSet[int],
    ) -> UnknownStartResetSample:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not self.minimum_seed <= seed <= self.maximum_seed
        ):
            raise ValueError("unknown-start reset requires a reserved seed")
        if seed in forbidden_seeds:
            raise ValueError("unknown-start reset seed was already used")
        plan = replace(
            build_exploration_plan(seed, DatasetSplit.HELD_OUT),
            socket_scale=self.bounds.socket_scale[0],
        )
        sample = UnknownStartResetSample(
            seed=plan.seed,
            split=plan.split,
            initial_arm_offset_radians=plan.initial_arm_offset_radians,
            camera_offset_m=plan.camera_offset_m,
            scene_offset_m=plan.scene_offset_m,
            socket_scale=plan.socket_scale,
            light_exposure_delta=plan.light_exposure_delta,
        )
        self.validate_sample(sample)
        return sample

    def validate_sample(self, sample: UnknownStartResetSample) -> None:
        if (
            not isinstance(sample, UnknownStartResetSample)
            or not self.minimum_seed <= sample.seed <= self.maximum_seed
            or sample.split is not DatasetSplit.HELD_OUT
            or not self.bounds.contains(sample)
        ):
            raise ValueError("unknown-start reset sample exceeds its contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": UNKNOWN_START_RESET_SCHEMA,
            "seed_namespace": {
                "minimum": self.minimum_seed,
                "maximum": self.maximum_seed,
            },
            "split": DatasetSplit.HELD_OUT.value,
            "sampler": "build_exploration_plan.v1",
            "sampler_source_fingerprint": sha256(
                Path(__file__).with_name("exploration.py").read_bytes()
            ).hexdigest(),
            "bounds": self.bounds.to_dict(),
            "workspace_bounds_m": self.workspace.to_dict(),
            "initialization": "direct_state_setting_once",
            "direct_state_setting_count": 1,
            "prefix_replay_frames": 0,
            "runtime_motion": "drive_only",
            "maximum_initial_contact_force_newtons": 0.0,
            "require_unattached": True,
            "require_collision_free": True,
            "authority": {
                "reset_authentication_only": True,
                "apply_actions": False,
                "train": False,
                "film": False,
            },
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class UnknownStartResetEvidence:
    sample: UnknownStartResetSample
    workspace: UnknownStartWorkspaceState
    realized_sample_fingerprint: str
    plug_attached: bool
    collision_detected: bool
    contact_force_newtons: float
    direct_state_setting_count: int
    prefix_replay_frames: int
    applied_actions: int
    phase: str

    def validate(self, contract: UnknownStartResetContract) -> None:
        contract.validate_sample(self.sample)
        expected = contract.draw(self.sample.seed, forbidden_seeds=set())
        if (
            self.sample != expected
            or not contract.workspace.contains(self.workspace)
            or not _valid_fingerprint(self.realized_sample_fingerprint)
            or self.realized_sample_fingerprint != self.sample.fingerprint
            or self.plug_attached
            or self.collision_detected
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons != 0.0
            or self.direct_state_setting_count != 1
            or self.prefix_replay_frames != 0
            or self.applied_actions != 0
            or self.phase != "reset_authentication"
        ):
            raise ValueError("unknown-start reset evidence is unsafe or inauthentic")


UNKNOWN_START_RESET_CONTRACT = UnknownStartResetContract()
