"""Deterministic, simulator-independent reset contract for milestone 20."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from math import isfinite
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
    split: str
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
            or self.split != DatasetSplit.HELD_OUT.value
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
            "split": self.split,
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
class UnknownStartResetContract:
    minimum_seed: int = 62600
    maximum_seed: int = 62699
    socket_scale: float = 1.05

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_seed, bool)
            or not isinstance(self.minimum_seed, int)
            or isinstance(self.maximum_seed, bool)
            or not isinstance(self.maximum_seed, int)
            or self.minimum_seed < 0
            or self.maximum_seed < self.minimum_seed
            or not isfinite(self.socket_scale)
            or self.socket_scale <= 0.0
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
            socket_scale=self.socket_scale,
        )
        sample = UnknownStartResetSample(
            seed=plan.seed,
            split=plan.split.value,
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
            or sample.split != DatasetSplit.HELD_OUT.value
            or any(abs(value) > 0.01 for value in sample.initial_arm_offset_radians)
            or any(abs(value) > 0.012 for value in sample.camera_offset_m)
            or sample.scene_offset_m[0] != 0.0
            or abs(sample.scene_offset_m[1]) > 0.025
            or abs(sample.scene_offset_m[2]) > 0.015
            or sample.socket_scale != self.socket_scale
            or abs(sample.light_exposure_delta) > 0.4
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
            "bounds": {
                "initial_arm_offset_radians": [-0.01, 0.01],
                "camera_offset_m": [-0.012, 0.012],
                "scene_offset_m": {
                    "x": [0.0, 0.0],
                    "y": [-0.025, 0.025],
                    "z": [-0.015, 0.015],
                },
                "socket_scale": [self.socket_scale, self.socket_scale],
                "light_exposure_delta": [-0.4, 0.4],
            },
            "initialization": "direct_state_setting_once",
            "prefix_replay_frames": 0,
            "runtime_motion": "drive_only",
            "maximum_initial_contact_force_newtons": 0.0,
            "require_unattached": True,
            "require_collision_free": True,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class UnknownStartResetEvidence:
    sample: UnknownStartResetSample
    realized_sample_fingerprint: str
    plug_attached: bool
    collision_detected: bool
    contact_force_newtons: float
    prefix_replay_frames: int
    runtime_motion: str

    def validate(self, contract: UnknownStartResetContract) -> None:
        contract.validate_sample(self.sample)
        expected = contract.draw(self.sample.seed, forbidden_seeds=set())
        if (
            self.sample != expected
            or not _valid_fingerprint(self.realized_sample_fingerprint)
            or self.realized_sample_fingerprint != self.sample.fingerprint
            or self.plug_attached
            or self.collision_detected
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons != 0.0
            or self.prefix_replay_frames != 0
            or self.runtime_motion != "drive_only"
        ):
            raise ValueError("unknown-start reset evidence is unsafe or inauthentic")


UNKNOWN_START_RESET_CONTRACT = UnknownStartResetContract()
