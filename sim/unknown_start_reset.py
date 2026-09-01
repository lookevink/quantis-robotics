"""Deterministic, simulator-independent reset contract for milestone 20."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from typing import Any, AbstractSet

from sim.exploration import DatasetSplit, build_exploration_plan


UNKNOWN_START_RESET_SCHEMA = "quantis.unknown_start_reset_contract.v2"
UNKNOWN_START_RESET_SAMPLE_SCHEMA = "quantis.unknown_start_reset_sample.v1"
UNKNOWN_START_RESET_EVIDENCE_SCHEMA = "quantis.unknown_start_reset_evidence.v2"
UNKNOWN_START_GRIPPER_FRAME = "right_gripper_control_frame"


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

    @classmethod
    def from_dict(cls, payload: Any) -> UnknownStartResetSample:
        if not isinstance(payload, dict) or payload.get("schema") != UNKNOWN_START_RESET_SAMPLE_SCHEMA:
            raise ValueError("unknown-start reset sample payload is invalid")
        return cls(
            seed=payload.get("seed"),
            split=DatasetSplit(payload.get("split")),
            initial_arm_offset_radians=tuple(payload.get("initial_arm_offset_radians", ())),
            camera_offset_m=tuple(payload.get("camera_offset_m", ())),
            scene_offset_m=tuple(payload.get("scene_offset_m", ())),
            socket_scale=payload.get("socket_scale"),
            light_exposure_delta=payload.get("light_exposure_delta"),
        )

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
    gripper_control_frame_position_m: tuple[float, ...]
    socket_scale: float

    def __post_init__(self) -> None:
        if any(
            len(position) != 3 or not all(isfinite(value) for value in position)
            for position in (
                self.connector_position_m,
                self.socket_position_m,
                self.gripper_control_frame_position_m,
            )
        ) or not isfinite(self.socket_scale):
            raise ValueError("unknown-start workspace state is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "connector_position_m": list(self.connector_position_m),
            "socket_position_m": list(self.socket_position_m),
            "gripper_control_frame": UNKNOWN_START_GRIPPER_FRAME,
            "gripper_control_frame_position_m": list(
                self.gripper_control_frame_position_m
            ),
            "socket_scale": self.socket_scale,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> UnknownStartWorkspaceState:
        if (
            not isinstance(payload, dict)
            or payload.get("gripper_control_frame")
            != UNKNOWN_START_GRIPPER_FRAME
        ):
            raise ValueError("unknown-start workspace payload is invalid")
        return cls(
            connector_position_m=tuple(payload.get("connector_position_m", ())),
            socket_position_m=tuple(payload.get("socket_position_m", ())),
            gripper_control_frame_position_m=tuple(
                payload.get("gripper_control_frame_position_m", ())
            ),
            socket_scale=payload.get("socket_scale"),
        )


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
    initial_gripper_control_frame: tuple[tuple[float, float], ...] = (
        (0.22, 0.28),
        (-0.30, -0.20),
        (1.43, 1.53),
    )
    connector_baseline_m: tuple[float, float, float] = (-0.0256, -0.25025, 1.32)
    socket_baseline_m: tuple[float, float, float] = (-0.071, -0.25, 1.32)
    realization_position_tolerance_m: float = 1e-5
    realization_scale_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        groups = (self.connector, self.socket, self.initial_gripper_control_frame)
        if any(
            len(group) != 3
            or any(
                len(interval) != 2
                or not all(isfinite(value) for value in interval)
                or interval[0] > interval[1]
                for interval in group
            )
            for group in groups
        ) or (
            not all(isfinite(value) for value in self.connector_baseline_m)
            or not all(isfinite(value) for value in self.socket_baseline_m)
            or not isfinite(self.realization_position_tolerance_m)
            or self.realization_position_tolerance_m < 0.0
            or not isfinite(self.realization_scale_tolerance)
            or self.realization_scale_tolerance < 0.0
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
                state.gripper_control_frame_position_m,
                self.initial_gripper_control_frame,
            )
        )

    def matches_sample(
        self,
        sample: UnknownStartResetSample,
        state: UnknownStartWorkspaceState,
    ) -> bool:
        expected_connector = tuple(
            baseline + offset
            for baseline, offset in zip(
                self.connector_baseline_m,
                sample.scene_offset_m,
            )
        )
        expected_socket = tuple(
            baseline + offset
            for baseline, offset in zip(
                self.socket_baseline_m,
                sample.scene_offset_m,
            )
        )
        return (
            all(
                abs(actual - expected) <= self.realization_position_tolerance_m
                for actual, expected in zip(
                    state.connector_position_m,
                    expected_connector,
                )
            )
            and all(
                abs(actual - expected) <= self.realization_position_tolerance_m
                for actual, expected in zip(
                    state.socket_position_m,
                    expected_socket,
                )
            )
            and abs(state.socket_scale - sample.socket_scale)
            <= self.realization_scale_tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        def axes(bounds: tuple[tuple[float, float], ...]) -> dict[str, list[float]]:
            return {
                axis: list(interval)
                for axis, interval in zip(("x", "y", "z"), bounds)
            }

        return {
            "baseline_m": {
                "connector": list(self.connector_baseline_m),
                "socket": list(self.socket_baseline_m),
            },
            "connector": axes(self.connector),
            "socket": axes(self.socket),
            "initial_gripper_control_frame": {
                "frame": UNKNOWN_START_GRIPPER_FRAME,
                "bounds": axes(self.initial_gripper_control_frame),
            },
            "realization_tolerances": {
                "position_m": self.realization_position_tolerance_m,
                "socket_scale": self.realization_scale_tolerance,
            },
        }


@dataclass(frozen=True)
class UnknownStartSampleRealization:
    initial_arm_offset_radians: tuple[float, ...]
    camera_offset_m: tuple[float, ...]
    light_exposure_delta: float

    def __post_init__(self) -> None:
        if (
            len(self.initial_arm_offset_radians) != 7
            or len(self.camera_offset_m) != 3
            or not all(isfinite(value) for value in self.initial_arm_offset_radians)
            or not all(isfinite(value) for value in self.camera_offset_m)
            or not isfinite(self.light_exposure_delta)
        ):
            raise ValueError("unknown-start sample realization is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_arm_offset_radians": list(self.initial_arm_offset_radians),
            "camera_offset_m": list(self.camera_offset_m),
            "light_exposure_delta": self.light_exposure_delta,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> UnknownStartSampleRealization:
        if not isinstance(payload, dict):
            raise ValueError("unknown-start realization payload is invalid")
        return cls(
            initial_arm_offset_radians=tuple(payload.get("initial_arm_offset_radians", ())),
            camera_offset_m=tuple(payload.get("camera_offset_m", ())),
            light_exposure_delta=payload.get("light_exposure_delta"),
        )


@dataclass(frozen=True)
class UnknownStartSampleRealizationTolerances:
    initial_arm_offset_radians: float = 1e-5
    camera_offset_m: float = 1e-6
    light_exposure_delta: float = 1e-6

    def __post_init__(self) -> None:
        if any(
            not isfinite(tolerance) or tolerance < 0.0
            for tolerance in (
                self.initial_arm_offset_radians,
                self.camera_offset_m,
                self.light_exposure_delta,
            )
        ):
            raise ValueError("unknown-start sample tolerances are invalid")

    def matches(
        self,
        sample: UnknownStartResetSample,
        realized: UnknownStartSampleRealization,
    ) -> bool:
        return (
            all(
                abs(actual - intended) <= self.initial_arm_offset_radians
                for actual, intended in zip(
                    realized.initial_arm_offset_radians,
                    sample.initial_arm_offset_radians,
                )
            )
            and all(
                abs(actual - intended) <= self.camera_offset_m
                for actual, intended in zip(
                    realized.camera_offset_m,
                    sample.camera_offset_m,
                )
            )
            and abs(realized.light_exposure_delta - sample.light_exposure_delta)
            <= self.light_exposure_delta
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "initial_arm_offset_radians": self.initial_arm_offset_radians,
            "camera_offset_m": self.camera_offset_m,
            "light_exposure_delta": self.light_exposure_delta,
        }


@dataclass(frozen=True)
class UnknownStartResetContract:
    minimum_seed: int = 62600
    maximum_seed: int = 62699
    bounds: UnknownStartResetBounds = UnknownStartResetBounds()
    workspace: UnknownStartWorkspaceBounds = UnknownStartWorkspaceBounds()
    sampler_source_fingerprint: str = (
        "0ec746dbf12fbed61c66b3c64dee6717fa15f015be4aad23e0f40e5b47a5228d"
    )
    realization_tolerances: UnknownStartSampleRealizationTolerances = (
        UnknownStartSampleRealizationTolerances()
    )

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
            or not _valid_fingerprint(self.sampler_source_fingerprint)
            or not isinstance(
                self.realization_tolerances,
                UnknownStartSampleRealizationTolerances,
            )
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
        current_sampler_fingerprint = sha256(
            Path(__file__).with_name("exploration.py").read_bytes()
        ).hexdigest()
        if current_sampler_fingerprint != self.sampler_source_fingerprint:
            raise ValueError("unknown-start reset sampler source changed")
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
            "sampler_source_fingerprint": self.sampler_source_fingerprint,
            "bounds": self.bounds.to_dict(),
            "sample_realization_tolerances": self.realization_tolerances.to_dict(),
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


class UnknownStartResetPhase(str, Enum):
    RESET_AUTHENTICATION = "reset_authentication"


@dataclass(frozen=True)
class UnknownStartResetEvidence:
    sample: UnknownStartResetSample
    workspace: UnknownStartWorkspaceState
    realization: UnknownStartSampleRealization
    observed_arm_positions_radians: tuple[float, ...]
    observed_gripper_width_m: float
    realized_sample_fingerprint: str
    plug_attached: bool
    collision_detected: bool
    contact_force_newtons: float
    direct_state_setting_count: int
    prefix_replay_frames: int
    applied_actions: int
    phase: UnknownStartResetPhase

    def validate(self, contract: UnknownStartResetContract) -> None:
        contract.validate_sample(self.sample)
        expected = contract.draw(self.sample.seed, forbidden_seeds=set())
        if (
            self.sample != expected
            or not contract.workspace.contains(self.workspace)
            or not contract.workspace.matches_sample(self.sample, self.workspace)
            or not contract.realization_tolerances.matches(
                self.sample,
                self.realization,
            )
            or not _valid_fingerprint(self.realized_sample_fingerprint)
            or self.realized_sample_fingerprint != self.sample.fingerprint
            or len(self.observed_arm_positions_radians) != 7
            or not all(isfinite(value) for value in self.observed_arm_positions_radians)
            or not isfinite(self.observed_gripper_width_m)
            or not 0.0 <= self.observed_gripper_width_m <= 0.08
            or self.plug_attached
            or self.collision_detected
            or not isfinite(self.contact_force_newtons)
            or self.contact_force_newtons != 0.0
            or self.direct_state_setting_count != 1
            or self.prefix_replay_frames != 0
            or self.applied_actions != 0
            or self.phase is not UnknownStartResetPhase.RESET_AUTHENTICATION
        ):
            raise ValueError("unknown-start reset evidence is unsafe or inauthentic")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": UNKNOWN_START_RESET_EVIDENCE_SCHEMA,
            "sample": self.sample.to_dict(),
            "workspace": self.workspace.to_dict(),
            "realization": self.realization.to_dict(),
            "observed_arm_positions_radians": list(
                self.observed_arm_positions_radians
            ),
            "observed_gripper_width_m": self.observed_gripper_width_m,
            "realized_sample_fingerprint": self.realized_sample_fingerprint,
            "plug_attached": self.plug_attached,
            "collision_detected": self.collision_detected,
            "contact_force_newtons": self.contact_force_newtons,
            "direct_state_setting_count": self.direct_state_setting_count,
            "prefix_replay_frames": self.prefix_replay_frames,
            "applied_actions": self.applied_actions,
            "phase": self.phase.value,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> UnknownStartResetEvidence:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != UNKNOWN_START_RESET_EVIDENCE_SCHEMA
            or not isinstance(payload.get("plug_attached"), bool)
            or not isinstance(payload.get("collision_detected"), bool)
        ):
            raise ValueError("unknown-start reset evidence payload is invalid")
        evidence = cls(
            sample=UnknownStartResetSample.from_dict(payload.get("sample")),
            workspace=UnknownStartWorkspaceState.from_dict(payload.get("workspace")),
            realization=UnknownStartSampleRealization.from_dict(
                payload.get("realization")
            ),
            observed_arm_positions_radians=tuple(
                payload.get("observed_arm_positions_radians", ())
            ),
            observed_gripper_width_m=payload.get("observed_gripper_width_m"),
            realized_sample_fingerprint=payload.get("realized_sample_fingerprint"),
            plug_attached=payload["plug_attached"],
            collision_detected=payload["collision_detected"],
            contact_force_newtons=payload.get("contact_force_newtons"),
            direct_state_setting_count=payload.get("direct_state_setting_count"),
            prefix_replay_frames=payload.get("prefix_replay_frames"),
            applied_actions=payload.get("applied_actions"),
            phase=UnknownStartResetPhase(payload.get("phase")),
        )
        evidence.validate(UNKNOWN_START_RESET_CONTRACT)
        return evidence


UNKNOWN_START_RESET_CONTRACT = UnknownStartResetContract()
