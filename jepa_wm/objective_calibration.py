"""Realized action-response calibration for shadow-only task reranking."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from jepa_wm.action import ACTION_DIMENSIONS, DroidAction, DroidPose, action_between


ACTION_RESPONSE_CALIBRATION_SCHEMA = "quantis.jepa_wm_action_response_calibration.v1"


@dataclass(frozen=True)
class CalibrationIdentity:
    path: Path
    fingerprint: str

    def __post_init__(self) -> None:
        if (
            not self.path.is_absolute()
            or len(self.fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.fingerprint)
        ):
            raise ValueError("calibration identity is invalid")

    @classmethod
    def from_calibration(
        cls, path: Path, calibration: ActionResponseCalibration
    ) -> CalibrationIdentity:
        return cls(path.resolve(), calibration.fingerprint)

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Any) -> CalibrationIdentity:
        if not isinstance(payload, Mapping):
            raise ValueError("calibration identity is incomplete")
        try:
            return cls(Path(payload["path"]), str(payload["fingerprint"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("calibration identity is incomplete") from error


@dataclass(frozen=True)
class AxisResponse:
    scale: float
    alignment: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.scale)
            or self.scale <= 0.0
            or not isfinite(self.alignment)
            or not -1.0 <= self.alignment <= 1.0
        ):
            raise ValueError("axis response is invalid")


@dataclass(frozen=True)
class ActionResponseTrial:
    trial_id: str
    seed: int
    proposed: DroidAction
    realized: DroidAction

    def __post_init__(self) -> None:
        if not self.trial_id or self.seed < 0:
            raise ValueError("action-response trial identity is invalid")

    @staticmethod
    def _vector_response(
        proposed: Sequence[float],
        realized: Sequence[float],
    ) -> AxisResponse:
        expected = np.asarray(proposed, dtype=np.float64)
        actual = np.asarray(realized, dtype=np.float64)
        expected_norm = float(np.linalg.norm(expected))
        actual_norm = float(np.linalg.norm(actual))
        if expected_norm <= 1e-12:
            raise ValueError("calibration trial has an inactive proposed axis")
        alignment = (
            float(np.dot(expected, actual) / (expected_norm * actual_norm))
            if actual_norm > 1e-12
            else 0.0
        )
        return AxisResponse(actual_norm / expected_norm, max(-1.0, min(1.0, alignment)))

    @property
    def axis_responses(self) -> tuple[AxisResponse, AxisResponse, AxisResponse]:
        translation = self._vector_response(
            self.proposed.values[:3], self.realized.values[:3]
        )
        rotation = self._vector_response(
            self.proposed.values[3:6], self.realized.values[3:6]
        )
        if abs(self.proposed.values[6]) <= 1e-12:
            raise ValueError("calibration trial has an inactive proposed gripper")
        gripper = AxisResponse(
            abs(self.realized.values[6] / self.proposed.values[6]),
            1.0
            if self.proposed.values[6] * self.realized.values[6] > 0.0
            else 0.0,
        )
        return translation, rotation, gripper

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "seed": self.seed,
            "proposed": list(self.proposed.values),
            "realized": list(self.realized.values),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ActionResponseTrial:
        if not isinstance(payload, Mapping):
            raise ValueError("action-response trial is incomplete")
        try:
            return cls(
                str(payload["trial_id"]),
                int(payload["seed"]),
                DroidAction(tuple(payload["proposed"])),
                DroidAction(tuple(payload["realized"])),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("action-response trial is incomplete") from error


@dataclass(frozen=True)
class ActionResponseCalibration:
    trials: tuple[ActionResponseTrial, ...]

    def __post_init__(self) -> None:
        if (
            len(self.trials) < 3
            or len({trial.trial_id for trial in self.trials}) != len(self.trials)
        ):
            raise ValueError("action-response calibration is invalid")

    @property
    def trial_ids(self) -> tuple[str, ...]:
        return tuple(trial.trial_id for trial in self.trials)

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(trial.seed for trial in self.trials)

    @property
    def responses(self) -> tuple[AxisResponse, AxisResponse, AxisResponse]:
        trial_responses = tuple(trial.axis_responses for trial in self.trials)
        aggregated = tuple(
            AxisResponse(
                median(response[axis].scale for response in trial_responses),
                median(response[axis].alignment for response in trial_responses),
            )
            for axis in range(3)
        )
        return aggregated[0], aggregated[1], aggregated[2]

    @property
    def translation(self) -> AxisResponse:
        return self.responses[0]

    @property
    def rotation(self) -> AxisResponse:
        return self.responses[1]

    @property
    def gripper(self) -> AxisResponse:
        return self.responses[2]

    @staticmethod
    def _direction_count(vectors: Sequence[Sequence[float]]) -> int:
        directions = set()
        for vector in vectors:
            values = np.asarray(vector, dtype=np.float64)
            norm = np.linalg.norm(values)
            if norm > 1e-12:
                directions.add(tuple(np.round(values / norm, decimals=6)))
        return len(directions)

    @property
    def translation_direction_count(self) -> int:
        return self._direction_count(
            tuple(trial.proposed.values[:3] for trial in self.trials)
        )

    @property
    def rotation_direction_count(self) -> int:
        return self._direction_count(
            tuple(trial.proposed.values[3:6] for trial in self.trials)
        )

    @property
    def gripper_direction_count(self) -> int:
        return len(
            {
                1 if trial.proposed.values[6] > 0.0 else -1
                for trial in self.trials
                if abs(trial.proposed.values[6]) > 1e-12
            }
        )

    @property
    def distinct_action_directions(self) -> int:
        return self._direction_count(
            tuple(trial.proposed.values for trial in self.trials)
        )

    @property
    def translation_scale(self) -> float:
        return self.translation.scale

    @property
    def rotation_scale(self) -> float:
        return self.rotation.scale

    @property
    def gripper_scale(self) -> float:
        return self.gripper.scale

    @property
    def translation_alignment(self) -> float:
        return self.translation.alignment

    @property
    def rotation_alignment(self) -> float:
        return self.rotation.alignment

    @property
    def gripper_alignment(self) -> float:
        return self.gripper.alignment

    @classmethod
    def fit(
        cls,
        trials: Sequence[ActionResponseTrial],
    ) -> ActionResponseCalibration:
        evidence = tuple(trials)
        if len(evidence) < 3:
            raise ValueError("action-response calibration requires three trials")
        return cls(evidence)

    @property
    def trial_count(self) -> int:
        return len(self.trial_ids)

    @property
    def ready_for_reranking(self) -> bool:
        return min(
            self.translation_alignment,
            self.rotation_alignment,
            self.gripper_alignment,
        ) >= 0.5 and min(
            self.translation_direction_count,
            self.rotation_direction_count,
        ) >= 3 and self.gripper_direction_count >= 1

    def apply(self, action: DroidAction) -> DroidAction:
        scales = (
            *(self.translation_scale,) * 3,
            *(self.rotation_scale,) * 3,
            self.gripper_scale,
        )
        return DroidAction(
            tuple(value * scale for value, scale in zip(action.values, scales))
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": ACTION_RESPONSE_CALIBRATION_SCHEMA,
            "trials": [trial.to_dict() for trial in self.trials],
            "trial_ids": list(self.trial_ids),
            "seeds": list(self.seeds),
            "trial_count": self.trial_count,
            "translation_scale": self.translation_scale,
            "rotation_scale": self.rotation_scale,
            "gripper_scale": self.gripper_scale,
            "translation_alignment": self.translation_alignment,
            "rotation_alignment": self.rotation_alignment,
            "gripper_alignment": self.gripper_alignment,
            "distinct_action_directions": self.distinct_action_directions,
            "translation_direction_count": self.translation_direction_count,
            "rotation_direction_count": self.rotation_direction_count,
            "gripper_direction_count": self.gripper_direction_count,
            "ready_for_reranking": self.ready_for_reranking,
            "production_authority_granted": False,
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActionResponseCalibration:
        if (
            payload.get("schema") != ACTION_RESPONSE_CALIBRATION_SCHEMA
            or payload.get("production_authority_granted") is not False
        ):
            raise ValueError("action-response calibration schema is invalid")
        try:
            calibration = cls(
                tuple(ActionResponseTrial.from_dict(item) for item in payload["trials"])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("action-response calibration is incomplete") from error
        if (
            payload.get("trial_ids") != list(calibration.trial_ids)
            or payload.get("seeds") != list(calibration.seeds)
            or payload.get("trial_count") != calibration.trial_count
            or payload.get("ready_for_reranking")
            is not calibration.ready_for_reranking
            or any(
                not np.isclose(float(payload[name]), expected, rtol=0.0, atol=1e-12)
                for name, expected in (
                    ("translation_scale", calibration.translation_scale),
                    ("rotation_scale", calibration.rotation_scale),
                    ("gripper_scale", calibration.gripper_scale),
                    ("translation_alignment", calibration.translation_alignment),
                    ("rotation_alignment", calibration.rotation_alignment),
                    ("gripper_alignment", calibration.gripper_alignment),
                )
            )
            or any(
                payload.get(name) != expected
                for name, expected in (
                    ("distinct_action_directions", calibration.distinct_action_directions),
                    ("translation_direction_count", calibration.translation_direction_count),
                    ("rotation_direction_count", calibration.rotation_direction_count),
                    ("gripper_direction_count", calibration.gripper_direction_count),
                )
            )
        ):
            raise ValueError("action-response calibration claims are inconsistent")
        return calibration

    @classmethod
    def load(cls, path: Path) -> ActionResponseCalibration:
        try:
            payload = json.loads(path.resolve().read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("action-response calibration file is invalid") from error
        if not isinstance(payload, Mapping):
            raise ValueError("action-response calibration file must be an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class TaskProgress:
    translation_meters: float
    rotation_radians: float
    gripper_closedness: float


@dataclass(frozen=True)
class TaskAxisTolerances:
    translation_meters: float = 1e-4
    rotation_radians: float = 1e-3
    gripper_closedness: float = 0.01

    def __post_init__(self) -> None:
        if not all(
            isfinite(value) and value > 0.0
            for value in (
                self.translation_meters,
                self.rotation_radians,
                self.gripper_closedness,
            )
        ):
            raise ValueError("task-axis tolerances must be finite and positive")

    def to_dict(self) -> dict[str, float]:
        return {
            "translation_meters": self.translation_meters,
            "rotation_radians": self.rotation_radians,
            "gripper_closedness": self.gripper_closedness,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskAxisTolerances:
        try:
            return cls(
                translation_meters=float(payload["translation_meters"]),
                rotation_radians=float(payload["rotation_radians"]),
                gripper_closedness=float(payload["gripper_closedness"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("task-axis tolerances are incomplete") from error


def _axis_error(pose: DroidPose, target: DroidPose) -> TaskProgress:
    delta = action_between(pose, target)
    return TaskProgress(
        float(np.linalg.norm(delta.values[:3])),
        float(Rotation.from_euler("xyz", delta.values[3:6]).magnitude()),
        abs(delta.values[6]),
    )


@dataclass(frozen=True)
class TaskProgressObjective:
    """Penalize latent winners predicted to regress on any task-space axis."""

    start: DroidPose
    target: DroidPose
    calibration: ActionResponseCalibration
    failure_penalty: float = 1.0
    tolerances: TaskAxisTolerances = TaskAxisTolerances()

    def __post_init__(self) -> None:
        if not self.calibration.ready_for_reranking:
            raise ValueError("action-response calibration is not ready for reranking")
        if not isfinite(self.failure_penalty) or self.failure_penalty <= 0.0:
            raise ValueError("task-progress failure penalty must be positive")

    def penalty(self, candidates: np.ndarray) -> np.ndarray:
        values = np.asarray(candidates, dtype=np.float64)
        if (
            values.ndim != 3
            or values.shape[1] != 3
            or values.shape[2] != ACTION_DIMENSIONS
            or not np.all(np.isfinite(values))
        ):
            raise ValueError("task-progress scorer received invalid candidates")
        initial_error = _axis_error(self.start, self.target)
        initial = (
            initial_error.translation_meters,
            initial_error.rotation_radians,
            initial_error.gripper_closedness,
        )
        tolerances = (
            self.tolerances.translation_meters,
            self.tolerances.rotation_radians,
            self.tolerances.gripper_closedness,
        )
        penalties = []
        for candidate in values:
            action = self.calibration.apply(DroidAction(tuple(candidate[0])))
            final_error = _axis_error(self.start.applied(action), self.target)
            final = (
                final_error.translation_meters,
                final_error.rotation_radians,
                final_error.gripper_closedness,
            )
            axis_penalties = []
            for before, after, tolerance in zip(initial, final, tolerances):
                if before > tolerance:
                    regression = after - before
                    axis_penalties.append(
                        0.0
                        if regression < 0.0
                        else self.failure_penalty + regression / tolerance
                    )
                else:
                    excess = after - tolerance
                    axis_penalties.append(
                        0.0
                        if excess <= 0.0
                        else self.failure_penalty + excess / tolerance
                    )
            penalties.append(sum(axis_penalties))
        return np.asarray(penalties, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_pose": list(self.start.values),
            "target_pose": list(self.target.values),
            "calibration": self.calibration.to_dict(),
            "failure_penalty": self.failure_penalty,
            "tolerances": self.tolerances.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskProgressObjective:
        try:
            return cls(
                start=DroidPose(tuple(payload["start_pose"])),
                target=DroidPose(tuple(payload["target_pose"])),
                calibration=ActionResponseCalibration.from_dict(
                    payload["calibration"]
                ),
                failure_penalty=float(payload["failure_penalty"]),
                tolerances=TaskAxisTolerances.from_dict(payload["tolerances"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("task-progress objective is incomplete") from error
