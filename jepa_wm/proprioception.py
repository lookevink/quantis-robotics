"""Normalization contracts for proposal proprioception and task inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from jepa_wm.action import ACTION_DIMENSIONS


@dataclass(frozen=True)
class ScalarNormalization:
    mean: float
    standard_deviation: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.mean)
            or not np.isfinite(self.standard_deviation)
            or self.standard_deviation <= 0.0
        ):
            raise ValueError("scalar normalization must be finite and positive")

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        *,
        minimum_standard_deviation: float = 1e-3,
    ) -> ScalarNormalization:
        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        if (
            not len(values)
            or not np.all(np.isfinite(values))
            or minimum_standard_deviation <= 0.0
        ):
            raise ValueError("scalar samples and deviation floor must be valid")
        return cls(
            float(values.mean()),
            max(float(values.std()), minimum_standard_deviation),
        )


@dataclass(frozen=True)
class DroidValueNormalization:
    mean: np.ndarray
    standard_deviation: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float32)
        standard_deviation = np.asarray(self.standard_deviation, dtype=np.float32)
        expected_shape = (ACTION_DIMENSIONS,)
        if mean.shape != expected_shape or standard_deviation.shape != expected_shape:
            raise ValueError("DROID normalization must have shape [7]")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(standard_deviation)):
            raise ValueError("DROID normalization must be finite")
        if np.any(standard_deviation <= 0):
            raise ValueError("DROID normalization standard deviation must be positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", standard_deviation)

    @classmethod
    def from_samples(
        cls,
        samples: np.ndarray,
        *,
        minimum_standard_deviation: float = 1e-3,
    ) -> DroidValueNormalization:
        values = np.asarray(samples, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != ACTION_DIMENSIONS:
            raise ValueError("DROID values must have shape [samples, 7]")
        if not len(values) or not np.all(np.isfinite(values)):
            raise ValueError("DROID values must contain finite samples")
        if minimum_standard_deviation <= 0:
            raise ValueError("minimum standard deviation must be positive")
        return cls(
            values.mean(axis=0),
            np.maximum(values.std(axis=0), minimum_standard_deviation),
        )

    def standardize(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float32)
        if values.shape[-1:] != (ACTION_DIMENSIONS,):
            raise ValueError("DROID values must end with a seven-value dimension")
        return (values - self.mean) / self.standard_deviation

    def restore(self, standardized_samples: np.ndarray) -> np.ndarray:
        values = np.asarray(standardized_samples, dtype=np.float32)
        if values.shape[-1:] != (ACTION_DIMENSIONS,):
            raise ValueError(
                "standardized values must end with a seven-value dimension"
            )
        return values * self.standard_deviation + self.mean
