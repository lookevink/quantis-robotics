"""Shared provenance and sidecar contract for trained JEPA-WM artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from jepa_wm.contract import MODEL_ID


def validate_artifact_fingerprint(fingerprint: str) -> str:
    if (
        len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("training artifact fingerprint is invalid")
    return fingerprint


def _validate_artifact_identity(path: Path, fingerprint: str) -> None:
    if not path.is_absolute():
        raise ValueError("training artifact identity is invalid")
    try:
        validate_artifact_fingerprint(fingerprint)
    except ValueError as error:
        raise ValueError("training artifact identity is invalid") from error


@dataclass(frozen=True)
class ProposalConditioningCapabilities:
    proprioception: bool
    action_history: bool
    goal_delta: bool
    task_progress: bool = False

    def __post_init__(self) -> None:
        if self.action_history and not self.proprioception:
            raise ValueError("action-history conditioning requires proprioception")

    @classmethod
    def from_dict(cls, payload: Any) -> ProposalConditioningCapabilities:
        legacy_fields = {"proprioception", "action_history", "goal_delta"}
        fields = legacy_fields | {"task_progress"}
        if (
            not isinstance(payload, dict)
            or set(payload) not in (legacy_fields, fields)
            or any(type(payload[field]) is not bool for field in payload)
        ):
            raise ValueError("proposal conditioning capabilities are invalid")
        values = dict(payload)
        values.setdefault("task_progress", False)
        return cls(**values)

    def to_dict(self) -> dict[str, bool]:
        return {
            "proprioception": self.proprioception,
            "action_history": self.action_history,
            "goal_delta": self.goal_delta,
            "task_progress": self.task_progress,
        }


@dataclass(frozen=True)
class TrainingRecordingSelection:
    recording: str
    context_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not self.recording
            or not self.context_indices
            or any(index < 0 for index in self.context_indices)
        ):
            raise ValueError("training recording selection is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording": self.recording,
            "context_indices": list(self.context_indices),
        }


@dataclass(frozen=True)
class ArtifactIdentity:
    path: Path
    fingerprint: str

    def __post_init__(self) -> None:
        _validate_artifact_identity(self.path, self.fingerprint)

    @classmethod
    def from_artifact(cls, artifact: Path) -> ArtifactIdentity:
        resolved = artifact.resolve()
        return cls(resolved, artifact_fingerprint(resolved))

    def to_dict(self) -> dict[str, str]:
        return {"path": str(self.path), "fingerprint": self.fingerprint}


ProposalArtifactIdentity = ArtifactIdentity


@dataclass(frozen=True)
class TrainingCorpusIdentity:
    base_model: str
    source_revision: str
    camera: str
    training_recordings: tuple[str, ...]


@dataclass(frozen=True)
class TrainingArtifactIdentity(ArtifactIdentity):
    metadata: TrainingArtifactMetadata

    @classmethod
    def from_artifact(
        cls,
        artifact: Path,
        *,
        fingerprint_field: str,
    ) -> TrainingArtifactIdentity:
        resolved = artifact.resolve()
        payload = load_training_report(resolved)
        actual_fingerprint = artifact_fingerprint(resolved)
        if payload.get(fingerprint_field) != actual_fingerprint:
            raise ValueError(
                f"{fingerprint_field} does not match the training artifact"
            )
        return cls(
            resolved,
            actual_fingerprint,
            TrainingArtifactMetadata.from_dict(payload.get("metadata")),
        )

@dataclass(frozen=True)
class TrainingArtifactMetadata:
    base_model: str
    source_revision: str
    camera: str
    training_recordings: tuple[str, ...]
    training_steps: int

    def __post_init__(self) -> None:
        if (
            self.base_model != MODEL_ID
            or not self.source_revision
            or not self.camera
            or not self.training_recordings
            or len(set(self.training_recordings)) != len(self.training_recordings)
            or any(not recording for recording in self.training_recordings)
            or self.training_steps <= 0
        ):
            raise ValueError("training artifact metadata is invalid")

    @classmethod
    def from_dict(cls, payload: Any) -> TrainingArtifactMetadata:
        if not isinstance(payload, dict):
            raise ValueError("training artifact metadata is missing")
        try:
            recordings = payload["training_recordings"]
            if not isinstance(recordings, (list, tuple)) or not all(
                isinstance(recording, str) for recording in recordings
            ):
                raise ValueError("training recordings must be a string list")
            return cls(
                base_model=str(payload["base_model"]),
                source_revision=str(payload["source_revision"]),
                camera=str(payload["camera"]),
                training_recordings=tuple(recordings),
                training_steps=int(payload["training_steps"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("training artifact metadata is incomplete") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "source_revision": self.source_revision,
            "camera": self.camera,
            "training_recordings": list(self.training_recordings),
            "training_steps": self.training_steps,
        }

    @property
    def corpus_identity(self) -> TrainingCorpusIdentity:
        return TrainingCorpusIdentity(
            self.base_model,
            self.source_revision,
            self.camera,
            self.training_recordings,
        )


def training_report_path(artifact: Path) -> Path:
    return artifact.with_suffix(artifact.suffix + ".json")


def artifact_fingerprint(artifact: Path) -> str:
    digest = sha256()
    with artifact.resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rollout_training_selection_fingerprint(payload: Any) -> str:
    """Fingerprint the exact rollout-selection contract used for training."""

    if not isinstance(payload, dict) or set(payload) != {
        "window",
        "selection_bounds",
        "recording_selections",
        "rollouts",
    }:
        raise ValueError("rollout training selection contract is invalid")
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(canonical).hexdigest()


def training_configuration_fingerprint(payload: Any) -> str:
    """Fingerprint the complete persisted training configuration."""

    if not isinstance(payload, dict) or not payload:
        raise ValueError("training configuration contract is invalid")
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("training configuration contract is invalid") from error
    return sha256(canonical).hexdigest()


def load_training_report(artifact: Path) -> dict[str, Any]:
    report = training_report_path(artifact)
    if not report.is_file():
        raise ValueError(f"training report does not exist: {report}")
    payload = json.loads(report.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"training report must be an object: {report}")
    return payload


def load_training_report_metadata(artifact: Path) -> TrainingArtifactMetadata:
    payload = load_training_report(artifact)
    return TrainingArtifactMetadata.from_dict(payload.get("metadata"))
