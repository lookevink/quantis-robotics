"""Shared path and nonce policy for simulator control sessions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.training_artifact import ArtifactIdentity
from sim.recording import validate_recording_id


CONTROL_PROPOSAL_ROOT = Path("/home/ubuntu/docker/jepa-wm/checkpoints")


def requires_authenticated_control_proposal(
    policy: ControlExecutionPolicy,
) -> bool:
    """Match model-command policies to their required checkpoint evidence."""

    return policy in (
        ControlExecutionPolicy.DIRECT,
        ControlExecutionPolicy.CALIBRATION_COLLECTION,
        ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
        ControlExecutionPolicy.INSERTION_RESET_TRIAL,
        ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
        ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT,
    )


@dataclass(frozen=True)
class ControlProposalRef:
    """Authenticated checkpoint bytes resolved from one suffix-free name."""

    name: str
    checkpoint: ArtifactIdentity
    metadata: ArtifactIdentity

    def __post_init__(self) -> None:
        validate_recording_id(self.name)
        if self.name.endswith(".pth") or self.name.endswith(".json"):
            raise ValueError("control proposal requires a logical proposal name")
        if (
            self.checkpoint.path.name != f"{self.name}.pth"
            or self.metadata.path != self.checkpoint.path.with_suffix(".pth.json")
        ):
            raise ValueError("control proposal reference path is invalid")
        try:
            metadata = json.loads(self.metadata.path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "control proposal reference metadata is invalid"
            ) from error
        if (
            not isinstance(metadata, dict)
            or metadata.get("proposal_fingerprint") != self.checkpoint.fingerprint
        ):
            raise ValueError("control proposal fingerprint does not match its metadata")

    @property
    def path(self) -> Path:
        return self.checkpoint.path

    @property
    def fingerprint(self) -> str:
        return self.checkpoint.fingerprint

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        root: Path = CONTROL_PROPOSAL_ROOT,
    ) -> ControlProposalRef:
        validate_recording_id(name)
        if name.endswith(".pth") or name.endswith(".json"):
            raise ValueError("control proposal requires a logical proposal name")
        path = (root / f"{name}.pth").resolve()
        metadata_path = path.with_suffix(".pth.json")
        if not path.is_file() or not metadata_path.is_file():
            raise ValueError("control proposal reference artifact is missing")
        return cls(
            name,
            ArtifactIdentity.from_artifact(path),
            ArtifactIdentity.from_artifact(metadata_path),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> ControlProposalRef:
        if not isinstance(payload, dict) or set(payload) != {
            "name",
            "checkpoint",
            "metadata",
        }:
            raise ValueError("control proposal reference payload is invalid")
        try:
            expected_checkpoint = ArtifactIdentity.from_dict(payload["checkpoint"])
            expected_metadata = ArtifactIdentity.from_dict(payload["metadata"])
            checkpoint = ArtifactIdentity.from_artifact(expected_checkpoint.path)
            metadata = ArtifactIdentity.from_artifact(expected_metadata.path)
            if checkpoint != expected_checkpoint or metadata != expected_metadata:
                raise ValueError(
                    "control proposal reference fingerprint does not match"
                )
            return cls(str(payload["name"]), checkpoint, metadata)
        except (TypeError, ValueError) as error:
            raise ValueError("control proposal reference payload is invalid") from error

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "checkpoint": self.checkpoint.to_dict(),
            "metadata": self.metadata.to_dict(),
        }


def observation_id_for_session(session_id: str) -> int:
    """Derive a stable nonzero 64-bit request nonce from one session ID."""

    validate_recording_id(session_id)
    identifier = int.from_bytes(sha256(session_id.encode()).digest()[:8], "big")
    return identifier or 1


def control_proposal_path(proposal_name: str) -> Path:
    validate_recording_id(proposal_name)
    return CONTROL_PROPOSAL_ROOT / f"{proposal_name}.pth"
