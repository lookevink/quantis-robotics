"""Typed provenance and replay metrics for a validated grasp visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jepa_wm.grasp_task import ReachAndGraspDecision
from jepa_wm.replay_verification import ReplayVerification
from jepa_wm.training_artifact import ArtifactIdentity
from sim.recording import validate_recording_id


GRASP_DEMO_SCHEMA = "quantis.jepa_wm_grasp_demo.v1"


@dataclass(frozen=True)
class GraspDemoMetadata:
    readiness_id: str
    baseline_experiment_id: str
    rollout_id: str
    seed: int
    proposal: ArtifactIdentity
    source_steps: int
    task_outcome: ReachAndGraspDecision
    replay: ReplayVerification

    def __post_init__(self) -> None:
        for value in (
            self.readiness_id,
            self.baseline_experiment_id,
            self.rollout_id,
        ):
            validate_recording_id(value)
        if (
            self.seed < 0
            or self.source_steps <= 0
            or not self.task_outcome.passed
            or self.task_outcome.attached_observations > self.source_steps
            or not self.replay.tracking_passed
            or not self.replay.safety_passed
        ):
            raise ValueError("grasp demo metadata is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GRASP_DEMO_SCHEMA,
            "visualization_only": True,
            "readiness_id": self.readiness_id,
            "baseline_experiment_id": self.baseline_experiment_id,
            "rollout_id": self.rollout_id,
            "seed": self.seed,
            "proposal": self.proposal.to_dict(),
            "source_steps": self.source_steps,
            "task_outcome": self.task_outcome.to_dict(),
            **self.replay.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GraspDemoMetadata:
        if (
            payload.get("schema") != GRASP_DEMO_SCHEMA
            or payload.get("visualization_only") is not True
        ):
            raise ValueError("grasp demo metadata schema is invalid")
        try:
            proposal = payload["proposal"]
            outcome = payload["task_outcome"]
            if not isinstance(proposal, Mapping) or not isinstance(outcome, Mapping):
                raise ValueError("grasp demo nested metadata is invalid")
            decision = ReachAndGraspDecision.from_dict(outcome)
            return cls(
                readiness_id=str(payload["readiness_id"]),
                baseline_experiment_id=str(payload["baseline_experiment_id"]),
                rollout_id=str(payload["rollout_id"]),
                seed=int(payload["seed"]),
                proposal=ArtifactIdentity(
                    Path(proposal["path"]), str(proposal["fingerprint"])
                ),
                source_steps=int(payload["source_steps"]),
                task_outcome=decision,
                replay=ReplayVerification.from_dict(payload),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("grasp demo metadata is incomplete") from error

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> GraspDemoMetadata | None:
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping) or "grasp_demo" not in metadata:
            return None
        payload = metadata["grasp_demo"]
        if not isinstance(payload, Mapping):
            raise ValueError("grasp demo manifest metadata is invalid")
        return cls.from_dict(payload)
