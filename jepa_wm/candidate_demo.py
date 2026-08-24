"""Typed metadata contract for a realized-candidate visualization."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from jepa_wm.action import DroidAction, DroidActionScale
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.planner import CEMConfig
from sim.recording import validate_recording_id


CANDIDATE_DEMO_SCHEMA = "quantis.jepa_wm_candidate_demo.v1"


@dataclass(frozen=True)
class CandidateDemoMetadata:
    report_id: str
    candidate_session: str
    source_session: str
    seed: int
    policy: ControlExecutionPolicy
    selected_action_scale: DroidActionScale
    candidates_scored: int
    planner: CEMConfig
    energy_improvement: float
    actual_action: DroidAction
    tracking_passed: bool
    maximum_replay_joint_error_rad: float
    maximum_replay_gripper_error_m: float
    maximum_replay_contact_force_newtons: float
    replay_collision_detected: bool

    def __post_init__(self) -> None:
        for value in (self.report_id, self.candidate_session, self.source_session):
            validate_recording_id(value)
        scalars = (
            self.energy_improvement,
            self.maximum_replay_joint_error_rad,
            self.maximum_replay_gripper_error_m,
            self.maximum_replay_contact_force_newtons,
        )
        if (
            self.policy is not ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
            or self.seed < 0
            or self.candidates_scored <= 0
            or self.candidates_scored
            != self.planner.iterations * self.planner.samples
            or not all(isfinite(value) and value >= 0.0 for value in scalars)
            or not isinstance(self.tracking_passed, bool)
            or not isinstance(self.replay_collision_detected, bool)
        ):
            raise ValueError("candidate demo metadata is invalid")

    @property
    def action_scale_label(self) -> str:
        return (
            "FULL"
            if self.selected_action_scale == DroidActionScale.uniform(1.0)
            else "PROJECTED"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CANDIDATE_DEMO_SCHEMA,
            "visualization_only": True,
            "report_id": self.report_id,
            "candidate_session": self.candidate_session,
            "source_session": self.source_session,
            "seed": self.seed,
            "policy": self.policy.value,
            "selected_action_scale": self.selected_action_scale.to_dict(),
            "candidates_scored": self.candidates_scored,
            "planner": self.planner.to_dict(),
            "energy_improvement": self.energy_improvement,
            "actual_action": list(self.actual_action.values),
            "tracking_passed": self.tracking_passed,
            "maximum_replay_joint_error_rad": self.maximum_replay_joint_error_rad,
            "maximum_replay_gripper_error_m": self.maximum_replay_gripper_error_m,
            "maximum_replay_contact_force_newtons": (
                self.maximum_replay_contact_force_newtons
            ),
            "replay_collision_detected": self.replay_collision_detected,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CandidateDemoMetadata:
        if (
            payload.get("schema") != CANDIDATE_DEMO_SCHEMA
            or payload.get("visualization_only") is not True
        ):
            raise ValueError("candidate demo metadata schema is invalid")
        try:
            planner_payload = payload["planner"]
            if not isinstance(planner_payload, Mapping):
                raise ValueError("candidate demo planner is invalid")
            planner = CEMConfig(
                horizon=int(planner_payload["horizon"]),
                iterations=int(planner_payload["iterations"]),
                samples=int(planner_payload["samples"]),
                elites=int(planner_payload["elites"]),
                seed=int(planner_payload["seed"]),
                minimum_standard_deviation=float(
                    planner_payload["minimum_standard_deviation"]
                ),
            )
            return cls(
                report_id=str(payload["report_id"]),
                candidate_session=str(payload["candidate_session"]),
                source_session=str(payload["source_session"]),
                seed=int(payload["seed"]),
                policy=ControlExecutionPolicy(payload["policy"]),
                selected_action_scale=DroidActionScale.from_payload(
                    payload["selected_action_scale"]
                ),
                candidates_scored=int(payload["candidates_scored"]),
                planner=planner,
                energy_improvement=float(payload["energy_improvement"]),
                actual_action=DroidAction(tuple(payload["actual_action"])),
                tracking_passed=payload["tracking_passed"],
                maximum_replay_joint_error_rad=float(
                    payload["maximum_replay_joint_error_rad"]
                ),
                maximum_replay_gripper_error_m=float(
                    payload["maximum_replay_gripper_error_m"]
                ),
                maximum_replay_contact_force_newtons=float(
                    payload["maximum_replay_contact_force_newtons"]
                ),
                replay_collision_detected=payload["replay_collision_detected"],
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("candidate demo metadata is incomplete") from error

    @classmethod
    def from_manifest(
        cls, manifest: Mapping[str, Any]
    ) -> CandidateDemoMetadata | None:
        metadata = manifest.get("metadata")
        if not isinstance(metadata, Mapping) or "candidate_demo" not in metadata:
            return None
        payload = metadata["candidate_demo"]
        if not isinstance(payload, Mapping):
            raise ValueError("candidate demo manifest metadata is invalid")
        return cls.from_dict(payload)
