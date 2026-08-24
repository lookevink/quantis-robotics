"""Realized pose-outcome comparison for control-policy promotion evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from math import isfinite
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from jepa_wm.action import ActionSelectionBounds, DroidAction, DroidPose
from jepa_wm.control_protocol import (
    DROID_ACTION_HORIZON,
    ControlObservation,
    ProposedControl,
)
from jepa_wm.control_rollout import ControlRolloutReport, ControlStepSummary, PoseError
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.trajectory import load_rollout_at
from jepa_wm.trial_equivalence import (
    ResetEquivalenceTolerances,
    TrialResetState,
    validate_reset_equivalence,
)
from sim.exploration import DatasetSplit
from sim.recording import validate_recording_id


class ControlPolicy(str, Enum):
    ZERO = "zero"
    DIRECT = "direct"
    SCRIPTED = "scripted"
    EXPERIMENTAL_CANDIDATE = "experimental_candidate"


class NonModelBaselinePolicy(str, Enum):
    ZERO = "zero"
    SCRIPTED = "scripted"


def build_baseline_response(
    observation: ControlObservation,
    policy: NonModelBaselinePolicy,
    *,
    scripted_actions: tuple[DroidAction, ...] | None = None,
    created_at_unix_seconds: float | None = None,
) -> ProposedControl:
    """Build a non-model response bound to an explicit baseline identity."""

    if observation.expected_proposal.stem != f"baseline_{policy.value}":
        raise ValueError("control observation expects a different baseline policy")
    if policy is NonModelBaselinePolicy.ZERO:
        if scripted_actions is not None:
            raise ValueError("zero baseline cannot contain scripted actions")
        actions = (DroidAction((0.0,) * 7),) * DROID_ACTION_HORIZON
    else:
        if scripted_actions is None or len(scripted_actions) != DROID_ACTION_HORIZON:
            raise ValueError("scripted baseline requires one native action horizon")
        actions = scripted_actions
    return ProposedControl(
        observation.observation_id,
        time() if created_at_unix_seconds is None else created_at_unix_seconds,
        actions,
        observation.expected_proposal,
    )


def scripted_actions_at(
    recording: DomainRecording,
    context_index: int,
) -> tuple[DroidAction, ...]:
    return load_rollout_at(
        recording.path,
        camera="wrist",
        context_index=context_index,
        bounds=ActionSelectionBounds(minimum_action_norm=0.0),
    ).actions


def load_held_out_reference(path: Path, expected_seed: int) -> DomainRecording:
    recording = DomainRecording.from_path(
        path,
        expected_split=DatasetSplit.HELD_OUT,
    )
    if recording.seed != expected_seed:
        raise ValueError("baseline reference seed does not match the live session")
    return recording


@dataclass(frozen=True)
class RealizedPolicyOutcome:
    policy: ControlPolicy
    initial_error: PoseError
    final_error: PoseError

    @classmethod
    def from_step(
        cls,
        policy: ControlPolicy,
        step: ControlStepSummary,
        target: DroidPose,
    ) -> RealizedPolicyOutcome:
        final = step.post_action_pose
        if not step.is_applied or final is None:
            raise ValueError(f"{policy.value} trial did not apply one measured action")
        return cls(
            policy,
            PoseError.between(step.observation.pose, target),
            PoseError.between(final, target),
        )

    @property
    def translation_progress_meters(self) -> float:
        return self.initial_error.translation_meters - self.final_error.translation_meters

    @property
    def rotation_progress_radians(self) -> float:
        return self.initial_error.rotation_radians - self.final_error.rotation_radians

    @property
    def gripper_progress(self) -> float:
        return self.initial_error.gripper_closedness - self.final_error.gripper_closedness

    def improvement_over(self, other: RealizedPolicyOutcome) -> PoseAxisDecision:
        return PoseAxisDecision(
            self.translation_progress_meters > other.translation_progress_meters,
            self.rotation_progress_radians > other.rotation_progress_radians,
            self.gripper_progress > other.gripper_progress,
        )

    def reaches_within(
        self,
        other: RealizedPolicyOutcome,
        tolerances: ScriptedOutcomeTolerances,
    ) -> PoseAxisDecision:
        return PoseAxisDecision(
            self.final_error.translation_meters
            <= other.final_error.translation_meters
            + tolerances.maximum_translation_difference_meters,
            self.final_error.rotation_radians
            <= other.final_error.rotation_radians
            + tolerances.maximum_rotation_difference_radians,
            self.final_error.gripper_closedness
            <= other.final_error.gripper_closedness
            + tolerances.maximum_gripper_difference,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_error": self.initial_error.to_dict(),
            "final_error": self.final_error.to_dict(),
            "translation_progress_meters": self.translation_progress_meters,
            "rotation_progress_radians": self.rotation_progress_radians,
            "gripper_progress": self.gripper_progress,
        }


@dataclass(frozen=True)
class PoseAxisDecision:
    translation: bool
    rotation: bool
    gripper: bool

    @property
    def passed(self) -> bool:
        return self.translation and self.rotation and self.gripper

    def to_dict(self) -> dict[str, bool]:
        return {
            "translation": self.translation,
            "rotation": self.rotation,
            "gripper": self.gripper,
        }


@dataclass(frozen=True)
class ScriptedOutcomeTolerances:
    maximum_translation_difference_meters: float = 5e-4
    maximum_rotation_difference_radians: float = 3e-3
    maximum_gripper_difference: float = 0.01

    def __post_init__(self) -> None:
        values = (
            self.maximum_translation_difference_meters,
            self.maximum_rotation_difference_radians,
            self.maximum_gripper_difference,
        )
        if not all(isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("scripted outcome tolerances must be finite and nonnegative")


@dataclass(frozen=True)
class BaselineComparisonLimits:
    scripted_outcome: ScriptedOutcomeTolerances = ScriptedOutcomeTolerances()
    reset_equivalence: ResetEquivalenceTolerances = ResetEquivalenceTolerances()


@dataclass(frozen=True)
class RealizedBaselineComparison:
    zero: RealizedPolicyOutcome
    direct: RealizedPolicyOutcome
    scripted: RealizedPolicyOutcome
    tolerances: ScriptedOutcomeTolerances = ScriptedOutcomeTolerances()

    @classmethod
    def from_poses(
        cls,
        *,
        initial: DroidPose,
        target: DroidPose,
        zero_final: DroidPose,
        direct_final: DroidPose,
        scripted_final: DroidPose,
        tolerances: ScriptedOutcomeTolerances = ScriptedOutcomeTolerances(),
    ) -> RealizedBaselineComparison:
        return cls.from_trial_poses(
            target=target,
            zero_initial=initial,
            zero_final=zero_final,
            direct_initial=initial,
            direct_final=direct_final,
            scripted_initial=initial,
            scripted_final=scripted_final,
            tolerances=tolerances,
        )

    @classmethod
    def from_trial_poses(
        cls,
        *,
        target: DroidPose,
        zero_initial: DroidPose,
        zero_final: DroidPose,
        direct_initial: DroidPose,
        direct_final: DroidPose,
        scripted_initial: DroidPose,
        scripted_final: DroidPose,
        tolerances: ScriptedOutcomeTolerances = ScriptedOutcomeTolerances(),
    ) -> RealizedBaselineComparison:
        return cls(
            zero=RealizedPolicyOutcome(
                ControlPolicy.ZERO,
                PoseError.between(zero_initial, target),
                PoseError.between(zero_final, target),
            ),
            direct=RealizedPolicyOutcome(
                ControlPolicy.DIRECT,
                PoseError.between(direct_initial, target),
                PoseError.between(direct_final, target),
            ),
            scripted=RealizedPolicyOutcome(
                ControlPolicy.SCRIPTED,
                PoseError.between(scripted_initial, target),
                PoseError.between(scripted_final, target),
            ),
            tolerances=tolerances,
        )

    @property
    def direct_improves_over_zero(self) -> PoseAxisDecision:
        return self.direct.improvement_over(self.zero)

    @property
    def direct_reaches_scripted_tolerance(self) -> PoseAxisDecision:
        return self.direct.reaches_within(self.scripted, self.tolerances)

    @property
    def direct_baseline_gate_passed(self) -> bool:
        return (
            self.direct_improves_over_zero.passed
            and self.direct_reaches_scripted_tolerance.passed
        )

    @property
    def candidate_authority_granted(self) -> bool:
        """Baseline evidence alone can never promote a shadow candidate."""

        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcomes": {
                outcome.policy.value: outcome.to_dict()
                for outcome in (self.zero, self.direct, self.scripted)
            },
            "direct_improves_over_zero": self.direct_improves_over_zero.to_dict(),
            "direct_reaches_scripted_tolerance": (
                self.direct_reaches_scripted_tolerance.to_dict()
            ),
            "direct_baseline_gate_passed": self.direct_baseline_gate_passed,
            "candidate_authority_granted": self.candidate_authority_granted,
        }


BASELINE_REPORT_SCHEMA = "quantis.jepa_wm_realized_baselines.v1"


@dataclass(frozen=True)
class RolloutArtifact:
    rollout_id: str
    session_ids: tuple[str, ...]
    proposal: Path

    def __post_init__(self) -> None:
        validate_recording_id(self.rollout_id)
        if not self.session_ids or not self.proposal.is_absolute():
            raise ValueError("rollout artifact is invalid")


@dataclass(frozen=True)
class RealizedBaselineFirstStep:
    zero: RealizedPolicyOutcome
    direct: RealizedPolicyOutcome
    scripted: RealizedPolicyOutcome
    target_pose: DroidPose
    target_frame: Path
    direct_session_id: str


@dataclass(frozen=True)
class RealizedBaselineReport:
    experiment_id: str
    reference_recording: str
    seed: int
    direct: ControlRolloutReport
    zero: ControlRolloutReport
    scripted: ControlRolloutReport
    comparison: RealizedBaselineComparison

    @classmethod
    def from_sessions(
        cls,
        data_root: Path,
        experiment_id: str,
        *,
        reference_recording: str,
        seed: int,
        requested_steps: int,
        direct: RolloutArtifact,
        zero: RolloutArtifact,
        scripted: RolloutArtifact,
    ) -> RealizedBaselineReport:
        validate_recording_id(experiment_id)

        def load(artifact: RolloutArtifact) -> ControlRolloutReport:
            return ControlRolloutReport.from_sessions(
                data_root,
                artifact.rollout_id,
                artifact.session_ids,
                reference_recording=reference_recording,
                seed=seed,
                proposal=artifact.proposal,
                requested_steps=requested_steps,
            )

        report = cls.from_rollouts(
            experiment_id,
            load(direct),
            load(zero),
            load(scripted),
        )
        report._validate_scripted_responses(data_root)
        return report

    @classmethod
    def load_persisted(
        cls,
        data_root: Path,
        experiment_id: str,
    ) -> RealizedBaselineReport:
        validate_recording_id(experiment_id)
        path = data_root / "control_baselines" / experiment_id / "report.json"
        payload = json.loads(path.read_text())
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != BASELINE_REPORT_SCHEMA
            or payload.get("experiment_id") != experiment_id
        ):
            raise ValueError("realized baseline report identity is invalid")
        trials = payload.get("trials")
        if not isinstance(trials, dict):
            raise ValueError("realized baseline report has no trial provenance")

        def artifact(role: ControlPolicy) -> RolloutArtifact:
            trial = trials.get(role.value)
            if not isinstance(trial, dict) or not isinstance(
                trial.get("sessions"), list
            ):
                raise ValueError(f"baseline report has no {role.value} trial")
            return RolloutArtifact(
                str(trial["rollout_id"]),
                tuple(str(session) for session in trial["sessions"]),
                Path(trial["proposal"]),
            )

        direct = artifact(ControlPolicy.DIRECT)
        zero = artifact(ControlPolicy.ZERO)
        scripted = artifact(ControlPolicy.SCRIPTED)
        if not (
            len(direct.session_ids)
            == len(zero.session_ids)
            == len(scripted.session_ids)
        ):
            raise ValueError("baseline report trial lengths differ")
        return cls.from_sessions(
            data_root,
            experiment_id,
            reference_recording=str(payload["reference_recording"]),
            seed=int(payload["seed"]),
            requested_steps=len(direct.session_ids),
            direct=direct,
            zero=zero,
            scripted=scripted,
        )

    def first_step(self) -> RealizedBaselineFirstStep:
        target = self.direct.target_pose
        if target is None:
            raise ValueError("baseline report has no target pose")

        direct_step = self.direct.complete_steps[0]
        return RealizedBaselineFirstStep(
            zero=RealizedPolicyOutcome.from_step(
                ControlPolicy.ZERO, self.zero.complete_steps[0], target
            ),
            direct=RealizedPolicyOutcome.from_step(
                ControlPolicy.DIRECT, direct_step, target
            ),
            scripted=RealizedPolicyOutcome.from_step(
                ControlPolicy.SCRIPTED, self.scripted.complete_steps[0], target
            ),
            target_pose=target,
            target_frame=direct_step.observation.target_frame,
            direct_session_id=direct_step.session_id,
        )

    def _validate_scripted_responses(self, data_root: Path) -> None:
        recording = load_held_out_reference(
            data_root / "recordings" / self.reference_recording,
            self.seed,
        )
        for step in self.scripted.complete_steps:
            expected = scripted_actions_at(recording, step.observation.warmup_frames)
            if not np.allclose(
                [action.values for action in step.response.actions],
                [action.values for action in expected],
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("scripted baseline response does not match its reference")

    @staticmethod
    def _initial_pose(report: ControlRolloutReport) -> DroidPose:
        return report.complete_steps[0].observation.pose

    @staticmethod
    def _final_pose(report: ControlRolloutReport) -> DroidPose:
        pose = report.applied_steps[-1].post_action_pose
        if pose is None:
            raise ValueError("applied baseline trial has no final pose")
        return pose

    @classmethod
    def from_rollouts(
        cls,
        experiment_id: str,
        direct: ControlRolloutReport,
        zero: ControlRolloutReport,
        scripted: ControlRolloutReport,
        *,
        limits: BaselineComparisonLimits = BaselineComparisonLimits(),
    ) -> RealizedBaselineReport:
        reports = (direct, zero, scripted)
        if (
            len({report.reference_recording for report in reports}) != 1
            or len({report.seed for report in reports}) != 1
            or len({report.requested_steps for report in reports}) != 1
            or any(
                len(report.applied_steps) != report.requested_steps
                or report.orchestration_failure is not None
                or report.target_pose is None
                for report in reports
            )
        ):
            raise ValueError("baseline trials are incomplete or do not share provenance")
        if (
            zero.proposal.stem != "baseline_zero"
            or scripted.proposal.stem != "baseline_scripted"
            or direct.proposal.stem in {"baseline_zero", "baseline_scripted"}
        ):
            raise ValueError("baseline trial roles are invalid")
        if any(
            any(value != 0.0 for value in action.values)
            for step in zero.complete_steps
            for action in step.response.actions
        ):
            raise ValueError("zero baseline contains a nonzero response")
        if any(
            step.shadow is not None or step.shadow_safety is not None
            for report in (zero, scripted)
            for step in report.complete_steps
        ):
            raise ValueError("non-model baselines cannot contain shadow evidence")
        target = direct.target_pose
        if any(report.target_pose != target for report in reports[1:]):
            raise ValueError("baseline trials do not share one target pose")
        direct_initial = cls._initial_pose(direct)
        direct_state = direct.complete_steps[0].state
        reset_tolerances = limits.reset_equivalence
        for report in reports[1:]:
            state = report.complete_steps[0].state
            validate_reset_equivalence(
                TrialResetState(
                    direct_initial,
                    direct_state.current_joint_positions,
                    direct_state.collision_detected,
                    direct_state.contact_force_newtons,
                    direct_state.plug_position,
                    direct_state.plug_attached,
                ),
                TrialResetState(
                    cls._initial_pose(report),
                    state.current_joint_positions,
                    state.collision_detected,
                    state.contact_force_newtons,
                    state.plug_position,
                    state.plug_attached,
                ),
                tolerances=reset_tolerances,
            )
        comparison = RealizedBaselineComparison.from_trial_poses(
            target=target,
            zero_initial=cls._initial_pose(zero),
            zero_final=cls._final_pose(zero),
            direct_initial=direct_initial,
            direct_final=cls._final_pose(direct),
            scripted_initial=cls._initial_pose(scripted),
            scripted_final=cls._final_pose(scripted),
            tolerances=limits.scripted_outcome,
        )
        return cls(
            experiment_id,
            direct.reference_recording,
            direct.seed,
            direct,
            zero,
            scripted,
            comparison,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": BASELINE_REPORT_SCHEMA,
            "experiment_id": self.experiment_id,
            "reference_recording": self.reference_recording,
            "seed": self.seed,
            "trials": {
                policy.value: {
                    "rollout_id": report.rollout_id,
                    "proposal": str(report.proposal),
                    "sessions": [
                        step.session_id for step in report.complete_steps
                    ],
                }
                for policy, report in (
                    (ControlPolicy.DIRECT, self.direct),
                    (ControlPolicy.ZERO, self.zero),
                    (ControlPolicy.SCRIPTED, self.scripted),
                )
            },
            **self.comparison.to_dict(),
        }
