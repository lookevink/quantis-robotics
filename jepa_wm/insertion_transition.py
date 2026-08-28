"""Train-only supervision for the grasp-to-insertion distribution gap."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
from hashlib import sha256
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_protocol import (
    DROID_ACTION_HORIZON,
    ControlObservation,
    ProposedControl,
)
from jepa_wm.control_safety import ControlGateReason, INSERTION_TARGET_PROGRESS
from jepa_wm.insertion_contract import (
    MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.identifiers import validate_safe_identifier
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.training_artifact import (
    artifact_fingerprint,
    load_training_report,
    training_report_path,
)
from jepa_wm.trajectory import RecordedFrame, RecordedRollout

if TYPE_CHECKING:
    from jepa_wm.proposal import ProposalTrainingFingerprints


INSERTION_TRANSITION_EXAMPLE_SCHEMA = "quantis.jepa_wm_insertion_transition_example.v1"
INSERTION_TRANSITION_FINETUNE_SCHEMA = (
    "quantis.jepa_wm_insertion_transition_finetune.v1"
)
INSERTION_TRANSITION_OUTPUT_CONSTRAINT = "translation_only"
INSERTION_TRANSITION_ZERO_AXIS_TOLERANCE = 1e-9
INSERTION_PROPOSAL_HANDOFF_SCHEMA = "quantis.jepa_wm_insertion_proposal_handoff.v1"
INSERTION_TRANSITION_DESCENDANT_HANDOFF_SCHEMA = (
    "quantis.jepa_wm_insertion_transition_descendant_handoff.v1"
)
_LEGACY_INSERTION_TRANSITION_SELECTION_FIELDS = {
    "parent",
    "transition_example",
    "transition_example_fingerprint",
    "evaluation_exclusions",
    "rehearsal_recordings",
    "rehearsal_context_indices",
}
_CURRENT_INSERTION_TRANSITION_SELECTION_FIELDS = (
    _LEGACY_INSERTION_TRANSITION_SELECTION_FIELDS | {"rehearsal_transition_examples"}
)


def transition_training_selection_fingerprint(payload: Any) -> str:
    """Fingerprint the complete persisted bridge-training selection."""

    return InsertionTransitionTrainingSelection.from_dict(payload).fingerprint


def bounded_insertion_transition_cosine(value: float) -> float:
    """Bound one finite measured cosine to its mathematical range."""

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError("insertion transition prediction cosine is invalid")
    return max(-1.0, min(1.0, float(value)))


def _checkpoint_training_fingerprints(
    proposal: Path,
) -> ProposalTrainingFingerprints:
    from jepa_wm.proposal import action_proposal_training_fingerprints

    return action_proposal_training_fingerprints(proposal)


def _frame_index(path: Path) -> int:
    name = path.stem
    if not name.startswith("frame_"):
        raise ValueError("insertion transition frame name is invalid")
    suffix = name.removeprefix("frame_")
    if not suffix.isdigit():
        raise ValueError("insertion transition frame index is invalid")
    return int(suffix)


@dataclass(frozen=True)
class InsertionTransitionSupervisionPolicy:
    """Label a safe translation-only bridge toward the selected insertion goal."""

    action_horizon: int = DROID_ACTION_HORIZON
    maximum_action_translation_meters: float | None = (
        0.96 * MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS
    )

    def __post_init__(self) -> None:
        if self.action_horizon != DROID_ACTION_HORIZON or (
            self.maximum_action_translation_meters is not None
            and (
                not isfinite(self.maximum_action_translation_meters)
                or self.maximum_action_translation_meters <= 0.0
                or self.maximum_action_translation_meters
                > MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS
            )
        ):
            raise ValueError("insertion transition must use the native action horizon")

    def actions(self, start: DroidPose, target: DroidPose) -> tuple[DroidAction, ...]:
        translation = tuple(
            (target.values[axis] - start.values[axis]) / self.action_horizon
            for axis in range(3)
        )
        magnitude = sqrt(sum(value * value for value in translation))
        if not all(isfinite(value) for value in translation) or magnitude <= 0.0:
            raise ValueError("insertion transition target must require translation")
        if (
            self.maximum_action_translation_meters is not None
            and magnitude > self.maximum_action_translation_meters
        ):
            scale = self.maximum_action_translation_meters / magnitude
            translation = tuple(value * scale for value in translation)
        action = DroidAction((*translation, 0.0, 0.0, 0.0, 0.0))
        return (action,) * self.action_horizon

    def to_dict(self) -> dict[str, int | float]:
        payload: dict[str, int | float] = {"action_horizon": self.action_horizon}
        if self.maximum_action_translation_meters is not None:
            payload["maximum_action_translation_meters"] = (
                self.maximum_action_translation_meters
            )
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTransitionSupervisionPolicy:
        if (
            not isinstance(payload, dict)
            or not set(payload).issubset(
                {"action_horizon", "maximum_action_translation_meters"}
            )
            or "action_horizon" not in payload
        ):
            raise ValueError("insertion transition supervision policy is invalid")
        value = payload["action_horizon"]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("insertion transition action horizon is invalid")
        maximum = payload.get("maximum_action_translation_meters")
        if maximum is None:
            return cls(value, None)
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
            raise ValueError("insertion transition action limit is invalid")
        return cls(value, float(maximum))


@dataclass(frozen=True)
class InsertionTransitionExample:
    """One authenticated no-actuation failure relabeled as train-only supervision."""

    source_session_id: str
    reference_recording: str
    seed: int
    observation_id: int
    context_frame: Path
    target_frame: Path
    context_pose: DroidPose
    target_pose: DroidPose
    previous_action: DroidAction
    task_context_index: int
    source_proposal: ArtifactIdentity
    actions: tuple[DroidAction, ...]
    supervision: InsertionTransitionSupervisionPolicy = (
        InsertionTransitionSupervisionPolicy()
    )

    def __post_init__(self) -> None:
        if (
            not self.source_session_id
            or not self.reference_recording
            or isinstance(self.seed, bool)
            or self.seed < 0
            or isinstance(self.observation_id, bool)
            or self.observation_id <= 0
            or isinstance(self.task_context_index, bool)
            or self.task_context_index < 0
            or self.context_frame.is_absolute()
            or self.target_frame.is_absolute()
            or len(self.actions) != DROID_ACTION_HORIZON
            or self.actions
            != self.supervision.actions(self.context_pose, self.target_pose)
        ):
            raise ValueError("insertion transition example is invalid")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()

    @property
    def target_progress_admissible(self) -> bool:
        """Whether any safe full-scale translation can pass this target gate."""

        return INSERTION_TARGET_PROGRESS.translation_bound_can_satisfy(
            self.context_pose,
            self.target_pose,
            MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS,
        )

    def rollout(self, data_root: Path) -> RecordedRollout:
        root = data_root.resolve()
        context_path = (root / self.context_frame).resolve()
        target_path = (root / self.target_frame).resolve()
        for path in (context_path, target_path):
            try:
                path.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    "insertion transition frame escapes the data root"
                ) from error
            if not path.is_file():
                raise ValueError(f"insertion transition frame is missing: {path}")
        return RecordedRollout(
            context=(RecordedFrame(self.task_context_index, context_path),),
            context_pose=self.context_pose,
            previous_action=self.previous_action,
            target=RecordedFrame(_frame_index(self.target_frame), target_path),
            target_pose=self.target_pose,
            actions=self.actions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INSERTION_TRANSITION_EXAMPLE_SCHEMA,
            "source_session_id": self.source_session_id,
            "reference_recording": self.reference_recording,
            "seed": self.seed,
            "observation_id": self.observation_id,
            "context_frame": str(self.context_frame),
            "target_frame": str(self.target_frame),
            "context_pose": list(self.context_pose.values),
            "target_pose": list(self.target_pose.values),
            "previous_action": list(self.previous_action.values),
            "task_context_index": self.task_context_index,
            "source_proposal": self.source_proposal.to_dict(),
            "actions": [list(action.values) for action in self.actions],
            "supervision": self.supervision.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTransitionExample:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != INSERTION_TRANSITION_EXAMPLE_SCHEMA
        ):
            raise ValueError("insertion transition example schema is invalid")
        try:
            seed = payload["seed"]
            observation_id = payload["observation_id"]
            task_context_index = payload["task_context_index"]
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (seed, observation_id, task_context_index)
            ):
                raise ValueError("insertion transition integer provenance is invalid")
            return cls(
                source_session_id=str(payload["source_session_id"]),
                reference_recording=str(payload["reference_recording"]),
                seed=seed,
                observation_id=observation_id,
                context_frame=Path(payload["context_frame"]),
                target_frame=Path(payload["target_frame"]),
                context_pose=DroidPose(tuple(payload["context_pose"])),
                target_pose=DroidPose(tuple(payload["target_pose"])),
                previous_action=DroidAction(tuple(payload["previous_action"])),
                task_context_index=task_context_index,
                source_proposal=ArtifactIdentity.from_dict(payload["source_proposal"]),
                actions=tuple(
                    DroidAction(tuple(values)) for values in payload["actions"]
                ),
                supervision=InsertionTransitionSupervisionPolicy.from_dict(
                    payload["supervision"]
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion transition example is incomplete") from error


@dataclass(frozen=True)
class InsertionTransitionTrainingSelection:
    """Exact hard-example and rehearsal contract for one transition artifact."""

    parent: ArtifactIdentity
    transition_example: InsertionTransitionExample
    evaluation_exclusions: tuple[str, ...]
    rehearsal_recordings: tuple[str, ...]
    rehearsal_context_indices: tuple[int, ...]
    rehearsal_transition_examples: tuple[InsertionTransitionExample, ...] | None = None

    def __post_init__(self) -> None:
        if (
            self.transition_example.source_proposal != self.parent
            or not self.evaluation_exclusions
            or not self.rehearsal_recordings
            or not self.rehearsal_context_indices
            or any(
                not isinstance(value, str) or not value
                for value in (*self.evaluation_exclusions, *self.rehearsal_recordings)
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.rehearsal_context_indices
            )
        ):
            raise ValueError("insertion transition training selection is invalid")
        for value in (*self.evaluation_exclusions, *self.rehearsal_recordings):
            validate_safe_identifier(value)
        if self.rehearsal_transition_examples is not None:
            expected_exclusions = tuple(
                dict.fromkeys(
                    item.reference_recording
                    for item in (
                        *self.rehearsal_transition_examples,
                        self.transition_example,
                    )
                )
            )
            if self.evaluation_exclusions != expected_exclusions:
                raise ValueError("insertion transition rehearsal provenance is invalid")

    @property
    def actionable_rehearsal_transition_examples(
        self,
    ) -> tuple[InsertionTransitionExample, ...]:
        return tuple(
            example
            for example in self.rehearsal_transition_examples or ()
            if example.target_progress_admissible
        )

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return sha256(canonical).hexdigest()

    def validate_rehearsal(
        self,
        expected: tuple[InsertionTransitionExample, ...],
    ) -> None:
        if self.rehearsal_transition_examples is None:
            return
        if self.rehearsal_transition_examples != expected:
            raise ValueError("insertion transition rehearsal provenance is invalid")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "parent": self.parent.to_dict(),
            "transition_example": self.transition_example.to_dict(),
            "transition_example_fingerprint": self.transition_example.fingerprint,
            "evaluation_exclusions": list(self.evaluation_exclusions),
            "rehearsal_recordings": list(self.rehearsal_recordings),
            "rehearsal_context_indices": list(self.rehearsal_context_indices),
        }
        if self.rehearsal_transition_examples is not None:
            payload["rehearsal_transition_examples"] = [
                example.to_dict() for example in self.rehearsal_transition_examples
            ]
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTransitionTrainingSelection:
        if not isinstance(payload, dict) or set(payload) not in (
            _LEGACY_INSERTION_TRANSITION_SELECTION_FIELDS,
            _CURRENT_INSERTION_TRANSITION_SELECTION_FIELDS,
        ):
            raise ValueError("insertion transition training selection is invalid")
        try:
            example = InsertionTransitionExample.from_dict(
                payload["transition_example"]
            )
            if payload["transition_example_fingerprint"] != example.fingerprint:
                raise ValueError("insertion transition example fingerprint is invalid")
            rehearsal_payload = payload.get("rehearsal_transition_examples")
            rehearsal_examples = (
                None
                if rehearsal_payload is None
                else tuple(
                    InsertionTransitionExample.from_dict(item)
                    for item in rehearsal_payload
                )
            )
            return cls(
                parent=ArtifactIdentity.from_dict(payload["parent"]),
                transition_example=example,
                evaluation_exclusions=tuple(payload["evaluation_exclusions"]),
                rehearsal_recordings=tuple(payload["rehearsal_recordings"]),
                rehearsal_context_indices=tuple(payload["rehearsal_context_indices"]),
                rehearsal_transition_examples=rehearsal_examples,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "insertion transition training selection is invalid"
            ) from error


@dataclass(frozen=True)
class InsertionTransitionHardExampleEvaluation:
    """Reconstructible result of checking one retained hard example."""

    source_session_id: str
    first_action_goal_cosine: float
    predicted_actions: tuple[DroidAction, ...]
    failure_reason: ControlGateReason | None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_session_id, str)
            or not self.source_session_id
            or isinstance(self.first_action_goal_cosine, bool)
            or not isinstance(self.first_action_goal_cosine, (int, float))
            or not isfinite(self.first_action_goal_cosine)
            or not -1.0 <= self.first_action_goal_cosine <= 1.0
            or len(self.predicted_actions) != DROID_ACTION_HORIZON
            or (
                self.failure_reason is not None
                and not isinstance(self.failure_reason, ControlGateReason)
            )
        ):
            raise ValueError("insertion transition hard evaluation is invalid")

    @property
    def passed(self) -> bool:
        return self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_session_id": self.source_session_id,
            "first_action_goal_cosine": self.first_action_goal_cosine,
            "passed": self.passed,
            "failure_reason": (
                None if self.failure_reason is None else self.failure_reason.value
            ),
            "predicted_actions": [
                list(action.values) for action in self.predicted_actions
            ],
        }

    @classmethod
    def from_prediction(
        cls,
        example: InsertionTransitionExample,
        predicted_actions: tuple[DroidAction, ...],
    ) -> InsertionTransitionHardExampleEvaluation:
        """Build evaluated evidence from one model prediction."""

        evaluation = cls(
            source_session_id=example.source_session_id,
            first_action_goal_cosine=_prediction_goal_cosine(
                example,
                predicted_actions,
            ),
            predicted_actions=predicted_actions,
            failure_reason=_hard_prediction_failure_reason(
                example,
                predicted_actions,
            ),
        )
        evaluation.validate_example(example)
        return evaluation

    def validate_example(self, example: InsertionTransitionExample) -> None:
        reconstructed_cosine = _prediction_goal_cosine(
            example,
            self.predicted_actions,
        )
        reason = _hard_prediction_failure_reason(example, self.predicted_actions)
        if (
            self.source_session_id != example.source_session_id
            or self.failure_reason is not reason
            or not isclose(
                self.first_action_goal_cosine,
                reconstructed_cosine,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            raise ValueError("insertion transition hard evaluation is not bound")

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTransitionHardExampleEvaluation:
        if not isinstance(payload, dict) or set(payload) != {
            "source_session_id",
            "first_action_goal_cosine",
            "passed",
            "failure_reason",
            "predicted_actions",
        }:
            raise ValueError("insertion transition hard evaluation is invalid")
        try:
            source_session_id = payload["source_session_id"]
            cosine = payload["first_action_goal_cosine"]
            passed = payload["passed"]
            if (
                not isinstance(source_session_id, str)
                or isinstance(cosine, bool)
                or not isinstance(cosine, (int, float))
                or not isinstance(passed, bool)
            ):
                raise ValueError("insertion transition hard evaluation type is invalid")
            reason_payload = payload["failure_reason"]
            evaluation = cls(
                source_session_id=source_session_id,
                first_action_goal_cosine=float(cosine),
                predicted_actions=tuple(
                    DroidAction(tuple(values))
                    for values in payload["predicted_actions"]
                ),
                failure_reason=(
                    None
                    if reason_payload is None
                    else ControlGateReason(reason_payload)
                ),
            )
            if passed is not evaluation.passed:
                raise ValueError("insertion transition pass claim is invalid")
            return evaluation
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "insertion transition hard evaluation is invalid"
            ) from error


@dataclass(frozen=True, order=True)
class InsertionTransitionCandidateRank:
    """Lexicographic rank for retaining one trained transition candidate."""

    failed_hard_examples: int
    hard_objective: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.failed_hard_examples, bool)
            or not isinstance(self.failed_hard_examples, int)
            or self.failed_hard_examples < 0
            or isinstance(self.hard_objective, bool)
            or not isinstance(self.hard_objective, (int, float))
            or not isfinite(self.hard_objective)
            or self.hard_objective < 0.0
        ):
            raise ValueError("insertion transition candidate rank is invalid")

    @classmethod
    def from_evaluations(
        cls,
        evaluations: tuple[InsertionTransitionHardExampleEvaluation, ...],
        *,
        hard_objective: float,
    ) -> InsertionTransitionCandidateRank:
        if not evaluations:
            raise ValueError("insertion transition candidate evaluations are empty")
        return cls(
            failed_hard_examples=sum(
                not evaluation.passed for evaluation in evaluations
            ),
            hard_objective=hard_objective,
        )


def _prediction_goal_cosine(
    example: InsertionTransitionExample,
    predicted_actions: tuple[DroidAction, ...],
) -> float:
    if len(predicted_actions) != DROID_ACTION_HORIZON:
        raise ValueError("insertion transition hard prediction is invalid")
    first_translation = predicted_actions[0].values[:3]
    goal_translation = tuple(
        example.target_pose.values[axis] - example.context_pose.values[axis]
        for axis in range(3)
    )
    first_norm = sqrt(sum(value * value for value in first_translation))
    goal_norm = sqrt(sum(value * value for value in goal_translation))
    return bounded_insertion_transition_cosine(
        0.0
        if first_norm <= 1e-12 or goal_norm <= 1e-12
        else sum(
            action * goal for action, goal in zip(first_translation, goal_translation)
        )
        / (first_norm * goal_norm)
    )


def _hard_prediction_failure_reason(
    example: InsertionTransitionExample,
    predicted_actions: tuple[DroidAction, ...],
) -> ControlGateReason | None:
    if len(predicted_actions) != DROID_ACTION_HORIZON:
        raise ValueError("insertion transition hard prediction is invalid")
    if any(
        sqrt(sum(value * value for value in action.values[:3]))
        > MAXIMUM_FULL_SCALE_INSERTION_TRANSLATION_METERS + 1e-12
        or any(
            abs(value) > INSERTION_TRANSITION_ZERO_AXIS_TOLERANCE
            for value in action.values[3:]
        )
        for action in predicted_actions
    ):
        return ControlGateReason.ACTION_OUT_OF_BOUNDS
    return INSERTION_TARGET_PROGRESS.failure_reason(
        example.context_pose,
        example.target_pose,
        example.context_pose.applied(predicted_actions[0]),
    )


def transition_hard_evaluations_fingerprint(
    evaluations: tuple[InsertionTransitionHardExampleEvaluation, ...],
) -> str:
    """Fingerprint the exact reconstructed hard-example acceptance roster."""

    if not evaluations:
        raise ValueError("insertion transition hard evaluations are empty")
    encoded = json.dumps(
        [evaluation.to_dict() for evaluation in evaluations],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(encoded).hexdigest()


def _validated_hard_evaluations(
    report: dict[str, Any],
    selection: InsertionTransitionTrainingSelection,
    prior_examples: tuple[InsertionTransitionExample, ...],
    checkpoint_fingerprints: ProposalTrainingFingerprints,
) -> tuple[InsertionTransitionHardExampleEvaluation, ...]:
    if selection.rehearsal_transition_examples is None:
        if (
            report.get("hard_example_evaluations") is not None
            or report.get("hard_example_evaluations_fingerprint") is not None
            or checkpoint_fingerprints.evaluation is not None
        ):
            raise ValueError("legacy insertion transition evaluation is invalid")
        return ()
    payload = report.get("hard_example_evaluations")
    if not isinstance(payload, list):
        raise ValueError("insertion transition hard evaluations are missing")
    try:
        evaluations = tuple(
            InsertionTransitionHardExampleEvaluation.from_dict(item) for item in payload
        )
    except ValueError as error:
        raise ValueError("insertion transition hard evaluations are invalid") from error
    examples = (
        selection.transition_example,
        *selection.actionable_rehearsal_transition_examples,
    )
    if len(evaluations) != len(examples):
        raise ValueError("insertion transition hard evaluation roster is invalid")
    for evaluation, example in zip(evaluations, examples):
        evaluation.validate_example(example)
    fingerprint = transition_hard_evaluations_fingerprint(evaluations)
    if (
        not all(evaluation.passed for evaluation in evaluations)
        or report.get("hard_example_evaluations_fingerprint") != fingerprint
        or checkpoint_fingerprints.evaluation != fingerprint
    ):
        raise ValueError("insertion transition hard evaluation fingerprint is invalid")
    return evaluations


def transition_parent_proposal(bridge_path: Path) -> ArtifactIdentity:
    """Authenticate a constrained bridge artifact and return its frozen parent."""

    bridge = bridge_path.resolve()
    report = load_training_report(bridge)
    try:
        selection = InsertionTransitionTrainingSelection.from_dict(
            report.get("training_selection")
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("insertion transition parent provenance is invalid") from error
    checkpoint_fingerprints = _checkpoint_training_fingerprints(bridge)
    if (
        report.get("schema") != INSERTION_TRANSITION_FINETUNE_SCHEMA
        or report.get("status") != "trained"
        or report.get("output_constraint") != INSERTION_TRANSITION_OUTPUT_CONSTRAINT
        or Path(str(report.get("proposal"))).resolve() != bridge
        or report.get("proposal_fingerprint") != artifact_fingerprint(bridge)
        or report.get("training_selection_fingerprint") != selection.fingerprint
        or checkpoint_fingerprints.selection != selection.fingerprint
        or ArtifactIdentity.from_artifact(selection.parent.path) != selection.parent
        or selection.parent.path.resolve() == bridge
    ):
        raise ValueError("insertion transition parent fingerprint is invalid")
    prior_examples = _transition_examples_for_parent(selection.parent)
    selection.validate_rehearsal(prior_examples)
    _validated_hard_evaluations(
        report,
        selection,
        prior_examples,
        checkpoint_fingerprints,
    )
    return selection.parent


def _transition_examples_for_parent(
    parent: ArtifactIdentity,
) -> tuple[InsertionTransitionExample, ...]:
    report_path = training_report_path(parent.path.resolve())
    if report_path.is_file():
        report = load_training_report(parent.path)
        if report.get("schema") == INSERTION_TRANSITION_FINETUNE_SCHEMA:
            return transition_training_examples(parent.path)
    from jepa_wm.insertion_proposal_readiness import validate_insertion_proposal

    if validate_insertion_proposal(parent.path).identity != parent:
        raise ValueError("insertion transition base parent is invalid")
    return ()


def transition_training_examples(
    proposal_path: Path,
) -> tuple[InsertionTransitionExample, ...]:
    """Return every authenticated hard example in one transition lineage."""

    proposal = proposal_path.resolve()
    parent = transition_parent_proposal(proposal)
    report = load_training_report(proposal)
    try:
        selection = InsertionTransitionTrainingSelection.from_dict(
            report["training_selection"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("insertion transition example lineage is invalid") from error
    return (
        *_transition_examples_for_parent(parent),
        selection.transition_example,
    )


def validate_insertion_transition_proposal(
    proposal_path: Path,
) -> ArtifactIdentity:
    """Authenticate one constrained transition child as a reusable parent."""

    proposal = proposal_path.resolve()
    transition_parent_proposal(proposal)
    return ArtifactIdentity.from_artifact(proposal)


@dataclass(frozen=True)
class InsertionProposalHandoff:
    """Host-authenticated one-way transition from a bridge to its frozen parent."""

    bridge: ArtifactIdentity
    parent: ArtifactIdentity

    @property
    def previous(self) -> ArtifactIdentity:
        return self.bridge

    @property
    def requested(self) -> ArtifactIdentity:
        return self.parent

    def __post_init__(self) -> None:
        if self.bridge == self.parent:
            raise ValueError("insertion proposal handoff requires distinct artifacts")

    @classmethod
    def from_bridge(
        cls,
        bridge_path: Path,
        requested_parent_path: Path,
    ) -> InsertionProposalHandoff:
        bridge = ArtifactIdentity.from_artifact(bridge_path.resolve())
        parent = transition_parent_proposal(bridge.path)
        if ArtifactIdentity.from_artifact(requested_parent_path.resolve()) != parent:
            raise ValueError(
                "insertion follow-up proposal is not the bridge's exact frozen parent"
            )
        return cls(bridge, parent)

    def resolve(
        self,
        previous_proposal: Path,
        previous_proposal_fingerprint: str | None,
        requested_proposal: Path,
    ) -> Path:
        if (
            previous_proposal.resolve() != self.bridge.path.resolve()
            or previous_proposal_fingerprint != self.bridge.fingerprint
        ):
            raise ValueError("insertion proposal handoff bridge response is invalid")
        requested = requested_proposal.resolve()
        if requested != self.parent.path.resolve():
            raise ValueError(
                "insertion follow-up proposal is not the bridge's exact frozen parent"
            )
        return requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INSERTION_PROPOSAL_HANDOFF_SCHEMA,
            "bridge": self.bridge.to_dict(),
            "parent": self.parent.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionProposalHandoff:
        if (
            not isinstance(payload, dict)
            or set(payload) != {"schema", "bridge", "parent"}
            or payload.get("schema") != INSERTION_PROPOSAL_HANDOFF_SCHEMA
        ):
            raise ValueError("insertion proposal handoff is invalid")
        try:
            return cls(
                ArtifactIdentity.from_dict(payload["bridge"]),
                ArtifactIdentity.from_dict(payload["parent"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("insertion proposal handoff is invalid") from error


@dataclass(frozen=True)
class InsertionTransitionDescendantHandoff:
    """Authorize a child trained on an exact no-actuation continuation source."""

    previous: ArtifactIdentity
    training_parent: ArtifactIdentity
    descendant: ArtifactIdentity
    source_session_id: str

    def __post_init__(self) -> None:
        validate_safe_identifier(self.source_session_id)
        if self.descendant == self.previous or self.descendant == self.training_parent:
            raise ValueError("insertion transition descendant handoff is invalid")

    @property
    def requested(self) -> ArtifactIdentity:
        return self.descendant

    def resolve(
        self,
        previous_proposal: Path,
        previous_proposal_fingerprint: str | None,
        requested_proposal: Path,
    ) -> Path:
        if (
            previous_proposal.resolve() != self.previous.path.resolve()
            or previous_proposal_fingerprint != self.previous.fingerprint
        ):
            raise ValueError(
                "insertion descendant handoff previous response is invalid"
            )
        requested = requested_proposal.resolve()
        if requested != self.descendant.path.resolve():
            raise ValueError("insertion descendant handoff proposal is invalid")
        return requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INSERTION_TRANSITION_DESCENDANT_HANDOFF_SCHEMA,
            "previous": self.previous.to_dict(),
            "training_parent": self.training_parent.to_dict(),
            "descendant": self.descendant.to_dict(),
            "source_session_id": self.source_session_id,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionTransitionDescendantHandoff:
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "schema",
                "previous",
                "training_parent",
                "descendant",
                "source_session_id",
            }
            or payload.get("schema") != INSERTION_TRANSITION_DESCENDANT_HANDOFF_SCHEMA
        ):
            raise ValueError("insertion transition descendant handoff is invalid")
        try:
            return cls(
                ArtifactIdentity.from_dict(payload["previous"]),
                ArtifactIdentity.from_dict(payload["training_parent"]),
                ArtifactIdentity.from_dict(payload["descendant"]),
                str(payload["source_session_id"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "insertion transition descendant handoff is invalid"
            ) from error


InsertionProposalContinuation = Union[
    InsertionProposalHandoff,
    InsertionTransitionDescendantHandoff,
]


def insertion_proposal_continuation_from_dict(
    payload: Any,
) -> InsertionProposalContinuation:
    if (
        isinstance(payload, dict)
        and payload.get("schema") == INSERTION_PROPOSAL_HANDOFF_SCHEMA
    ):
        return InsertionProposalHandoff.from_dict(payload)
    return InsertionTransitionDescendantHandoff.from_dict(payload)


def resolve_insertion_followup_proposal(
    previous_proposal: Path,
    requested_proposal: Path,
    *,
    previous_proposal_fingerprint: str | None = None,
    handoff: InsertionProposalContinuation | None = None,
) -> Path:
    """Allow an unchanged proposal or the bridge's exact authenticated parent."""

    previous = previous_proposal.resolve()
    requested = requested_proposal.resolve()
    if previous == requested:
        return requested
    if handoff is not None:
        return handoff.resolve(
            previous,
            previous_proposal_fingerprint,
            requested,
        )
    parent = transition_parent_proposal(previous)
    if ArtifactIdentity.from_artifact(requested) != parent:
        raise ValueError(
            "insertion follow-up proposal is not the bridge's exact frozen parent"
        )
    return requested


def write_insertion_proposal_handoff(
    previous_request: Path,
    previous_response: Path,
    requested_parent: Path,
    output: Path | None = None,
    *,
    data_root: Path | None = None,
    previous_session_id: str | None = None,
) -> InsertionProposalContinuation:
    """Authenticate prior session/model bytes on the host and persist the handoff."""

    observation = ControlObservation.from_dict(json.loads(previous_request.read_text()))
    response = ProposedControl.from_dict(json.loads(previous_response.read_text()))
    try:
        handoff: InsertionProposalContinuation = InsertionProposalHandoff.from_bridge(
            observation.expected_proposal,
            requested_parent,
        )
    except ValueError:
        if data_root is None or previous_session_id is None:
            raise
        validate_safe_identifier(previous_session_id)
        descendant = ArtifactIdentity.from_artifact(requested_parent.resolve())
        training_parent = transition_parent_proposal(descendant.path)
        report = load_training_report(descendant.path)
        selection = InsertionTransitionTrainingSelection.from_dict(
            report["training_selection"]
        )
        example = selection.transition_example
        source_root = (
            data_root.resolve() / "control_sessions" / example.source_session_id
        )
        source_observation = ControlObservation.from_dict(
            json.loads((source_root / "request.json").read_text())
        )
        source_response = ProposedControl.from_dict(
            json.loads((source_root / "response.json").read_text())
        )
        source_state = json.loads((source_root / "state.json").read_text())
        previous = ArtifactIdentity.from_artifact(
            observation.expected_proposal.resolve()
        )
        if (
            response.proposal_fingerprint != previous.fingerprint
            or source_state.get("previous_session_id") != previous_session_id
            or source_observation.expected_proposal.resolve()
            != training_parent.path.resolve()
            or source_response.proposal_fingerprint != training_parent.fingerprint
            or example.source_proposal != training_parent
        ):
            raise ValueError("insertion transition descendant source is invalid")
        handoff = InsertionTransitionDescendantHandoff(
            previous,
            training_parent,
            descendant,
            example.source_session_id,
        )
    handoff.resolve(
        observation.expected_proposal,
        response.proposal_fingerprint,
        requested_parent,
    )
    if output is not None:
        write_json_atomic(output.resolve(), handoff.to_dict())
    return handoff


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous-request", type=Path, required=True)
    parser.add_argument("--previous-response", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--previous-session")
    args = parser.parse_args()
    handoff = write_insertion_proposal_handoff(
        args.previous_request,
        args.previous_response,
        args.parent,
        args.output,
        data_root=args.data_root,
        previous_session_id=args.previous_session,
    )
    print(json.dumps(handoff.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
