"""Benchmark bounded JEPA-WM candidate search on held-out Isaac recordings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
import torch

from jepa_wm.action import DEFAULT_ACTION_SELECTION_BOUNDS, ActionSelectionBounds
from jepa_wm.action_prior import (
    ActionLibrary,
    ActionPriorConfig,
    EmpiricalActionPrior,
)
from jepa_wm.adapter import apply_action_adapter
from jepa_wm.contract import MODEL_ID
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.frames import video_batch
from jepa_wm.model import load_headless_model
from jepa_wm.planner import (
    CandidateTrustRegion,
    CEMConfig,
    CEMPlanner,
    PlannerActionBounds,
    ProposalCenteredBounds,
)
from jepa_wm.planner_objective import evaluate_planner_objective
from jepa_wm.planning_scoring import LatentGoalScorer
from jepa_wm.planner_report import (
    CandidateEvaluation,
    PlannerBenchmarkProvenance,
    PlannerBenchmarkReport,
    PlannerInitialization,
    PlannerRolloutEvaluation,
    PlannerRunSummary,
    PlannerTimings,
)
from jepa_wm.planner_policy import PlannerTaskPolicy
from jepa_wm.proposal import ProposalInputs, load_action_proposal
from jepa_wm.trajectory import RolloutWindow, load_rollouts
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    TrainingArtifactIdentity,
)
from sim.exploration import DatasetSplit


PROPOSAL_REFINEMENT_PRIOR = ActionPriorConfig(penalty_weight=1e-2)


@dataclass(frozen=True)
class ProposalBenchmarkArtifact:
    identity: TrainingArtifactIdentity

    @classmethod
    def from_path(cls, path: Path) -> ProposalBenchmarkArtifact:
        return cls(
            TrainingArtifactIdentity.from_artifact(
                path, fingerprint_field="proposal_fingerprint"
            )
        )

    @property
    def path(self) -> Path:
        return self.identity.path

    @property
    def report_metadata(self) -> TrainingArtifactMetadata:
        return self.identity.metadata

    def load(self, device: torch.device):
        model, checkpoint_metadata = load_action_proposal(self.path, device=device)
        if checkpoint_metadata != self.report_metadata:
            raise ValueError(
                "proposal checkpoint metadata does not match its training report"
            )
        return model


@dataclass(frozen=True)
class AdapterBenchmarkArtifact:
    identity: TrainingArtifactIdentity

    @classmethod
    def from_path(cls, path: Path) -> AdapterBenchmarkArtifact:
        return cls(
            TrainingArtifactIdentity.from_artifact(
                path, fingerprint_field="adapter_fingerprint"
            )
        )

    def apply(self, model: Any, *, source_revision: str) -> None:
        checkpoint_metadata = apply_action_adapter(
            model,
            self.identity.path,
            expected_source_revision=source_revision,
        )
        if checkpoint_metadata != self.identity.metadata:
            raise ValueError(
                "adapter checkpoint metadata does not match its training report"
            )


@dataclass(frozen=True)
class BenchmarkInitialization:
    kind: PlannerInitialization
    prior: ActionPriorConfig
    adapter: AdapterBenchmarkArtifact
    proposal: ProposalBenchmarkArtifact | None = None

    def __post_init__(self) -> None:
        if (self.kind is PlannerInitialization.PROPOSAL) != (
            self.proposal is not None
        ):
            raise ValueError("planner initialization kind and proposal disagree")

    @classmethod
    def create(
        cls,
        *,
        adapter: Path,
        proposal: Path | None,
        library_prior: ActionPriorConfig,
        proposal_prior: ActionPriorConfig,
    ) -> BenchmarkInitialization:
        adapter_artifact = AdapterBenchmarkArtifact.from_path(adapter)
        if proposal is None:
            return cls(
                PlannerInitialization.LIBRARY, library_prior, adapter_artifact
            )
        artifact = ProposalBenchmarkArtifact.from_path(proposal)
        return cls(
            PlannerInitialization.PROPOSAL,
            proposal_prior,
            adapter_artifact,
            artifact,
        )

    @property
    def output_label(self) -> str:
        return (
            f"{self.kind.value}_prior_"
            f"{_prior_output_token(self.prior.penalty_weight)}"
        )


@dataclass(frozen=True)
class EffectiveBenchmarkIdentity:
    source_revision: str
    base_checkpoint: ArtifactIdentity
    recording: DomainRecording
    camera: str
    window: RolloutWindow
    selection_bounds: ActionSelectionBounds
    planner_bounds: PlannerActionBounds
    planner_config: CEMConfig
    scoring_batch_size: int
    initialization: BenchmarkInitialization
    task_policy: PlannerTaskPolicy = PlannerTaskPolicy()

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": MODEL_ID,
            "source_revision": self.source_revision,
            "base_checkpoint": self.base_checkpoint.to_dict(),
            "adapter": self.initialization.adapter.identity.to_dict(),
            "proposal": (
                self.initialization.proposal.identity.to_dict()
                if self.initialization.proposal is not None
                else None
            ),
            "recording": {
                "name": self.recording.name,
                "split": self.recording.split.value,
                "seed": self.recording.seed,
            },
            "camera": self.camera,
            "window": self.window.to_dict(),
            "selection_bounds": self.selection_bounds.to_dict(),
            "planner_bounds": self.planner_bounds.to_dict(),
            "planner_config": self.planner_config.to_dict(),
            "scoring_batch_size": self.scoring_batch_size,
            "initialization": {
                "kind": self.initialization.kind.value,
                "prior": self.initialization.prior.to_dict(),
            },
            "task_policy": self.task_policy.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256(encoded).hexdigest()


def validate_benchmark_recording(
    recording: Path,
    *,
    expected_split: DatasetSplit,
    adapter_metadata: TrainingArtifactMetadata,
    proposal_metadata: TrainingArtifactMetadata | None,
) -> DomainRecording:
    domain_recording = DomainRecording.from_path(
        recording, expected_split=expected_split
    )
    artifact_metadata = tuple(
        metadata
        for metadata in (adapter_metadata, proposal_metadata)
        if metadata is not None
    )
    memberships = tuple(
        domain_recording.name in metadata.training_recordings
        for metadata in artifact_metadata
    )
    if expected_split is DatasetSplit.TRAIN:
        if proposal_metadata is None:
            raise ValueError("training calibration requires a proposal artifact")
        if not all(memberships):
            raise ValueError(
                "training calibration recording must belong to every model artifact"
            )
    if expected_split is DatasetSplit.HELD_OUT and any(memberships):
        raise ValueError("held-out planner recording was used for model training")
    return domain_recording


def _prior_output_token(weight: float) -> str:
    return repr(weight).replace("-", "m").replace("+", "p").replace(".", "d")


def benchmark_recording(
    source: Path,
    checkpoint: Path,
    recording: Path,
    *,
    camera: str,
    window: RolloutWindow,
    adapter: Path,
    planner_config: CEMConfig,
    planner_bounds: PlannerActionBounds,
    selection_bounds: ActionSelectionBounds = DEFAULT_ACTION_SELECTION_BOUNDS,
    scoring_batch_size: int = 64,
    prior_config: ActionPriorConfig = ActionPriorConfig(),
    proposal: Path | None = None,
    proposal_prior_config: ActionPriorConfig = PROPOSAL_REFINEMENT_PRIOR,
    task_policy: PlannerTaskPolicy = PlannerTaskPolicy(),
    expected_split: DatasetSplit = DatasetSplit.HELD_OUT,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM planner benchmark requires CUDA")
    source_revision = os.environ.get("JEPA_WM_REVISION", "unknown")
    initialization_config = BenchmarkInitialization.create(
        adapter=adapter,
        proposal=proposal,
        library_prior=prior_config,
        proposal_prior=proposal_prior_config,
    )
    adapter_metadata = initialization_config.adapter.identity.metadata
    if adapter_metadata.camera != camera:
        raise ValueError("adapter camera does not match planner camera")
    domain_recording = validate_benchmark_recording(
        recording,
        expected_split=expected_split,
        adapter_metadata=adapter_metadata,
        proposal_metadata=(
            initialization_config.proposal.report_metadata
            if initialization_config.proposal is not None
            else None
        ),
    )
    rollouts = window.select(
        load_rollouts(domain_recording.path, camera=camera, bounds=selection_bounds)
    )
    training_rollouts = tuple(
        training_rollout
        for training_name in adapter_metadata.training_recordings
        for training_rollout in load_rollouts(
            domain_recording.path.parent / training_name,
            camera=camera,
            bounds=ActionSelectionBounds(minimum_action_norm=0.0),
        )
    )
    action_library = ActionLibrary(
        np.asarray(
            [
                [action.values for action in training_rollout.actions]
                for training_rollout in training_rollouts
            ],
            dtype=np.float64,
        )
    )
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    base_checkpoint_identity = ArtifactIdentity.from_artifact(checkpoint)
    load_started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
    )
    initialization_config.adapter.apply(model, source_revision=source_revision)
    load_seconds = monotonic() - load_started
    encoding_started = monotonic()
    with torch.inference_mode():
        contexts = model.encode(
            video_batch([rollout.context_paths for rollout in rollouts])
        )
        targets = model.encode(
            video_batch([rollout.target_clip for rollout in rollouts])
        )
    proposal_actions = None
    if initialization_config.proposal is not None:
        proposal_model = initialization_config.proposal.load(device)
        if initialization_config.proposal.report_metadata.camera != camera:
            raise ValueError("proposal camera does not match planner camera")
        inputs = ProposalInputs.from_rollouts(
            rollouts,
            conditioning=proposal_model.conditioning,
            device=device,
            dtype=contexts.dtype,
        )
        with torch.inference_mode():
            proposal_actions = planner_bounds.clip(
                proposal_model(contexts, targets, inputs)
                .cpu()
                .numpy()
            )
    encoding_seconds = monotonic() - encoding_started

    planning_started = monotonic()
    evaluations = []
    for index, rollout in enumerate(rollouts):
        scorer = LatentGoalScorer(
            model,
            contexts[index : index + 1],
            targets[index : index + 1],
            device=device,
            batch_size=scoring_batch_size,
        )
        proposed_actions = (
            proposal_actions[index] if proposal_actions is not None else None
        )
        if proposed_actions is not None:
            library_energies = None
            prior = EmpiricalActionPrior.fit(
                proposed_actions[None, :, :],
                initialization_config.prior,
            )
        else:
            library_energies = scorer(action_library.sequences)
            prior = action_library.goal_conditioned_prior(
                library_energies,
                elites=planner_config.elites,
                config=prior_config,
            )

        goal_alignment = task_policy.goal_action_alignment
        task_penalty = (
            (
                lambda candidates: goal_alignment.penalty(
                    candidates,
                    rollout.goal_action,
                )
            )
            if goal_alignment is not None
            else None
        )

        def evaluate_objective(
            candidates: np.ndarray,
            *,
            latent_energy: np.ndarray | None = None,
        ):
            return evaluate_planner_objective(
                candidates,
                scorer,
                prior,
                task_penalty,
                latent_energy=latent_energy,
            )

        def planning_objective(candidates: np.ndarray) -> np.ndarray:
            return evaluate_objective(candidates).total

        config = CEMConfig(
            horizon=planner_config.horizon,
            iterations=planner_config.iterations,
            samples=planner_config.samples,
            elites=planner_config.elites,
            seed=planner_config.seed + rollout.context[0].index,
            minimum_standard_deviation=planner_config.minimum_standard_deviation,
        )
        candidate_bounds = (
            ProposalCenteredBounds(
                proposed_actions,
                planner_bounds,
                task_policy.proposal_trust_region,
            )
            if (
                proposed_actions is not None
                and task_policy.proposal_trust_region is not None
            )
            else planner_bounds
        )
        result = CEMPlanner(config, candidate_bounds).plan(
            planning_objective,
            initial_distribution=prior.distribution,
        )
        recorded_actions = np.asarray(
            [action.values for action in rollout.actions], dtype=np.float64
        )
        if library_energies is not None:
            library_components = evaluate_objective(
                action_library.sequences,
                latent_energy=library_energies,
            )
            library_index = int(np.argmin(library_components.total))
            library_action = action_library.sequences[library_index]
            library_scores = library_components.candidate(library_index)
        else:
            library_action = None
            library_scores = None
        comparison_actions = [
            recorded_actions,
            np.zeros_like(recorded_actions),
            result.actions,
        ]
        if proposed_actions is not None:
            comparison_actions.append(proposed_actions)
        comparison_components = evaluate_objective(np.stack(comparison_actions))
        proposal_scores = (
            comparison_components.candidate(3)
            if proposed_actions is not None
            else None
        )
        if proposed_actions is not None:
            if proposal_scores is None:
                raise ValueError("proposal initialization metrics are incomplete")
            rollout_initialization = PlannerInitialization.PROPOSAL
            initial_candidate = CandidateEvaluation(
                proposed_actions,
                proposal_scores,
            )
        else:
            if library_action is None or library_scores is None:
                raise ValueError("library initialization metrics are incomplete")
            rollout_initialization = PlannerInitialization.LIBRARY
            initial_candidate = CandidateEvaluation(
                library_action,
                library_scores,
            )
        evaluations.append(
            PlannerRolloutEvaluation(
                context_index=rollout.context[0].index,
                target_index=rollout.target.index,
                recorded_actions=recorded_actions,
                recorded_energy=float(comparison_components.latent_energy[0]),
                zero_energy=float(comparison_components.latent_energy[1]),
                initialization=rollout_initialization,
                initial_candidate=initial_candidate,
                searched_candidate=CandidateEvaluation(
                    result.actions,
                    comparison_components.candidate(2),
                ),
                goal_action=rollout.goal_action,
            )
        )
    torch.cuda.synchronize(device)
    planning_seconds = monotonic() - planning_started
    benchmark_identity = EffectiveBenchmarkIdentity(
        source_revision=source_revision,
        base_checkpoint=base_checkpoint_identity,
        recording=domain_recording,
        camera=camera,
        window=window,
        selection_bounds=selection_bounds,
        planner_bounds=planner_bounds,
        planner_config=planner_config,
        scoring_batch_size=scoring_batch_size,
        initialization=initialization_config,
        task_policy=task_policy,
    )
    output_path = domain_recording.path / "jepa_wm" / (
        f"{camera}_cem_benchmark_"
        f"{window.start_index:06d}_{window.count:03d}_"
        f"{expected_split.value}_{initialization_config.output_label}_"
        f"{benchmark_identity.fingerprint}.json"
    )
    return PlannerBenchmarkReport(
        provenance=PlannerBenchmarkProvenance(
            model=MODEL_ID,
            source_revision=source_revision,
            adapter=initialization_config.adapter.identity,
            proposal=(
                initialization_config.proposal.identity
                if initialization_config.proposal is not None
                else None
            ),
            base_checkpoint=base_checkpoint_identity,
            recording=domain_recording,
            camera=camera,
            window=window,
            selection_bounds=selection_bounds,
            scoring_batch_size=scoring_batch_size,
        ),
        planner=PlannerRunSummary(
            config=planner_config,
            training_action_library=len(action_library.sequences),
            prior_penalty_weight=initialization_config.prior.penalty_weight,
            initialization=initialization_config.kind,
            task_policy=task_policy,
        ),
        bounds=planner_bounds,
        timings=PlannerTimings(
            load_seconds=load_seconds,
            encoding_seconds=encoding_seconds,
            planning_seconds=planning_seconds,
            peak_allocated_gib=(
                torch.cuda.max_memory_allocated(device_index) / 2**30
            ),
        ),
        evaluations=tuple(evaluations),
    ).write(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--proposal", type=Path)
    parser.add_argument(
        "--recording-split",
        choices=tuple(split.value for split in DatasetSplit),
        default=DatasetSplit.HELD_OUT.value,
    )
    parser.add_argument(
        "--proposal-prior-weight",
        type=float,
        default=PROPOSAL_REFINEMENT_PRIOR.penalty_weight,
    )
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=CEMConfig.iterations)
    parser.add_argument("--samples", type=int, default=CEMConfig.samples)
    parser.add_argument("--elites", type=int, default=CEMConfig.elites)
    parser.add_argument("--seed", type=int, default=CEMConfig.seed)
    parser.add_argument("--scoring-batch-size", type=int, default=64)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark_recording(
                args.source,
                args.checkpoint,
                args.recording,
                camera=args.camera,
                window=RolloutWindow(args.start_index, args.count, args.stride),
                adapter=args.adapter,
                proposal=args.proposal,
                proposal_prior_config=ActionPriorConfig(
                    penalty_weight=args.proposal_prior_weight
                ),
                expected_split=DatasetSplit(args.recording_split),
                planner_config=CEMConfig(
                    iterations=args.iterations,
                    samples=args.samples,
                    elites=args.elites,
                    seed=args.seed,
                ),
                planner_bounds=PlannerActionBounds(),
                scoring_batch_size=args.scoring_batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
