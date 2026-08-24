"""Benchmark bounded JEPA-WM candidate search on held-out Isaac recordings."""

from __future__ import annotations

import argparse
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
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import video_batch
from jepa_wm.model import load_headless_model
from jepa_wm.planner import CEMConfig, CEMPlanner, PlannerActionBounds
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
from jepa_wm.proposal import ProposalInputs, load_action_proposal
from jepa_wm.trajectory import RolloutWindow, load_rollouts
from jepa_wm.training_artifact import load_training_report_metadata


PROPOSAL_REFINEMENT_PRIOR = ActionPriorConfig(penalty_weight=1e-4)


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
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("JEPA-WM planner benchmark requires CUDA")
    rollouts = window.select(
        load_rollouts(recording, camera=camera, bounds=selection_bounds)
    )
    adapter_metadata = load_training_report_metadata(adapter)
    if adapter_metadata.camera != camera:
        raise ValueError("adapter camera does not match planner camera")
    training_rollouts = tuple(
        training_rollout
        for training_name in adapter_metadata.training_recordings
        for training_rollout in load_rollouts(
            recording.resolve().parent / training_name,
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

    load_started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=adapter,
    )
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
    proposal_metadata = None
    if proposal is not None:
        proposal_model, proposal_metadata = load_action_proposal(
            proposal,
            device=device,
        )
        if proposal_metadata.camera != camera:
            raise ValueError("proposal camera does not match planner camera")
        if recording.name in proposal_metadata.training_recordings:
            raise ValueError("planner proposal was trained on the evaluation recording")
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
                PROPOSAL_REFINEMENT_PRIOR,
            )
        else:
            library_energies = scorer(action_library.sequences)
            prior = action_library.goal_conditioned_prior(
                library_energies,
                elites=planner_config.elites,
                config=prior_config,
            )

        def planning_objective(candidates: np.ndarray) -> np.ndarray:
            return scorer(candidates) + prior.penalty(candidates)

        config = CEMConfig(
            horizon=planner_config.horizon,
            iterations=planner_config.iterations,
            samples=planner_config.samples,
            elites=planner_config.elites,
            seed=planner_config.seed + rollout.context[0].index,
            minimum_standard_deviation=planner_config.minimum_standard_deviation,
        )
        result = CEMPlanner(config, planner_bounds).plan(
            planning_objective,
            initial_distribution=prior.distribution,
        )
        recorded_actions = np.asarray(
            [action.values for action in rollout.actions], dtype=np.float64
        )
        if library_energies is not None:
            library_objectives = library_energies + prior.penalty(
                action_library.sequences
            )
            library_index = int(np.argmin(library_objectives))
            library_action = action_library.sequences[library_index]
            library_energy = float(library_energies[library_index])
            library_objective = float(library_objectives[library_index])
        else:
            library_action = None
            library_energy = None
            library_objective = None
        comparison_actions = [
            recorded_actions,
            np.zeros_like(recorded_actions),
            result.actions,
        ]
        if proposed_actions is not None:
            comparison_actions.append(proposed_actions)
        comparison_energies = scorer(np.stack(comparison_actions))
        proposal_energy = (
            float(comparison_energies[3]) if proposed_actions is not None else None
        )
        proposal_objective = (
            proposal_energy + float(prior.penalty(proposed_actions[None, :, :])[0])
            if proposed_actions is not None and proposal_energy is not None
            else None
        )
        if proposed_actions is not None:
            if proposal_energy is None or proposal_objective is None:
                raise ValueError("proposal initialization metrics are incomplete")
            initialization = PlannerInitialization.PROPOSAL
            initial_candidate = CandidateEvaluation(
                proposed_actions,
                proposal_energy,
                proposal_objective,
            )
        else:
            if (
                library_action is None
                or library_energy is None
                or library_objective is None
            ):
                raise ValueError("library initialization metrics are incomplete")
            initialization = PlannerInitialization.LIBRARY
            initial_candidate = CandidateEvaluation(
                library_action,
                library_energy,
                library_objective,
            )
        evaluations.append(
            PlannerRolloutEvaluation(
                context_index=rollout.context[0].index,
                target_index=rollout.target.index,
                recorded_actions=recorded_actions,
                recorded_energy=float(comparison_energies[0]),
                zero_energy=float(comparison_energies[1]),
                initialization=initialization,
                initial_candidate=initial_candidate,
                planned_candidate=CandidateEvaluation(
                    result.actions,
                    float(comparison_energies[2]),
                    result.energy,
                ),
            )
        )
    torch.cuda.synchronize(device)
    planning_seconds = monotonic() - planning_started
    initialization_name = (
        f"{adapter.stem}_{proposal.stem}"
        if proposal is not None
        else adapter.stem
    )
    output_path = recording.resolve() / "jepa_wm" / (
        f"{camera}_{initialization_name}_cem_benchmark_"
        f"{window.start_index:06d}_{window.count:03d}.json"
    )
    return PlannerBenchmarkReport(
        provenance=PlannerBenchmarkProvenance(
            model=MODEL_ID,
            source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
            adapter=adapter,
            proposal=proposal,
            recording=recording,
            camera=camera,
            window=window,
        ),
        planner=PlannerRunSummary(
            config=planner_config,
            training_action_library=len(action_library.sequences),
            prior_penalty_weight=(
                PROPOSAL_REFINEMENT_PRIOR.penalty_weight
                if proposal is not None
                else prior_config.penalty_weight
            ),
            initialization=(
                PlannerInitialization.PROPOSAL
                if proposal is not None
                else PlannerInitialization.LIBRARY
            ),
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
