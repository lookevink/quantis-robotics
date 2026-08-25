"""Run the pinned insertion planner profile on one held-out recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.benchmark_planner import benchmark_recording
from jepa_wm.insertion_proposal_readiness import (
    validate_insertion_proposal,
)
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.insertion_planner import INSERTION_PLANNER_PROFILE
from jepa_wm.insertion_wm_readiness import validate_insertion_adapter
from jepa_wm.planner import PlannerActionBounds
from sim.exploration import DatasetSplit


def validate_insertion_benchmark_inputs(
    recording: Path,
    adapter_path: Path,
    proposal_path: Path,
) -> None:
    """Bind the planner to one exact held-out task and training corpus."""

    ContactInsertionEvidence.from_recording(
        recording,
        expected_split=DatasetSplit.HELD_OUT.value,
    )
    adapter = validate_insertion_adapter(adapter_path)
    proposal = validate_insertion_proposal(proposal_path)
    if (
        adapter.contract.metadata.camera != "wrist"
        or proposal.metadata.camera != "wrist"
        or adapter.contract.metadata.corpus_identity
        != proposal.metadata.corpus_identity
    ):
        raise ValueError(
            "insertion adapter and proposal require one wrist-camera training corpus"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument(
        "--scoring-batch-size",
        type=int,
        default=INSERTION_PLANNER_PROFILE.scoring_batch_size,
    )
    arguments = parser.parse_args()
    if arguments.camera != "wrist":
        parser.error("insertion planner benchmark requires the wrist camera")
    profile = INSERTION_PLANNER_PROFILE
    validate_insertion_benchmark_inputs(
        arguments.recording,
        arguments.adapter,
        arguments.proposal,
    )
    print(
        json.dumps(
            benchmark_recording(
                arguments.source,
                arguments.checkpoint,
                arguments.recording,
                camera=arguments.camera,
                window=profile.window,
                adapter=arguments.adapter,
                proposal=arguments.proposal,
                planner_config=profile.planner,
                planner_bounds=PlannerActionBounds(),
                selection_bounds=ActionSelectionBounds(minimum_action_norm=0.0),
                scoring_batch_size=arguments.scoring_batch_size,
                proposal_prior_config=profile.prior,
                task_policy=profile.task_policy,
                expected_split=DatasetSplit.HELD_OUT,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
