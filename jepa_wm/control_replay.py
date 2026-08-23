"""Build a control observation from one synchronized recorded rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

from jepa_wm.action import ActionSelectionBounds
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.trajectory import load_rollouts


def observation_from_recording(
    recording: Path,
    *,
    camera: str,
    context_index: int,
    expected_proposal: Path,
    observation_id: int = 1,
    captured_at_unix_seconds: float | None = None,
) -> ControlObservation:
    matches = tuple(
        rollout
        for rollout in load_rollouts(
            recording,
            camera=camera,
            bounds=ActionSelectionBounds(minimum_action_norm=0.0),
        )
        if rollout.context[0].index == context_index
    )
    if len(matches) != 1:
        raise ValueError(
            f"recording has {len(matches)} rollouts at context index {context_index}"
        )
    rollout = matches[0]
    return ControlObservation(
        observation_id=observation_id,
        captured_at_unix_seconds=(
            time()
            if captured_at_unix_seconds is None
            else captured_at_unix_seconds
        ),
        context_frame=rollout.context[0].path,
        target_frame=rollout.target.path,
        expected_proposal=expected_proposal,
        pose=rollout.context_pose,
        previous_action=rollout.previous_action,
        warmup_frames=context_index,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--context-index", type=int, required=True)
    parser.add_argument("--observation-id", type=int, default=1)
    parser.add_argument("--proposal", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            observation_from_recording(
                args.recording,
                camera=args.camera,
                context_index=args.context_index,
                expected_proposal=args.proposal.resolve(),
                observation_id=args.observation_id,
            ).to_dict()
        )
    )


if __name__ == "__main__":
    main()
