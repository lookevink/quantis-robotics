"""Persist an explicit non-model response for one baseline control session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.control_baselines import (
    NonModelBaselinePolicy,
    build_baseline_response,
    load_held_out_reference,
    scripted_actions_at,
)
from sim.control_session import ControlSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument(
        "--policy",
        type=NonModelBaselinePolicy,
        choices=(NonModelBaselinePolicy.ZERO, NonModelBaselinePolicy.SCRIPTED),
        required=True,
    )
    args = parser.parse_args()
    session = ControlSession.at(args.data_root / "control_sessions", args.session)
    observation, state = session.load_capture()
    recording = load_held_out_reference(
        args.data_root / "recordings" / state.reference_recording,
        state.seed,
    )
    response = build_baseline_response(
        observation,
        args.policy,
        scripted_actions=(
            scripted_actions_at(recording, observation.warmup_frames)
            if args.policy is NonModelBaselinePolicy.SCRIPTED
            else None
        ),
    )
    session.write_response(response)
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()
