"""Persist non-model baseline responses inside Isaac's resident runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jepa_wm.control_baselines import (
    NonModelBaselinePolicy,
    build_baseline_response,
    load_held_out_reference,
    scripted_actions_at,
)
from sim.control_session import CONTROL_ROOT, ControlSession


def persist_baseline_response(
    session_id: str,
    policy: str,
    *,
    control_root: Path = CONTROL_ROOT,
) -> dict[str, Any]:
    """Build and bind a zero or scripted response after resident validation."""

    baseline_policy = NonModelBaselinePolicy(policy)
    session = ControlSession.at(control_root, session_id)
    observation, state = session.load_capture()
    recording = load_held_out_reference(
        control_root.parent / "recordings" / state.reference_recording,
        state.seed,
    )
    response = build_baseline_response(
        observation,
        baseline_policy,
        scripted_actions=(
            scripted_actions_at(recording, observation.warmup_frames)
            if baseline_policy is NonModelBaselinePolicy.SCRIPTED
            else None
        ),
    )
    session.write_response(response)
    return response.to_dict()
