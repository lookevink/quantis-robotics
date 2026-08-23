"""Persist one reset-bound experimental shadow-candidate response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.experimental_candidate import build_experimental_candidate_response
from sim.control_session import ControlSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--source-session", required=True)
    args = parser.parse_args()
    root = args.data_root / "control_sessions"
    session = ControlSession.at(root, args.session)
    source = ControlSession.at(root, args.source_session)
    observation, _ = session.load_capture()
    shadow = source.load_shadow()
    safety = source.load_shadow_safety()
    binding, response = build_experimental_candidate_response(
        execution_session_id=args.session,
        source_session_id=args.source_session,
        observation=observation,
        shadow=shadow,
        safety=safety,
    )
    session.write_candidate_binding(binding)
    session.write_response(response)
    print(
        json.dumps(
            {"binding": binding.to_dict(), "response": response.to_dict()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
