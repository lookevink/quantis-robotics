"""Persist one reset-bound experimental shadow-candidate response."""

from __future__ import annotations

import argparse
from pathlib import Path

from sim.isaac_candidate_binding import persist_experimental_candidate_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--source-session", required=True)
    args = parser.parse_args()
    import json

    print(
        json.dumps(
            persist_experimental_candidate_response(
                args.session,
                args.source_session,
                control_root=args.data_root / "control_sessions",
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
