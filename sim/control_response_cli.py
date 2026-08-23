"""Persist one worker response through the session-owned binding contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from jepa_wm.control_protocol import ProposedControl
from sim.control_session import ControlSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    response = ProposedControl.from_dict(json.load(sys.stdin))
    session = ControlSession.at(args.data_root / "control_sessions", args.session)
    session.write_response(response)
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    main()
