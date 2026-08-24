"""Persist an explicit non-model response for one baseline control session."""

from __future__ import annotations

import argparse
from pathlib import Path

from jepa_wm.control_baselines import NonModelBaselinePolicy
from sim.isaac_baseline_response import persist_baseline_response


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
    response = persist_baseline_response(
        args.session,
        args.policy.value,
        control_root=args.data_root / "control_sessions",
    )
    import json

    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
