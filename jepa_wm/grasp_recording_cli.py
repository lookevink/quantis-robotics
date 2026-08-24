"""Validate one seeded reach-and-grasp demonstration recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from jepa_wm.grasp_recording import GraspDemonstrationEvidence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("split", choices=("train", "held_out"))
    arguments = parser.parse_args(argv)
    evidence = GraspDemonstrationEvidence.from_recording(
        arguments.recording,
        expected_split=arguments.split,
    )
    print(json.dumps(evidence.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
