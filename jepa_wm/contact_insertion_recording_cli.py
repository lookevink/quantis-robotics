"""Validate one contact-aware scripted insertion recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.insertion_recording import ContactInsertionEvidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("split", choices=("train", "held_out"))
    args = parser.parse_args()
    evidence = ContactInsertionEvidence.from_recording(
        args.recording,
        expected_split=args.split,
    )
    print(json.dumps(evidence.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
