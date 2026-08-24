"""Classify one persisted contact-insertion artifact for safe corpus resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jepa_wm.insertion_recording import ContactInsertionEvidence
from sim.recording_jobs import running_job_is_stale


def recording_status(recording: Path, split: str, seed: int) -> str:
    """Return a resumable state without mistaking live work for stale output."""

    job = recording.parent.parent / "recording_jobs" / f"{recording.name}.json"
    if job.is_file():
        try:
            job_payload = json.loads(job.read_text())
            job_status = job_payload.get("status")
        except (OSError, ValueError):
            return "invalid"
        if job_status == "running":
            return "partial" if running_job_is_stale(job_payload) else "running"
        if job_status == "error":
            return "partial"
        if job_status != "complete":
            return "invalid"
    if not recording.exists():
        return "invalid" if job.is_file() else "missing"
    if not (recording / "manifest.json").is_file():
        return "invalid"
    try:
        ContactInsertionEvidence.from_recording(
            recording,
            expected_split=split,
            expected_seed=seed,
        )
    except (OSError, ValueError):
        return "invalid"
    return "valid"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("split", choices=("train", "held_out"))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()
    print(recording_status(args.recording, args.split, args.seed))


if __name__ == "__main__":
    main()
