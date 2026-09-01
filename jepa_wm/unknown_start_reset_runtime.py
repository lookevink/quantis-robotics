"""Authenticated source roster for the milestone-20 reset-only runtime."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Sequence


UNKNOWN_START_RESET_RUNTIME_FILES = (
    "jepa/contract.py",
    "jepa_wm/action.py",
    "jepa_wm/identifiers.py",
    "jepa_wm/insertion_contract.py",
    "jepa_wm/insertion_task.py",
    "jepa_wm/persistence.py",
    "jepa_wm/unknown_start_reset_lifecycle.py",
    "jepa_wm/unknown_start_reset_runtime.py",
    "ops/backup_state.sh",
    "ops/run_unknown_start_reset.sh",
    "ops/shell_helpers.sh",
    "sim/demo_sequence.py",
    "sim/exploration.py",
    "sim/isaac_control_runtime.py",
    "sim/isaac_demo.py",
    "sim/isaac_demo_camera.py",
    "sim/isaac_demo_kinematics.py",
    "sim/isaac_demo_runtime.py",
    "sim/isaac_demo_scene.py",
    "sim/isaac_exploration.py",
    "sim/isaac_unknown_start_reset.py",
    "sim/recording.py",
    "sim/recording_jobs.py",
    "sim/runtime_loader.py",
    "sim/unknown_start_reset.py",
)


def runtime_source_fingerprint(repository: Path | None = None) -> str:
    root = repository or Path(__file__).resolve().parents[1]
    digest = sha256()
    for relative_name in UNKNOWN_START_RESET_RUNTIME_FILES:
        path = root / relative_name
        if not path.is_file():
            raise ValueError(f"unknown-start reset runtime source is missing: {relative_name}")
        encoded_name = relative_name.encode()
        contents = path.read_bytes()
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def authenticate_runtime_source(expected_fingerprint: str) -> str:
    actual = runtime_source_fingerprint()
    if actual != expected_fingerprint:
        raise ValueError("unknown-start reset runtime source changed")
    return actual


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("fingerprint", "authenticate"))
    parser.add_argument("--expected")
    arguments = parser.parse_args(argv)
    if arguments.command == "authenticate":
        if not arguments.expected:
            parser.error("authenticate requires --expected")
        value = authenticate_runtime_source(arguments.expected)
    else:
        value = runtime_source_fingerprint()
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
