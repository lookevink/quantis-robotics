"""Exclusive claim and terminal failure for one unknown-start reset gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.identifiers import validate_safe_identifier
from sim.unknown_start_reset import UNKNOWN_START_RESET_CONTRACT


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise ValueError("unknown-start reset was already claimed") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _fingerprint(path: Path) -> str:
    if not path.is_file():
        raise ValueError(f"unknown-start reset artifact is missing: {path}")
    return sha256(path.read_bytes()).hexdigest()


def _validate_source_revision(source_revision: str) -> None:
    if len(source_revision) != 40 or any(
        character not in "0123456789abcdef" for character in source_revision
    ):
        raise ValueError("unknown-start reset source revision is invalid")


def claim(
    path: Path,
    recording_id: str,
    seed: int,
    source_revision: str,
) -> dict[str, Any]:
    validate_safe_identifier(recording_id)
    _validate_source_revision(source_revision)
    sample = UNKNOWN_START_RESET_CONTRACT.draw(seed, forbidden_seeds=set())
    payload = {
        "schema": "quantis.unknown_start_reset_claim.v1",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "recording_id": recording_id,
        "seed": seed,
        "contract_fingerprint": UNKNOWN_START_RESET_CONTRACT.fingerprint,
        "sample_fingerprint": sample.fingerprint,
        "source_revision": source_revision,
        "evaluations_claimed": 1,
        "applied_actions": 0,
    }
    _write_exclusive(path, payload)
    return payload


def failure(path: Path, claim_path: Path, error: str) -> dict[str, Any]:
    if not claim_path.is_file():
        raise ValueError("unknown-start reset has no claim")
    payload = {
        "schema": "quantis.unknown_start_reset_failure.v1",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "error": error,
        "claim_fingerprint": _fingerprint(claim_path),
        "retry_authorized": False,
        "applied_actions": 0,
    }
    _write_exclusive(path, payload)
    return payload


def finalize_recovery(
    primary_recording: Path,
    recovery_recording: Path,
    primary_claim: Path,
    recovery_claim: Path,
    source_revision: str,
) -> dict[str, Any]:
    """Write success only after exact reset evidence exists on recovery storage."""

    _validate_source_revision(source_revision)
    artifact_names = (
        "manifest.json",
        "unknown_start_reset_evidence.json",
        "CAPTURE.json",
    )
    primary_fingerprints = {
        name: _fingerprint(primary_recording / name) for name in artifact_names
    }
    recovery_fingerprints = {
        name: _fingerprint(recovery_recording / name) for name in artifact_names
    }
    if primary_fingerprints != recovery_fingerprints:
        raise ValueError("unknown-start reset recovery recording changed")
    primary_claim_fingerprint = _fingerprint(primary_claim)
    if primary_claim_fingerprint != _fingerprint(recovery_claim):
        raise ValueError("unknown-start reset recovery claim changed")
    capture = json.loads((primary_recording / "CAPTURE.json").read_text())
    claim_payload = json.loads(primary_claim.read_text())
    if (
        capture.get("status") != "captured"
        or capture.get("applied_actions") != 0
        or capture.get("source_revision") != source_revision
        or claim_payload.get("source_revision") != source_revision
        or capture.get("recording_id") != claim_payload.get("recording_id")
        or capture.get("contract_fingerprint")
        != claim_payload.get("contract_fingerprint")
        or capture.get("sample_fingerprint")
        != claim_payload.get("sample_fingerprint")
    ):
        raise ValueError("unknown-start reset recovery identity is invalid")
    payload = {
        "schema": "quantis.unknown_start_reset_terminal.v1",
        "status": "passed",
        "passed": True,
        "recording_id": capture["recording_id"],
        "source_revision": source_revision,
        "contract_fingerprint": capture["contract_fingerprint"],
        "sample_fingerprint": capture["sample_fingerprint"],
        "claim_fingerprint": primary_claim_fingerprint,
        "artifacts": primary_fingerprints,
        "recovery_verified": True,
        "applied_actions": 0,
        "training_authorized": False,
        "filming_authorized": False,
    }
    primary_result = primary_recording / "RESULT.json"
    recovery_result = recovery_recording / "RESULT.json"
    _write_exclusive(primary_result, payload)
    _write_exclusive(recovery_result, payload)
    if _fingerprint(primary_result) != _fingerprint(recovery_result):
        raise ValueError("unknown-start reset terminal recovery changed")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("claim", "failure", "finalize-recovery"))
    parser.add_argument("--path", type=Path)
    parser.add_argument("--claim-path", type=Path)
    parser.add_argument("--recording-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-revision")
    parser.add_argument("--error")
    parser.add_argument("--primary-recording", type=Path)
    parser.add_argument("--recovery-recording", type=Path)
    parser.add_argument("--recovery-claim", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "claim":
        if (
            arguments.path is None
            or arguments.recording_id is None
            or arguments.seed is None
            or arguments.source_revision is None
        ):
            parser.error("claim requires path, recording id, seed, and source revision")
        payload = claim(
            arguments.path,
            arguments.recording_id,
            arguments.seed,
            arguments.source_revision,
        )
    elif arguments.command == "failure":
        if arguments.path is None or arguments.claim_path is None or not arguments.error:
            parser.error("failure requires --claim-path and --error")
        payload = failure(arguments.path, arguments.claim_path, arguments.error)
    else:
        if (
            arguments.primary_recording is None
            or arguments.recovery_recording is None
            or arguments.claim_path is None
            or arguments.recovery_claim is None
            or arguments.source_revision is None
        ):
            parser.error("finalize-recovery requires primary and recovery identities")
        payload = finalize_recovery(
            arguments.primary_recording,
            arguments.recovery_recording,
            arguments.claim_path,
            arguments.recovery_claim,
            arguments.source_revision,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
