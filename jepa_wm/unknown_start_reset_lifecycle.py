"""Exclusive claim and terminal failure for one unknown-start reset gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import struct
from typing import Any, Sequence
import zlib

from jepa_wm.identifiers import validate_safe_identifier
from sim.unknown_start_reset import (
    UNKNOWN_START_RESET_CONTRACT,
    UnknownStartResetEvidence,
)


UNKNOWN_START_RESET_RECORDING_ID = "unknown-start-reset-v2-62601"
UNKNOWN_START_RESET_SEED = 62601
UNKNOWN_START_RESET_PREVIOUS_SEEDS = frozenset({62600})
UNKNOWN_START_RESET_CLAIM_NAME = "milestone-20-unknown-start-reset-v2-claim.json"
UNKNOWN_START_RESET_FAILURE_NAME = "milestone-20-unknown-start-reset-v2-failure.json"


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("unknown-start reset artifact already exists") from error
    finally:
        temporary.unlink(missing_ok=True)
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


def _validate_fingerprint(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"unknown-start reset {label} fingerprint is invalid")


def terminal_paths(ledger_root: Path) -> tuple[Path, Path]:
    return (
        ledger_root / UNKNOWN_START_RESET_CLAIM_NAME,
        ledger_root / UNKNOWN_START_RESET_FAILURE_NAME,
    )


def claim(
    ledger_root: Path,
    recording_id: str,
    seed: int,
    source_revision: str,
    runtime_source_fingerprint: str,
) -> dict[str, Any]:
    validate_safe_identifier(recording_id)
    _validate_source_revision(source_revision)
    _validate_fingerprint(runtime_source_fingerprint, "runtime source")
    if (
        recording_id != UNKNOWN_START_RESET_RECORDING_ID
        or seed != UNKNOWN_START_RESET_SEED
    ):
        raise ValueError("unknown-start reset claim does not match the frozen run")
    path, failure_path = terminal_paths(ledger_root)
    if failure_path.exists() or any(ledger_root.glob("*-claim.json")):
        raise ValueError("unknown-start reset milestone was already claimed")
    sample = UNKNOWN_START_RESET_CONTRACT.draw(
        seed,
        forbidden_seeds=UNKNOWN_START_RESET_PREVIOUS_SEEDS,
    )
    payload = {
        "schema": "quantis.unknown_start_reset_claim.v1",
        "claimed_at": datetime.now(timezone.utc).isoformat(),
        "recording_id": recording_id,
        "seed": seed,
        "contract_fingerprint": UNKNOWN_START_RESET_CONTRACT.fingerprint,
        "sample_fingerprint": sample.fingerprint,
        "source_revision": source_revision,
        "runtime_source_fingerprint": runtime_source_fingerprint,
        "evaluations_claimed": 1,
        "applied_actions": 0,
    }
    _write_exclusive(path, payload)
    return payload


def failure(ledger_root: Path, error: str) -> dict[str, Any]:
    claim_path, path = terminal_paths(ledger_root)
    if not claim_path.is_file():
        raise ValueError("unknown-start reset has no claim")
    primary_result = (
        ledger_root.parent
        / "recordings"
        / UNKNOWN_START_RESET_RECORDING_ID
        / "RESULT.json"
    )
    if primary_result.exists():
        raise ValueError("unknown-start reset is already terminally passed")
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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unknown-start reset {label} is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"unknown-start reset {label} is invalid")
    return payload


def _validate_manifest_and_observation(
    recording: Path,
    manifest: dict[str, Any],
    evidence: UnknownStartResetEvidence,
    claim_payload: dict[str, Any],
) -> None:
    metadata = manifest.get("metadata")
    if (
        manifest.get("schema") != "quantis.demo_recording.v9"
        or manifest.get("recording_id") != claim_payload.get("recording_id")
        or manifest.get("fps") != 4
        or manifest.get("frames") != 1
        or manifest.get("stage_frames") != {"approaching_cable": 1}
        or manifest.get("cameras") != ["wrist"]
        or manifest.get("resolutions") != {"wrist": [512, 512]}
        or not isinstance(metadata, dict)
        or metadata.get("task") != "unknown_start_reset_authentication"
        or metadata.get("contract_fingerprint")
        != claim_payload.get("contract_fingerprint")
        or metadata.get("sample_fingerprint")
        != claim_payload.get("sample_fingerprint")
        or metadata.get("source_revision") != claim_payload.get("source_revision")
        or metadata.get("runtime_source_fingerprint")
        != claim_payload.get("runtime_source_fingerprint")
        or metadata.get("sample") != evidence.sample.to_dict()
        or metadata.get("applied_actions") != 0
        or metadata.get("prefix_replay_frames") != 0
    ):
        raise ValueError("unknown-start reset manifest identity is invalid")
    lines = (recording / "steps.jsonl").read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("unknown-start reset observation roster is invalid")
    try:
        step = json.loads(lines[0])
    except json.JSONDecodeError as error:
        raise ValueError("unknown-start reset observation is invalid") from error
    _validate_rgb_png(recording / "wrist/frame_000000.png", 512, 512)
    if (
        not isinstance(step, dict)
        or step.get("index") != 0
        or step.get("phase") != "initial"
        or step.get("stage") != "approaching_cable"
        or step.get("frames") != {"wrist": "wrist/frame_000000.png"}
        or step.get("action_from_previous") is not None
        or step.get("plug_attached") is not False
        or step.get("collision_detected") is not False
        or step.get("contact_force_newtons") != 0.0
        or step.get("arm_positions")
        != list(evidence.observed_arm_positions_radians)
        or step.get("gripper_width_m") != evidence.observed_gripper_width_m
        or step.get("plug_position") != list(evidence.workspace.connector_position_m)
        or step.get("end_effector_world_position")
        != list(evidence.workspace.end_effector_position_m)
    ):
        raise ValueError("unknown-start reset observation is inauthentic")


def _validate_rgb_png(path: Path, expected_width: int, expected_height: int) -> None:
    contents = path.read_bytes()
    if not contents.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("unknown-start reset camera frame is not PNG")
    offset = 8
    chunks: list[tuple[bytes, bytes]] = []
    while offset + 12 <= len(contents):
        length = struct.unpack(">I", contents[offset : offset + 4])[0]
        chunk_type = contents[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(contents):
            raise ValueError("unknown-start reset PNG is truncated")
        data = contents[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", contents[offset + 8 + length : end])[0]
        if zlib.crc32(chunk_type + data) & 0xFFFFFFFF != expected_crc:
            raise ValueError("unknown-start reset PNG checksum is invalid")
        chunks.append((chunk_type, data))
        offset = end
        if chunk_type == b"IEND":
            break
    if offset != len(contents) or not chunks or chunks[-1] != (b"IEND", b""):
        raise ValueError("unknown-start reset PNG terminal chunk is invalid")
    if chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise ValueError("unknown-start reset PNG header is invalid")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", chunks[0][1])
    )
    if (
        (width, height) != (expected_width, expected_height)
        or (bit_depth, color_type, compression, filtering, interlace)
        != (8, 2, 0, 0, 0)
    ):
        raise ValueError("unknown-start reset PNG format is invalid")
    compressed = b"".join(data for kind, data in chunks if kind == b"IDAT")
    try:
        pixels = zlib.decompress(compressed)
    except zlib.error as error:
        raise ValueError("unknown-start reset PNG pixels are invalid") from error
    row_bytes = width * 3
    if len(pixels) != height * (row_bytes + 1) or any(
        pixels[row * (row_bytes + 1)] > 4 for row in range(height)
    ):
        raise ValueError("unknown-start reset PNG raster is invalid")


def finalize_recovery(
    primary_recording: Path,
    recovery_recording: Path,
    primary_claim: Path,
    recovery_claim: Path,
    source_revision: str,
    runtime_source_fingerprint: str,
) -> dict[str, Any]:
    """Write success only after exact reset evidence exists on recovery storage."""

    _validate_source_revision(source_revision)
    _validate_fingerprint(runtime_source_fingerprint, "runtime source")
    if primary_claim.with_name(UNKNOWN_START_RESET_FAILURE_NAME).exists():
        raise ValueError("unknown-start reset is already terminally failed")
    artifact_names = (
        "manifest.json",
        "steps.jsonl",
        "wrist/frame_000000.png",
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
    capture = _load_json(primary_recording / "CAPTURE.json", "capture")
    claim_payload = _load_json(primary_claim, "claim")
    evidence_payload = _load_json(
        primary_recording / "unknown_start_reset_evidence.json",
        "evidence",
    )
    evidence = UnknownStartResetEvidence.from_dict(evidence_payload)
    manifest = _load_json(primary_recording / "manifest.json", "manifest")
    _validate_manifest_and_observation(
        primary_recording,
        manifest,
        evidence,
        claim_payload,
    )
    if (
        claim_payload.get("schema") != "quantis.unknown_start_reset_claim.v1"
        or claim_payload.get("recording_id") != UNKNOWN_START_RESET_RECORDING_ID
        or claim_payload.get("seed") != UNKNOWN_START_RESET_SEED
        or claim_payload.get("evaluations_claimed") != 1
        or claim_payload.get("applied_actions") != 0
        or claim_payload.get("contract_fingerprint")
        != UNKNOWN_START_RESET_CONTRACT.fingerprint
        or claim_payload.get("sample_fingerprint") != evidence.sample.fingerprint
        or capture.get("status") != "captured"
        or capture.get("applied_actions") != 0
        or capture.get("source_revision") != source_revision
        or claim_payload.get("source_revision") != source_revision
        or capture.get("runtime_source_fingerprint") != runtime_source_fingerprint
        or claim_payload.get("runtime_source_fingerprint")
        != runtime_source_fingerprint
        or capture.get("recording_id") != claim_payload.get("recording_id")
        or capture.get("contract_fingerprint")
        != claim_payload.get("contract_fingerprint")
        or capture.get("sample_fingerprint")
        != claim_payload.get("sample_fingerprint")
        or capture.get("sample_fingerprint") != evidence.sample.fingerprint
        or capture.get("evidence_fingerprint")
        != primary_fingerprints["unknown_start_reset_evidence.json"]
    ):
        raise ValueError("unknown-start reset recovery identity is invalid")
    payload = {
        "schema": "quantis.unknown_start_reset_terminal.v1",
        "status": "passed",
        "passed": True,
        "recording_id": capture["recording_id"],
        "source_revision": source_revision,
        "runtime_source_fingerprint": runtime_source_fingerprint,
        "contract_fingerprint": capture["contract_fingerprint"],
        "sample_fingerprint": capture["sample_fingerprint"],
        "claim_fingerprint": primary_claim_fingerprint,
        "artifacts": primary_fingerprints,
        "recovery_verified": True,
        "applied_actions": 0,
        "training_authorized": False,
        "filming_authorized": False,
    }
    recovery_marker = recovery_recording / "RECOVERY_VERIFIED.json"
    recovery_payload = {
        "schema": "quantis.unknown_start_reset_recovery.v1",
        "status": "recovery_verified",
        "passed": False,
        "recording_id": capture["recording_id"],
        "source_revision": source_revision,
        "runtime_source_fingerprint": runtime_source_fingerprint,
        "claim_fingerprint": primary_claim_fingerprint,
        "artifacts": recovery_fingerprints,
    }
    _write_or_validate(recovery_marker, recovery_payload)
    payload["recovery_marker_fingerprint"] = _fingerprint(recovery_marker)
    primary_result = primary_recording / "RESULT.json"
    _write_or_validate(primary_result, payload)
    return payload


def _write_or_validate(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if _load_json(path, "terminal artifact") != payload:
            raise ValueError("unknown-start reset terminal artifact changed")
        return
    _write_exclusive(path, payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("claim", "failure", "finalize-recovery"))
    parser.add_argument("--ledger-root", type=Path)
    parser.add_argument("--claim-path", type=Path)
    parser.add_argument("--recording-id")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--source-revision")
    parser.add_argument("--runtime-source-fingerprint")
    parser.add_argument("--error")
    parser.add_argument("--primary-recording", type=Path)
    parser.add_argument("--recovery-recording", type=Path)
    parser.add_argument("--recovery-claim", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.command == "claim":
        if (
            arguments.ledger_root is None
            or arguments.recording_id is None
            or arguments.seed is None
            or arguments.source_revision is None
            or arguments.runtime_source_fingerprint is None
        ):
            parser.error("claim requires path, recording id, seed, and source revision")
        payload = claim(
            arguments.ledger_root,
            arguments.recording_id,
            arguments.seed,
            arguments.source_revision,
            arguments.runtime_source_fingerprint,
        )
    elif arguments.command == "failure":
        if arguments.ledger_root is None or not arguments.error:
            parser.error("failure requires --ledger-root and --error")
        payload = failure(arguments.ledger_root, arguments.error)
    else:
        if (
            arguments.primary_recording is None
            or arguments.recovery_recording is None
            or arguments.claim_path is None
            or arguments.recovery_claim is None
            or arguments.source_revision is None
            or arguments.runtime_source_fingerprint is None
        ):
            parser.error("finalize-recovery requires primary and recovery identities")
        payload = finalize_recovery(
            arguments.primary_recording,
            arguments.recovery_recording,
            arguments.claim_path,
            arguments.recovery_claim,
            arguments.source_revision,
            arguments.runtime_source_fingerprint,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
