"""Shared path and nonce policy for simulator control sessions."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from sim.recording import validate_recording_id


CONTROL_PROPOSAL_ROOT = Path("/home/ubuntu/docker/jepa-wm/checkpoints")


def observation_id_for_session(session_id: str) -> int:
    """Derive a stable nonzero 64-bit request nonce from one session ID."""

    validate_recording_id(session_id)
    identifier = int.from_bytes(sha256(session_id.encode()).digest()[:8], "big")
    return identifier or 1


def control_proposal_path(proposal_name: str) -> Path:
    validate_recording_id(proposal_name)
    return CONTROL_PROPOSAL_ROOT / f"{proposal_name}.pth"
