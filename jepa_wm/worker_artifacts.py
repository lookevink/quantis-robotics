"""One persisted identity for the model artifacts loaded by a control worker."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from jepa_wm.persistence import write_json_atomic


WORKER_ARTIFACTS_SCHEMA = "quantis.jepa_wm_control_worker_artifacts.v1"


@dataclass(frozen=True)
class ControlWorkerArtifacts:
    proposal: Path
    adapter: Path
    calibration: Path | None = None

    def __post_init__(self) -> None:
        paths = (self.proposal, self.adapter)
        if any(not path.is_absolute() for path in paths) or (
            self.calibration is not None and not self.calibration.is_absolute()
        ):
            raise ValueError("control worker artifact paths must be absolute")

    @property
    def calibrated(self) -> bool:
        return self.calibration is not None

    def to_dict(self, *, relative_to: Path | None = None) -> dict[str, Any]:
        def encoded(path: Path | None) -> str | None:
            if path is None:
                return None
            if relative_to is None:
                return str(path)
            return str(path.relative_to(relative_to))

        return {
            "schema": WORKER_ARTIFACTS_SCHEMA,
            "proposal": encoded(self.proposal),
            "adapter": encoded(self.adapter),
            "calibration": encoded(self.calibration),
            "calibrated": self.calibrated,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, relative_to: Path
    ) -> ControlWorkerArtifacts:
        if payload.get("schema") != WORKER_ARTIFACTS_SCHEMA:
            raise ValueError("control worker artifact schema is invalid")

        def resolved(value: Any, *, optional: bool = False) -> Path | None:
            if value is None and optional:
                return None
            if not isinstance(value, str) or not value:
                raise ValueError("control worker artifact path is invalid")
            path = Path(value)
            return path.resolve() if path.is_absolute() else (relative_to / path).resolve()

        proposal = resolved(payload.get("proposal"))
        adapter = resolved(payload.get("adapter"))
        if proposal is None or adapter is None:
            raise ValueError("control worker artifact paths are required")
        artifacts = cls(
            proposal=proposal,
            adapter=adapter,
            calibration=resolved(payload.get("calibration"), optional=True),
        )
        if payload.get("calibrated") is not artifacts.calibrated:
            raise ValueError("control worker artifact claims are inconsistent")
        return artifacts

    @classmethod
    def load(cls, manifest: Path) -> ControlWorkerArtifacts:
        path = manifest.resolve()
        payload = json.loads(path.read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("control worker artifact manifest must be an object")
        return cls.from_dict(payload, relative_to=path.parent)

    def write(self, manifest: Path) -> None:
        path = manifest.resolve()
        write_json_atomic(path, self.to_dict(relative_to=path.parent))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("--output", type=Path, required=True)
    write_parser.add_argument("--proposal", type=Path, required=True)
    write_parser.add_argument("--adapter", type=Path, required=True)
    write_parser.add_argument("--calibration", type=Path)
    proposal_parser = subparsers.add_parser("proposal-name")
    proposal_parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "proposal-name":
        print(ControlWorkerArtifacts.load(args.manifest).proposal.stem)
        return
    ControlWorkerArtifacts(
        args.proposal.resolve(),
        args.adapter.resolve(),
        args.calibration.resolve() if args.calibration is not None else None,
    ).write(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
