"""Command-line preflight for one frozen demo run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from jepa_wm.persistence import write_json_atomic
from sim.demo_run import DemoArtifactRole, DemoWorkerRole, validate_demo_run_spec
from sim.recording import validate_recording_id


def _artifact_bindings(values: Sequence[str]) -> dict[DemoArtifactRole, Path]:
    bindings: dict[DemoArtifactRole, Path] = {}
    for value in values:
        raw_role, separator, raw_path = value.partition("=")
        try:
            role = DemoArtifactRole(raw_role)
        except ValueError as error:
            raise ValueError("demo run artifact argument is invalid") from error
        if not separator or not role or not raw_path or role in bindings:
            raise ValueError("demo run artifact argument is invalid")
        bindings[role] = Path(raw_path)
    return bindings


def _worker_bindings(
    values: Sequence[str],
) -> dict[DemoWorkerRole, tuple[str, Path]]:
    bindings: dict[DemoWorkerRole, tuple[str, Path]] = {}
    for value in values:
        raw_role, separator, remainder = value.partition("=")
        identity, identity_separator, raw_path = remainder.partition("=")
        try:
            role = DemoWorkerRole(raw_role)
        except ValueError as error:
            raise ValueError("demo run worker argument is invalid") from error
        if (
            not separator
            or not identity_separator
            or not identity
            or not raw_path
            or role in bindings
        ):
            raise ValueError("demo run worker argument is invalid")
        bindings[role] = (identity, Path(raw_path))
    return bindings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--spec", type=Path, required=True)
    verify.add_argument("--fingerprint", required=True)
    verify.add_argument("--recording-root", type=Path, required=True)
    verify.add_argument("--source-revision", required=True)
    verify.add_argument("--container-image-digest", required=True)
    verify.add_argument("--run-id", required=True)
    verify.add_argument("--binding-output", type=Path, required=True)
    verify.add_argument("--grasp-actions", type=int, required=True)
    verify.add_argument("--insertion-actions", type=int, required=True)
    verify.add_argument("--reference-recording", required=True)
    verify.add_argument("--exploration-seed", type=int, required=True)
    verify.add_argument("--artifact", action="append", default=[])
    verify.add_argument("--worker", action="append", default=[])
    arguments = parser.parse_args(argv)
    if arguments.command != "verify":
        raise ValueError("unsupported demo run command")
    validate_recording_id(arguments.run_id)
    required_artifacts = _artifact_bindings(arguments.artifact)
    required_workers = _worker_bindings(arguments.worker)
    spec = validate_demo_run_spec(
        arguments.spec,
        expected_fingerprint=arguments.fingerprint,
        recording_root=arguments.recording_root,
        source_revision=arguments.source_revision,
        container_image_digest=arguments.container_image_digest,
        required_artifacts=required_artifacts,
        required_workers=required_workers,
        reference_recording=arguments.reference_recording,
        exploration_seed=arguments.exploration_seed,
        grasp_actions=arguments.grasp_actions,
        insertion_actions=arguments.insertion_actions,
    )
    result = {
        "status": "demo_run_spec_authenticated",
        "fingerprint": spec.fingerprint,
    }
    write_json_atomic(
        arguments.binding_output,
        {
            "schema": "quantis.demo_run_binding.v1",
            "run_id": arguments.run_id,
            "spec": str(arguments.spec.resolve()),
            "spec_fingerprint": spec.fingerprint,
            "source_revision": spec.source_revision,
            "container_image_digest": spec.container_image_digest,
            "artifacts": {
                role.value: str(path.resolve())
                for role, path in sorted(required_artifacts.items())
            },
            "workers": {
                role.value: {
                    "identity": identity,
                    "manifest": str(manifest.resolve()),
                }
                for role, (identity, manifest) in sorted(required_workers.items())
            },
            "selection": spec.selection.to_dict(),
        },
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
