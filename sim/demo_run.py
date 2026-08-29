"""Immutable, authenticated contract for one bounded demo experiment."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from jepa_wm.persistence import write_json_atomic
from jepa_wm.insertion_contract import CONTACT_INSERTION_RECORDING
from jepa_wm.insertion_corpus import InsertionCorpusRoster
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.task_windows import CONTACT_GRASP_PROPOSAL_WINDOW
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    validate_artifact_fingerprint,
)
from jepa_wm.worker_artifacts import ControlWorkerArtifacts
from sim.exploration import DatasetSplit
from sim.control_capture_schedule import (
    canonical_control_fingerprint,
)
from sim.demo_behavior import (
    DemoBehavioralContract,
    DemoTerminalContract,
    current_demo_behavioral_contract,
)
from sim.exploration import build_exploration_plan
from sim.control_identity import ControlProposalRef
from sim.recording import validate_recording_id


DEMO_RUN_SPEC_SCHEMA = "quantis.demo_run_spec.v1"


@dataclass(frozen=True)
class DemoCorpusEntry:
    recording: str
    split: DatasetSplit
    seed: int
    fingerprint: str

    def __post_init__(self) -> None:
        validate_recording_id(self.recording)
        try:
            validate_artifact_fingerprint(self.fingerprint)
        except ValueError as error:
            raise ValueError("demo corpus entry is invalid") from error
        if (
            not isinstance(self.split, DatasetSplit)
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("demo corpus entry is invalid")

    @classmethod
    def from_recording(cls, recording: Path) -> DemoCorpusEntry:
        """Authenticate one roster entry from canonical manifest and telemetry."""

        recording = recording.resolve()
        try:
            manifest = json.loads((recording / "manifest.json").read_text())
            metadata = manifest["metadata"]
            recording_id = manifest["recording_id"]
            split = DatasetSplit(metadata["split"])
            seed = metadata["seed"]
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError("demo corpus recording identity is invalid") from error
        if recording_id != recording.name:
            raise ValueError("demo corpus recording identity is invalid")
        try:
            ContactInsertionEvidence.from_recording(
                recording,
                expected_split=split.value,
                expected_seed=seed,
            )
            steps = tuple(
                json.loads(line)
                for line in (recording / "steps.jsonl").read_text().splitlines()
                if line
            )
            cameras = manifest.get("cameras")
            if (
                not isinstance(cameras, list)
                or not cameras
                or not all(isinstance(camera, str) for camera in cameras)
                or len(set(cameras)) != len(cameras)
                or any(
                    not isinstance(step, dict)
                    or not isinstance(step.get("frames"), dict)
                    or set(step["frames"]) != set(cameras)
                    or any(
                        Path(step["frames"][camera]).is_absolute()
                        or Path(step["frames"][camera]).parts
                        != (camera, f"frame_{index:06d}.png")
                        or not (recording / step["frames"][camera]).is_file()
                        for camera in cameras
                    )
                    for index, step in enumerate(steps)
                )
            ):
                raise ValueError("demo corpus RGB evidence is incomplete")
            fingerprint = canonical_control_fingerprint(
                {
                    path.relative_to(recording).as_posix(): (
                        ArtifactIdentity.from_artifact(path).fingerprint
                    )
                    for path in sorted(recording.rglob("*"))
                    if path.is_file()
                }
            )
        except (OSError, TypeError, ValueError) as error:
            raise ValueError("demo corpus recording evidence is incomplete") from error
        return cls(recording_id, split, seed, fingerprint)

    @classmethod
    def from_dict(cls, payload: Any) -> DemoCorpusEntry:
        if not isinstance(payload, dict) or set(payload) != {
            "recording",
            "split",
            "seed",
            "fingerprint",
        }:
            raise ValueError("demo corpus entry payload is invalid")
        try:
            return cls(
                str(payload["recording"]),
                DatasetSplit(str(payload["split"])),
                payload["seed"],
                str(payload["fingerprint"]),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("demo corpus entry payload is invalid") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording": self.recording,
            "split": self.split.value,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
        }


class DemoArtifactRole(str, Enum):
    STAGE_ASSET = "stage_asset"


class DemoWorkerRole(str, Enum):
    GRASP = "grasp"
    INSERTION = "insertion"


@dataclass(frozen=True)
class DemoArtifactBinding:
    role: DemoArtifactRole
    identity: ArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.role, DemoArtifactRole):
            raise ValueError("demo artifact role is invalid")
        try:
            actual = ArtifactIdentity.from_artifact(self.identity.path)
        except (OSError, ValueError) as error:
            raise ValueError(
                "demo artifact fingerprint cannot be authenticated"
            ) from error
        if actual != self.identity:
            raise ValueError("demo artifact fingerprint does not match frozen bytes")

    @classmethod
    def from_artifact(
        cls, role: DemoArtifactRole, artifact: Any
    ) -> DemoArtifactBinding:
        return cls(role, ArtifactIdentity.from_artifact(artifact))

    @classmethod
    def from_dict(cls, payload: Any) -> DemoArtifactBinding:
        if not isinstance(payload, dict) or set(payload) != {"role", "identity"}:
            raise ValueError("demo artifact binding payload is invalid")
        try:
            role = DemoArtifactRole(payload["role"])
            identity = ArtifactIdentity.from_dict(payload["identity"])
        except (TypeError, ValueError) as error:
            raise ValueError("demo artifact binding payload is invalid") from error
        return cls(role, identity)

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "identity": self.identity.to_dict()}


@dataclass(frozen=True)
class DemoWorkerBinding:
    role: DemoWorkerRole
    identity: str
    manifest: ArtifactIdentity
    proposal: ArtifactIdentity
    proposal_metadata: ArtifactIdentity
    adapter: ArtifactIdentity
    calibration: ArtifactIdentity | None
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, DemoWorkerRole):
            raise ValueError("demo worker role is invalid")
        validate_recording_id(self.identity)
        validate_artifact_fingerprint(self.configuration_fingerprint)
        expected_manifest = f"{self.identity}.worker.json"
        if self.manifest.path.name != expected_manifest:
            raise ValueError("demo worker identity does not match its manifest")
        try:
            current = type(self).from_manifest(
                self.role,
                self.identity,
                self.manifest.path,
            )
        except (OSError, ValueError) as error:
            raise ValueError("demo worker artifacts cannot be authenticated") from error
        if current != self:
            raise ValueError("demo worker artifacts do not match frozen bytes")

    @classmethod
    def from_manifest(
        cls,
        role: DemoWorkerRole,
        identity: str,
        manifest: Path,
    ) -> DemoWorkerBinding:
        manifest = manifest.resolve()
        validate_recording_id(identity)
        if manifest.name != f"{identity}.worker.json":
            raise ValueError("demo worker identity does not match its manifest")
        artifacts = ControlWorkerArtifacts.load(manifest)
        proposal_ref = ControlProposalRef.from_name(
            artifacts.proposal.stem,
            root=artifacts.proposal.parent,
        )
        return cls._from_authenticated_artifacts(
            role,
            identity,
            manifest,
            artifacts,
            proposal_ref,
        )

    @classmethod
    def _from_authenticated_artifacts(
        cls,
        role: DemoWorkerRole,
        identity: str,
        manifest: Path,
        artifacts: ControlWorkerArtifacts,
        proposal_ref: ControlProposalRef,
    ) -> DemoWorkerBinding:
        instance = object.__new__(cls)
        object.__setattr__(instance, "role", role)
        object.__setattr__(instance, "identity", identity)
        object.__setattr__(
            instance, "manifest", ArtifactIdentity.from_artifact(manifest)
        )
        object.__setattr__(instance, "proposal", proposal_ref.checkpoint)
        object.__setattr__(instance, "proposal_metadata", proposal_ref.metadata)
        object.__setattr__(
            instance, "adapter", ArtifactIdentity.from_artifact(artifacts.adapter)
        )
        object.__setattr__(
            instance,
            "calibration",
            (
                ArtifactIdentity.from_artifact(artifacts.calibration)
                if artifacts.calibration is not None
                else None
            ),
        )
        object.__setattr__(
            instance,
            "configuration_fingerprint",
            canonical_control_fingerprint(
                artifacts.to_dict(relative_to=manifest.parent)
            ),
        )
        return instance

    @classmethod
    def from_dict(cls, payload: Any) -> DemoWorkerBinding:
        fields = {
            "role",
            "identity",
            "manifest",
            "proposal",
            "proposal_metadata",
            "adapter",
            "calibration",
            "configuration_fingerprint",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("demo worker binding payload is invalid")
        try:
            return cls(
                DemoWorkerRole(payload["role"]),
                str(payload["identity"]),
                ArtifactIdentity.from_dict(payload["manifest"]),
                ArtifactIdentity.from_dict(payload["proposal"]),
                ArtifactIdentity.from_dict(payload["proposal_metadata"]),
                ArtifactIdentity.from_dict(payload["adapter"]),
                (
                    ArtifactIdentity.from_dict(payload["calibration"])
                    if payload["calibration"] is not None
                    else None
                ),
                str(payload["configuration_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("demo worker binding payload is invalid") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "identity": self.identity,
            "manifest": self.manifest.to_dict(),
            "proposal": self.proposal.to_dict(),
            "proposal_metadata": self.proposal_metadata.to_dict(),
            "adapter": self.adapter.to_dict(),
            "calibration": (
                self.calibration.to_dict() if self.calibration is not None else None
            ),
            "configuration_fingerprint": self.configuration_fingerprint,
        }


@dataclass(frozen=True)
class DemoRunSelection:
    reference_recording: str
    exploration_seed: int
    reference_fingerprint: str
    context_index: int
    exploration_plan_fingerprint: str

    def __post_init__(self) -> None:
        validate_recording_id(self.reference_recording)
        for fingerprint in (
            self.reference_fingerprint,
            self.exploration_plan_fingerprint,
        ):
            validate_artifact_fingerprint(fingerprint)
        if (
            isinstance(self.exploration_seed, bool)
            or not isinstance(self.exploration_seed, int)
            or self.exploration_seed < 0
            or self.context_index != CONTACT_GRASP_PROPOSAL_WINDOW.start_index
        ):
            raise ValueError("demo run selection is invalid")

    @classmethod
    def from_reference(
        cls,
        recording: Path,
        exploration_seed: int,
    ) -> DemoRunSelection:
        entry = DemoCorpusEntry.from_recording(recording)
        if entry.split is not DatasetSplit.HELD_OUT or entry.seed != exploration_seed:
            raise ValueError("demo run selection requires its exact held-out seed")
        plan = replace(
            build_exploration_plan(exploration_seed, entry.split),
            socket_scale=CONTACT_INSERTION_RECORDING.socket_scale,
        )
        return cls(
            entry.recording,
            exploration_seed,
            entry.fingerprint,
            CONTACT_GRASP_PROPOSAL_WINDOW.start_index,
            canonical_control_fingerprint(plan.metadata()),
        )

    @classmethod
    def from_dict(cls, payload: Any) -> DemoRunSelection:
        fields = {
            "reference_recording",
            "exploration_seed",
            "reference_fingerprint",
            "context_index",
            "exploration_plan_fingerprint",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("demo run selection payload is invalid")
        return cls(
            str(payload["reference_recording"]),
            payload["exploration_seed"],
            str(payload["reference_fingerprint"]),
            payload["context_index"],
            str(payload["exploration_plan_fingerprint"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_recording": self.reference_recording,
            "exploration_seed": self.exploration_seed,
            "reference_fingerprint": self.reference_fingerprint,
            "context_index": self.context_index,
            "exploration_plan_fingerprint": self.exploration_plan_fingerprint,
        }


@dataclass(frozen=True)
class DemoRunSpec:
    source_revision: str
    container_image_digest: str
    corpus: tuple[DemoCorpusEntry, ...]
    artifacts: tuple[DemoArtifactBinding, ...]
    workers: tuple[DemoWorkerBinding, ...]
    selection: DemoRunSelection
    behavior: DemoBehavioralContract

    def __post_init__(self) -> None:
        if (
            len(self.source_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.source_revision
            )
            or not self.container_image_digest.startswith("sha256:")
            or len(self.container_image_digest) != 71
            or any(
                character not in "0123456789abcdef"
                for character in self.container_image_digest[7:]
            )
        ):
            raise ValueError("demo run source identity is invalid")
        if not isinstance(self.behavior, DemoBehavioralContract) or not isinstance(
            self.selection, DemoRunSelection
        ):
            raise ValueError("demo run frozen contract is invalid")
        train = tuple(
            entry for entry in self.corpus if entry.split is DatasetSplit.TRAIN
        )
        held_out = tuple(
            entry for entry in self.corpus if entry.split is DatasetSplit.HELD_OUT
        )
        if len(train) != 12 or len(held_out) != 2:
            raise ValueError(
                "demo run requires exactly 12 TRAIN and two HELD_OUT entries"
            )
        experiment_id = train[0].recording.removesuffix("-train-00")
        canonical_roster = InsertionCorpusRoster.create(
            experiment_id,
            train[0].seed,
        ).recordings
        if tuple(
            (entry.recording, entry.split.value, entry.seed) for entry in self.corpus
        ) != tuple(
            (entry.recording_id, entry.split, entry.seed) for entry in canonical_roster
        ):
            raise ValueError("demo corpus roster is not canonical")
        if len({entry.recording for entry in self.corpus}) != len(self.corpus) or len(
            {entry.seed for entry in self.corpus}
        ) != len(self.corpus):
            raise ValueError("demo corpus roster overlaps")
        roles = tuple(binding.role for binding in self.artifacts)
        if len(set(roles)) != len(roles) or set(roles) != set(DemoArtifactRole):
            raise ValueError("demo artifact roster is invalid")
        worker_roles = tuple(binding.role for binding in self.workers)
        if len(set(worker_roles)) != len(worker_roles) or set(worker_roles) != set(
            DemoWorkerRole
        ):
            raise ValueError("demo worker roster is invalid")
        selected_entry = next(
            (
                entry
                for entry in self.corpus
                if entry.recording == self.selection.reference_recording
            ),
            None,
        )
        if (
            selected_entry is None
            or selected_entry.split is not DatasetSplit.HELD_OUT
            or selected_entry.seed != self.selection.exploration_seed
            or selected_entry.fingerprint != self.selection.reference_fingerprint
        ):
            raise ValueError("demo run selection is outside its corpus")

    @property
    def action_cap(self) -> int:
        return self.behavior.action_cap

    @property
    def terminal_contract(self) -> DemoTerminalContract:
        return self.behavior.terminal

    @classmethod
    def from_dict(cls, payload: Any) -> DemoRunSpec:
        fields = {
            "schema",
            "source_revision",
            "container_image_digest",
            "corpus",
            "artifacts",
            "workers",
            "selection",
            "behavior",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != fields
            or payload.get("schema") != DEMO_RUN_SPEC_SCHEMA
            or not isinstance(payload.get("corpus"), list)
            or not isinstance(payload.get("artifacts"), list)
            or not isinstance(payload.get("workers"), list)
        ):
            raise ValueError("demo run spec payload is invalid")
        try:
            return cls(
                source_revision=str(payload["source_revision"]),
                container_image_digest=str(payload["container_image_digest"]),
                corpus=tuple(
                    DemoCorpusEntry.from_dict(entry) for entry in payload["corpus"]
                ),
                artifacts=tuple(
                    DemoArtifactBinding.from_dict(binding)
                    for binding in payload["artifacts"]
                ),
                workers=tuple(
                    DemoWorkerBinding.from_dict(binding)
                    for binding in payload["workers"]
                ),
                selection=DemoRunSelection.from_dict(payload["selection"]),
                behavior=DemoBehavioralContract.from_dict(payload["behavior"]),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("demo run spec payload is invalid") from error

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DEMO_RUN_SPEC_SCHEMA,
            "source_revision": self.source_revision,
            "container_image_digest": self.container_image_digest,
            "corpus": [entry.to_dict() for entry in self.corpus],
            "artifacts": [binding.to_dict() for binding in self.artifacts],
            "workers": [binding.to_dict() for binding in self.workers],
            "selection": self.selection.to_dict(),
            "behavior": self.behavior.to_dict(),
        }

    def authenticate_corpus(self, recording_root: Path) -> None:
        """Recompute the full roster before granting one demo run."""

        root = recording_root.resolve()
        for expected in self.corpus:
            try:
                actual = DemoCorpusEntry.from_recording(root / expected.recording)
            except ValueError as error:
                raise ValueError(
                    f"demo corpus fingerprint does not match: {expected.recording}"
                ) from error
            if actual != expected:
                raise ValueError(
                    f"demo corpus fingerprint does not match: {expected.recording}"
                )

    def persist(self, path: Path) -> None:
        """Atomically publish the immutable manifest before a live experiment."""

        write_json_atomic(path, self.to_dict())

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return sha256(encoded).hexdigest()


def validate_demo_run_spec(
    spec_path: Path,
    *,
    expected_fingerprint: str,
    recording_root: Path,
    source_revision: str,
    container_image_digest: str,
    required_artifacts: Mapping[DemoArtifactRole, Path],
    required_workers: Mapping[DemoWorkerRole, tuple[str, Path]],
    reference_recording: str,
    exploration_seed: int,
    grasp_actions: int,
    insertion_actions: int,
) -> DemoRunSpec:
    """Fail before live work unless the complete frozen run still authenticates."""

    try:
        validate_artifact_fingerprint(expected_fingerprint)
        payload = json.loads(spec_path.resolve().read_text())
        spec = DemoRunSpec.from_dict(payload)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("demo run spec cannot be authenticated") from error
    if spec.fingerprint != expected_fingerprint:
        raise ValueError("demo run frozen fingerprint does not match")
    if (
        spec.source_revision != source_revision
        or spec.container_image_digest != container_image_digest
    ):
        raise ValueError("demo run source identity does not match")
    current_behavior = current_demo_behavioral_contract()
    if spec.behavior != current_behavior:
        raise ValueError("demo run behavioral contract does not match current code")
    if (
        isinstance(grasp_actions, bool)
        or not isinstance(grasp_actions, int)
        or isinstance(insertion_actions, bool)
        or not isinstance(insertion_actions, int)
        or grasp_actions != current_behavior.terminal.grasp_actions
        or insertion_actions != current_behavior.terminal.insertion_actions
    ):
        raise ValueError("demo run orchestration action allocation does not match")
    spec.authenticate_corpus(recording_root)
    current_selection = DemoRunSelection.from_reference(
        recording_root.resolve() / reference_recording,
        exploration_seed,
    )
    if spec.selection != current_selection:
        raise ValueError("demo run selection does not match")
    bindings = {binding.role: binding for binding in spec.artifacts}
    if set(required_artifacts) != set(DemoArtifactRole):
        raise ValueError("demo run required artifact roster is invalid")
    for role, path in required_artifacts.items():
        binding = bindings.get(role)
        if binding is None or ArtifactIdentity.from_artifact(path.resolve()) != (
            binding.identity
        ):
            raise ValueError(f"demo run artifact does not match: {role}")
    worker_bindings = {binding.role: binding for binding in spec.workers}
    if set(required_workers) != set(DemoWorkerRole):
        raise ValueError("demo run required worker roster is invalid")
    for role, (identity, manifest) in required_workers.items():
        if DemoWorkerBinding.from_manifest(role, identity, manifest) != (
            worker_bindings.get(role)
        ):
            raise ValueError(f"demo run worker does not match: {role.value}")
    return spec
