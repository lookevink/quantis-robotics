"""Typed, serializable roster for the fixed insertion-domain corpus."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.identifiers import validate_safe_identifier
from jepa_wm.training_artifact import validate_artifact_fingerprint


TRAINING_RECORDINGS = 12
HELD_OUT_RECORDINGS = 2
HELD_OUT_SEED_OFFSET = 10_000
FRESH_HELD_OUT_RECORDINGS = 2


def _canonical_recordings(
    experiment_id: str,
    base_seed: int,
) -> tuple[InsertionCorpusRecording, ...]:
    return tuple(
        InsertionCorpusRecording(
            f"{experiment_id}-train-{index:02d}",
            "train",
            base_seed + index,
        )
        for index in range(TRAINING_RECORDINGS)
    ) + tuple(
        InsertionCorpusRecording(
            f"{experiment_id}-held-{index:02d}",
            "held_out",
            base_seed + HELD_OUT_SEED_OFFSET + index,
        )
        for index in range(HELD_OUT_RECORDINGS)
    )


def _canonical_fresh_recordings(
    evaluation_id: str,
    base_seed: int,
) -> tuple[InsertionCorpusRecording, ...]:
    return tuple(
        InsertionCorpusRecording(
            f"{evaluation_id}-held-{index:02d}",
            "held_out",
            base_seed + index,
        )
        for index in range(FRESH_HELD_OUT_RECORDINGS)
    )


@dataclass(frozen=True)
class InsertionCorpusRecording:
    recording_id: str
    split: str
    seed: int

    def __post_init__(self) -> None:
        if (
            not self.recording_id
            or self.split not in {"train", "held_out"}
            or self.seed < 0
        ):
            raise ValueError("insertion corpus recording is invalid")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "recording_id": self.recording_id,
            "split": self.split,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionCorpusRecording:
        if not isinstance(payload, dict):
            raise ValueError("insertion corpus recording must be an object")
        recording_id = payload.get("recording_id")
        split = payload.get("split")
        seed = payload.get("seed")
        if (
            not isinstance(recording_id, str)
            or not isinstance(split, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ValueError("insertion corpus recording fields are invalid")
        return cls(recording_id, split, seed)


@dataclass(frozen=True)
class InsertionCorpusRoster:
    experiment_id: str
    base_seed: int
    recordings: tuple[InsertionCorpusRecording, ...]

    def __post_init__(self) -> None:
        validate_safe_identifier(self.experiment_id)
        if (
            self.base_seed < 0
            or self.recordings
            != _canonical_recordings(self.experiment_id, self.base_seed)
        ):
            raise ValueError("insertion corpus roster is inconsistent")

    @classmethod
    def create(cls, experiment_id: str, base_seed: int) -> InsertionCorpusRoster:
        return cls(
            experiment_id,
            base_seed,
            _canonical_recordings(experiment_id, base_seed),
        )

    def for_split(self, split: str) -> tuple[InsertionCorpusRecording, ...]:
        return tuple(
            recording for recording in self.recordings if recording.split == split
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "quantis.jepa_wm_insertion_corpus.v1",
            "experiment_id": self.experiment_id,
            "base_seed": self.base_seed,
            "recordings": [recording.to_dict() for recording in self.recordings],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionCorpusRoster:
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != "quantis.jepa_wm_insertion_corpus.v1"
            or not isinstance(payload.get("experiment_id"), str)
            or isinstance(payload.get("base_seed"), bool)
            or not isinstance(payload.get("base_seed"), int)
            or not isinstance(payload.get("recordings"), list)
        ):
            raise ValueError("insertion corpus roster payload is invalid")
        return cls(
            payload["experiment_id"],
            payload["base_seed"],
            tuple(
                InsertionCorpusRecording.from_dict(recording)
                for recording in payload["recordings"]
            ),
        )

    @classmethod
    def load(cls, path: Path) -> InsertionCorpusRoster:
        return cls.from_dict(json.loads(path.read_text()))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


@dataclass(frozen=True)
class FrozenInsertionAdapter:
    name: str
    fingerprint: str

    def __post_init__(self) -> None:
        validate_safe_identifier(self.name)
        validate_artifact_fingerprint(self.fingerprint)

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, payload: Any) -> FrozenInsertionAdapter:
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("name"), str)
            or not isinstance(payload.get("fingerprint"), str)
        ):
            raise ValueError("frozen insertion adapter identity is invalid")
        return cls(payload["name"], payload["fingerprint"])


@dataclass(frozen=True)
class InsertionFreshEvaluationRoster:
    evaluation_id: str
    base_seed: int
    source_corpus: InsertionCorpusRoster
    adapter: FrozenInsertionAdapter
    recordings: tuple[InsertionCorpusRecording, ...]

    def __post_init__(self) -> None:
        validate_safe_identifier(self.evaluation_id)
        source_seeds = {recording.seed for recording in self.source_corpus.recordings}
        source_recording_ids = {
            recording.recording_id for recording in self.source_corpus.recordings
        }
        if (
            self.base_seed < 0
            or self.evaluation_id == self.source_corpus.experiment_id
            or self.recordings
            != _canonical_fresh_recordings(self.evaluation_id, self.base_seed)
            or source_seeds.intersection(
                recording.seed for recording in self.recordings
            )
            or source_recording_ids.intersection(
                recording.recording_id for recording in self.recordings
            )
        ):
            raise ValueError("fresh insertion evaluation roster is inconsistent")

    @classmethod
    def create(
        cls,
        evaluation_id: str,
        base_seed: int,
        source_corpus: InsertionCorpusRoster,
        adapter: FrozenInsertionAdapter,
    ) -> InsertionFreshEvaluationRoster:
        return cls(
            evaluation_id,
            base_seed,
            source_corpus,
            adapter,
            _canonical_fresh_recordings(evaluation_id, base_seed),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "quantis.jepa_wm_insertion_fresh_evaluation.v1",
            "evaluation_id": self.evaluation_id,
            "base_seed": self.base_seed,
            "source_corpus": self.source_corpus.to_dict(),
            "adapter": self.adapter.to_dict(),
            "recordings": [recording.to_dict() for recording in self.recordings],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> InsertionFreshEvaluationRoster:
        if (
            not isinstance(payload, dict)
            or payload.get("schema")
            != "quantis.jepa_wm_insertion_fresh_evaluation.v1"
            or not isinstance(payload.get("evaluation_id"), str)
            or isinstance(payload.get("base_seed"), bool)
            or not isinstance(payload.get("base_seed"), int)
            or not isinstance(payload.get("recordings"), list)
        ):
            raise ValueError("fresh insertion evaluation roster payload is invalid")
        return cls(
            payload["evaluation_id"],
            payload["base_seed"],
            InsertionCorpusRoster.from_dict(payload.get("source_corpus")),
            FrozenInsertionAdapter.from_dict(payload.get("adapter")),
            tuple(
                InsertionCorpusRecording.from_dict(recording)
                for recording in payload["recordings"]
            ),
        )

    @classmethod
    def load(cls, path: Path) -> InsertionFreshEvaluationRoster:
        return cls.from_dict(json.loads(path.read_text()))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


def _emit(roster: InsertionCorpusRoster, output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(roster.to_dict()))
    elif output_format == "tsv":
        for recording in roster.recordings:
            print(recording.recording_id, recording.seed, recording.split, sep="\t")
    elif output_format in {"train-csv", "held-out-csv"}:
        split = "train" if output_format == "train-csv" else "held_out"
        print(",".join(recording.recording_id for recording in roster.for_split(split)))
    else:
        raise ValueError(f"unsupported roster output: {output_format}")


def _emit_fresh(
    roster: InsertionFreshEvaluationRoster,
    output_format: str,
) -> None:
    if output_format == "json":
        print(json.dumps(roster.to_dict()))
    elif output_format == "tsv":
        for recording in roster.recordings:
            print(recording.recording_id, recording.seed, recording.split, sep="\t")
    elif output_format == "held-out-csv":
        print(",".join(recording.recording_id for recording in roster.recordings))
    elif output_format == "evaluation-id":
        print(roster.evaluation_id)
    elif output_format == "adapter-name":
        print(roster.adapter.name)
    else:
        raise ValueError(f"unsupported fresh roster output: {output_format}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--experiment-id", required=True)
    create.add_argument("--base-seed", type=int, required=True)
    create.add_argument("--output", type=Path, required=True)
    create_fresh = subparsers.add_parser("create-fresh")
    create_fresh.add_argument("--evaluation-id", required=True)
    create_fresh.add_argument("--base-seed", type=int, required=True)
    create_fresh.add_argument("--source-roster", type=Path, required=True)
    create_fresh.add_argument("--adapter-name", required=True)
    create_fresh.add_argument("--adapter-fingerprint", required=True)
    create_fresh.add_argument("--output", type=Path, required=True)
    show = subparsers.add_parser("show")
    show.add_argument("--roster", type=Path, required=True)
    show.add_argument(
        "--format",
        choices=("json", "tsv", "train-csv", "held-out-csv"),
        default="json",
    )
    show_fresh = subparsers.add_parser("show-fresh")
    show_fresh.add_argument("--roster", type=Path, required=True)
    show_fresh.add_argument(
        "--format",
        choices=(
            "json",
            "tsv",
            "held-out-csv",
            "evaluation-id",
            "adapter-name",
        ),
        default="json",
    )
    arguments = parser.parse_args(argv)
    if arguments.command == "create":
        InsertionCorpusRoster.create(
            arguments.experiment_id,
            arguments.base_seed,
        ).write(arguments.output)
        return 0
    if arguments.command == "create-fresh":
        InsertionFreshEvaluationRoster.create(
            arguments.evaluation_id,
            arguments.base_seed,
            InsertionCorpusRoster.load(arguments.source_roster),
            FrozenInsertionAdapter(
                arguments.adapter_name,
                arguments.adapter_fingerprint,
            ),
        ).write(arguments.output)
        return 0
    if arguments.command == "show-fresh":
        _emit_fresh(
            InsertionFreshEvaluationRoster.load(arguments.roster),
            arguments.format,
        )
        return 0
    _emit(InsertionCorpusRoster.load(arguments.roster), arguments.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
