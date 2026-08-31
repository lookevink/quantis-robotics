"""Frozen TRAIN-only probe for semantic physical-state routing."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import torch

from jepa_wm.action_conditioning_experiment import (
    EXPERIMENT_WINDOW,
    TRAINING_RECORDINGS,
)
from jepa_wm.action_routing_experiment import (
    CONTROL_ARTIFACT_FINGERPRINT,
    _authenticate_base_model,
    _validated_selection,
)
from jepa_wm.causal_context_routing_experiment import (
    _passthrough_owned_route_activations,
    _route_counts_by_slice,
    _selected_corpus_input_fingerprint,
)
from jepa_wm.causal_routing import RecordedMotionLabelSpec
from jepa_wm.insertion_layout import (
    CONTACT_INSERTION_LAYOUT,
    CONTACT_INSERTION_PASSTHROUGH_SEGMENTS,
)
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.persistence import write_json_atomic
from jepa_wm.physical_observation import (
    PHYSICAL_ROUTING_FEATURE_NAMES,
    PHYSICAL_ROUTING_OBSERVATION_SCHEMA,
    PhysicalRoutingObservation,
)
from jepa_wm.physical_route_probe import (
    PhysicalRouteProbeConfig,
    PhysicalRouteProbeDataset,
    run_grouped_physical_route_probe,
)
from jepa_wm.physical_routing import PhysicalStateRoutingSpec
from jepa_wm.rollout_scoring import rollout_action_tensor
from jepa_wm.rollout_training import RolloutTrainingSelection
from jepa_wm.training_artifact import ArtifactIdentity


EXPERIMENT_SCHEMA = "quantis.jepa_wm_physical_state_routing_probe.v1"
FROZEN_EXPERIMENT_CONFIG_FINGERPRINT = (
    "f889fcd39704f9d242e6f4e45965fc5d857e6cf2727109aad86b3cc34361f2d5"
)
OUTPUT_ROOT = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/quantis_physical_state_routing_v2"
)
PROBE_PATH = OUTPUT_ROOT / "route-probe.json"


def _load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT:
        raise ValueError("physical route probe configuration changed")
    payload = json.loads(encoded)
    corpus = payload.get("corpus", {})
    routing = payload.get("routing", {})
    labeling = payload.get("labeling", {})
    expected_holds = [
        segment.value for segment in CONTACT_INSERTION_PASSTHROUGH_SEGMENTS
    ]
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != EXPERIMENT_SCHEMA
        or tuple(corpus.get("training_recordings", ())) != TRAINING_RECORDINGS
        or corpus.get("window") != EXPERIMENT_WINDOW.to_dict()
        or routing.get("observation_schema") != PHYSICAL_ROUTING_OBSERVATION_SCHEMA
        or routing.get("feature_names") != list(PHYSICAL_ROUTING_FEATURE_NAMES)
        or routing.get("runtime_inputs") != ["physical_observation"]
        or set(routing.get("forbidden_runtime_inputs", ()))
        != {
            "visual_context_latent",
            "candidate_action",
            "recorded_future_action",
            "scripted_phase",
            "context_index",
            "seed",
        }
        or sorted(labeling.get("semantic_hold_segments", ())) != sorted(expected_holds)
        or payload.get("probe", {}).get("maximum_passthrough_owned_route_activations")
        != 0
        or payload.get("stopping_rules", {}).get("no_residual_training") is not True
    ):
        raise ValueError("physical route probe contract is invalid")
    return payload


def _routing_spec(experiment: dict[str, Any]) -> PhysicalStateRoutingSpec:
    payload = dict(experiment["routing"])
    payload.pop("runtime_inputs")
    payload.pop("forbidden_runtime_inputs")
    return PhysicalStateRoutingSpec.from_dict(payload)


def _labeling_spec(experiment: dict[str, Any]) -> RecordedMotionLabelSpec:
    payload = dict(experiment["labeling"])
    payload.pop("source")
    payload.pop("semantic_hold_segments")
    return RecordedMotionLabelSpec.from_dict(payload)


def _authenticate_training_recordings(recordings: Sequence[Path]) -> None:
    if tuple(path.name for path in recordings) != TRAINING_RECORDINGS:
        raise ValueError("physical route inputs do not match the exact TRAIN roster")
    for offset, recording in enumerate(recordings):
        ContactInsertionEvidence.from_recording(
            recording,
            expected_split="train",
            expected_seed=2600 + offset,
        )


def _physical_dataset(
    recordings: Sequence[Path],
    selection: RolloutTrainingSelection,
    labeling: RecordedMotionLabelSpec,
) -> PhysicalRouteProbeDataset:
    features = []
    segments = []
    groups = []
    offset = 0
    for group, (recording, selected) in enumerate(
        zip(recordings, selection.recordings)
    ):
        manifest = json.loads((recording / "manifest.json").read_text())
        try:
            insertion_target = manifest["metadata"]["insertion_target"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "physical route recording has no insertion target"
            ) from error
        steps = tuple(
            json.loads(line)
            for line in (recording / "steps.jsonl").read_text().splitlines()
            if line
        )
        count = len(selected.context_indices)
        rollouts = selection.rollouts[offset : offset + count]
        if (
            len(rollouts) != count
            or tuple(rollout.context[0].index for rollout in rollouts)
            != selected.context_indices
        ):
            raise ValueError("physical route rollout selection changed")
        for rollout in rollouts:
            context_index = rollout.context[0].index
            if not 0 <= context_index < len(steps):
                raise ValueError("physical route context index is outside telemetry")
            features.append(
                PhysicalRoutingObservation.from_recorded_step(
                    steps[context_index],
                    insertion_target,
                    rollout.previous_action,
                ).values
            )
            segments.append(CONTACT_INSERTION_LAYOUT.segment_for_index(context_index))
            groups.append(group)
        offset += count
    if offset != len(selection.rollouts) or len(recordings) != len(
        selection.recordings
    ):
        raise ValueError("physical route corpus selection is incomplete")
    labels = labeling.classify_recorded_horizons(
        rollout_action_tensor(selection.rollouts),
        segments,
    )
    return PhysicalRouteProbeDataset(
        features=torch.tensor(features, dtype=torch.float32),
        labels=labels,
        groups=torch.tensor(groups, dtype=torch.long),
        slices=tuple(segment.value for segment in segments),
    )


def probe(
    recordings: Sequence[Path],
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if output.resolve() != PROBE_PATH or output.exists():
        raise ValueError(f"physical route probe output must be new at {PROBE_PATH}")
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    _authenticate_training_recordings(recordings)
    selection = _validated_selection(recordings)
    selected_input_fingerprint = _selected_corpus_input_fingerprint(
        recordings,
        selection,
    )
    dataset = _physical_dataset(recordings, selection, _labeling_spec(experiment))
    routing = _routing_spec(experiment)
    probe_config = experiment["probe"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = monotonic()
    result = run_grouped_physical_route_probe(
        dataset,
        routing,
        PhysicalRouteProbeConfig(
            steps=int(probe_config["steps"]),
            learning_rate=float(probe_config["learning_rate"]),
            weight_decay=float(probe_config["weight_decay"]),
            seed=int(probe_config["seed"]),
        ),
        device=device,
    )
    overall = result["overall"]
    by_route = overall["by_route"]
    passthrough = _passthrough_owned_route_activations(
        result["by_slice"],
        experiment["ownership"]["passthrough"],
    )
    passed = (
        overall["accuracy"] >= probe_config["minimum_overall_accuracy"]
        and by_route["retreat"]["recall"] >= probe_config["minimum_retreat_recall"]
        and by_route["advance"]["recall"] >= probe_config["minimum_advance_recall"]
        and result["by_slice"]["grasp_attach"]["accuracy"]
        >= probe_config["minimum_grasp_attach_accuracy"]
        and result["failed_closed_fraction"]
        <= probe_config["maximum_failed_closed_fraction"]
        and max(passthrough.values())
        <= probe_config["maximum_passthrough_owned_route_activations"]
    )
    report = {
        **result,
        "status": "passed" if passed else "failed",
        "outcome": (
            "physical_state_route_probe_candidate"
            if passed
            else "physical_state_route_probe_failed"
        ),
        "scope": "grouped TRAIN-only physical route probe; not action-model evidence",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "control_artifact": control_identity.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "selected_input_fingerprint": selected_input_fingerprint,
        "semantic_label_counts_by_slice": _route_counts_by_slice(
            dataset.slices,
            dataset.labels,
        ),
        "gate": {
            "passed": passed,
            "minimum_overall_accuracy": probe_config["minimum_overall_accuracy"],
            "minimum_retreat_recall": probe_config["minimum_retreat_recall"],
            "minimum_advance_recall": probe_config["minimum_advance_recall"],
            "minimum_grasp_attach_accuracy": probe_config[
                "minimum_grasp_attach_accuracy"
            ],
            "maximum_failed_closed_fraction": probe_config[
                "maximum_failed_closed_fraction"
            ],
            "passthrough_owned_route_activations": passthrough,
            "maximum_passthrough_owned_route_activations": probe_config[
                "maximum_passthrough_owned_route_activations"
            ],
        },
        "probe_seconds": round(monotonic() - started, 3),
        "held_out_accessed": False,
        "canonical_accessed": False,
        "residual_training_authorized": False,
        "live_action_authorized": False,
    }
    write_json_atomic(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=Path, action="append", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--control-adapter", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = probe(
        arguments.recording,
        arguments.source,
        arguments.checkpoint,
        arguments.control_adapter,
        arguments.output,
        experiment_config=arguments.experiment_config,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
