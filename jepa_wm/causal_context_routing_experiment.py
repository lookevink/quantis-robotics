"""Frozen feasibility and grouped probe for causal context routing."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic
from typing import Any, Sequence

import torch

from jepa_wm.action_conditioning_experiment import (
    EXPERIMENT_WINDOW,
    TRAINING_BOUNDS,
    TRAINING_RECORDINGS,
)
from jepa_wm.action_routing_experiment import (
    CONTROL_ARTIFACT_FINGERPRINT,
    _authenticate_base_model,
    _selected_input_fingerprint,
    _validated_selection,
)
from jepa_wm.causal_route_probe import (
    CausalRouteProbeConfig,
    CausalRouteProbeDataset,
    run_grouped_causal_route_probe,
)
from jepa_wm.causal_routing import (
    CAUSAL_MOTION_ROUTE_NAMES,
    CausalContextRoutingSpec,
)
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_layout import CONTACT_INSERTION_LAYOUT
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.model import load_headless_model
from jepa_wm.owned_slice_gate import (
    SliceEvaluation,
    SliceRequirement,
    TrainingFeasibility,
)
from jepa_wm.persistence import write_json_atomic
from jepa_wm.rollout_scoring import (
    RolloutEnergies,
    rollout_action_tensor,
    score_recorded_against_zero,
)
from jepa_wm.rollout_training import RolloutTrainingSelection
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.trajectory import RecordedRollout


EXPERIMENT_SCHEMA = "quantis.jepa_wm_causal_context_routing_probe.v1"
FROZEN_EXPERIMENT_CONFIG_FINGERPRINT = (
    "f16aa06fd0930a3cfda7503575427e7c113d1044811439f22239cb12f0a5c812"
)
OUTPUT_ROOT = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/quantis_causal_context_routing_v1"
)
PREFLIGHT_PATH = OUTPUT_ROOT / "preflight.json"
PROBE_PATH = OUTPUT_ROOT / "route-probe.json"


def _load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT:
        raise ValueError("causal route probe configuration changed")
    payload = json.loads(encoded)
    corpus = payload.get("corpus", {})
    routing = payload.get("routing", {})
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != EXPERIMENT_SCHEMA
        or tuple(corpus.get("training_recordings", ())) != TRAINING_RECORDINGS
        or corpus.get("window") != EXPERIMENT_WINDOW.to_dict()
        or routing.get("runtime_inputs")
        != ["visual_context_latent", "context_pose", "previous_action"]
        or "candidate_action" not in routing.get("forbidden_runtime_inputs", ())
        or "recorded_future_action" not in routing.get("forbidden_runtime_inputs", ())
        or payload.get("probe", {}).get("maximum_passthrough_owned_route_activations")
        != 0
        or payload.get("stopping_rules", {}).get("no_residual_training") is not True
    ):
        raise ValueError("causal route probe contract is invalid")
    return payload


def _routing_spec(experiment: dict[str, Any]) -> CausalContextRoutingSpec:
    payload = dict(experiment["routing"])
    payload.pop("runtime_inputs")
    payload.pop("forbidden_runtime_inputs")
    return CausalContextRoutingSpec.from_dict(payload)


def _validate_output(output: Path, expected: Path) -> None:
    if output.resolve() != expected or output.exists():
        raise ValueError(f"causal route probe output must be new at {expected}")


def _context_pose_tensor(rollouts: Sequence[RecordedRollout]) -> torch.Tensor:
    return torch.tensor(
        [rollout.context_pose.values for rollout in rollouts],
        dtype=torch.float32,
    )


def _previous_action_tensor(rollouts: Sequence[RecordedRollout]) -> torch.Tensor:
    return torch.tensor(
        [rollout.previous_action.values for rollout in rollouts],
        dtype=torch.float32,
    )


def _slice_names(rollouts: Sequence[RecordedRollout]) -> tuple[str, ...]:
    return tuple(
        CONTACT_INSERTION_LAYOUT.segment_for_index(rollout.context[0].index).value
        for rollout in rollouts
    )


def _route_counts_by_slice(
    slices: Sequence[str],
    routes: torch.Tensor,
) -> dict[str, dict[str, int]]:
    counts = Counter(
        (name, CAUSAL_MOTION_ROUTE_NAMES[int(route)])
        for name, route in zip(slices, routes)
    )
    return {
        name: {
            route_name: counts[(name, route_name)]
            for route_name in CAUSAL_MOTION_ROUTE_NAMES
            if counts[(name, route_name)]
        }
        for name in sorted(set(slices))
    }


def _selected_corpus_input_fingerprint(
    recordings: Sequence[Path],
    selection: RolloutTrainingSelection,
) -> str:
    if len(recordings) != len(selection.recordings):
        raise ValueError("causal route corpus selection is incomplete")
    digest = sha256()
    offset = 0
    for recording_path, recording_selection in zip(
        recordings,
        selection.recordings,
    ):
        recording = recording_path.resolve()
        if recording.name != recording_selection.recording:
            raise ValueError("causal route corpus order changed")
        count = len(recording_selection.context_indices)
        selected_rollouts = selection.rollouts[offset : offset + count]
        if len(selected_rollouts) != count:
            raise ValueError("causal route corpus rollout selection is incomplete")
        digest.update(recording.name.encode())
        digest.update(b"\0")
        digest.update(
            _selected_input_fingerprint(recording, selected_rollouts).encode()
        )
        digest.update(b"\0")
        offset += count
    if offset != len(selection.rollouts):
        raise ValueError("causal route corpus has unowned rollouts")
    return digest.hexdigest()


def _authenticate_training_recordings(recordings: Sequence[Path]) -> None:
    if tuple(path.name for path in recordings) != TRAINING_RECORDINGS:
        raise ValueError("causal route inputs do not match the exact TRAIN roster")
    for offset, recording in enumerate(recordings):
        ContactInsertionEvidence.from_recording(
            recording,
            expected_split="train",
            expected_seed=2600 + offset,
        )


def _slice_metrics(
    slices: Sequence[str],
    recorded: torch.Tensor,
    zero: torch.Tensor,
) -> dict[str, SliceEvaluation]:
    return {
        name: SliceEvaluation(
            recorded_win_rate=float((recorded[indices] < zero[indices]).float().mean()),
            mean_improvement_over_zero=float(
                (zero[indices] - recorded[indices]).mean()
            ),
            signed_order_fraction=0.0,
        )
        for name in sorted(set(slices))
        if (indices := torch.tensor([value == name for value in slices])).any()
    }


def _passthrough_owned_route_activations(
    by_slice: dict[str, Any],
    passthrough_slices: Sequence[str],
) -> dict[str, int]:
    activations = {}
    for name in passthrough_slices:
        try:
            predictions = by_slice[name]["predictions"]
            activations[name] = int(predictions["retreat"]) + int(
                predictions["advance"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"causal route probe lacks passthrough metrics for {name}"
            ) from error
    return activations


def _requirements(experiment: dict[str, Any]) -> dict[str, SliceRequirement]:
    ownership = experiment["ownership"]
    owned_gate = ownership["owned_gate"]
    requirements = {
        name: SliceRequirement.owned(
            minimum_win_rate=float(owned_gate["minimum_win_rate"]),
            require_positive_mean=bool(owned_gate["requires_positive_mean"]),
        )
        for name in ownership["owned"]
    }
    requirements.update(
        {name: SliceRequirement.passthrough() for name in ownership["passthrough"]}
    )
    return requirements


def _score_baseline_batches(
    model: Any,
    contexts: torch.Tensor,
    targets: torch.Tensor,
    actions: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 2,
) -> RolloutEnergies:
    recorded_chunks = []
    zero_chunks = []
    with torch.inference_mode():
        for start in range(0, contexts.shape[0], batch_size):
            stop = start + batch_size
            energies = score_recorded_against_zero(
                model,
                contexts[start:stop].to(device),
                targets[start:stop].to(device),
                actions[:, start:stop].to(device),
            )
            recorded_chunks.append(energies.recorded.cpu())
            zero_chunks.append(energies.zero.cpu())
    return RolloutEnergies(
        recorded=torch.cat(recorded_chunks),
        zero=torch.cat(zero_chunks),
    )


def preflight(
    recordings: Sequence[Path],
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("causal route preflight requires CUDA")
    _validate_output(output, PREFLIGHT_PATH)
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
    rollouts = selection.rollouts
    routing = _routing_spec(experiment)
    actions = rollout_action_tensor(rollouts)
    future_routes = routing.classify_action_horizons(actions)
    slices = _slice_names(rollouts)
    route_counts = _route_counts_by_slice(slices, future_routes)
    feasibility = TrainingFeasibility.evaluate(
        requirements=_requirements(experiment),
        route_counts=route_counts,
        trainable_routes=set(experiment["ownership"]["trainable_routes"]),
    )
    report = {
        "schema": "quantis.jepa_wm_causal_context_preflight.v1",
        "status": "passed" if feasibility.feasible else "blocked",
        "scope": "TRAIN-only feasibility; no residual training or live action",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "control_artifact": control_identity.to_dict(),
        **selection.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "selected_input_fingerprint": selected_input_fingerprint,
        "routing": routing.to_dict(),
        "future_route_counts_by_slice": route_counts,
        "training_feasibility": {
            "feasible": feasibility.feasible,
            "reasons": list(feasibility.reasons),
        },
        "held_out_accessed": False,
        "canonical_accessed": False,
        "residual_training_authorized": False,
        "live_action_authorized": False,
    }
    if not feasibility.feasible:
        report.update(
            {
                "baseline_by_slice": {},
                "load_seconds": 0.0,
                "encoding_seconds": 0.0,
            }
        )
        write_json_atomic(output, report)
        return report
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    load_started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=control_adapter,
    )
    load_seconds = monotonic() - load_started
    encoding_started = monotonic()
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=4,
    )
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=4,
    )
    encoding_seconds = monotonic() - encoding_started
    energies = _score_baseline_batches(
        model,
        contexts,
        targets,
        actions,
        device=device,
    )
    baseline = _slice_metrics(
        slices,
        energies.recorded.cpu(),
        energies.zero.cpu(),
    )
    report.update(
        {
            "baseline_by_slice": {
                name: evaluation.__dict__ for name, evaluation in baseline.items()
            },
            "load_seconds": round(load_seconds, 3),
            "encoding_seconds": round(encoding_seconds, 3),
        }
    )
    write_json_atomic(output, report)
    return report


def probe(
    recordings: Sequence[Path],
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("causal route probe requires CUDA")
    _validate_output(output, PROBE_PATH)
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    if not PREFLIGHT_PATH.is_file():
        raise ValueError("causal route probe requires the authenticated preflight")
    preflight_report = json.loads(PREFLIGHT_PATH.read_text())
    if (
        preflight_report.get("schema") != "quantis.jepa_wm_causal_context_preflight.v1"
        or preflight_report.get("status") != "passed"
        or preflight_report.get("experiment_config_fingerprint")
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
        or preflight_report.get("control_artifact", {}).get("fingerprint")
        != control_identity.fingerprint
    ):
        raise ValueError("causal route preflight did not pass")
    _authenticate_training_recordings(recordings)
    selection = _validated_selection(recordings)
    if preflight_report.get("training_selection_fingerprint") != selection.fingerprint:
        raise ValueError("causal route probe TRAIN selection changed")
    selected_input_fingerprint = _selected_corpus_input_fingerprint(
        recordings,
        selection,
    )
    if preflight_report.get("selected_input_fingerprint") != selected_input_fingerprint:
        raise ValueError("causal route probe TRAIN contents changed")
    rollouts = selection.rollouts
    routing = _routing_spec(experiment)
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    model = load_headless_model(source, checkpoint, device=device)
    encoding_started = monotonic()
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=4,
    )
    encoding_seconds = monotonic() - encoding_started
    groups = torch.arange(len(TRAINING_RECORDINGS)).repeat_interleave(
        EXPERIMENT_WINDOW.count
    )
    dataset = CausalRouteProbeDataset.build(
        contexts,
        _context_pose_tensor(rollouts),
        _previous_action_tensor(rollouts),
        rollout_action_tensor(rollouts),
        groups,
        _slice_names(rollouts),
        routing=routing,
    )
    probe_config = experiment["probe"]
    probe_started = monotonic()
    result = run_grouped_causal_route_probe(
        dataset,
        routing,
        CausalRouteProbeConfig(
            steps=int(probe_config["steps"]),
            learning_rate=float(probe_config["learning_rate"]),
            weight_decay=float(probe_config["weight_decay"]),
            seed=int(probe_config["seed"]),
        ),
        device=device,
    )
    probe_seconds = monotonic() - probe_started
    overall = result["overall"]
    by_route = overall["by_route"]
    attach = result["by_slice"]["grasp_attach"]
    passthrough_activations = _passthrough_owned_route_activations(
        result["by_slice"],
        experiment["ownership"]["passthrough"],
    )
    maximum_passthrough_activations = probe_config[
        "maximum_passthrough_owned_route_activations"
    ]
    passed = (
        overall["accuracy"] >= probe_config["minimum_overall_accuracy"]
        and by_route["retreat"]["recall"] >= probe_config["minimum_retreat_recall"]
        and by_route["advance"]["recall"] >= probe_config["minimum_advance_recall"]
        and attach["accuracy"] >= probe_config["minimum_grasp_attach_accuracy"]
        and result["failed_closed_fraction"]
        <= probe_config["maximum_failed_closed_fraction"]
        and max(passthrough_activations.values()) <= maximum_passthrough_activations
    )
    report = {
        **result,
        "status": "passed" if passed else "failed",
        "outcome": (
            "causal_route_probe_candidate" if passed else "causal_route_probe_failed"
        ),
        "scope": "grouped TRAIN-only route probe; not action-model evidence",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "control_artifact": control_identity.to_dict(),
        "preflight": ArtifactIdentity.from_artifact(PREFLIGHT_PATH).to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "selected_input_fingerprint": selected_input_fingerprint,
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
            "passthrough_owned_route_activations": passthrough_activations,
            "maximum_passthrough_owned_route_activations": (
                maximum_passthrough_activations
            ),
        },
        "encoding_seconds": round(encoding_seconds, 3),
        "probe_seconds": round(probe_seconds, 3),
        "held_out_accessed": False,
        "canonical_accessed": False,
        "residual_training_authorized": False,
        "live_action_authorized": False,
    }
    write_json_atomic(output, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "probe"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument(
            "--recording", type=Path, action="append", required=True
        )
        command_parser.add_argument("--source", type=Path, required=True)
        command_parser.add_argument("--checkpoint", type=Path, required=True)
        command_parser.add_argument("--control-adapter", type=Path, required=True)
        command_parser.add_argument("--experiment-config", type=Path, required=True)
        command_parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    function = preflight if arguments.command == "preflight" else probe
    result = function(
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
