"""Frozen offline experiment for runtime-command action routing."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import monotonic
from typing import Any, Sequence

import numpy as np
import torch

from jepa_wm.action_conditioning import (
    ACTION_CONDITIONING_SCHEMA,
    BASE_COMMAND_ROUTE,
    COMMAND_ROUTE_NAMES,
    NEGATIVE_X_COMMAND_ROUTE,
    POSITIVE_X_COMMAND_ROUTE,
    ActionConditioningContract,
    ActionConditioningKind,
    ActionConditioningSpec,
    LoadedActionConditioning,
    RuntimeCommandRoutingSpec,
    action_conditioning_parameters,
    install_action_conditioning,
    save_action_conditioning,
)
from jepa_wm.action_conditioning_experiment import (
    CANONICAL_HELD_OUT,
    EXPERIMENT_WINDOW,
    TRAINING_BOUNDS,
    TRAINING_RECORDINGS,
    EvaluationEnergies,
    _metrics,
    _regime_for_context,
    _score_evaluation_batches,
    _segment_for_context,
)
from jepa_wm.action_conditioning_training import (
    AlternatingCommandRouteSampler,
    signed_x_margin_loss,
    signed_x_negatives,
)
from jepa_wm.adapt_recording import (
    ContrastiveTermConfig,
    _sample_mismatched_indices,
    mismatched_negative_candidates,
    validated_training_recordings,
)
from jepa_wm.candidate_negatives import (
    CandidateMiningConfig,
    mine_lowest_energy_candidates,
    sample_local_candidates,
)
from jepa_wm.contract import MODEL_ID
from jepa_wm.frames import encode_clips
from jepa_wm.insertion_layout import ContactInsertionSegment
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.model import load_headless_model
from jepa_wm.persistence import write_json_atomic
from jepa_wm.readiness import ActionControlGate
from jepa_wm.rollout_scoring import (
    rollout_action_tensor,
    score_actions,
    score_recorded_against_mismatched,
)
from jepa_wm.rollout_training import RolloutTrainingSelection
from jepa_wm.training_artifact import (
    ArtifactIdentity,
    TrainingArtifactMetadata,
    artifact_fingerprint,
    load_training_report,
    training_configuration_fingerprint,
    training_report_path,
)
from jepa_wm.trajectory import RecordedRollout, load_rollouts


EXPERIMENT_SCHEMA = "quantis.jepa_wm_runtime_command_routing_experiment.v1"
FROZEN_EXPERIMENT_CONFIG_FINGERPRINT = (
    "98fc2af503919d52a3853d3181bf007d56360136e5c1d27cd1a08a4db18bf66d"
)
FRESH_CANARY = "contact-insertion-v10-drive-slow-72601-held-00"
FRESH_CANARY_SEED = 72601
CONTROL_ARTIFACT_FINGERPRINT = (
    "e2fea116de2aca46bb9a3e72e3d971e49dfc64936f8fc27469353da102ffa0ed"
)
EVALUATION_OUTPUT_ROOT = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/quantis_action_routing_v1"
)
ROUTING_SPEC = RuntimeCommandRoutingSpec(
    signed_x_deadband=0.0001,
    translation_activity_deadband=0.0001,
    rotation_activity_deadband=0.001,
    gripper_activity_deadband=0.005,
)
TREATMENT_SPEC = ActionConditioningSpec(
    ActionConditioningKind.RUNTIME_COMMAND_RESIDUAL,
    runtime_routing=ROUTING_SPEC,
)
EXPECTED_ROUTE_ROSTER = {
    "negative_x": 564,
    "positive_x": 1296,
    "base_neutral": 106,
    "base_active_non_x": 50,
    "total": 2016,
}


def _authenticate_base_model(
    experiment: dict[str, Any],
    source: Path,
    checkpoint: Path,
) -> dict[str, str]:
    base = experiment["base_model"]
    expected_revision = str(base["source_revision"])
    configured_revision = os.environ.get("JEPA_WM_REVISION")
    try:
        actual_revision = subprocess.run(
            ("git", "-C", str(source.resolve()), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_changes = subprocess.run(
            (
                "git",
                "-C",
                str(source.resolve()),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("JEPA-WM source revision cannot be authenticated") from error
    checkpoint_identity = ArtifactIdentity.from_artifact(checkpoint)
    if (
        configured_revision != expected_revision
        or actual_revision != expected_revision
        or tracked_changes
        or checkpoint_identity.fingerprint != base["checkpoint_fingerprint"]
    ):
        raise ValueError("frozen JEPA-WM base identity changed")
    return {
        "source": str(source.resolve()),
        "source_revision": actual_revision,
        "checkpoint": str(checkpoint_identity.path),
        "checkpoint_fingerprint": checkpoint_identity.fingerprint,
    }


def _expected_evaluation_output(recording: str, treatment: str) -> Path:
    return EVALUATION_OUTPUT_ROOT / f"evaluation-{recording}-{treatment}.json"


def _selected_input_fingerprint(
    recording: Path,
    rollouts: Sequence[RecordedRollout],
) -> str:
    root = recording.resolve()
    selected = {root / "manifest.json", root / "steps.jsonl"}
    for rollout in rollouts:
        selected.update(rollout.context_paths)
        selected.update(rollout.target_clip)
    digest = sha256()
    for path in sorted(selected):
        resolved = path.resolve()
        relative = resolved.relative_to(root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_experiment_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT:
        raise ValueError("runtime-routing experiment configuration fingerprint changed")
    payload = json.loads(encoded)
    corpus = payload.get("corpus", {})
    canary = corpus.get("fresh_development_canary", {})
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != EXPERIMENT_SCHEMA
        or tuple(corpus.get("training_recordings", ())) != TRAINING_RECORDINGS
        or canary.get("recording") != FRESH_CANARY
        or canary.get("seed") != FRESH_CANARY_SEED
        or payload.get("train_only_route_roster") != EXPECTED_ROUTE_ROSTER
    ):
        raise ValueError("runtime-routing experiment configuration is invalid")
    if payload.get("router") != {
        "kind": "runtime_command_residual",
        **ROUTING_SPEC.to_dict(),
        "negative_route": "frozen_base_plus_linear_negative_x_residual",
        "positive_route": "frozen_base_plus_linear_positive_x_residual",
        "base_route": "frozen_base_only",
        "active_non_x_fallback": "frozen_base_only",
        "runtime_inputs": ["candidate_droid_action"],
        "forbidden_runtime_inputs": ["scripted_phase", "context_index", "seed"],
        "residual_output_zero_initialized": True,
        "trainable_base": False,
    }:
        raise ValueError("runtime-routing specification changed")
    return payload


def _functional_routes(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if actions.ndim != 3 or actions.shape[-1] != 7:
        raise ValueError("rollout actions must have shape [horizon, batch, 7]")
    return ROUTING_SPEC.classify(actions.transpose(0, 1))


def _route_name(route: int, active: bool) -> str:
    if route == NEGATIVE_X_COMMAND_ROUTE:
        return "negative_x"
    if route == POSITIVE_X_COMMAND_ROUTE:
        return "positive_x"
    return "base_active_non_x" if active else "base_neutral"


def route_roster(rollouts: Sequence[RecordedRollout]) -> dict[str, int]:
    actions = rollout_action_tensor(rollouts)
    routes, active = _functional_routes(actions)
    counts = Counter(
        _route_name(int(route), bool(is_active))
        for route, is_active in zip(routes, active)
    )
    return {
        "negative_x": counts["negative_x"],
        "positive_x": counts["positive_x"],
        "base_neutral": counts["base_neutral"],
        "base_active_non_x": counts["base_active_non_x"],
        "total": len(rollouts),
    }


def _validated_selection(recordings: Sequence[Path]) -> RolloutTrainingSelection:
    validated = validated_training_recordings(recordings)
    if tuple(recording.name for recording in validated) != TRAINING_RECORDINGS:
        raise ValueError("training inputs do not match the frozen TRAIN roster")
    selection = RolloutTrainingSelection.load(
        tuple(recording.path for recording in validated),
        camera="wrist",
        bounds=TRAINING_BOUNDS,
        window=EXPERIMENT_WINDOW,
    )
    if route_roster(selection.rollouts) != EXPECTED_ROUTE_ROSTER:
        raise ValueError("TRAIN command-route roster changed")
    return selection


def preflight(
    recordings: Sequence[Path],
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"runtime-routing preflight already exists: {output}")
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    selection = _validated_selection(recordings)
    report = {
        "schema": "quantis.jepa_wm_runtime_command_routing_preflight.v1",
        "status": "passed",
        "scope": "offline routing experiment only; no live JEPA action",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "control_artifact": control_identity.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "router": ROUTING_SPEC.to_dict(),
        "route_roster": route_roster(selection.rollouts),
        "fresh_canary": {"recording": FRESH_CANARY, "seed": FRESH_CANARY_SEED},
        "seed_72600_excluded": True,
        "canonical_accessed": False,
    }
    write_json_atomic(output, report)
    return report


def _training_config(experiment: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "treatment": "R",
        "spec": TREATMENT_SPEC.to_dict(),
        "control_artifact_fingerprint": CONTROL_ARTIFACT_FINGERPRINT,
        "training": experiment["training"],
    }


def train_router(
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    recordings: Sequence[Path],
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("runtime-routing training requires CUDA")
    expected_output = EVALUATION_OUTPUT_ROOT / "runtime_command_router.pth"
    if output.resolve() != expected_output:
        raise ValueError(f"runtime-routing training output must be {expected_output}")
    if output.exists() or training_report_path(output).exists():
        raise ValueError(f"runtime-routing output already exists: {output}")
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    selection = _validated_selection(recordings)
    rollouts = selection.rollouts
    training = experiment["training"]
    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)

    load_started = monotonic()
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=control_adapter,
    )
    encoder = install_action_conditioning(model, TREATMENT_SPEC)
    load_seconds = monotonic() - load_started
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = action_conditioning_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)
    frozen_base = tuple(
        value.detach().clone() for value in encoder.base.state_dict().values()
    )

    encoding_started = monotonic()
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=int(training["encoding_batch_size"]),
    )
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=int(training["encoding_batch_size"]),
    )
    encoding_seconds = monotonic() - encoding_started
    actions = rollout_action_tensor(rollouts)
    routes, _ = _functional_routes(actions)
    goal_actions = torch.tensor(
        [rollout.goal_action.values for rollout in rollouts],
        dtype=actions.dtype,
    )
    mismatched_candidates = mismatched_negative_candidates(rollouts)
    candidate_config = CandidateMiningConfig.from_dict(
        training["candidate_mining"]
    )
    terms = {
        name: ContrastiveTermConfig(**training["objective"][name])
        for name in (
            "zero_negative",
            "mismatched_negative",
            "candidate_negative",
            "signed_x_zero_negative",
            "signed_x_opposed_negative",
        )
    }
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sampler = AlternatingCommandRouteSampler(routes, seed=seed)
    mismatch_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    candidate_generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    model.eval()
    training_started = monotonic()
    for _ in range(int(training["steps"])):
        indices = torch.tensor((sampler.next_index(),), dtype=torch.long)
        context = contexts[indices].to(device)
        target = targets[indices].to(device)
        action_batch = actions[:, indices].to(device)
        negative_indices = _sample_mismatched_indices(
            mismatched_candidates,
            indices,
            mismatch_generator,
        )
        mismatched = actions[:, negative_indices].to(device)
        local_candidates = sample_local_candidates(
            action_batch,
            config=candidate_config,
            generator=candidate_generator,
            goal_actions=goal_actions[indices].to(device),
        )
        mined = mine_lowest_energy_candidates(
            model,
            context,
            target,
            local_candidates,
            scoring_batch_size=candidate_config.scoring_batch_size,
        )
        energies = score_recorded_against_mismatched(
            model,
            context,
            target,
            action_batch,
            mismatched,
        )
        candidate_energy = score_actions(model, context, target, mined)
        x_zero, x_opposed = signed_x_negatives(action_batch)
        x_zero_energy = score_actions(model, context, target, x_zero)
        x_opposed_energy = score_actions(model, context, target, x_opposed)
        loss = (
            energies.recorded.mean()
            + terms["zero_negative"].loss(energies.recorded, energies.zero)
            + terms["mismatched_negative"].loss(
                energies.recorded,
                energies.mismatched_negative,
            )
            + terms["candidate_negative"].loss(
                energies.recorded,
                candidate_energy,
            )
            + signed_x_margin_loss(
                energies.recorded,
                x_zero_energy,
                action_batch,
                weight=terms["signed_x_zero_negative"].weight,
                margin=terms["signed_x_zero_negative"].margin,
                minimum_activity=float(
                    training["objective"]["signed_x_activity_threshold"]
                ),
            )
            + signed_x_margin_loss(
                energies.recorded,
                x_opposed_energy,
                action_batch,
                weight=terms["signed_x_opposed_negative"].weight,
                margin=terms["signed_x_opposed_negative"].margin,
                minimum_activity=float(
                    training["objective"]["signed_x_activity_threshold"]
                ),
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    torch.cuda.synchronize(device)
    training_seconds = monotonic() - training_started
    if any(
        not torch.equal(before, after)
        for before, after in zip(frozen_base, encoder.base.state_dict().values())
    ):
        raise ValueError("frozen control action map changed during router training")

    metadata = TrainingArtifactMetadata(
        MODEL_ID,
        os.environ.get("JEPA_WM_REVISION", "unknown"),
        "wrist",
        TRAINING_RECORDINGS,
        int(training["steps"]),
    )
    config_payload = _training_config(experiment)
    config_fingerprint = training_configuration_fingerprint(config_payload)
    contract = ActionConditioningContract(
        ACTION_CONDITIONING_SCHEMA,
        metadata,
        selection.fingerprint,
        config_fingerprint,
        FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        TREATMENT_SPEC,
    )
    save_action_conditioning(model, output, contract)
    report = {
        "schema": "quantis.jepa_wm_runtime_command_routing_training.v1",
        "status": "trained",
        "treatment": "R",
        "scope": "offline action routing only; no live JEPA action",
        "artifact": str(output.resolve()),
        "artifact_fingerprint": artifact_fingerprint(output),
        "metadata": metadata.to_dict(),
        "contract": contract.to_dict(),
        "config": config_payload,
        "training_config_fingerprint": config_fingerprint,
        **selection.to_dict(),
        "training_selection_fingerprint": selection.fingerprint,
        "route_roster": route_roster(rollouts),
        "sampling": sampler.to_dict(),
        "control_artifact": control_identity.to_dict(),
        "base_model": base_identity,
        "base_map_unchanged": True,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": float(np.min(losses)),
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "training_seconds": round(training_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
    }
    report["report"] = str(training_report_path(output).resolve())
    write_json_atomic(training_report_path(output), report)
    return report


def smoke(
    source: Path,
    checkpoint: Path,
    control_adapter: Path,
    recording: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    """Exercise both learned routes on real TRAIN latents without saving."""

    if not torch.cuda.is_available():
        raise RuntimeError("runtime-routing smoke requires CUDA")
    experiment = _load_experiment_config(experiment_config)
    _authenticate_base_model(experiment, source, checkpoint)
    validated = validated_training_recordings((recording,))
    if validated[0].name != TRAINING_RECORDINGS[0] or validated[0].seed != 2600:
        raise ValueError("runtime-routing smoke requires frozen TRAIN recording 00")
    control_identity = ArtifactIdentity.from_artifact(control_adapter)
    if control_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
        raise ValueError("frozen control action map identity changed")
    all_rollouts = load_rollouts(
        recording,
        camera="wrist",
        bounds=TRAINING_BOUNDS,
    )
    by_context = {rollout.context[0].index: rollout for rollout in all_rollouts}
    rollouts = (by_context[120], by_context[180])
    actions = rollout_action_tensor(rollouts)
    routes, _ = _functional_routes(actions)
    if tuple(int(route) for route in routes) != (
        NEGATIVE_X_COMMAND_ROUTE,
        POSITIVE_X_COMMAND_ROUTE,
    ):
        raise ValueError("runtime-routing smoke contexts changed command routes")

    device = torch.device("cuda", torch.cuda.current_device())
    seed = int(experiment["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = load_headless_model(
        source,
        checkpoint,
        device=device,
        adapter=control_adapter,
    )
    encoder = install_action_conditioning(model, TREATMENT_SPEC)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = action_conditioning_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)
    base_before = tuple(
        value.detach().clone() for value in encoder.base.state_dict().values()
    )
    contexts = encode_clips(
        model,
        [rollout.context_paths for rollout in rollouts],
        batch_size=2,
    ).to(device)
    targets = encode_clips(
        model,
        [rollout.target_clip for rollout in rollouts],
        batch_size=2,
    ).to(device)
    action_batch = actions.to(device)
    recorded = score_actions(model, contexts, targets, action_batch)
    zero = score_actions(model, contexts, targets, torch.zeros_like(action_batch))
    loss = recorded.mean() + ContrastiveTermConfig().loss(recorded, zero)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(experiment["training"]["learning_rate"]),
        weight_decay=float(experiment["training"]["weight_decay"]),
    )
    before = tuple(parameter.detach().clone() for parameter in trainable)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if any(
        parameter.grad is None or not torch.isfinite(parameter.grad).all()
        for parameter in trainable
    ):
        raise ValueError("runtime-routing smoke gradient is invalid")
    optimizer.step()
    changed = tuple(
        not torch.equal(previous, parameter.detach())
        for previous, parameter in zip(before, trainable)
    )
    if changed != (True, True):
        raise ValueError("runtime-routing smoke did not update both residuals")
    if any(
        not torch.equal(previous, current)
        for previous, current in zip(base_before, encoder.base.state_dict().values())
    ):
        raise ValueError("runtime-routing smoke changed the frozen base map")
    return {
        "status": "smoke_passed",
        "recording": recording.name,
        "contexts": [rollout.context[0].index for rollout in rollouts],
        "routes": [COMMAND_ROUTE_NAMES[int(route)] for route in routes],
        "loss": float(loss.detach().cpu()),
        "residuals_updated": list(changed),
        "base_map_unchanged": True,
        "artifacts_written": 0,
        "held_out_accessed": False,
    }


def _expected_evaluation_seed(recording: str) -> int | None:
    roster = {
        FRESH_CANARY: FRESH_CANARY_SEED,
        CANONICAL_HELD_OUT[0]: 12600,
        CANONICAL_HELD_OUT[1]: 12601,
    }
    return roster.get(recording)


def _authorize_evaluation(
    recording: str,
    treatment: str,
    artifact: ArtifactIdentity,
    canary_summary: Path | None,
    experiment: dict[str, Any],
) -> None:
    if recording == FRESH_CANARY:
        if canary_summary is not None:
            raise ValueError("fresh canary evaluation cannot consume its own summary")
        return
    if recording not in CANONICAL_HELD_OUT or treatment != "R":
        raise ValueError("canonical routing evaluation is not authorized")
    if canary_summary is None:
        raise ValueError("canonical routing evaluation requires a passing canary summary")
    expected_summary = EVALUATION_OUTPUT_ROOT / "canary-summary.json"
    if canary_summary.resolve() != expected_summary:
        raise ValueError("canonical routing evaluation requires the frozen canary summary")
    payload = json.loads(canary_summary.resolve().read_text())
    selected = payload.get("selected_artifact")
    source_reports = payload.get("source_reports")
    if (
        payload.get("schema")
        != "quantis.jepa_wm_runtime_command_routing_canary_summary.v1"
        or payload.get("experiment_config_fingerprint")
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
        or payload.get("selected_treatment") != "R"
        or not payload.get("canonical_authorized_offline")
        or not isinstance(selected, dict)
        or selected != artifact.to_dict()
        or not isinstance(source_reports, list)
        or len(source_reports) != 2
    ):
        raise ValueError("canonical routing evaluation lacks matching canary authority")
    reports_by_treatment = {}
    expected_report_paths = {
        _expected_evaluation_output(FRESH_CANARY, item) for item in ("A", "R")
    }
    for identity_payload in source_reports:
        identity = ArtifactIdentity.from_dict(identity_payload)
        if identity.path not in expected_report_paths:
            raise ValueError("canonical routing canary report path changed")
        actual = ArtifactIdentity.from_artifact(identity.path)
        if actual != identity:
            raise ValueError("canonical routing canary report identity changed")
        report = json.loads(identity.path.read_text())
        reports_by_treatment[report.get("treatment")] = report
    if set(reports_by_treatment) != {"A", "R"}:
        raise ValueError("canonical routing canary reports are incomplete")
    for report_treatment, report in reports_by_treatment.items():
        _validate_evaluation_report(
            report,
            treatment=report_treatment,
            experiment=experiment,
        )
    router_report = reports_by_treatment["R"]
    router_identity = ArtifactIdentity(
        Path(router_report["artifact"]),
        router_report["artifact_fingerprint"],
    )
    if (
        not router_report["experimental_gate"]["passed"]
        or selected != router_identity.to_dict()
        or payload.get("status") != "complete"
        or payload.get("outcome") != "runtime_router_candidate"
        or payload.get("live_action_authorized") is not False
    ):
        raise ValueError("canonical routing canary did not select a passing router")


def _experimental_gate(
    energies: EvaluationEnergies,
    rollouts: Sequence[RecordedRollout],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    return _gate_for_context_indices(
        energies,
        tuple(rollout.context[0].index for rollout in rollouts),
    )


def _gate_for_context_indices(
    energies: EvaluationEnergies,
    context_indices: Sequence[int],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], bool]:
    aggregate = _metrics(list(range(len(context_indices))), energies)
    control_gate = ActionControlGate().evaluate(
        mean_improvement_over_zero=aggregate["mean_improvement_over_zero"],
        recorded_action_win_rate=aggregate["recorded_action_win_rate"],
    )
    by_segment = {}
    for segment in ContactInsertionSegment:
        indices = [
            index
            for index, context in enumerate(context_indices)
            if _segment_for_context(context) == segment.value
        ]
        if indices:
            by_segment[segment.value] = _metrics(indices, energies)
    retained_indices = [
        index
        for index, context in enumerate(context_indices)
        if _regime_for_context(context) == 0
    ]
    post_indices = [
        index
        for index, context in enumerate(context_indices)
        if _regime_for_context(context) == 1
    ]
    retained = _metrics(retained_indices, energies)
    post = _metrics(post_indices, energies)
    passed = (
        control_gate.passed
        and aggregate["recorded_action_win_rate"] >= 0.90
        and retained["recorded_action_win_rate"] >= 0.85
        and post["recorded_action_win_rate"] >= 0.95
        and all(
            segment["mean_improvement_over_zero"] > 0.0
            for segment in by_segment.values()
        )
        and by_segment[ContactInsertionSegment.RETREAT.value][
            "signed_order_fraction"
        ]
        >= 0.75
        and by_segment[ContactInsertionSegment.ALIGN.value][
            "signed_order_fraction"
        ]
        >= 0.75
    )
    aggregate["control_gate"] = control_gate.to_dict()
    return aggregate, retained, post, by_segment, passed


def evaluate(
    source: Path,
    checkpoint: Path,
    recording: Path,
    artifact: Path,
    output: Path,
    *,
    treatment: str,
    expected_seed: int,
    experiment_config: Path,
    canary_summary: Path | None = None,
) -> dict[str, Any]:
    if treatment not in ("A", "R"):
        raise ValueError("runtime-routing evaluation treatment must be A or R")
    if not torch.cuda.is_available():
        raise RuntimeError("runtime-routing evaluation requires CUDA")
    expected_output = _expected_evaluation_output(recording.name, treatment)
    if output.resolve() != expected_output:
        raise ValueError(
            f"runtime-routing evaluation output must be {expected_output}"
        )
    if output.exists():
        raise ValueError(f"runtime-routing evaluation already exists: {output}")
    experiment = _load_experiment_config(experiment_config)
    base_identity = _authenticate_base_model(experiment, source, checkpoint)
    if _expected_evaluation_seed(recording.name) != expected_seed:
        raise ValueError("evaluation input is outside the frozen held-out roster")
    artifact_identity = ArtifactIdentity.from_artifact(artifact)
    _authorize_evaluation(
        recording.name,
        treatment,
        artifact_identity,
        canary_summary,
        experiment,
    )
    ContactInsertionEvidence.from_recording(
        recording,
        expected_split="held_out",
        expected_seed=expected_seed,
    )
    rollouts = EXPERIMENT_WINDOW.select(
        load_rollouts(recording, camera="wrist", bounds=TRAINING_BOUNDS)
    )
    if len(rollouts) != EXPERIMENT_WINDOW.count:
        raise ValueError("evaluation recording lacks the frozen insertion window")
    input_fingerprint = _selected_input_fingerprint(recording, rollouts)

    device_index = torch.cuda.current_device()
    device = torch.device("cuda", device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    load_started = monotonic()
    if treatment == "A":
        if artifact_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
            raise ValueError("control artifact identity changed")
        model = load_headless_model(
            source,
            checkpoint,
            device=device,
            adapter=artifact,
        )
        artifact_spec = {"kind": "frozen_control_global_linear"}
    else:
        model = load_headless_model(source, checkpoint, device=device)
        loaded = LoadedActionConditioning.load(
            artifact,
            expected_identity=artifact_identity,
        )
        if (
            loaded.contract.experiment_config_fingerprint
            != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
            or loaded.contract.spec != TREATMENT_SPEC
        ):
            raise ValueError("router artifact disagrees with the frozen treatment")
        loaded.apply(
            model,
            expected_source_revision=os.environ.get("JEPA_WM_REVISION", "unknown"),
        )
        artifact_identity = loaded.identity
        artifact_spec = loaded.contract.spec.to_dict()
        training_report = load_training_report(artifact)
        if (
            training_report.get("artifact_fingerprint")
            != artifact_identity.fingerprint
            or training_report.get("treatment") != "R"
            or not training_report.get("base_map_unchanged")
        ):
            raise ValueError("router training report disagrees with its artifact")
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
    actions = rollout_action_tensor(rollouts)
    evaluation_started = monotonic()
    energies = _score_evaluation_batches(
        model,
        contexts,
        targets,
        actions,
        None,
        device=device,
    )
    torch.cuda.synchronize(device)
    evaluation_seconds = monotonic() - evaluation_started
    aggregate, retained, post, by_segment, passed = _experimental_gate(
        energies,
        rollouts,
    )
    routes, active = _functional_routes(actions)
    results = []
    for index, rollout in enumerate(rollouts):
        recorded = float(energies.recorded[index])
        zero = float(energies.zero[index])
        x_zero = float(energies.x_zero[index])
        x_opposed = float(energies.x_opposed[index])
        route = int(routes[index])
        results.append(
            {
                "context_index": rollout.context[0].index,
                "target_index": rollout.target.index,
                "segment": _segment_for_context(rollout.context[0].index),
                "route": _route_name(route, bool(active[index])),
                "functional_route": COMMAND_ROUTE_NAMES[route],
                "recorded_energy": recorded,
                "zero_energy": zero,
                "x_zero_energy": x_zero,
                "x_opposed_energy": x_opposed,
                "improvement_over_zero": zero - recorded,
                "recorded_action_wins": recorded < zero,
                "signed_order_passed": recorded < x_zero < x_opposed,
            }
        )
    report = {
        "schema": "quantis.jepa_wm_runtime_command_routing_evaluation.v1",
        "status": "evaluated",
        "scope": "offline insertion energy only; no live JEPA action",
        "treatment": treatment,
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "base_model": base_identity,
        "artifact": str(artifact_identity.path),
        "artifact_fingerprint": artifact_identity.fingerprint,
        "artifact_spec": artifact_spec,
        "recording": recording.name,
        "recording_path": str(recording.resolve()),
        "selected_input_fingerprint": input_fingerprint,
        "expected_seed": expected_seed,
        "window": EXPERIMENT_WINDOW.to_dict(),
        "route_roster": route_roster(rollouts),
        "aggregate": aggregate,
        "retained": retained,
        "post": post,
        "by_segment": by_segment,
        "experimental_gate": {
            "passed": passed,
            "minimum_overall_win_rate": 0.90,
            "minimum_retained_win_rate": 0.85,
            "minimum_post_win_rate": 0.95,
            "minimum_retreat_signed_order_fraction": 0.75,
            "minimum_alignment_signed_order_fraction": 0.75,
            "requires_positive_mean_each_segment": True,
        },
        "load_seconds": round(load_seconds, 3),
        "encoding_seconds": round(encoding_seconds, 3),
        "evaluation_seconds": round(evaluation_seconds, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(device_index) / 2**30,
            3,
        ),
        "results": results,
    }
    write_json_atomic(output, report)
    return report


def _validate_evaluation_report(
    payload: dict[str, Any],
    *,
    treatment: str,
    experiment: dict[str, Any],
) -> None:
    artifact_path = Path(str(payload.get("artifact", "")))
    recording_path = Path(str(payload.get("recording_path", "")))
    base = payload.get("base_model")
    if (
        payload.get("schema")
        != "quantis.jepa_wm_runtime_command_routing_evaluation.v1"
        or payload.get("status") != "evaluated"
        or payload.get("treatment") != treatment
        or payload.get("experiment_config_fingerprint")
        != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
        or payload.get("recording") != FRESH_CANARY
        or payload.get("expected_seed") != FRESH_CANARY_SEED
        or payload.get("window") != EXPERIMENT_WINDOW.to_dict()
        or not isinstance(base, dict)
    ):
        raise ValueError("canary evaluation report identity is invalid")
    authenticated_base = _authenticate_base_model(
        experiment,
        Path(str(base.get("source", ""))),
        Path(str(base.get("checkpoint", ""))),
    )
    if base != authenticated_base:
        raise ValueError("canary evaluation base identity is invalid")
    artifact_identity = ArtifactIdentity.from_artifact(artifact_path)
    if artifact_identity.fingerprint != payload.get("artifact_fingerprint"):
        raise ValueError("canary evaluation artifact identity is invalid")
    if treatment == "A":
        if artifact_identity.fingerprint != CONTROL_ARTIFACT_FINGERPRINT:
            raise ValueError("canary control artifact identity is invalid")
    else:
        loaded = LoadedActionConditioning.load(
            artifact_path,
            expected_identity=artifact_identity,
        )
        if (
            loaded.contract.experiment_config_fingerprint
            != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
            or loaded.contract.spec != TREATMENT_SPEC
            or loaded.contract.metadata.source_revision
            != experiment["base_model"]["source_revision"]
        ):
            raise ValueError("canary router artifact contract is invalid")
        training_report = load_training_report(artifact_path)
        if (
            training_report.get("artifact_fingerprint")
            != artifact_identity.fingerprint
            or training_report.get("treatment") != "R"
            or not training_report.get("base_map_unchanged")
            or training_report.get("base_model") != authenticated_base
        ):
            raise ValueError("canary router training provenance is invalid")

    ContactInsertionEvidence.from_recording(
        recording_path,
        expected_split="held_out",
        expected_seed=FRESH_CANARY_SEED,
    )
    rollouts = EXPERIMENT_WINDOW.select(
        load_rollouts(recording_path, camera="wrist", bounds=TRAINING_BOUNDS)
    )
    if (
        recording_path.name != FRESH_CANARY
        or payload.get("selected_input_fingerprint")
        != _selected_input_fingerprint(recording_path, rollouts)
        or payload.get("route_roster") != route_roster(rollouts)
    ):
        raise ValueError("canary evaluation input identity is invalid")

    results = payload.get("results")
    if not isinstance(results, list) or len(results) != EXPERIMENT_WINDOW.count:
        raise ValueError("canary evaluation results are incomplete")
    contexts = tuple(result.get("context_index") for result in results)
    if contexts != EXPERIMENT_WINDOW.context_indices:
        raise ValueError("canary evaluation contexts are invalid")
    actions = rollout_action_tensor(rollouts)
    routes, active = _functional_routes(actions)
    energy_fields = (
        "recorded_energy",
        "zero_energy",
        "x_zero_energy",
        "x_opposed_energy",
    )
    if any(
        not isinstance(result, dict)
        or result.get("target_index") != context + 3
        or result.get("segment") != _segment_for_context(context)
        or result.get("functional_route") != COMMAND_ROUTE_NAMES[int(routes[index])]
        or result.get("route")
        != _route_name(int(routes[index]), bool(active[index]))
        or any(
            isinstance(result.get(field), bool)
            or not isinstance(result.get(field), (int, float))
            or not np.isfinite(float(result[field]))
            for field in energy_fields
        )
        for index, (result, context) in enumerate(zip(results, contexts))
    ):
        raise ValueError("canary evaluation result contract is invalid")
    energies = EvaluationEnergies(
        **{
            name: torch.tensor(
                [float(result[field]) for result in results],
                dtype=torch.float32,
            )
            for name, field in (
                ("recorded", "recorded_energy"),
                ("zero", "zero_energy"),
                ("x_zero", "x_zero_energy"),
                ("x_opposed", "x_opposed_energy"),
            )
        }
    )
    aggregate, retained, post, by_segment, passed = _gate_for_context_indices(
        energies,
        contexts,
    )
    for index, result in enumerate(results):
        recorded = float(energies.recorded[index])
        zero = float(energies.zero[index])
        x_zero = float(energies.x_zero[index])
        x_opposed = float(energies.x_opposed[index])
        if (
            result.get("improvement_over_zero") != zero - recorded
            or result.get("recorded_action_wins") is not (recorded < zero)
            or result.get("signed_order_passed")
            is not (recorded < x_zero < x_opposed)
        ):
            raise ValueError("canary evaluation result claims are invalid")
    expected_gate = {
        "passed": passed,
        "minimum_overall_win_rate": 0.90,
        "minimum_retained_win_rate": 0.85,
        "minimum_post_win_rate": 0.95,
        "minimum_retreat_signed_order_fraction": 0.75,
        "minimum_alignment_signed_order_fraction": 0.75,
        "requires_positive_mean_each_segment": True,
    }
    if (
        payload.get("aggregate") != aggregate
        or payload.get("retained") != retained
        or payload.get("post") != post
        or payload.get("by_segment") != by_segment
        or payload.get("experimental_gate") != expected_gate
    ):
        raise ValueError("canary evaluation gate cannot be reproduced")


def summarize_canary(
    reports: Sequence[Path],
    output: Path,
    *,
    experiment_config: Path,
) -> dict[str, Any]:
    expected_output = EVALUATION_OUTPUT_ROOT / "canary-summary.json"
    if output.resolve() != expected_output:
        raise ValueError(f"runtime-routing canary summary must be {expected_output}")
    if output.exists():
        raise ValueError(f"runtime-routing canary summary already exists: {output}")
    experiment = _load_experiment_config(experiment_config)
    expected_reports = {
        _expected_evaluation_output(FRESH_CANARY, treatment)
        for treatment in ("A", "R")
    }
    resolved_reports = {report.resolve() for report in reports}
    if resolved_reports != expected_reports:
        raise ValueError("canary reports do not use the frozen evaluation paths")
    payloads = [json.loads(report.read_text()) for report in sorted(resolved_reports)]
    if (
        len(payloads) != 2
        or {payload.get("treatment") for payload in payloads} != {"A", "R"}
        or any(payload.get("recording") != FRESH_CANARY for payload in payloads)
        or any(
            payload.get("experiment_config_fingerprint")
            != FROZEN_EXPERIMENT_CONFIG_FINGERPRINT
            for payload in payloads
        )
    ):
        raise ValueError("canary reports do not match the frozen routing experiment")
    by_treatment = {payload["treatment"]: payload for payload in payloads}
    for treatment, payload in by_treatment.items():
        _validate_evaluation_report(
            payload,
            treatment=treatment,
            experiment=experiment,
        )
    router_passed = bool(by_treatment["R"]["experimental_gate"]["passed"])
    summary = {
        "schema": "quantis.jepa_wm_runtime_command_routing_canary_summary.v1",
        "status": "complete",
        "experiment_config_fingerprint": FROZEN_EXPERIMENT_CONFIG_FINGERPRINT,
        "outcome": "runtime_router_candidate" if router_passed else "router_failed",
        "selected_treatment": "R" if router_passed else None,
        "selected_artifact": (
            {
                "path": by_treatment["R"]["artifact"],
                "fingerprint": by_treatment["R"]["artifact_fingerprint"],
            }
            if router_passed
            else None
        ),
        "source_reports": [
            ArtifactIdentity.from_artifact(report).to_dict()
            for report in sorted(resolved_reports)
        ],
        "control": {
            "aggregate": by_treatment["A"]["aggregate"],
            "retained": by_treatment["A"]["retained"],
            "post": by_treatment["A"]["post"],
            "experimental_gate": by_treatment["A"]["experimental_gate"],
        },
        "router": {
            "artifact_fingerprint": by_treatment["R"]["artifact_fingerprint"],
            "aggregate": by_treatment["R"]["aggregate"],
            "retained": by_treatment["R"]["retained"],
            "post": by_treatment["R"]["post"],
            "experimental_gate": by_treatment["R"]["experimental_gate"],
        },
        "canonical_authorized_offline": router_passed,
        "live_action_authorized": False,
    }
    write_json_atomic(output, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--recording", type=Path, action="append", required=True)
    preflight_parser.add_argument("--source", type=Path, required=True)
    preflight_parser.add_argument("--checkpoint", type=Path, required=True)
    preflight_parser.add_argument("--control-adapter", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument("--experiment-config", type=Path, required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--source", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--control-adapter", type=Path, required=True)
    train_parser.add_argument("--recording", type=Path, action="append", required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--experiment-config", type=Path, required=True)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--source", type=Path, required=True)
    smoke_parser.add_argument("--checkpoint", type=Path, required=True)
    smoke_parser.add_argument("--control-adapter", type=Path, required=True)
    smoke_parser.add_argument("--recording", type=Path, required=True)
    smoke_parser.add_argument("--experiment-config", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--source", type=Path, required=True)
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--recording", type=Path, required=True)
    evaluate_parser.add_argument("--artifact", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--treatment", choices=("A", "R"), required=True)
    evaluate_parser.add_argument("--expected-seed", type=int, required=True)
    evaluate_parser.add_argument("--experiment-config", type=Path, required=True)
    evaluate_parser.add_argument("--canary-summary", type=Path)
    summarize_parser = subparsers.add_parser("summarize-canary")
    summarize_parser.add_argument("--report", type=Path, action="append", required=True)
    summarize_parser.add_argument("--output", type=Path, required=True)
    summarize_parser.add_argument("--experiment-config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "preflight":
        result = preflight(
            arguments.recording,
            arguments.source,
            arguments.checkpoint,
            arguments.control_adapter,
            arguments.output,
            experiment_config=arguments.experiment_config,
        )
    elif arguments.command == "train":
        result = train_router(
            arguments.source,
            arguments.checkpoint,
            arguments.control_adapter,
            arguments.recording,
            arguments.output,
            experiment_config=arguments.experiment_config,
        )
    elif arguments.command == "smoke":
        result = smoke(
            arguments.source,
            arguments.checkpoint,
            arguments.control_adapter,
            arguments.recording,
            experiment_config=arguments.experiment_config,
        )
    elif arguments.command == "evaluate":
        result = evaluate(
            arguments.source,
            arguments.checkpoint,
            arguments.recording,
            arguments.artifact,
            arguments.output,
            treatment=arguments.treatment,
            expected_seed=arguments.expected_seed,
            experiment_config=arguments.experiment_config,
            canary_summary=arguments.canary_summary,
        )
    else:
        result = summarize_canary(
            arguments.report,
            arguments.output,
            experiment_config=arguments.experiment_config,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
