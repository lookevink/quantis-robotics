"""Adjudicate one immutable physical-residual TRAIN report without rescoring."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jepa_wm.persistence import write_json_atomic
from jepa_wm.readiness import ResidualTrainGate
from jepa_wm.training_artifact import ArtifactIdentity


ADJUDICATION_SCHEMA = "quantis.jepa_wm_physical_state_residual_gate_adjudication.v1"
FROZEN_CONFIG_FINGERPRINT = (
    "92986c48b5c05bbf0e93c5d7bc265d904eaa23daa56bd4ce13611d4c3a1a437e"
)
OUTPUT_PATH = Path(
    "/home/ubuntu/docker/jepa-wm/checkpoints/"
    "quantis_physical_state_residual_v1/gate-adjudication-v1.json"
)


def _load_config(path: Path) -> dict[str, Any]:
    encoded = path.resolve().read_bytes()
    if sha256(encoded).hexdigest() != FROZEN_CONFIG_FINGERPRINT:
        raise ValueError("physical residual adjudication configuration changed")
    payload = json.loads(encoded)
    source = payload.get("source_evaluation", {})
    gate = payload.get("gate", {})
    execution = payload.get("execution", {})
    if (
        payload.get("schema") != ADJUDICATION_SCHEMA
        or source.get("fingerprint")
        != "a5527c50eeb4a3223b74779584f7dad1f6edc29121534a778abf9147bf4d6bdd"
        or gate.get("maximum_residual_ratio") != 0.15
        or gate.get("residual_ratio_absolute_tolerance") != 1e-6
        or gate.get("require_positive_mean_each_segment") is not True
        or gate.get("require_exact_base_in_semantic_holds") is not True
        or gate.get("require_final_router_gate") is not True
        or execution
        != {
            "adjudications": 1,
            "load_model": False,
            "load_recordings": False,
            "rescore": False,
            "train": False,
            "access_held_out": False,
            "access_canonical": False,
            "run_isaac": False,
            "issue_live_action": False,
        }
    ):
        raise ValueError("physical residual adjudication contract is invalid")
    return payload


def _expected_original_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    signed = float(gate["minimum_signed_order_fraction"])
    return {
        "passed": False,
        "minimum_overall_win_rate": float(gate["minimum_overall_win_rate"]),
        "minimum_retained_win_rate": float(gate["minimum_retained_win_rate"]),
        "minimum_post_win_rate": float(gate["minimum_post_win_rate"]),
        "minimum_signed_order_fraction": {
            segment: signed for segment in gate["required_signed_segments"]
        },
        "requires_positive_mean_each_segment": True,
        "maximum_applied_residual_to_base_embedding_ratio": float(
            gate["maximum_residual_ratio"]
        ),
        "requires_exact_base_in_semantic_holds": True,
    }


def adjudicate_report(
    report: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-evaluate only the stored terminal predicates under numeric tolerance."""

    if (
        report.get("schema")
        != "quantis.jepa_wm_physical_state_residual_train_evaluation.v1"
        or report.get("status") != "evaluated"
        or report.get("outcome") != "physical_state_residual_train_failed"
        or report.get("experimental_gate") != _expected_original_gate(gate)
        or report.get("held_out_accessed") is not False
        or report.get("canonical_accessed") is not False
        or report.get("live_action_authorized") is not False
    ):
        raise ValueError("source physical residual evaluation is not adjudicable")
    decision = ResidualTrainGate(
        minimum_overall_win_rate=float(gate["minimum_overall_win_rate"]),
        minimum_retained_win_rate=float(gate["minimum_retained_win_rate"]),
        minimum_post_win_rate=float(gate["minimum_post_win_rate"]),
        minimum_signed_order_fraction=float(gate["minimum_signed_order_fraction"]),
        maximum_residual_ratio=float(gate["maximum_residual_ratio"]),
        residual_ratio_tolerance=float(gate["residual_ratio_absolute_tolerance"]),
        required_signed_segments=tuple(gate["required_signed_segments"]),
    ).evaluate(
        aggregate=report["aggregate"],
        retained=report["retained"],
        post=report["post"],
        by_segment=report["by_segment"],
        maximum_residual_ratio=float(
            report["residual_ratios"]["maximum_applied_ratio"]
        ),
    )
    reasons = [reason.value for reason in decision.reasons]
    if report["final_router"].get("gate_passed") is not True:
        reasons.append("final_router_gate")
    if report["residual_ratios"].get("semantic_holds_exact_base") is not True:
        reasons.append("semantic_hold_base_passthrough")
    passed = not reasons
    return {
        "passed": passed,
        "reasons": reasons,
        "original_outcome": report["outcome"],
        "outcome": (
            "physical_state_residual_train_candidate"
            if passed
            else "physical_state_residual_train_failed"
        ),
        "residual_ratio": decision.to_dict(),
        "final_router_gate_passed": report["final_router"]["gate_passed"],
        "semantic_holds_exact_base": report["residual_ratios"][
            "semantic_holds_exact_base"
        ],
        "model_loaded": False,
        "recordings_loaded": False,
        "rescored": False,
        "trained": False,
    }


def adjudicate(
    evaluation: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = _load_config(config_path)
    source = config["source_evaluation"]
    if evaluation.resolve() != Path(source["path"]) or output.resolve() != OUTPUT_PATH:
        raise ValueError("physical residual adjudication paths changed")
    if output.exists():
        raise ValueError("physical residual adjudication already exists")
    evaluation_identity = ArtifactIdentity.from_artifact(evaluation)
    if evaluation_identity.fingerprint != source["fingerprint"]:
        raise ValueError("source physical residual evaluation identity changed")
    report = json.loads(evaluation.read_text())
    if (
        report.get("experiment_config_fingerprint")
        != source["experiment_config_fingerprint"]
        or report.get("artifact", {}).get("fingerprint")
        != source["artifact_fingerprint"]
        or report.get("training_report", {}).get("fingerprint")
        != source["training_report_fingerprint"]
        or report.get("training_selection_fingerprint")
        != source["training_selection_fingerprint"]
        or report.get("selected_input_fingerprint")
        != source["selected_input_fingerprint"]
    ):
        raise ValueError("source physical residual provenance changed")
    decision = adjudicate_report(report, config["gate"])
    result = {
        "schema": ADJUDICATION_SCHEMA,
        "status": "adjudicated",
        "scope": "immutable TRAIN report predicate correction only",
        "config": ArtifactIdentity.from_artifact(config_path).to_dict(),
        "source_evaluation": evaluation_identity.to_dict(),
        **decision,
        "eligible_for_separately_frozen_held_out_gate_proposal": decision["passed"],
        "held_out_gate_authorized": False,
        "held_out_accessed": False,
        "canonical_accessed": False,
        "isaac_run": False,
        "live_action_authorized": False,
        "filming_authorized": False,
        "production_authority_granted": False,
    }
    write_json_atomic(output, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = adjudicate(arguments.evaluation, arguments.config, arguments.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
