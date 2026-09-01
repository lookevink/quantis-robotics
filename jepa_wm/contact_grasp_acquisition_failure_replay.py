"""Replay the preserved V4 gripper failure against the acquisition proposal."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.contact_grasp_target import ContactGraspTargetPolicy
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.persistence import write_json_atomic
from jepa_wm.planner import PlannerActionBounds
from jepa_wm.training_artifact import artifact_fingerprint
from jepa_wm.worker_artifacts import ControlWorkerArtifacts
from sim.control_session import ControlResult, ControlSessionState


REPLAY_SCHEMA = "quantis.jepa_wm_contact_grasp_acquisition_failure_replay.v1"
SOURCE_SESSION_ID = "unknown-start-e2e-v4-62605-grasp-03"
SOURCE_FINGERPRINTS = {
    "request.json": "7d74adb159ec0e781b04204bb58dbfdea2718ac4e57914c9ee397021c3810394",
    "state.json": "736c5250b5ecfd052e68ef3d1f74ca1626ee4fa00c2b5f798b0fd5b992342509",
    "response.json": "716046e8c67571b72a1f9ae1be0b0befdfaf6d7bb2484a62b2ecb8d5c7b1cd64",
    "result.json": "480921a5daee6a0bd53d9b3e3b98922f1d1423bee80c157e4af2af85cdb9589a",
    "context.png": "9e32575a43ab6202eb95eafb40d9e776bf78e9d480fdcba07d1e0378c48c14b4",
}
EXPECTED_PROPOSAL_FINGERPRINT = (
    "16cdb2a36f1d80d8e07b321c9607a6a91747429a2e73895b22bc0e7e4e2f4dfa"
)
EXPECTED_WORKER_FINGERPRINT = (
    "1cd9d7adee65af5817bcebb7abd03e5ba82a12d95b1674fe5138c74256b8c51d"
)
EXPECTED_READINESS_FINGERPRINT = (
    "1cf2d752e17325ed737c5c761de06fd2934a1ada06c703bc0a2e43372bcdfc4a"
)


def _norm(values: Sequence[float]) -> float:
    return sqrt(sum(float(value) ** 2 for value in values))


def evaluate_replay_action(
    observation: ControlObservation,
    control: ProposedControl,
) -> dict[str, Any]:
    """Require the replacement to recover goal progress without opening."""

    action = control.first_action
    goal = observation.goal_action
    next_translation = tuple(
        current + delta
        for current, delta in zip(
            observation.pose.values[:3], action.values[:3]
        )
    )
    next_gripper = observation.pose.values[6] + action.values[6]
    next_goal = DroidAction(
        (
            *(target - current for target, current in zip(
                observation.target_pose.values[:3], next_translation
            )),
            *goal.values[3:6],
            observation.target_pose.values[6] - next_gripper,
        )
    )
    translation_before = _norm(goal.values[:3])
    translation_after = _norm(next_goal.values[:3])
    gripper_before = abs(goal.values[6])
    gripper_after = abs(next_goal.values[6])
    passed = (
        PlannerActionBounds().accepts(control.actions)
        and 0.0 <= next_gripper <= 1.0
        and action.values[6] >= 0.0
        and translation_after < translation_before
        and gripper_after <= gripper_before
    )
    return {
        "passed": passed,
        "first_action": list(action.values),
        "translation_error_before_m": translation_before,
        "translation_error_after_m": translation_after,
        "gripper_error_before": gripper_before,
        "gripper_error_after": gripper_after,
        "next_translation": list(next_translation),
        "next_gripper_closedness": next_gripper,
    }


def run_failure_replay(
    source: Path,
    checkpoint: Path,
    data_root: Path,
    worker_manifest: Path,
    readiness: Path,
    output: Path,
) -> dict[str, Any]:
    session = data_root / "control_sessions" / SOURCE_SESSION_ID
    actual_source_fingerprints = {
        name: artifact_fingerprint(session / name) for name in SOURCE_FINGERPRINTS
    }
    if actual_source_fingerprints != SOURCE_FINGERPRINTS:
        raise ValueError("contact-grasp acquisition replay source changed")
    if artifact_fingerprint(worker_manifest) != EXPECTED_WORKER_FINGERPRINT:
        raise ValueError("contact-grasp acquisition replay worker changed")
    if artifact_fingerprint(readiness) != EXPECTED_READINESS_FINGERPRINT:
        raise ValueError("contact-grasp acquisition replay readiness changed")
    readiness_payload = json.loads(readiness.read_text())
    if readiness_payload.get("passed") is not True:
        raise ValueError("contact-grasp acquisition proposal is not ready")

    source_observation = ControlObservation.from_dict(
        json.loads((session / "request.json").read_text())
    )
    source_state = ControlSessionState.from_dict(
        json.loads((session / "state.json").read_text())
    )
    source_control = ProposedControl.from_dict(
        json.loads((session / "response.json").read_text())
    )
    source_result = ControlResult.from_dict(
        json.loads((session / "result.json").read_text())
    )
    if (
        source_result.status.value != "blocked"
        or tuple(reason.value for reason in source_result.gate.reasons)
        != ("gripper_violation",)
        or source_control.first_action.values[6] >= 0.0
    ):
        raise ValueError("contact-grasp acquisition replay is not the V4 failure")

    previous_policy = source_state.require_current_contact_grasp_policy()
    policy = ContactGraspTargetPolicy.for_scene_translation(
        previous_policy.scene_translation_m
    )
    reference = data_root / "recordings" / source_state.reference_recording
    target = policy.initial_target(
        reference,
        frame_root=data_root,
        live_pose=source_observation.pose,
    )
    artifacts = ControlWorkerArtifacts.load(worker_manifest)
    if artifact_fingerprint(artifacts.proposal) != EXPECTED_PROPOSAL_FINGERPRINT:
        raise ValueError("contact-grasp acquisition replay proposal changed")
    observation = replace(
        source_observation,
        target=target,
        expected_proposal=artifacts.proposal,
        warmup_frames=policy.context_index_for_target(target.frame),
    )

    from jepa_wm.control_worker import FrozenProposalPredictor

    predictor = FrozenProposalPredictor(
        source,
        checkpoint,
        artifacts,
        frame_root=data_root,
    )
    control = predictor.predict(observation)
    evaluation = evaluate_replay_action(observation, control)
    payload = {
        "schema": REPLAY_SCHEMA,
        "status": "passed" if evaluation["passed"] else "failed",
        "passed": evaluation["passed"],
        "source_session": SOURCE_SESSION_ID,
        "source_fingerprints": SOURCE_FINGERPRINTS,
        "source_failure_first_action": list(source_control.first_action.values),
        "proposal": str(artifacts.proposal),
        "proposal_fingerprint": EXPECTED_PROPOSAL_FINGERPRINT,
        "worker_manifest": str(worker_manifest.resolve()),
        "worker_fingerprint": EXPECTED_WORKER_FINGERPRINT,
        "readiness": str(readiness.resolve()),
        "readiness_fingerprint": EXPECTED_READINESS_FINGERPRINT,
        "target": target.to_dict(),
        "task_context_index": observation.warmup_frames,
        "control": control.to_dict(),
        "evaluation": evaluation,
        "simulator_action_authorized": False,
    }
    write_json_atomic(output.resolve(), payload)
    if not evaluation["passed"]:
        raise ValueError("contact-grasp acquisition replay did not repair V4")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    payload = run_failure_replay(
        arguments.source,
        arguments.checkpoint,
        arguments.data_root,
        arguments.worker,
        arguments.readiness,
        arguments.output,
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
