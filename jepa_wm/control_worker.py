"""Resident JSONL worker that turns one control observation into three actions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from time import time
from typing import Any, Protocol, TextIO

from jepa_wm.action import DroidAction
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.shadow_planning import (
    CALIBRATED_SHADOW_SEARCH_CONFIG,
    SHADOW_REQUEST_SCHEMA,
    ShadowPlanningRequest,
    ShadowSearchConfig,
    ShadowSearchEvidence,
    plan_shadow_candidates,
)
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
    TaskProgressObjective,
)
from jepa_wm.worker_artifacts import ControlWorkerArtifacts


class ControlPredictor(Protocol):
    def predict(self, observation: ControlObservation) -> ProposedControl:
        ...

    def plan_shadow(self, request: ShadowPlanningRequest) -> ShadowSearchEvidence:
        ...


@dataclass(frozen=True)
class EncodedObservationCache:
    observation_id: int
    context: Any
    target: Any
    control: ProposedControl

class FrozenProposalPredictor:
    """Keep frozen JEPA-WM and the promoted proposal resident on one GPU."""

    def __init__(
        self,
        source: Path,
        checkpoint: Path,
        artifacts: ControlWorkerArtifacts,
        *,
        frame_root: Path | None = None,
    ) -> None:
        import torch

        from jepa_wm.model import load_headless_model
        from jepa_wm.proposal import load_action_proposal

        if not torch.cuda.is_available():
            raise RuntimeError("control inference requires CUDA")
        self._torch = torch
        self._device = torch.device("cuda", torch.cuda.current_device())
        self._model = load_headless_model(
            source,
            checkpoint,
            device=self._device,
            adapter=artifacts.adapter,
        )
        self._proposal, _ = load_action_proposal(
            artifacts.proposal, device=self._device
        )
        self._proposal_path = artifacts.proposal
        self._adapter_path = artifacts.adapter
        self._calibration = (
            ActionResponseCalibration.load(artifacts.calibration)
            if artifacts.calibration is not None
            else None
        )
        self._calibration_path = artifacts.calibration
        self._progress_margins = artifacts.progress_margins
        self._planner = artifacts.planner
        if (
            self._calibration is not None
            and not self._calibration.ready_for_reranking
        ):
            raise ValueError("action-response calibration is not ready for reranking")
        self._frame_root = frame_root.resolve() if frame_root is not None else None
        self._latest: EncodedObservationCache | None = None

    def _frame_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        if self._frame_root is None:
            raise ValueError("relative control frames require a frame root")
        resolved = (self._frame_root / path).resolve()
        resolved.relative_to(self._frame_root)
        return resolved

    def _encode_observation(self, observation: ControlObservation):
        from jepa_wm.frames import encode_clips

        if observation.expected_proposal.resolve() != self._proposal_path:
            raise ValueError(
                "control observation expects a different proposal checkpoint"
            )
        context_frame = self._frame_path(observation.context_frame)
        target_frame = self._frame_path(observation.target_frame)
        for name, path in (("context", context_frame), ("target", target_frame)):
            if not path.is_file():
                raise ValueError(f"control {name} frame does not exist: {path}")
        context = encode_clips(
            self._model, ((context_frame,),), batch_size=1
        ).to(self._device)
        target = encode_clips(
            self._model, ((target_frame,),), batch_size=1
        ).to(self._device)
        return context, target

    def predict(self, observation: ControlObservation) -> ProposedControl:
        from jepa_wm.planner import PlannerActionBounds
        from jepa_wm.proposal import ProposalInputs

        context, target = self._encode_observation(observation)
        inputs = ProposalInputs.from_observation(
            observation,
            conditioning=self._proposal.conditioning,
            device=self._device,
            dtype=context.dtype,
        )
        with self._torch.inference_mode():
            actions = PlannerActionBounds().clip(
                self._proposal(context, target, inputs)
                .cpu()
                .numpy()
            )[0]
        control = ProposedControl(
            observation_id=observation.observation_id,
            created_at_unix_seconds=time(),
            actions=tuple(
                DroidAction(tuple(action))
                for action in actions
            ),
            proposal=self._proposal_path,
        )
        self._latest = EncodedObservationCache(
            observation.observation_id,
            context,
            target,
            control,
        )
        return control

    def plan_shadow(self, request: ShadowPlanningRequest) -> ShadowSearchEvidence:
        from jepa_wm.planning_scoring import LatentGoalScorer

        if self._adapter_path is None:
            raise RuntimeError("shadow planning requires an adapted action encoder")
        if request.expected_adapter.resolve() != self._adapter_path:
            raise ValueError("shadow request expects a different action adapter")
        if (
            request.expected_calibration is not None
            and request.expected_calibration.path != self._calibration_path
        ) or (
            request.expected_calibration is None
            and self._calibration_path is not None
        ):
            raise ValueError("shadow request expects a different calibration")
        expected_fingerprint = (
            self._calibration.fingerprint if self._calibration is not None else None
        )
        request_fingerprint = (
            request.expected_calibration.fingerprint
            if request.expected_calibration is not None
            else None
        )
        if request_fingerprint != expected_fingerprint:
            raise ValueError("shadow request calibration fingerprint does not match")
        if request.expected_planner != self._planner:
            raise ValueError("shadow request expects a different planner configuration")
        if (
            self._latest is not None
            and request.observation.observation_id == self._latest.observation_id
            and request.direct_control == self._latest.control
        ):
            context, target = self._latest.context, self._latest.target
        else:
            context, target = self._encode_observation(request.observation)
        scorer = LatentGoalScorer(
            self._model,
            context,
            target,
            device=self._device,
        )
        task_progress = None
        if self._calibration is not None:
            target_pose = request.observation.target_pose
            if target_pose is None:
                raise ValueError("calibrated shadow planning requires a target pose")
            if self._progress_margins is None:
                raise ValueError("calibrated worker is missing task-progress margins")
            task_progress = TaskProgressObjective(
                request.observation.pose,
                target_pose,
                self._calibration,
                minimum_progress=self._progress_margins,
            )
        return plan_shadow_candidates(
            observation_id=request.observation.observation_id,
            direct_actions=request.direct_control.actions,
            score=scorer,
            proposal=self._proposal_path,
            adapter=self._adapter_path,
            config=replace(
                CALIBRATED_SHADOW_SEARCH_CONFIG
                if task_progress is not None
                else ShadowSearchConfig(),
                planner=self._planner,
            ),
            task_progress=task_progress,
        )


def serve_jsonl(
    input_stream: TextIO,
    output_stream: TextIO,
    predictor: ControlPredictor,
) -> None:
    for line in input_stream:
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("schema") == SHADOW_REQUEST_SCHEMA:
            response = predictor.plan_shadow(ShadowPlanningRequest.from_dict(payload))
        else:
            response = predictor.predict(ControlObservation.from_dict(payload))
        output_stream.write(json.dumps(response.to_dict(), separators=(",", ":")))
        output_stream.write("\n")
        output_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path)
    args = parser.parse_args()
    serve_jsonl(
        sys.stdin,
        sys.stdout,
        FrozenProposalPredictor(
            args.source,
            args.checkpoint,
            ControlWorkerArtifacts.load(args.artifacts),
            frame_root=args.frame_root,
        ),
    )


if __name__ == "__main__":
    main()
