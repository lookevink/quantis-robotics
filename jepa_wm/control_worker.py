"""Resident JSONL worker that turns one control observation into three actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import time
from typing import Protocol, TextIO

from jepa_wm.action import DroidAction
from jepa_wm.control_protocol import ControlObservation, ProposedControl


class ControlPredictor(Protocol):
    def predict(self, observation: ControlObservation) -> ProposedControl:
        ...


class FrozenProposalPredictor:
    """Keep frozen JEPA-WM and the promoted proposal resident on one GPU."""

    def __init__(
        self,
        source: Path,
        checkpoint: Path,
        proposal: Path,
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
        self._model = load_headless_model(source, checkpoint, device=self._device)
        self._proposal, _ = load_action_proposal(
            proposal.resolve(), device=self._device
        )
        self._proposal_path = proposal.resolve()
        self._frame_root = frame_root.resolve() if frame_root is not None else None

    def _frame_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        if self._frame_root is None:
            raise ValueError("relative control frames require a frame root")
        resolved = (self._frame_root / path).resolve()
        resolved.relative_to(self._frame_root)
        return resolved

    def predict(self, observation: ControlObservation) -> ProposedControl:
        from jepa_wm.frames import encode_clips
        from jepa_wm.planner import PlannerActionBounds

        if observation.expected_proposal.resolve() != self._proposal_path:
            raise ValueError(
                "control observation expects a different proposal checkpoint"
            )
        context_frame = self._frame_path(observation.context_frame)
        target_frame = self._frame_path(observation.target_frame)
        for name, path in (
            ("context", context_frame),
            ("target", target_frame),
        ):
            if not path.is_file():
                raise ValueError(f"control {name} frame does not exist: {path}")
        context = encode_clips(
            self._model, ((context_frame,),), batch_size=1
        ).to(self._device)
        target = encode_clips(
            self._model, ((target_frame,),), batch_size=1
        ).to(self._device)
        pose = self._torch.tensor(
            (observation.pose.values,),
            device=self._device,
            dtype=context.dtype,
        )
        previous_action = self._torch.tensor(
            (observation.previous_action.values,),
            device=self._device,
            dtype=context.dtype,
        )
        with self._torch.inference_mode():
            actions = PlannerActionBounds().clip(
                self._proposal(
                    context,
                    target,
                    pose if self._proposal.uses_proprioception else None,
                    previous_action if self._proposal.uses_action_history else None,
                )
                .cpu()
                .numpy()
            )[0]
        return ProposedControl(
            observation_id=observation.observation_id,
            created_at_unix_seconds=time(),
            actions=tuple(
                DroidAction(tuple(action))
                for action in actions
            ),
            proposal=self._proposal_path,
        )


def serve_jsonl(
    input_stream: TextIO,
    output_stream: TextIO,
    predictor: ControlPredictor,
) -> None:
    for line in input_stream:
        if not line.strip():
            continue
        observation = ControlObservation.from_dict(json.loads(line))
        proposal = predictor.predict(observation)
        output_stream.write(json.dumps(proposal.to_dict(), separators=(",", ":")))
        output_stream.write("\n")
        output_stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path)
    args = parser.parse_args()
    serve_jsonl(
        sys.stdin,
        sys.stdout,
        FrozenProposalPredictor(
            args.source,
            args.checkpoint,
            args.proposal,
            frame_root=args.frame_root,
        ),
    )


if __name__ == "__main__":
    main()
