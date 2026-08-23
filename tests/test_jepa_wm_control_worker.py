import io
import json
from pathlib import Path
import unittest

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_worker import serve_jsonl
from jepa_wm.planner import CEMConfig
from jepa_wm.shadow_planning import (
    ShadowPlanningRequest,
    ShadowSearchConfig,
    plan_shadow_candidates,
)

import numpy as np


class _Predictor:
    def predict(self, observation: ControlObservation) -> ProposedControl:
        return ProposedControl(
            observation_id=observation.observation_id,
            created_at_unix_seconds=101.0,
            actions=(DroidAction((0.0,) * 7),) * 3,
            proposal=Path("/tmp/proposal.pth"),
        )

    def plan_shadow(self, request: ShadowPlanningRequest):
        return plan_shadow_candidates(
            observation_id=request.observation.observation_id,
            direct_actions=request.direct_control.actions,
            score=lambda candidates: np.square(candidates).sum(axis=(1, 2)),
            proposal=request.direct_control.proposal,
            adapter=Path("/tmp/adapter.pth"),
            config=ShadowSearchConfig(
                planner=CEMConfig(iterations=1, samples=4, elites=2)
            ),
        )


class ControlWorkerTest(unittest.TestCase):
    def test_serves_versioned_observations_as_jsonl(self) -> None:
        request = ControlObservation(
            observation_id=7,
            captured_at_unix_seconds=100.0,
            context_frame=Path("context.png"),
            target_frame=Path("target.png"),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        output = io.StringIO()

        serve_jsonl(
            io.StringIO(json.dumps(request.to_dict()) + "\n"),
            output,
            _Predictor(),
        )

        response = ProposedControl.from_dict(json.loads(output.getvalue()))
        self.assertEqual(response.observation_id, 7)
        self.assertEqual(len(response.actions), 3)

    def test_routes_a_shadow_request_without_returning_a_control_command(self) -> None:
        observation = ControlObservation(
            observation_id=7,
            captured_at_unix_seconds=100.0,
            context_frame=Path("context.png"),
            target_frame=Path("target.png"),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=4,
        )
        direct = _Predictor().predict(observation)
        output = io.StringIO()

        serve_jsonl(
            io.StringIO(
                json.dumps(
                    ShadowPlanningRequest(
                        observation, direct, Path("/tmp/adapter.pth")
                    ).to_dict()
                )
                + "\n"
            ),
            output,
            _Predictor(),
        )

        payload = json.loads(output.getvalue())
        self.assertEqual(payload["authority"], "shadow_only")
        self.assertNotIn("created_at_unix_seconds", payload)


if __name__ == "__main__":
    unittest.main()
