from pathlib import Path
import tempfile
import threading
import unittest

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_client import request_control, request_shadow_plan
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_server import ControlUnixServer
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


class ControlServerTest(unittest.TestCase):
    def test_reuses_one_predictor_across_a_unix_socket(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            server = ControlUnixServer(Path(temp_dir) / "control.sock", _Predictor())
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                observation = ControlObservation(
                    observation_id=9,
                    captured_at_unix_seconds=100.0,
                    context_frame=Path("context.png"),
                    target=ControlTarget(Path("target.png")),
                    expected_proposal=Path("/tmp/proposal.pth"),
                    pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                    previous_action=DroidAction((0.0,) * 7),
                    warmup_frames=4,
                )

                response = request_control(server.socket_path, observation)

                self.assertEqual(response.observation_id, 9)
                shadow = request_shadow_plan(
                    server.socket_path,
                    ShadowPlanningRequest(
                        observation,
                        response,
                        Path("/tmp/adapter.pth"),
                        ShadowSearchConfig().planner,
                    ),
                )
                self.assertEqual(shadow.observation_id, 9)
                self.assertEqual(shadow.authority.value, "shadow_only")
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
