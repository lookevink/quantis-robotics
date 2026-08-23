import io
import json
from pathlib import Path
import unittest

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_worker import serve_jsonl


class _Predictor:
    def predict(self, observation: ControlObservation) -> ProposedControl:
        return ProposedControl(
            observation_id=observation.observation_id,
            created_at_unix_seconds=101.0,
            actions=(DroidAction((0.0,) * 7),) * 3,
            proposal=Path("/tmp/proposal.pth"),
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


if __name__ == "__main__":
    unittest.main()
