import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from sim.control_session import ControlSession, ControlSessionState


class ControlSessionTest(unittest.TestCase):
    def test_rejects_state_copied_from_another_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            session = ControlSession.at(Path(temp_dir), "session-b")
            observation = ControlObservation(
                observation_id=123,
                captured_at_unix_seconds=100.0,
                context_frame=Path("context.png"),
                target_frame=Path("target.png"),
                expected_proposal=Path("/tmp/proposal.pth"),
                pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                previous_action=DroidAction((0.0,) * 7),
                warmup_frames=4,
            )
            state = ControlSessionState(
                "session-a",
                "held-reference",
                11400,
                "control-session-a",
                (0.0,) * 7,
                False,
                0.0,
            )
            session.write_capture(observation, state)
            session.response_path.write_text(
                json.dumps(
                    ProposedControl(
                        123,
                        100.1,
                        (DroidAction((0.0,) * 7),) * 3,
                        Path("/tmp/proposal.pth"),
                    ).to_dict()
                )
            )

            with self.assertRaisesRegex(ValueError, "different session"):
                session.load()


if __name__ == "__main__":
    unittest.main()
