from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from sim.control_session import ControlSessionState
from sim.demo_sequence import Phase
from sim.isaac_demo_runtime import JointCommand
from sim.isaac_grasp_demo import record_grasp_demo, validate_grasp_replay_reset
from sim.recording import RecordingLabel, RecordingMoment, RecordingSnapshot


class GraspReplayResetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = DroidPose((0.4, -0.2, 0.5, 0.0, 0.0, 0.0, 0.5))
        self.observation = ControlObservation(
            observation_id=1,
            captured_at_unix_seconds=1.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png")),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=self.pose,
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=86,
        )
        self.joints = (0.0,) * 7
        self.plug = (-0.02, -0.25, 1.32)
        self.state = ControlSessionState(
            "session-1",
            "held-out-1",
            12401,
            "recording-1",
            self.joints,
            False,
            0.0,
            plug_position=self.plug,
            plug_attached=False,
        )
        self.actual = JointCommand(np.zeros(7), 0.04)

    def snapshot(
        self,
        *,
        pose: DroidPose | None = None,
        plug: tuple[float, float, float] | None = None,
    ) -> RecordingSnapshot:
        return RecordingSnapshot(
            RecordingLabel(RecordingMoment.MOTION, Phase.GRASP),
            ObservationStage.APPROACHING_CABLE,
            self.joints,
            0.04,
            plug or self.plug,
            False,
            pose or self.pose,
        )

    def test_accepts_the_exact_recreated_source_reset(self) -> None:
        validate_grasp_replay_reset(
            self.observation,
            self.state,
            self.snapshot(),
            self.actual,
            collision_detected=False,
            contact_force_newtons=0.0,
        )

    def test_rejects_pose_or_connector_drift(self) -> None:
        moved_pose = DroidPose((0.401, -0.2, 0.5, 0.0, 0.0, 0.0, 0.5))
        with self.assertRaisesRegex(ValueError, "same reset state"):
            validate_grasp_replay_reset(
                self.observation,
                self.state,
                self.snapshot(pose=moved_pose),
                self.actual,
                collision_detected=False,
                contact_force_newtons=0.0,
            )
        with self.assertRaisesRegex(ValueError, "same reset state"):
            validate_grasp_replay_reset(
                self.observation,
                self.state,
                self.snapshot(plug=(-0.019, -0.25, 1.32)),
                self.actual,
                collision_detected=False,
                contact_force_newtons=0.0,
            )

    def test_rejects_reset_contact(self) -> None:
        with self.assertRaisesRegex(ValueError, "collision or contact"):
            validate_grasp_replay_reset(
                self.observation,
                self.state,
                self.snapshot(),
                self.actual,
                collision_detected=False,
                contact_force_newtons=0.02,
            )

    def test_rejects_host_fingerprint_mismatch_before_reset(self) -> None:
        with patch(
            "sim.isaac_grasp_demo.GraspControlReadinessSummary."
            "load_container_reconstruction",
            side_effect=ValueError(
                "host-verified proposal fingerprint differs from grasp readiness"
            ),
        ) as load_readiness:
            with self.assertRaisesRegex(ValueError, "host-verified"):
                asyncio.run(
                    record_grasp_demo(
                        "grasp-readiness-v2",
                        12401,
                        "grasp-demo-v1",
                        "f" * 64,
                    )
                )
        load_readiness.assert_called_once_with(
            ANY,
            "grasp-readiness-v2",
            expected_proposal_fingerprint="f" * 64,
        )


if __name__ == "__main__":
    unittest.main()
