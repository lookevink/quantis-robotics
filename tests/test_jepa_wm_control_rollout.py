from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.control_protocol import ControlObservation, ProposedControl
from jepa_wm.control_rollout import (
    ControlRolloutReport,
    OrchestrationFailure,
    OrchestrationOperation,
)
from jepa_wm.control_safety import ControlGateDecision
from jepa_wm.control_tracking import ActionTrackingDecision
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSessionState,
    PostActionEvidence,
    SafetyProjectionAttempt,
)


class ControlRolloutTest(unittest.TestCase):
    def _write_step(
        self,
        root: Path,
        session_id: str,
        *,
        previous_session_id: str | None,
        pose_x: float,
        post_x: float,
        observation_id: int | None = None,
        target_frame: str = "recordings/reference/wrist/frame_000007.png",
        warmup_frames: int = 4,
        captured_at: float = 100.0,
        previous_action_x: float = 0.0,
    ) -> None:
        session = root / "control_sessions" / session_id
        session.mkdir(parents=True)
        observation = ControlObservation(
            observation_id=(
                100 + int(session_id[-1])
                if observation_id is None
                else observation_id
            ),
            captured_at_unix_seconds=captured_at,
            context_frame=Path(f"control_sessions/{session_id}/context.png"),
            target_frame=Path(target_frame),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=DroidPose((pose_x, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            previous_action=DroidAction((previous_action_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            warmup_frames=warmup_frames,
        )
        state = ControlSessionState(
            session_id,
            "reference",
            11400,
            "control-recording",
            (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
            False,
            0.0,
            previous_session_id,
        )
        raw_action = DroidAction(
            (post_x - pose_x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        )
        response = ProposedControl(
            observation.observation_id,
            captured_at + 0.2,
            (raw_action, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
            Path("/tmp/proposal.pth"),
        )
        scale = DroidActionScale(1.0, 0.25, 0.25)
        gate = ControlGateDecision(
            observation.observation_id,
            observation.pose.applied(scale.apply(raw_action)),
            (),
        )
        tracking = ActionTrackingDecision(1.0, 0.0, 0.0, 0.0, 0.0, ())
        result = ControlResult(
            ControlResultStatus.APPLIED,
            session_id,
            gate,
            (SafetyProjectionAttempt(scale, gate, 0.01),),
            scale,
            1.0,
            0.0,
            0.0,
            0.0,
            PostActionEvidence(
                raw_action,
                scale.apply(raw_action),
                scale.apply(raw_action),
                tracking,
                DroidPose((post_x, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
                state.current_joint_positions,
                0.0,
                0.0,
                False,
                {"path": "/tmp/post.png", "shape": [512, 512, 4]},
            ),
        )
        (session / "request.json").write_text(json.dumps(observation.to_dict()))
        (session / "response.json").write_text(json.dumps(response.to_dict()))
        (session / "state.json").write_text(json.dumps(state.to_dict()))
        (session / "result.json").write_text(json.dumps(result.to_dict()))

    def _report(
        self,
        root: Path,
        sessions: tuple[str, ...],
        *,
        requested_steps: int,
        orchestration_failure: OrchestrationFailure | None = None,
    ) -> dict:
        return ControlRolloutReport.from_sessions(
            root,
            "rollout-1",
            sessions,
            reference_recording="reference",
            seed=11400,
            proposal=Path("/tmp/proposal.pth"),
            requested_steps=requested_steps,
            orchestration_failure=orchestration_failure,
        ).to_dict()

    def _write_reference(self, root: Path) -> None:
        reference = root / "recordings" / "reference"
        reference.mkdir(parents=True)
        (reference / "steps.jsonl").write_text(
            json.dumps(
                {
                    "index": 7,
                    "end_effector_pose": [
                        0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                    ],
                }
            )
            + "\n"
        )

    def test_summarizes_a_provenance_bound_rollout_and_goal_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 7,
                        "end_effector_pose": [
                            0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        ],
                    }
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            report = self._report(
                root, ("session-0", "session-1"), requested_steps=2
            )

            self.assertTrue(report["all_steps_applied"])
            self.assertEqual(report["applied_steps"], 2)
            self.assertAlmostEqual(report["translation_progress_meters"], 0.02)

    def test_preserves_requested_count_when_a_rollout_stops_early(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 7,
                        "end_effector_pose": [
                            0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        ],
                    }
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )

            report = self._report(root, ("session-0",), requested_steps=3)

            self.assertEqual(report["requested_steps"], 3)
            self.assertEqual(report["attempted_steps"], 1)
            self.assertFalse(report["all_steps_applied"])

    def test_rejects_target_or_observation_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                observation_id=123,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                observation_id=123,
                target_frame="recordings/reference/wrist/frame_000008.png",
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            with self.assertRaisesRegex(ValueError, "observation IDs"):
                self._report(
                    root, ("session-0", "session-1"), requested_steps=2
                )

    def test_rejects_a_changed_target_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                target_frame="recordings/reference/wrist/frame_000008.png",
                warmup_frames=5,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            with self.assertRaisesRegex(ValueError, "target frame"):
                self._report(
                    root, ("session-0", "session-1"), requested_steps=2
                )

    def test_rejects_a_non_monotonic_warmup_chain(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_reference(root)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                warmup_frames=4,
            )
            self._write_step(
                root,
                "session-1",
                previous_session_id="session-0",
                pose_x=0.41,
                post_x=0.42,
                warmup_frames=4,
                captured_at=101.0,
                previous_action_x=0.01,
            )

            with self.assertRaisesRegex(ValueError, "warm-up"):
                self._report(
                    root, ("session-0", "session-1"), requested_steps=2
                )

    def test_persists_an_incomplete_terminal_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "control_sessions" / "session-0").mkdir(parents=True)

            report = self._report(
                root,
                ("session-0",),
                requested_steps=3,
                orchestration_failure=OrchestrationFailure(
                    OrchestrationOperation.INITIAL_CONTROL_STEP,
                    1,
                ),
            )

            self.assertEqual(report["attempted_steps"], 1)
            self.assertEqual(report["complete_steps"], 0)
            self.assertEqual(report["applied_steps"], 0)
            self.assertEqual(report["steps"][0]["status"], "orchestration_failed")
            self.assertEqual(
                report["orchestration_failure"]["operation"],
                "initial_control_step",
            )

    def test_parses_followup_failure_step_identity_explicitly(self) -> None:
        failure = OrchestrationFailure.parse("followup_capture_03:exit_124")

        self.assertEqual(failure.operation, OrchestrationOperation.FOLLOWUP_CAPTURE)
        self.assertEqual(failure.step_index, 3)
        self.assertEqual(failure.exit_code, 124)
        with self.assertRaisesRegex(ValueError, "malformed"):
            OrchestrationFailure.parse("followup_capture:exit_1")

    def test_rejects_a_stale_persisted_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            reference = root / "recordings" / "reference"
            reference.mkdir(parents=True)
            (reference / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 7,
                        "end_effector_pose": [
                            0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        ],
                    }
                )
                + "\n"
            )
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            result_path = root / "control_sessions" / "session-0" / "result.json"
            payload = json.loads(result_path.read_text())
            payload["inference_age_seconds"] = 2.1
            result_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "freshness"):
                self._report(root, ("session-0",), requested_steps=1)


if __name__ == "__main__":
    unittest.main()
