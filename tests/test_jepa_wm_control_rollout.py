from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jepa_wm.action import DroidAction, DroidActionScale, DroidPose
from jepa_wm.action_prior import ActionPriorConfig
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_rollout import (
    ControlRolloutReport,
    ControlStepSummary,
    OrchestrationFailure,
    OrchestrationOperation,
)
from jepa_wm.control_safety import ControlGateDecision, SafetyProjectionAttempt
from jepa_wm.control_tracking import ActionTrackingDecision
from jepa_wm.planner import CEMConfig
from jepa_wm.planner_readiness import FirstActionThresholds
from jepa_wm.objective_calibration import (
    ActionResponseCalibration,
    ActionResponseTrial,
    CalibrationIdentity,
    TaskProgressObjective,
)
from jepa_wm.shadow_planning import (
    ShadowPlanningRequest,
    ShadowSearchConfig,
    plan_shadow_candidates,
)
from jepa_wm.shadow_safety import ShadowSafetyEvidence
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
    PostActionEvidence,
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
        target_pose: DroidPose | None = None,
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
            target=ControlTarget(Path(target_frame), target_pose),
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
            (
                SafetyProjectionAttempt(
                    scale, gate, 0.01, state.current_joint_positions
                ),
            ),
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

    def test_summarizes_shadow_search_and_counterfactual_safety(self) -> None:
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
            session = root / "control_sessions" / "session-0"
            response = ProposedControl.from_dict(
                json.loads((session / "response.json").read_text())
            )
            observation = ControlObservation.from_dict(
                json.loads((session / "request.json").read_text())
            )
            target = np.asarray([action.values for action in response.actions]) * 0.8
            shadow = plan_shadow_candidates(
                observation_id=response.observation_id,
                direct_actions=response.actions,
                score=lambda candidates: np.square(
                    candidates - target[None, :, :]
                ).sum(axis=(1, 2)),
                proposal=response.proposal,
                adapter=Path("/tmp/adapter.pth"),
                config=ShadowSearchConfig(
                    planner=CEMConfig(iterations=5, samples=160, elites=12, seed=7),
                    prior=ActionPriorConfig(penalty_weight=1e-12),
                    first_action_thresholds=FirstActionThresholds(
                        minimum_active_cosine=-1.0
                    ),
                ),
            )
            (session / "shadow_request.json").write_text(
                json.dumps(
                    ShadowPlanningRequest(
                        observation,
                        response,
                        Path("/tmp/adapter.pth"),
                    ).to_dict()
                )
            )
            (session / "shadow.json").write_text(json.dumps(shadow.to_dict()))
            scale = DroidActionScale(1.0, 0.25, 0.25)
            gate = ControlGateDecision(
                response.observation_id,
                observation.pose.applied(scale.apply(shadow.planned.actions[0])),
                (),
            )
            safety = ShadowSafetyEvidence(
                response.observation_id,
                response.created_at_unix_seconds + 1.0,
                response.created_at_unix_seconds,
                shadow.planned.actions,
                (
                    SafetyProjectionAttempt(
                        scale,
                        gate,
                        0.0,
                        (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
                    ),
                ),
                scale,
            )
            (session / "shadow_safety.json").write_text(
                json.dumps(safety.to_dict())
            )

            report = self._report(root, ("session-0",), requested_steps=1)

            self.assertEqual(report["shadow_searches"], 1)
            self.assertEqual(
                report["shadow_gate_passes"], int(shadow.passes_shadow_gate)
            )
            self.assertEqual(report["shadow_safety_passes"], 1)
            self.assertGreater(report["mean_shadow_energy_improvement"], 0.0)

            safety_path = session / "shadow_safety.json"
            safety_payload = json.loads(safety_path.read_text())
            safety_payload["attempts"][0]["gate"]["next_pose"][0] += 0.01
            safety_path.write_text(json.dumps(safety_payload))
            with self.assertRaisesRegex(ValueError, "gate evidence"):
                self._report(root, ("session-0",), requested_steps=1)
            safety_path.write_text(json.dumps(safety.to_dict()))

            safety_payload = safety.to_dict()
            safety_payload["counterfactual_as_of_unix_seconds"] += 0.01
            safety_path.write_text(json.dumps(safety_payload))
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._report(root, ("session-0",), requested_steps=1)
            safety_path.write_text(json.dumps(safety.to_dict()))

            shadow_request_path = session / "shadow_request.json"
            shadow_request = json.loads(shadow_request_path.read_text())
            shadow_request["expected_adapter"] = "/tmp/other-adapter.pth"
            shadow_request_path.write_text(json.dumps(shadow_request))
            with self.assertRaisesRegex(ValueError, "not bound"):
                self._report(root, ("session-0",), requested_steps=1)

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

    def test_recomputes_realized_action_instead_of_trusting_result_claims(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
            )
            result_path = root / "control_sessions" / "session-0" / "result.json"
            result = json.loads(result_path.read_text())
            result["actual_action"][0] = 0.02
            result_path.write_text(json.dumps(result))

            with self.assertRaisesRegex(ValueError, "realization"):
                ControlStepSummary.from_session(
                    ControlSession.at(root / "control_sessions", "session-0")
                )

    def test_rejects_shadow_objective_rebound_to_another_start_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target_pose = DroidPose((0.43, 0.0, 0.5, 0.0, 0.0, 0.0, 0.6))
            self._write_step(
                root,
                "session-0",
                previous_session_id=None,
                pose_x=0.40,
                post_x=0.41,
                target_pose=target_pose,
            )
            session = ControlSession.at(root / "control_sessions", "session-0")
            observation, _ = session.load_capture()
            response = session.load_response()
            calibration = ActionResponseCalibration.fit(
                tuple(
                    ActionResponseTrial(
                        f"trial-{axis}",
                        axis,
                        DroidAction(
                            (
                                *(0.002 if index == axis else 0.0 for index in range(3)),
                                *(0.004 if index == axis else 0.0 for index in range(3)),
                                0.2,
                            )
                        ),
                        DroidAction(
                            (
                                *(0.001 if index == axis else 0.0 for index in range(3)),
                                *(0.001 if index == axis else 0.0 for index in range(3)),
                                0.05,
                            )
                        ),
                    )
                    for axis in range(3)
                )
            )
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps(calibration.to_dict()))
            request = ShadowPlanningRequest(
                observation,
                response,
                Path("/tmp/adapter.pth"),
                CalibrationIdentity.from_calibration(
                    calibration_path, calibration
                ),
            )
            session.shadow_request_path.write_text(json.dumps(request.to_dict()))
            tampered_start = DroidPose(
                (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
            )
            shadow = plan_shadow_candidates(
                observation_id=observation.observation_id,
                direct_actions=response.actions,
                score=lambda candidates: np.square(candidates).sum(axis=(1, 2)),
                proposal=response.proposal,
                adapter=Path("/tmp/adapter.pth"),
                config=ShadowSearchConfig(
                    planner=CEMConfig(iterations=1, samples=4, elites=2)
                ),
                task_progress=TaskProgressObjective(
                    tampered_start, target_pose, calibration
                ),
            )
            session.shadow_path.write_text(json.dumps(shadow.to_dict()))

            with self.assertRaisesRegex(ValueError, "not bound"):
                session.load_shadow()

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
            payload["observation_age_seconds"] = 3.1
            result_path.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "freshness"):
                self._report(root, ("session-0",), requested_steps=1)

    def test_reports_a_stale_step_that_the_gate_safely_blocked(self) -> None:
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
            result_path = root / "control_sessions" / "session-0" / "result.json"
            payload = json.loads(result_path.read_text())
            payload["status"] = "blocked"
            payload["gate"]["passed"] = False
            payload["gate"]["reasons"] = ["stale_observation"]
            for attempt in payload["safety_projection_attempts"]:
                attempt["gate"]["passed"] = False
                attempt["gate"]["reasons"] = ["stale_observation"]
            payload["selected_action_scale"] = None
            payload["observation_age_seconds"] = 3.0
            for field in (
                "raw_proposed_action",
                "commanded_action",
                "actual_action",
                "action_tracking",
                "post_action_pose",
                "post_action_joint_positions",
                "maximum_joint_tracking_error_rad",
                "post_action_contact_force_newtons",
                "post_action_collision_detected",
                "post_action_frame",
            ):
                payload.pop(field)
            result_path.write_text(json.dumps(payload))

            report = self._report(root, ("session-0",), requested_steps=2)

            self.assertEqual(report["complete_steps"], 1)
            self.assertEqual(report["applied_steps"], 0)
            self.assertEqual(report["steps"][0]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
