from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import tempfile

import numpy as np

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_resolution import (
    CONTROL_RESOLUTION_PROTOCOL,
    ControlResolutionEndpoint,
    ControlResolutionReport,
    ControlResolutionSample,
    retreat_direction,
)
from jepa_wm.control_safety import ControlInterlockEvidence
from jepa_wm.control_safety import ControlGateDecision, SafetyProjectionAttempt
from jepa_wm.action import DroidActionScale
from jepa_wm.direct_safety import ControlSafetySnapshot
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.trial_equivalence import TrialResetState
from sim.isaac_control_resolution import AttachedControlInterlock, _capture_endpoint
from sim.isaac_control_resolution import resolution_joint_target
from sim.isaac_demo_runtime import JointCommand
from sim.control_session import ControlSession, ControlSessionState


def _reset() -> TrialResetState:
    return TrialResetState(
        pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
        joint_positions=(0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
        collision_detected=False,
        contact_force_newtons=0.0,
        plug_position=(0.1, 0.0, 0.2),
        plug_attached=True,
    )


def _sample(index: int, magnitude: float) -> ControlResolutionSample:
    start = _reset()
    commanded = DroidAction((-magnitude, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    realized = 2e-5 if magnitude == 0.0 else magnitude * 0.8
    target_pose = start.pose.applied(commanded)
    actual_pose = start.pose.applied(
        DroidAction((-realized, 0.0, 0.0, 1e-5, 0.0, 0.0, 0.0))
    )
    actual_joints = (2e-4, *start.joint_positions[1:])
    return ControlResolutionSample(
        index=index,
        requested_translation_meters=magnitude,
        start_reset=start,
        commanded_action=commanded,
        target_pose=target_pose,
        projection=SafetyProjectionAttempt(
            DroidActionScale.uniform(1.0),
            ControlGateDecision(123, target_pose, ()),
            0.0,
            start.joint_positions,
        ),
        endpoint=ControlResolutionEndpoint(
            actual_pose,
            ControlSafetySnapshot(
                actual_joints,
                0.04,
                start.plug_position,
                0.0,
                False,
                True,
            ),
        ),
        interlock=ControlInterlockEvidence(0.0, False),
        rollback_reset=start,
    )


class ControlResolutionReportTest(unittest.TestCase):
    def test_capture_endpoint_returns_raw_pose_and_live_safety_state(self) -> None:
        command = JointCommand(np.zeros(7), 0.04)
        pose = _reset().pose
        runtime = SimpleNamespace(
            sensor=object(),
            actuators=SimpleNamespace(actual_command=Mock(return_value=command)),
            attachment=SimpleNamespace(
                attached=True,
                world_pose=Mock(return_value=((0.1, 0.0, 0.2), None)),
            ),
        )
        snapshot = SimpleNamespace(end_effector_pose=pose)

        with (
            patch(
                "sim.isaac_control_resolution.read_control_contact",
                return_value=(False, 0.25),
            ),
            patch(
                "sim.isaac_control_resolution.recording_snapshot",
                return_value=snapshot,
            ),
        ):
            actual, endpoint = _capture_endpoint(runtime)

        self.assertIs(actual, command)
        self.assertEqual(endpoint.pose, pose)
        self.assertEqual(endpoint.safety.contact_force_newtons, 0.25)
        self.assertTrue(endpoint.safety.plug_attached)

    def test_zero_probe_holds_its_exact_live_start_command(self) -> None:
        baseline = JointCommand(
            np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)), 0.04
        )
        drifted_start = JointCommand(
            baseline.arm_positions + 9e-5, baseline.gripper_width_m
        )

        target = resolution_joint_target(0.0, drifted_start, None)

        self.assertIs(target, drifted_start)

    def test_live_interlock_aborts_immediately_on_attachment_loss(self) -> None:
        contact = Mock()
        contact.observe.return_value = object()
        observer = AttachedControlInterlock(
            contact,
            type(
                "Runtime",
                (),
                {"attachment": type("Attachment", (), {"attached": False})()},
            )(),
        )

        with self.assertRaisesRegex(RuntimeError, "lost plug attachment"):
            observer.observe()

        contact.observe.assert_called_once_with()

    def test_retreat_direction_points_away_from_recorded_target(self) -> None:
        self.assertEqual(
            retreat_direction(
                DroidPose((0.4, 0.1, 0.5, 0.0, 0.0, 0.0, 0.5)),
                DroidPose((0.399, 0.1, 0.5, 0.0, 0.0, 0.0, 0.5)),
            ),
            (1.0, 0.0, 0.0),
        )

    def test_round_trip_reconstructs_noise_and_response_metrics(self) -> None:
        samples = tuple(
            _sample(index, magnitude)
            for index, magnitude in enumerate(
                CONTROL_RESOLUTION_PROTOCOL.requested_translations
            )
        )
        report = ControlResolutionReport(
            session_id="resolution-52600-c43",
            reference_recording="contact-insertion-held-00",
            seed=52600,
            context_index=43,
            observation_id=123,
            captured_pose=_reset().pose,
            recorded_target_pose=DroidPose(
                (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
            ),
            reference_reset=_reset(),
            samples=samples,
        )

        restored = ControlResolutionReport.from_dict(report.to_dict())

        self.assertEqual(restored, report)
        summary = restored.summary
        self.assertAlmostEqual(summary.zero_translation_drift_meters, 2e-5)
        self.assertAlmostEqual(summary.zero_orientation_drift_radians, 1e-5)
        self.assertEqual(
            tuple(result.requested_translation_meters for result in summary.responses),
            (3e-5, 1e-4, 2e-4),
        )
        self.assertTrue(summary.diagnostic_only)
        self.assertFalse(summary.multi_step_authority_granted)
        self.assertFalse(summary.production_authority_granted)
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=1.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png"), report.recorded_target_pose),
            expected_proposal=Path("/tmp/control-resolution-measurement.pth"),
            pose=report.captured_pose,
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=43,
        )
        state = {
            "session_id": report.session_id,
            "reference_recording": report.reference_recording,
            "seed": report.seed,
            "recording": "control-resolution-recording",
            "previous_session_id": None,
            "execution_policy": (
                ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT.value
            ),
            "current_joint_positions": list(report.reference_reset.joint_positions),
            "collision_detected": False,
            "contact_force_newtons": 0.0,
            "plug_position": list(report.reference_reset.plug_position),
            "plug_attached": True,
            "current_gripper_width_m": 0.04,
        }
        restored.validate_capture(observation.to_dict(), state)
        with self.assertRaisesRegex(ValueError, "bound to its capture"):
            restored.validate_capture(
                replace(
                    observation,
                    target=ControlTarget(
                        Path("target.png"),
                        DroidPose(
                            (0.402, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
                        ),
                    ),
                ).to_dict(),
                state,
            )
        with tempfile.TemporaryDirectory() as temp_dir:
            session = ControlSession.at(Path(temp_dir), report.session_id)
            session.write_capture(
                observation,
                ControlSessionState(
                    session_id=report.session_id,
                    reference_recording=report.reference_recording,
                    seed=report.seed,
                    recording="control-resolution-recording",
                    current_joint_positions=report.reference_reset.joint_positions,
                    collision_detected=False,
                    contact_force_newtons=0.0,
                    execution_policy=(
                        ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT
                    ),
                    plug_position=report.reference_reset.plug_position,
                    plug_attached=True,
                    current_gripper_width_m=0.04,
                ),
            )
            with self.assertRaisesRegex(ValueError, "restricted insertion"):
                session.claim_execution()

    def test_rejects_missing_roster_reset_drift_and_unsafe_evidence(self) -> None:
        samples = tuple(
            _sample(index, magnitude)
            for index, magnitude in enumerate(
                CONTROL_RESOLUTION_PROTOCOL.requested_translations
            )
        )
        arguments = dict(
            session_id="resolution-52600-c43",
            reference_recording="contact-insertion-held-00",
            seed=52600,
            context_index=43,
            observation_id=123,
            captured_pose=_reset().pose,
            recorded_target_pose=DroidPose(
                (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
            ),
            reference_reset=_reset(),
        )

        with self.assertRaisesRegex(ValueError, "sample roster"):
            ControlResolutionReport(samples=samples[:-1], **arguments)
        drifted = replace(
            samples[0],
            start_reset=replace(
                samples[0].start_reset,
                joint_positions=(0.01, *samples[0].start_reset.joint_positions[1:]),
            ),
        )
        with self.assertRaisesRegex(ValueError, "same reset"):
            ControlResolutionReport(samples=(drifted, *samples[1:]), **arguments)
        submillimeter_drift = replace(
            samples[0],
            start_reset=replace(
                samples[0].start_reset,
                pose=DroidPose(
                    (
                        samples[0].start_reset.pose.values[0] + 3e-5,
                        *samples[0].start_reset.pose.values[1:],
                    )
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "same reset"):
            ControlResolutionReport(
                samples=(submillimeter_drift, *samples[1:]), **arguments
            )
        unsafe = replace(
            samples[0], interlock=ControlInterlockEvidence(0.0, True)
        )
        with self.assertRaisesRegex(ValueError, "safety"):
            ControlResolutionReport(samples=(unsafe, *samples[1:]), **arguments)
        unsafe_ik = replace(
            samples[0],
            projection=replace(
                samples[0].projection,
                maximum_joint_delta_rad=3.0,
                proposed_joint_positions=(3.0, *samples[0].start_reset.joint_positions[1:]),
            ),
        )
        with self.assertRaisesRegex(ValueError, "protocol"):
            ControlResolutionReport(samples=(unsafe_ik, *samples[1:]), **arguments)
        detached = replace(
            samples[0],
            endpoint=replace(
                samples[0].endpoint,
                safety=replace(samples[0].endpoint.safety, plug_attached=False),
            ),
        )
        with self.assertRaisesRegex(ValueError, "safety"):
            ControlResolutionReport(samples=(detached, *samples[1:]), **arguments)


if __name__ == "__main__":
    unittest.main()
