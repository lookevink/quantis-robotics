from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch
import tempfile

import numpy as np

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_resolution import (
    CONTROL_RESOLUTION_FAILURE_SCHEMA,
    CONTROL_RESOLUTION_PROTOCOL,
    ControlResolutionEndpoint,
    ControlResolutionFailureEvidence,
    ControlResolutionProtocol,
    ControlResolutionProbeExecution,
    ControlResolutionProbePlan,
    ControlResolutionReport,
    ControlResolutionResetPhase,
    ControlResolutionSample,
    ControlResolutionSettlementEvidence,
    ControlResolutionSettlementAttempt,
    ControlResolutionSettlementTimeoutTrace,
    ControlResolutionMotionTimeout,
    ControlResolutionRollbackTimeout,
    ControlResolutionRollbackSuccess,
    ControlResolutionRollbackFailure,
    ControlResolutionForwardEvidence,
    FixedUpdateSettlement,
    ControlResolutionBaselineEvidence,
    ControlResolutionBaselineAttempt,
    ControlResolutionBaselinePolicy,
    ControlResolutionBaselineTrace,
    ControlResolutionCaptureIdentity,
    ControlResolutionDriveTarget,
    ControlResolutionLoad,
    ControlResolutionMotionTiming,
    RejectedControlResolutionReset,
    TrackedErrorSettlement,
    TrackedSettlementEvidence,
    retreat_direction,
)
from jepa_wm.control_safety import ControlInterlockEvidence
from jepa_wm.control_safety import ControlGateDecision, SafetyProjectionAttempt
from jepa_wm.action import DroidActionScale
from jepa_wm.direct_safety import ControlSafetySnapshot
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.trial_equivalence import (
    ResetEquivalenceMeasurement,
    TrialResetState,
)
from sim.isaac_control_resolution import (
    ResolutionControlInterlock,
    ControlResolutionResetMismatch,
    _capture_endpoint,
    _capture_reset_state,
    _require_resolution_reset,
    settle_resolution_motion,
    execute_resolution_probe_motion,
    stabilize_resolution_baseline,
    UnstableControlResolutionBaseline,
    UnsettledControlResolutionTarget,
    recover_resolution_drive_target,
    ResolutionDriveTargetRecovery,
    resolution_failure_evidence,
)
from sim.isaac_control_resolution import resolution_probe_observation
from sim.isaac_control_resolution import resolution_settlement_target
from sim.isaac_control_capture import resolution_capture_safety
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


def _baseline(
    reset: TrialResetState | None = None,
) -> ControlResolutionBaselineEvidence:
    state = reset or _reset()
    return ControlResolutionBaselineEvidence(
        ControlResolutionBaselineTrace(
            (state,) * 9,
            (0.25,) * 8,
            ControlInterlockEvidence(0.0, False),
            ControlResolutionDriveTarget(state.joint_positions, 0.04),
        )
    )


def _capture_identity() -> ControlResolutionCaptureIdentity:
    return ControlResolutionCaptureIdentity(
        "contact-insertion-held-00",
        52600,
        43,
        123,
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
        tracked_settlement=TrackedSettlementEvidence(
            motion=ControlResolutionSettlementEvidence(
                requested_joint_motion_radians=0.0,
                required_tracking_error_radians=5e-4,
                updates_used=2,
                passing_tracking_errors_radians=(3e-4, 2e-4),
            ),
            rollback=ControlResolutionSettlementEvidence(
                requested_joint_motion_radians=2e-4,
                required_tracking_error_radians=5e-4,
                updates_used=2,
                passing_tracking_errors_radians=(1e-4, 0.0),
            ),
            rollback_interlock=ControlInterlockEvidence(0.0, False),
            rollback_drive_command=ControlResolutionProbePlan.for_request(
                index, magnitude
            ).drive_command(
                None
                if magnitude == 0.0
                else (0.5 if magnitude == 1e-3 else 0.25)
            ),
        ),
        motion_timing=ControlResolutionMotionTiming(
            1.0,
            1.55 if magnitude == 1e-3 else 1.3,
        ),
    )


class ControlResolutionReportTest(unittest.TestCase):
    def test_corrected_protocol_uses_drive_safe_periods_for_zero_half_and_one_mm(self) -> None:
        self.assertEqual(
            CONTROL_RESOLUTION_PROTOCOL.translation_magnitudes_meters,
            (0.0, 5e-4, 1e-3),
        )
        self.assertEqual(CONTROL_RESOLUTION_PROTOCOL.repeats_per_magnitude, 3)
        self.assertEqual(
            CONTROL_RESOLUTION_PROTOCOL.motion_period_for(0.0),
            0.25,
        )
        self.assertEqual(
            CONTROL_RESOLUTION_PROTOCOL.motion_period_for(5e-4),
            0.25,
        )
        self.assertEqual(
            CONTROL_RESOLUTION_PROTOCOL.motion_period_for(1e-3),
            0.5,
        )

    def test_report_rejects_settling_timestamps_shorter_than_motion_period(self) -> None:
        samples = tuple(
            _sample(index, magnitude)
            for index, magnitude in enumerate(
                CONTROL_RESOLUTION_PROTOCOL.requested_translations
            )
        )
        one_millimeter_index = next(
            index
            for index, sample in enumerate(samples)
            if sample.requested_translation_meters == 1e-3
        )
        samples = (
            *samples[:one_millimeter_index],
            replace(
                samples[one_millimeter_index],
                motion_timing=ControlResolutionMotionTiming(1.0, 1.49),
            ),
            *samples[one_millimeter_index + 1 :],
        )

        with self.assertRaisesRegex(ValueError, "does not match its protocol"):
            ControlResolutionReport(
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
                baseline=_baseline(),
            )

    def test_baseline_requires_one_global_window_with_a_bounded_timeout(self) -> None:
        policy = ControlResolutionBaselinePolicy(required_consecutive_intervals=2)
        self.assertEqual(
            ControlResolutionBaselinePolicy().maximum_intervals
            * ControlResolutionBaselinePolicy().observation_period_seconds,
            20.0,
        )
        self.assertEqual(
            ControlResolutionBaselinePolicy().required_consecutive_intervals,
            8,
        )
        reference = _reset()
        drifted = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
        )
        stable_one = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((2.1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
        )
        stable_two = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((2.2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
        )
        evidence = ControlResolutionBaselineEvidence(
            ControlResolutionBaselineTrace(
                states=(reference, drifted, stable_one, stable_two),
                interval_seconds=(0.25, 0.25, 0.25),
                interlock=ControlInterlockEvidence(0.0, False),
            )
        )

        evidence.validate(
            policy,
            ControlResolutionLoad.ATTACHED,
        )

        with self.assertRaisesRegex(ValueError, "stable"):
            ControlResolutionBaselineEvidence(
                ControlResolutionBaselineTrace(
                    states=(reference, stable_one),
                    interval_seconds=(0.25,),
                    interlock=ControlInterlockEvidence(0.0, False),
                )
            ).validate(
                policy,
                ControlResolutionLoad.ATTACHED,
            )

        with self.assertRaisesRegex(ValueError, "stable"):
            ControlResolutionBaselineEvidence(
                ControlResolutionBaselineTrace(
                    states=(
                        reference,
                        drifted,
                        stable_one,
                        stable_two,
                        reference,
                        drifted,
                        stable_one,
                        stable_two,
                    ),
                    interval_seconds=(0.25,) * 7,
                    interlock=ControlInterlockEvidence(0.0, False),
                ),
            ).validate(
                policy,
                ControlResolutionLoad.ATTACHED,
            )

    def test_baseline_rejects_cumulative_drift_across_stability_window(self) -> None:
        policy = ControlResolutionBaselinePolicy(
            required_consecutive_intervals=2,
        )
        reference = _reset()
        states = tuple(
            replace(
                reference,
                pose=reference.pose.applied(
                    DroidAction(
                        (index * 1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    )
                ),
            )
            for index in range(3)
        )
        evidence = ControlResolutionBaselineEvidence(
            ControlResolutionBaselineTrace(
                states,
                (0.25, 0.25),
                ControlInterlockEvidence(0.0, False),
            )
        )

        with self.assertRaisesRegex(ValueError, "stable"):
            evidence.validate(policy, ControlResolutionLoad.ATTACHED)

    def test_legacy_fixed_settling_report_remains_reconstructible(self) -> None:
        protocol = ControlResolutionProtocol(
            translation_magnitudes_meters=(0.0, 3e-5, 1e-4, 2e-4),
            motion_period_overrides=(),
            baseline_policy=None,
            settlement=FixedUpdateSettlement(8),
        )
        samples = tuple(
            replace(
                _sample(index, magnitude),
                tracked_settlement=None,
                motion_timing=None,
            )
            for index, magnitude in enumerate(protocol.requested_translations)
        )
        report = ControlResolutionReport(
            session_id="resolution-legacy-52600-c43",
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
            protocol=protocol,
        )

        restored = ControlResolutionReport.from_dict(report.to_dict())

        self.assertEqual(restored, report)
        self.assertEqual(restored.protocol.settlement, FixedUpdateSettlement(8))

    def test_settlement_threshold_is_noise_floor_or_command_fraction(self) -> None:
        policy = TrackedErrorSettlement()

        self.assertEqual(policy.maximum_tracking_error(0.001), 5e-4)
        self.assertEqual(policy.maximum_tracking_error(0.004), 0.001)

    def test_hold_probe_binds_no_command_to_its_live_start(self) -> None:
        sample = _sample(0, 0.0)
        execution = ControlResolutionProbeExecution(
            CONTROL_RESOLUTION_PROTOCOL.probe_plan(0),
            DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            sample.start_reset,
            sample.commanded_action,
            sample.target_pose,
            sample.projection,
        )
        execution.validate(
            CONTROL_RESOLUTION_PROTOCOL,
            123,
        )

    def test_probe_retreat_direction_flips_after_crossing_recorded_target(self) -> None:
        recorded_target = DroidPose(
            (0.400189, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
        )
        crossed_start = DroidPose(
            (0.4003, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
        )

        action = CONTROL_RESOLUTION_PROTOCOL.probe_action(
            crossed_start, recorded_target, 1e-4
        )
        endpoint = crossed_start.applied(action)

        self.assertGreater(action.values[0], 0.0)
        self.assertGreater(
            abs(endpoint.values[0] - recorded_target.values[0]),
            abs(crossed_start.values[0] - recorded_target.values[0]),
        )

        samples = tuple(
            _sample(index, magnitude)
            for index, magnitude in enumerate(
                CONTROL_RESOLUTION_PROTOCOL.requested_translations
            )
        )
        target_directed = samples[3]
        crossed_reset = replace(target_directed.start_reset, pose=crossed_start)
        unsafe_target = crossed_start.applied(target_directed.commanded_action)
        target_directed = replace(
            target_directed,
            start_reset=crossed_reset,
            target_pose=unsafe_target,
            projection=replace(
                target_directed.projection,
                gate=replace(
                    target_directed.projection.gate,
                    next_pose=unsafe_target,
                ),
            ),
        )
        samples = (*samples[:3], target_directed, *samples[4:])
        with self.assertRaisesRegex(ValueError, "protocol"):
            ControlResolutionReport(
                session_id="resolution-52600-c43",
                reference_recording="contact-insertion-held-00",
                seed=52600,
                context_index=43,
                observation_id=123,
                captured_pose=_reset().pose,
                recorded_target_pose=recorded_target,
                reference_reset=_reset(),
                samples=samples,
                baseline=_baseline(),
            )

    def test_summary_exposes_bounded_reset_repeatability_noise(self) -> None:
        samples = tuple(
            _sample(index, magnitude)
            for index, magnitude in enumerate(
                CONTROL_RESOLUTION_PROTOCOL.requested_translations
            )
        )
        reference = _reset()
        noisy_rollback = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((2.18e-4, 0.0, 0.0, 5e-4, 0.0, 0.0, 0.0))
            ),
            joint_positions=(
                reference.joint_positions[0] + 3.96e-4,
                *reference.joint_positions[1:],
            ),
            plug_position=(
                reference.plug_position[0] + 2.42e-4,
                *reference.plug_position[1:],
            ),
        )
        samples = (
            replace(
                samples[0],
                rollback_reset=noisy_rollback,
                tracked_settlement=replace(
                    samples[0].tracked_settlement,
                    rollback=replace(
                        samples[0].tracked_settlement.rollback,
                        passing_tracking_errors_radians=(4e-4, 3.96e-4),
                    ),
                ),
            ),
            *samples[1:],
        )

        report = ControlResolutionReport(
            session_id="resolution-52600-c43",
            reference_recording="contact-insertion-held-00",
            seed=52600,
            context_index=43,
            observation_id=123,
            captured_pose=reference.pose,
            recorded_target_pose=DroidPose(
                (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
            ),
            reference_reset=reference,
            samples=samples,
            baseline=_baseline(reference),
        )

        repeatability = report.summary.rollback_repeatability
        self.assertAlmostEqual(repeatability.translation_difference_meters, 2.18e-4)
        self.assertAlmostEqual(
            repeatability.maximum_joint_difference_radians, 3.96e-4
        )
        self.assertAlmostEqual(
            repeatability.maximum_plug_axis_difference_meters, 2.42e-4
        )

    def test_rejected_reset_persists_exact_resolution_scale_drift(self) -> None:
        reference = _reset()
        candidate = replace(
            reference,
            joint_positions=(
                reference.joint_positions[0] + 6e-4,
                *reference.joint_positions[1:],
            ),
        )
        rejected = RejectedControlResolutionReset(
            phase=ControlResolutionResetPhase.ROLLBACK,
            sample_index=0,
            reference=reference,
            candidate=candidate,
            tolerances=CONTROL_RESOLUTION_PROTOCOL.reset_tolerances,
        )
        failure = ControlResolutionFailureEvidence(
            session_id="resolution-52600-c43",
            failed_at_unix_seconds=123.0,
            reference_reset=reference,
            completed_samples=(),
            error="ValueError: reset mismatch",
            rejected_reset=rejected,
            baseline=_baseline(reference),
            capture_identity=_capture_identity(),
        )

        restored = ControlResolutionFailureEvidence.from_dict(failure.to_dict())

        self.assertEqual(restored, failure)
        self.assertAlmostEqual(
            restored.rejected_reset.measurement.maximum_joint_difference_radians,
            6e-4,
        )
        self.assertFalse(
            restored.rejected_reset.measurement.passes(
                CONTROL_RESOLUTION_PROTOCOL.reset_tolerances
            )
        )
        malformed = failure.to_dict()
        malformed["rejected_reset"]["sample_index"] = 0.5
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ControlResolutionFailureEvidence.from_dict(malformed)
        malformed["rejected_reset"]["sample_index"] = True
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ControlResolutionFailureEvidence.from_dict(malformed)

    def test_prebaseline_failure_preserves_unloaded_mode(self) -> None:
        failure = ControlResolutionFailureEvidence(
            session_id="resolution-unloaded-52600-c43",
            failed_at_unix_seconds=123.0,
            reference_reset=None,
            completed_samples=(),
            error="RuntimeError: baseline did not stabilize",
            load=ControlResolutionLoad.UNLOADED,
            capture_identity=_capture_identity(),
        )

        payload = failure.to_dict()
        restored = ControlResolutionFailureEvidence.from_dict(payload)

        self.assertEqual(payload["schema"], CONTROL_RESOLUTION_FAILURE_SCHEMA)
        self.assertEqual(payload["load"], "unloaded")
        self.assertEqual(restored, failure)
        legacy_protocol = replace(
            CONTROL_RESOLUTION_PROTOCOL,
            baseline_policy=None,
            settlement=FixedUpdateSettlement(8),
        )
        with self.assertRaisesRegex(ValueError, "legacy.*capture identity"):
            replace(failure, protocol=legacy_protocol)

    def test_unstable_baseline_failure_preserves_every_observed_state(self) -> None:
        states = tuple(
            replace(
                _reset(),
                pose=_reset().pose.applied(
                    DroidAction((index * 2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                ),
            )
            for index in range(
                ControlResolutionBaselinePolicy().maximum_intervals + 1
            )
        )
        attempt = ControlResolutionBaselineAttempt(
            ControlResolutionBaselineTrace(
                states,
                (0.25,)
                * ControlResolutionBaselinePolicy().maximum_intervals,
                ControlInterlockEvidence(0.0, False),
            )
        )
        failure = ControlResolutionFailureEvidence(
            session_id="resolution-attached-52600-c43",
            failed_at_unix_seconds=123.0,
            reference_reset=None,
            completed_samples=(),
            error="RuntimeError: baseline did not stabilize",
            baseline_attempt=attempt,
            capture_identity=_capture_identity(),
        )

        restored = ControlResolutionFailureEvidence.from_dict(failure.to_dict())

        self.assertEqual(restored, failure)
        self.assertEqual(
            len(restored.baseline_attempt.trace.states),
            ControlResolutionBaselinePolicy().maximum_intervals + 1,
        )

        runtime_failure = resolution_failure_evidence(
            "resolution-attached-52600-c43",
            CONTROL_RESOLUTION_PROTOCOL,
            None,
            (),
            UnstableControlResolutionBaseline(attempt),
            ControlResolutionLoad.ATTACHED,
            None,
            _capture_identity(),
        )
        self.assertEqual(runtime_failure.baseline_attempt, attempt)

    def test_capture_failure_authenticates_rejected_reference_to_raw_state(self) -> None:
        baseline = _baseline()
        captured = replace(
            _reset(),
            joint_positions=(0.002, *_reset().joint_positions[1:]),
        )
        rejected = RejectedControlResolutionReset(
            phase=ControlResolutionResetPhase.CAPTURE_TO_BASELINE,
            sample_index=None,
            reference=captured,
            candidate=baseline.initial_reset,
            tolerances=CONTROL_RESOLUTION_PROTOCOL.capture_tolerances,
        )
        failure = ControlResolutionFailureEvidence(
            session_id="resolution-52600-c43",
            failed_at_unix_seconds=123.0,
            reference_reset=baseline.reference_reset,
            completed_samples=(),
            error="ValueError: capture changed before baseline",
            rejected_reset=rejected,
            baseline=baseline,
            capture_identity=_capture_identity(),
        )
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=1.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png"), _reset().pose),
            expected_proposal=Path("/tmp/control-resolution-measurement.pth"),
            pose=captured.pose,
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=43,
        )
        state = {
            "session_id": failure.session_id,
            "reference_recording": "contact-insertion-held-00",
            "seed": 52600,
            "recording": "control-resolution-recording",
            "previous_session_id": None,
            "execution_policy": (
                ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT.value
            ),
            "current_joint_positions": list(captured.joint_positions),
            "collision_detected": False,
            "contact_force_newtons": 0.0,
            "plug_position": list(captured.plug_position),
            "plug_attached": True,
            "current_gripper_width_m": 0.04,
        }

        restored = ControlResolutionFailureEvidence.from_dict(failure.to_dict())
        restored.validate_capture(observation.to_dict(), state)

        substituted = replace(
            captured,
            joint_positions=(0.003, *captured.joint_positions[1:]),
        )
        tampered = replace(
            restored,
            rejected_reset=replace(rejected, reference=substituted),
        )
        with self.assertRaisesRegex(ValueError, "capture reset"):
            tampered.validate_capture(observation.to_dict(), state)

    def test_unsafe_reset_is_captured_before_typed_rejection(self) -> None:
        command = JointCommand(np.zeros(7), 0.04)
        runtime = SimpleNamespace(
            sensor=object(),
            actuators=SimpleNamespace(actual_command=Mock(return_value=command)),
            attachment=SimpleNamespace(
                attached=True,
                world_pose=Mock(return_value=((0.1, 0.0, 0.2), None)),
            ),
        )
        snapshot = SimpleNamespace(end_effector_pose=_reset().pose)
        with (
            patch(
                "sim.isaac_control_resolution.read_control_contact",
                return_value=(True, 3.0),
            ),
            patch(
                "sim.isaac_control_resolution.recording_snapshot",
                return_value=snapshot,
            ),
        ):
            _, unsafe = _capture_reset_state(runtime)

        self.assertTrue(unsafe.collision_detected)
        self.assertEqual(unsafe.contact_force_newtons, 3.0)
        with self.assertRaises(ControlResolutionResetMismatch) as raised:
            _require_resolution_reset(
                _reset(),
                unsafe,
                CONTROL_RESOLUTION_PROTOCOL.reset_tolerances,
                ControlResolutionResetPhase.ROLLBACK,
                0,
            )
        self.assertEqual(raised.exception.evidence.candidate, unsafe)

    def test_probe_observation_uses_live_pose_and_probe_timestamp(self) -> None:
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=1.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png"), _reset().pose),
            expected_proposal=Path("/tmp/proposal.pth"),
            pose=_reset().pose,
            previous_action=DroidAction((0.0,) * 7),
            warmup_frames=43,
        )
        live_pose = observation.pose.applied(
            DroidAction((1e-5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        )

        probe = resolution_probe_observation(observation, live_pose, 2.0)

        self.assertEqual(probe.pose, live_pose)
        self.assertEqual(probe.captured_at_unix_seconds, 2.0)
        self.assertEqual(probe.observation_id, observation.observation_id)

    def test_failure_evidence_round_trip_requires_reset_for_samples(self) -> None:
        failure = ControlResolutionFailureEvidence(
            session_id="resolution-52600-c43",
            failed_at_unix_seconds=123.0,
            reference_reset=_reset(),
            completed_samples=(_sample(0, 0.0),),
            error="RuntimeError: stopped",
            baseline=_baseline(),
            capture_identity=_capture_identity(),
        )

        self.assertEqual(
            ControlResolutionFailureEvidence.from_dict(failure.to_dict()), failure
        )
        missing_settlement = failure.to_dict()
        del missing_settlement["completed_samples"][0]["tracked_settlement"]
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(missing_settlement)

        missing_rollback_interlock = failure.to_dict()
        del missing_rollback_interlock["completed_samples"][0][
            "tracked_settlement"
        ]["rollback_interlock"]
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(
                missing_rollback_interlock
            )

        unsafe_rollback = failure.to_dict()
        unsafe_rollback["completed_samples"][0]["tracked_settlement"][
            "rollback_interlock"
        ]["maximum_contact_force_newtons"] = 3.0
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(unsafe_rollback)

        unsafe_endpoint = failure.to_dict()
        unsafe_endpoint["completed_samples"][0]["endpoint"]["safety"][
            "plug_attached"
        ] = False
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(unsafe_endpoint)
        with self.assertRaisesRegex(ValueError, "reference reset"):
            replace(failure, reference_reset=None)
        missing_baseline = failure.to_dict()
        del missing_baseline["baseline"]
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(missing_baseline)

    def test_settlement_timeout_round_trip_retains_the_complete_tracking_trace(self) -> None:
        baseline = _baseline()
        final_positions = (
            baseline.drive_target.joint_positions[0] + 1e-3,
            *baseline.drive_target.joint_positions[1:],
        )
        probe = CONTROL_RESOLUTION_PROTOCOL.probe_plan(0)
        sample = _sample(0, 0.0)
        execution = ControlResolutionProbeExecution(
            probe,
            DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            sample.start_reset,
            sample.commanded_action,
            sample.target_pose,
            sample.projection,
        )
        settlement_failure = ControlResolutionMotionTimeout(
            trace=ControlResolutionSettlementTimeoutTrace(
                execution=execution,
                start_joint_positions=baseline.drive_target.joint_positions,
                target_joint_positions=baseline.drive_target.joint_positions,
                attempt=ControlResolutionSettlementAttempt(
                    requested_joint_motion_radians=0.0,
                    required_tracking_error_radians=5e-4,
                    tracking_errors_radians=(1e-3,) * 32,
                    final_joint_positions=final_positions,
                ),
                interlock=ControlInterlockEvidence(0.0, False),
                drive_command=probe.drive_command(None),
                timing=ControlResolutionMotionTiming(1.0, 2.0),
            ),
            rollback_outcome=ControlResolutionRollbackSuccess(
                start_joint_positions=final_positions,
                drive_command=probe.drive_command(None),
                settlement=ControlResolutionSettlementEvidence(
                    requested_joint_motion_radians=1e-3,
                    required_tracking_error_radians=5e-4,
                    updates_used=2,
                    passing_tracking_errors_radians=(4e-4, 0.0),
                ),
                interlock=ControlInterlockEvidence(0.0, False),
                reset=baseline.reference_reset,
            ),
        )
        failure = ControlResolutionFailureEvidence(
            session_id="resolution-52600-c43",
            failed_at_unix_seconds=123.0,
            reference_reset=baseline.reference_reset,
            completed_samples=(),
            error="ControlResolutionSettlementTimeout: bounded timeout",
            baseline=baseline,
            capture_identity=_capture_identity(),
            settlement_failure=settlement_failure,
        )

        restored = ControlResolutionFailureEvidence.from_dict(failure.to_dict())

        self.assertEqual(restored, failure)
        tampered = failure.to_dict()
        tampered["settlement_failure"]["target_joint_positions"][0] += 1e-3
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(tampered)

        settled_with_failure_fields = failure.to_dict()
        settled_with_failure_fields["settlement_failure"]["rollback_outcome"][
            "attempt"
        ] = None
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(
                settled_with_failure_fields
            )

        failed_with_settled_fields = failure.to_dict()
        contradictory_outcome = failed_with_settled_fields["settlement_failure"][
            "rollback_outcome"
        ]
        contradictory_outcome.update(
            status="failed",
            attempt=None,
            error="bounded recovery failure",
        )
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(failed_with_settled_fields)

    def test_rollback_timeout_retains_the_completed_forward_probe(self) -> None:
        baseline = _baseline()
        sample = _sample(0, 0.0)
        execution = ControlResolutionProbeExecution(
            CONTROL_RESOLUTION_PROTOCOL.probe_plan(0),
            DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            sample.start_reset,
            sample.commanded_action,
            sample.target_pose,
            sample.projection,
        )
        forward = ControlResolutionForwardEvidence(
            sample.endpoint,
            sample.tracked_settlement.motion,
            sample.interlock,
            sample.motion_timing,
        )
        timeout = ControlResolutionRollbackTimeout(
            trace=ControlResolutionSettlementTimeoutTrace(
                execution=execution,
                start_joint_positions=sample.endpoint.safety.joint_positions,
                target_joint_positions=baseline.drive_target.joint_positions,
                attempt=ControlResolutionSettlementAttempt(
                    requested_joint_motion_radians=2e-4,
                    required_tracking_error_radians=5e-4,
                    tracking_errors_radians=(1e-3,) * 32,
                    final_joint_positions=(
                        1e-3,
                        *baseline.drive_target.joint_positions[1:],
                    ),
                ),
                interlock=ControlInterlockEvidence(0.0, False),
                drive_command=CONTROL_RESOLUTION_PROTOCOL.probe_plan(
                    0
                ).drive_command(None),
                timing=ControlResolutionMotionTiming(2.0, 3.0),
            ),
            forward=forward,
        )
        failure = ControlResolutionFailureEvidence(
            session_id="resolution-52600-c43",
            failed_at_unix_seconds=123.0,
            reference_reset=baseline.reference_reset,
            completed_samples=(),
            error="ControlResolutionSettlementTimeout: rollback timeout",
            baseline=baseline,
            capture_identity=_capture_identity(),
            settlement_failure=timeout,
        )

        restored = ControlResolutionFailureEvidence.from_dict(failure.to_dict())

        self.assertEqual(restored.settlement_failure.forward, forward)
        self.assertEqual(
            restored.settlement_failure.execution.projection,
            sample.projection,
        )

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

    def test_zero_probe_uses_the_live_start_as_its_measurement_target(self) -> None:
        baseline = JointCommand(
            np.asarray((0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)), 0.04
        )
        drifted_start = JointCommand(
            baseline.arm_positions + 9e-5, baseline.gripper_width_m
        )

        target = resolution_settlement_target(
            CONTROL_RESOLUTION_PROTOCOL.probe_plan(0),
            None,
            drifted_start,
        )

        np.testing.assert_array_equal(
            target.arm_positions, drifted_start.arm_positions
        )
        self.assertEqual(target.gripper_width_m, drifted_start.gripper_width_m)

    def test_drive_target_rejects_controller_target_drift(self) -> None:
        target = _baseline().drive_target

        target.validate_active(target.joint_positions, target.gripper_width_m)
        with self.assertRaisesRegex(ValueError, "active drive target changed"):
            target.validate_active(
                (target.joint_positions[0] + 1e-6, *target.joint_positions[1:]),
                target.gripper_width_m,
            )

    def test_live_interlock_aborts_immediately_on_attachment_loss(self) -> None:
        contact = Mock()
        contact.observe.return_value = object()
        observer = ResolutionControlInterlock(
            contact,
            type(
                "Runtime",
                (),
                {"attachment": type("Attachment", (), {"attached": False})()},
            )(),
            True,
        )

        with self.assertRaisesRegex(RuntimeError, "load state changed"):
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
            baseline=_baseline(),
        )

        restored = ControlResolutionReport.from_dict(report.to_dict())

        self.assertEqual(restored, report)
        missing_drive_command = report.to_dict()
        del missing_drive_command["samples"][0]["tracked_settlement"][
            "rollback_drive_command"
        ]
        with self.assertRaisesRegex(ValueError, "report is incomplete"):
            ControlResolutionReport.from_dict(missing_drive_command)
        contradictory_drive_command = report.to_dict()
        contradictory_drive_command["samples"][0]["tracked_settlement"][
            "rollback_command_period_seconds"
        ] = 0.25
        with self.assertRaisesRegex(ValueError, "report is incomplete"):
            ControlResolutionReport.from_dict(contradictory_drive_command)
        summary = restored.summary
        self.assertAlmostEqual(summary.zero_translation_drift_meters, 2e-5)
        self.assertAlmostEqual(summary.zero_orientation_drift_radians, 1e-5)
        self.assertEqual(
            tuple(result.requested_translation_meters for result in summary.responses),
            (5e-4, 1e-3),
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
        lifecycle_drift = dict(state)
        lifecycle_drift["current_joint_positions"] = [
            state["current_joint_positions"][0] + 5e-4,
            *state["current_joint_positions"][1:],
        ]
        restored.validate_capture(observation.to_dict(), lifecycle_drift)
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

    def test_capture_is_bound_to_baseline_start_before_settling(self) -> None:
        reference = _reset()
        captured = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
            joint_positions=(
                reference.joint_positions[0] + 0.01,
                *reference.joint_positions[1:],
            ),
            plug_position=(
                reference.plug_position[0] + 0.01,
                *reference.plug_position[1:],
            ),
        )
        baseline = ControlResolutionBaselineEvidence(
            ControlResolutionBaselineTrace(
                (captured, *(reference,) * 9),
                (0.25,) * 9,
                ControlInterlockEvidence(0.0, False),
            )
        )
        report = ControlResolutionReport(
            session_id="resolution-52600-c43",
            reference_recording="contact-insertion-held-00",
            seed=52600,
            context_index=43,
            observation_id=123,
            captured_pose=captured.pose,
            recorded_target_pose=DroidPose(
                (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
            ),
            reference_reset=reference,
            samples=tuple(
                _sample(index, magnitude)
                for index, magnitude in enumerate(
                    CONTROL_RESOLUTION_PROTOCOL.requested_translations
                )
            ),
            baseline=baseline,
        )
        observation = ControlObservation(
            observation_id=123,
            captured_at_unix_seconds=1.0,
            context_frame=Path("context.png"),
            target=ControlTarget(Path("target.png"), report.recorded_target_pose),
            expected_proposal=Path("/tmp/control-resolution-measurement.pth"),
            pose=captured.pose,
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
            "current_joint_positions": list(captured.joint_positions),
            "collision_detected": False,
            "contact_force_newtons": 0.0,
            "plug_position": list(captured.plug_position),
            "plug_attached": True,
            "current_gripper_width_m": 0.04,
        }

        report.validate_capture(observation.to_dict(), state)

        drifted = dict(state)
        drifted["current_joint_positions"] = [
            captured.joint_positions[0] + 0.002,
            *captured.joint_positions[1:],
        ]
        with self.assertRaisesRegex(ValueError, "reset state"):
            report.validate_capture(observation.to_dict(), drifted)

    def test_unloaded_report_requires_detached_noncolliding_evidence(self) -> None:
        unloaded_reset = replace(_reset(), plug_attached=False)
        samples = tuple(
            replace(
                _sample(index, magnitude),
                start_reset=unloaded_reset,
                endpoint=replace(
                    _sample(index, magnitude).endpoint,
                    safety=replace(
                        _sample(index, magnitude).endpoint.safety,
                        plug_attached=False,
                    ),
                ),
                rollback_reset=unloaded_reset,
            )
            for index, magnitude in enumerate(
                CONTROL_RESOLUTION_PROTOCOL.requested_translations
            )
        )
        report = ControlResolutionReport(
            session_id="resolution-unloaded-52600-c43",
            reference_recording="contact-insertion-held-00",
            seed=52600,
            context_index=43,
            observation_id=123,
            captured_pose=_reset().pose,
            recorded_target_pose=DroidPose(
                (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
            ),
            reference_reset=unloaded_reset,
            samples=samples,
            load=ControlResolutionLoad.UNLOADED,
            baseline=_baseline(unloaded_reset),
        )

        restored = ControlResolutionReport.from_dict(report.to_dict())

        self.assertEqual(restored, report)
        self.assertEqual(restored.load, ControlResolutionLoad.UNLOADED)

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
            baseline=_baseline(),
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
                        samples[0].start_reset.pose.values[0] + 6e-4,
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


class ControlResolutionSettlementRuntimeTest(unittest.IsolatedAsyncioTestCase):
    def test_capture_persists_tracking_telemetry_without_a_second_baseline(self) -> None:
        command = JointCommand(np.zeros(7), 0.04)
        actual = JointCommand(
            np.asarray((2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            0.039,
        )
        actuators = SimpleNamespace(actual_command=Mock(return_value=actual))

        safety = resolution_capture_safety(
            actuators,
            command,
            ControlInterlockEvidence(0.25, False),
        )

        self.assertEqual(safety.arm_tracking_error_rad, 2e-4)
        self.assertAlmostEqual(safety.gripper_tracking_error_m, 0.001)
        self.assertFalse(safety.collision_detected)
        self.assertEqual(safety.contact_force_newtons, 0.25)

    @patch("sim.isaac_control_resolution.advance_simulation_period", new_callable=AsyncMock)
    @patch("sim.isaac_control_resolution._capture_reset_state")
    async def test_baseline_waits_for_two_consecutive_stable_intervals(
        self,
        capture_reset: Mock,
        advance_period: AsyncMock,
    ) -> None:
        reference = _reset()
        drifted = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
        )
        stable_one = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((2.1e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
        )
        stable_two = replace(
            reference,
            pose=reference.pose.applied(
                DroidAction((2.2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            ),
        )
        commands = tuple(
            JointCommand(np.asarray(state.joint_positions), 0.04)
            for state in (reference, drifted, stable_one, stable_two)
        )
        capture_reset.side_effect = tuple(
            zip(commands, (reference, drifted, stable_one, stable_two))
        )
        interlock = SimpleNamespace(
            observe=Mock(),
            contact=SimpleNamespace(
                evidence=ControlInterlockEvidence(0.0, False)
            ),
        )

        runtime = SimpleNamespace(
            actuators=SimpleNamespace(
                current_command=Mock(return_value=commands[0])
            )
        )
        command, evidence = await stabilize_resolution_baseline(
            runtime,
            interlock,
            ControlResolutionBaselinePolicy(required_consecutive_intervals=2),
            Mock(side_effect=(0.0, 0.25, 0.5, 0.75)),
        )

        self.assertIs(command, commands[-1])
        interlock.observe.assert_called_once_with()
        self.assertEqual(evidence.reference_reset, stable_two)
        self.assertEqual(advance_period.await_count, 3)
        advance_period.assert_awaited_with(0.25, interlock.observe)

    @patch("sim.isaac_control_resolution.advance_simulation_period", new_callable=AsyncMock)
    @patch("sim.isaac_control_resolution._capture_reset_state")
    async def test_unstable_baseline_retains_the_complete_attempt(
        self,
        capture_reset: Mock,
        advance_period: AsyncMock,
    ) -> None:
        states = tuple(
            replace(
                _reset(),
                pose=_reset().pose.applied(
                    DroidAction((index * 2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                ),
            )
            for index in range(9)
        )
        commands = tuple(
            JointCommand(np.asarray(state.joint_positions), 0.04)
            for state in states
        )
        capture_reset.side_effect = tuple(zip(commands, states))
        interlock = SimpleNamespace(
            observe=Mock(),
            contact=SimpleNamespace(
                evidence=ControlInterlockEvidence(0.0, False)
            ),
        )

        with self.assertRaises(UnstableControlResolutionBaseline) as raised:
            runtime = SimpleNamespace(
                actuators=SimpleNamespace(
                    current_command=Mock(return_value=commands[0])
                )
            )
            await stabilize_resolution_baseline(
                runtime,
                interlock,
                ControlResolutionBaselinePolicy(maximum_intervals=8),
                Mock(side_effect=tuple(index * 0.25 for index in range(9))),
            )

        self.assertEqual(raised.exception.attempt.trace.states, states)
        self.assertEqual(
            raised.exception.attempt.trace.interval_seconds,
            (0.25,) * 8,
        )
        self.assertEqual(advance_period.await_count, 8)

    async def test_settlement_requires_two_consecutive_tracked_updates(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        actual_commands = (
            JointCommand(np.asarray((4e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), 0.04),
            JointCommand(np.asarray((1e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), 0.04),
            JointCommand(np.asarray((4e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), 0.04),
            JointCommand(np.asarray((3e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)), 0.04),
        )
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(
                actual_command=Mock(side_effect=actual_commands)
            )
        )
        interlock = SimpleNamespace(observe=Mock())

        with patch(
            "sim.isaac_control_resolution.advance_physics_updates",
            new=AsyncMock(),
        ) as advance:
            evidence = await settle_resolution_motion(
                runtime,
                target,
                target,
                interlock,
                TrackedErrorSettlement(),
            )

        self.assertEqual(evidence.updates_used, 4)
        self.assertEqual(evidence.passing_tracking_errors_radians, (4e-4, 3e-4))
        self.assertEqual(evidence.final_tracking_error_radians, 3e-4)
        self.assertEqual(advance.await_count, 4)
        self.assertEqual(
            advance.await_args_list,
            [call(1, interlock.observe) for _ in range(4)],
        )

    async def test_settlement_fails_after_bounded_update_timeout(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(
                actual_command=Mock(
                    side_effect=(
                        JointCommand(
                            np.asarray((1e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
                            0.04,
                        )
                        for _ in range(3)
                    )
                )
            )
        )
        interlock = SimpleNamespace(observe=Mock())
        policy = TrackedErrorSettlement(maximum_updates=3)

        with patch(
            "sim.isaac_control_resolution.advance_physics_updates",
            new=AsyncMock(),
        ) as advance:
            with self.assertRaises(UnsettledControlResolutionTarget) as raised:
                await settle_resolution_motion(
                    runtime,
                    target,
                    target,
                    interlock,
                    policy,
                )

        self.assertEqual(advance.await_count, 3)
        self.assertEqual(
            raised.exception.attempt,
            ControlResolutionSettlementAttempt(
                requested_joint_motion_radians=0.0,
                required_tracking_error_radians=5e-4,
                tracking_errors_radians=(1e-3, 1e-3, 1e-3),
                final_joint_positions=(1e-3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            ),
        )

    async def test_zero_probe_advances_without_replacing_the_drive_target(self) -> None:
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(apply_drive_command=Mock()),
            attachment=object(),
        )
        command = JointCommand(np.zeros(7), 0.04)
        interlock = SimpleNamespace(observe=Mock())

        with (
            patch(
                "sim.isaac_control_resolution.advance_simulation_period",
                new=AsyncMock(),
            ) as advance,
            patch(
                "sim.isaac_control_resolution.move_joint_command",
                new=AsyncMock(),
            ) as move,
        ):
            await execute_resolution_probe_motion(
                runtime,
                command,
                command,
                CONTROL_RESOLUTION_PROTOCOL.probe_plan(0),
                0.25,
                interlock,
            )

        advance.assert_awaited_once_with(0.25, interlock.observe)
        move.assert_not_awaited()
        runtime.actuators.apply_drive_command.assert_not_called()

    async def test_interrupted_probe_attempts_interlocked_drive_target_recovery(self) -> None:
        baseline_target = _baseline().drive_target
        start = JointCommand(
            np.asarray(
                (
                    baseline_target.joint_positions[0] + 1e-3,
                    *baseline_target.joint_positions[1:],
                )
            ),
            0.04,
        )
        drive_target = JointCommand(
            np.asarray(baseline_target.joint_positions),
            baseline_target.gripper_width_m,
        )
        attempt = ControlResolutionSettlementAttempt(
            requested_joint_motion_radians=1e-3,
            required_tracking_error_radians=5e-4,
            tracking_errors_radians=(1e-3,) * 32,
            final_joint_positions=tuple(start.arm_positions),
        )
        interlock = SimpleNamespace(
            observe=Mock(),
            contact=SimpleNamespace(
                evidence=ControlInterlockEvidence(0.0, False)
            ),
        )

        actuators = SimpleNamespace(
            current_command=Mock(return_value=drive_target),
        )
        reference_reset = replace(
            _reset(),
            joint_positions=(
                _reset().joint_positions[0] - 1e-3,
                *_reset().joint_positions[1:],
            ),
        )
        with (
            patch(
                "sim.isaac_control_resolution.move_joint_command",
                new=AsyncMock(),
            ) as move,
            patch(
                "sim.isaac_control_resolution.settle_resolution_motion",
                new=AsyncMock(
                    side_effect=UnsettledControlResolutionTarget(attempt)
                ),
            ) as settle,
        ):
            outcome = await recover_resolution_drive_target(
                SimpleNamespace(actuators=actuators, attachment=object()),
                start,
                interlock,
                ResolutionDriveTargetRecovery(
                    CONTROL_RESOLUTION_PROTOCOL,
                    CONTROL_RESOLUTION_PROTOCOL.probe_plan(0),
                    baseline_target,
                    reference_reset,
                ),
            )

        self.assertIsInstance(outcome, ControlResolutionRollbackFailure)
        self.assertEqual(outcome.attempt, attempt)
        self.assertEqual(
            outcome.drive_command,
            CONTROL_RESOLUTION_PROTOCOL.probe_plan(0).drive_command(None),
        )
        move.assert_not_awaited()
        np.testing.assert_array_equal(
            settle.await_args.args[2].arm_positions,
            np.asarray(reference_reset.joint_positions),
        )
        actuators.current_command.assert_called()

    async def test_fixed_settlement_dispatches_exact_update_count(self) -> None:
        runtime = SimpleNamespace()
        command = JointCommand(np.zeros(7), 0.04)
        interlock = SimpleNamespace(observe=Mock())

        with patch(
            "sim.isaac_control_resolution.advance_physics_updates",
            new=AsyncMock(),
        ) as advance:
            evidence = await settle_resolution_motion(
                runtime,
                command,
                command,
                interlock,
                FixedUpdateSettlement(8),
            )

        self.assertIsNone(evidence)
        advance.assert_awaited_once_with(8, interlock.observe)


if __name__ == "__main__":
    unittest.main()
