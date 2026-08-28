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
    ControlResolutionProjectionFailure,
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
    ControlResolutionRollbackCorrection,
    ControlResolutionForwardEvidence,
    FixedUpdateSettlement,
    ControlResolutionBaselineEvidence,
    ControlResolutionBaselineAttempt,
    ControlResolutionBaselinePolicy,
    ControlResolutionBaselineTrace,
    ControlResolutionCaptureIdentity,
    ControlResolutionDriveTarget,
    DriveCommandApplied,
    ControlResolutionLoad,
    ControlResolutionMotionTiming,
    RejectedControlResolutionReset,
    TrackedErrorSettlement,
    TrackedSettlementEvidence,
    retreat_direction,
)
from jepa_wm.control_safety import ControlInterlockEvidence
from jepa_wm.joint_settlement import GripperSettlementTrace
from jepa_wm.control_safety import (
    ControlGateDecision,
    ControlGateReason,
    SafetyProjectionAttempt,
)
from jepa_wm.control_resolution_baseline import (
    CONTROL_RESOLUTION_CAPTURE_FAILURE_SCHEMA,
    ControlResolutionCaptureBaselineContract,
    ControlResolutionCaptureAttemptIdentity,
    ControlResolutionCaptureFailureEvidence,
    ControlResolutionCaptureSourceIdentity,
)
from jepa_wm.control_resolution_drive import ControlResolutionDriveBiasCompensation
from jepa_wm.action import DroidActionScale
from jepa_wm.insertion_refresh import ControlSafetySnapshot
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
    ControlResolutionProjectionRejected,
    resolution_failure_evidence,
)
from sim.isaac_control_resolution import resolution_probe_observation
from sim.isaac_control_resolution import resolution_settlement_target
from sim.isaac_control_capture import stabilize_resolution_capture
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
            ControlResolutionDriveTarget.for_command(
                state.joint_positions,
                0.04,
            ),
        )
    )


def _capture_identity() -> ControlResolutionCaptureIdentity:
    return ControlResolutionCaptureIdentity(
        ControlResolutionCaptureSourceIdentity(
            "contact-insertion-held-00",
            52600,
            43,
        ),
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
    actual_joints = start.joint_positions
    rollback_drive_target = ControlResolutionDriveTarget.for_command(
        start.joint_positions,
        0.04,
    )
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
                passing_tracking_errors_radians=(1e-4, 0.0),
            ),
            rollback=ControlResolutionSettlementEvidence(
                requested_joint_motion_radians=0.0,
                required_tracking_error_radians=5e-4,
                updates_used=2,
                passing_tracking_errors_radians=(0.0, 0.0),
            ),
            rollback_interlock=ControlInterlockEvidence(0.0, False),
            rollback_drive_command=ControlResolutionProbePlan.for_request(
                index, magnitude
            ).drive_command(
                None
                if magnitude == 0.0
                else 0.5,
                rollback_drive_target if magnitude > 0.0 else None,
            ),
        ),
        motion_timing=ControlResolutionMotionTiming(
            1.0,
            1.3 if magnitude == 0.0 else 1.55,
        ),
        drive_target_joint_positions=(
            rollback_drive_target.joint_positions
            if magnitude > 0.0
            else None
        ),
    )


class ControlResolutionReportTest(unittest.TestCase):
    def test_drive_target_canonicalizes_usd_float_storage(self) -> None:
        raw_joints = tuple(0.123456789 + index for index in range(7))
        raw_gripper = 0.017999999225139618

        target = ControlResolutionDriveTarget.for_command(
            raw_joints,
            raw_gripper,
        )
        active_joints = tuple(
            float(
                np.deg2rad(
                    np.float64(np.float32(np.rad2deg(value)))
                )
            )
            for value in raw_joints
        )
        active_gripper = float(np.float32(raw_gripper / 2.0)) * 2.0

        target.validate_active(active_joints, active_gripper)
        self.assertNotEqual(target.joint_positions, raw_joints)

    def test_drive_bias_compensation_is_bounded_and_velocity_safe(self) -> None:
        policy = ControlResolutionDriveBiasCompensation(0.002)
        reference = _reset()
        baseline_target = ControlResolutionDriveTarget(
            tuple(value + 0.001 for value in reference.joint_positions),
            0.04,
        )
        desired = tuple(value + 0.002 for value in reference.joint_positions)

        compensated = policy.compensated_joint_target(
            desired,
            baseline_target,
            reference.joint_positions,
            CONTROL_RESOLUTION_PROTOCOL.safety_limits,
        )

        np.testing.assert_allclose(
            compensated,
            tuple(value + 0.003 for value in reference.joint_positions),
        )
        self.assertEqual(
            ControlResolutionDriveBiasCompensation.from_dict(policy.to_dict()),
            policy,
        )
        with self.assertRaisesRegex(ValueError, "bias exceeds"):
            policy.compensated_joint_target(
                desired,
                ControlResolutionDriveTarget(
                    tuple(value + 0.003 for value in reference.joint_positions),
                    0.04,
                ),
                reference.joint_positions,
                CONTROL_RESOLUTION_PROTOCOL.safety_limits,
            )
        with self.assertRaisesRegex(ValueError, "velocity"):
            CONTROL_RESOLUTION_PROTOCOL.drive_joint_target(
                CONTROL_RESOLUTION_PROTOCOL.probe_plan(3),
                tuple(value + 0.3 for value in reference.joint_positions),
                ControlResolutionDriveTarget.for_command(
                    reference.joint_positions,
                    0.04,
                ),
                reference,
                reference.joint_positions,
            )

    def test_rollback_compensation_uses_the_observed_forward_equilibrium(self) -> None:
        reference = replace(
            _reset(),
            joint_positions=(
                0.8406665325164795,
                -1.3180197477340698,
                -1.1586148738861084,
                -2.6056032180786133,
                0.4334414601325989,
                3.4952261447906494,
                -0.7353371977806091,
            ),
        )
        active = ControlResolutionDriveTarget(
            (
                0.8408090914601275,
                -1.3172312999282176,
                -1.1576152726265365,
                -2.6050660951119227,
                0.43336481917493436,
                3.4953805549174786,
                -0.735293850122861,
            ),
            0.04,
        )
        forward_drive = (
            0.8394091725017465,
            -1.316645444949954,
            -1.159615478151655,
            -2.606631485873575,
            0.4267988150487443,
            3.495227201846615,
            -0.7293138671692664,
        )
        forward_actual = (
            0.8395864367485046,
            -1.3175078630447388,
            -1.1602394580841064,
            -2.606823205947876,
            0.42809009552001953,
            3.4952261447906494,
            -0.7304517030715942,
        )

        target, command = CONTROL_RESOLUTION_PROTOCOL.rollback_drive_command(
            CONTROL_RESOLUTION_PROTOCOL.probe_plan(3),
            active,
            reference,
            forward_drive,
            forward_actual,
            0.5,
        )

        np.testing.assert_allclose(
            target.joint_positions,
            tuple(
                desired + drive - realized
                for desired, drive, realized in zip(
                    reference.joint_positions,
                    forward_drive,
                    forward_actual,
                )
            ),
        )
        self.assertIsInstance(command, DriveCommandApplied)
        self.assertEqual(command.target, target)
        self.assertGreaterEqual(command.period_seconds, 0.5)
        stalled = (
            0.8403059244155884,
            -1.3178406953811646,
            -1.1591547727584839,
            -2.605978012084961,
            0.4316602051258087,
            3.4952261447906494,
            -0.7337121367454529,
        )
        corrected = CONTROL_RESOLUTION_PROTOCOL.rollback_feedback_command(
            CONTROL_RESOLUTION_PROTOCOL.probe_plan(3),
            reference.joint_positions,
            command,
            stalled,
            0.5,
        )
        np.testing.assert_allclose(
            corrected.target.joint_positions,
            tuple(
                drive + desired - actual
                for drive, desired, actual in zip(
                    target.joint_positions,
                    reference.joint_positions,
                    stalled,
                )
            ),
        )
        self.assertLessEqual(
            max(
                abs(desired - actual)
                for desired, actual in zip(
                    reference.joint_positions,
                    stalled,
                )
            ),
            0.002,
        )
        correction = ControlResolutionRollbackCorrection(
            ControlResolutionSettlementAttempt(
                requested_joint_motion_radians=max(
                    abs(start - desired)
                    for start, desired in zip(
                        forward_actual,
                        reference.joint_positions,
                    )
                ),
                required_tracking_error_radians=0.0005,
                tracking_errors_radians=(0.0017812550067901611,)
                * CONTROL_RESOLUTION_PROTOCOL.settlement.maximum_updates,
                final_joint_positions=stalled,
            ),
            corrected,
            ControlResolutionMotionTiming(34.5, 47.0),
        )
        correction.validate(
            CONTROL_RESOLUTION_PROTOCOL,
            CONTROL_RESOLUTION_PROTOCOL.probe_plan(3),
            forward_actual,
            reference.joint_positions,
            command,
            0.5,
        )
        self.assertEqual(
            ControlResolutionRollbackCorrection.from_dict(
                correction.to_dict()
            ),
            correction,
        )
        legacy = ControlResolutionDriveBiasCompensation.from_dict(
            {"maximum_bias_radians": 0.002}
        )
        self.assertFalse(legacy.path_dependent_rollback)

    def test_translation_rollback_settles_against_stable_reference(self) -> None:
        reference = _reset()
        drive_target = ControlResolutionDriveTarget(
            (
                reference.joint_positions[0] + 1e-3,
                *reference.joint_positions[1:],
            ),
            0.04,
        )
        probe = CONTROL_RESOLUTION_PROTOCOL.probe_plan(3)

        self.assertEqual(
            probe.rollback_joint_target(drive_target, reference),
            reference.joint_positions,
        )
        self.assertNotEqual(
            probe.rollback_joint_target(drive_target, reference),
            drive_target.joint_positions,
        )

    def test_legacy_settlement_protocol_omits_rollback_tracking_cap(self) -> None:
        payload = CONTROL_RESOLUTION_PROTOCOL.to_dict()
        payload["tracked_error_settlement"].pop(
            "rollback_tracking_error_cap_radians"
        )

        restored = ControlResolutionProtocol.from_dict(payload)

        self.assertIsNone(
            restored.settlement.rollback_tracking_error_cap_radians
        )
        self.assertNotIn(
            "rollback_tracking_error_cap_radians",
            restored.to_dict()["tracked_error_settlement"],
        )

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
            0.5,
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
            drive_bias_compensation=None,
        )
        samples = tuple(
            replace(
                _sample(index, magnitude),
                tracked_settlement=None,
                motion_timing=None,
                drive_target_joint_positions=None,
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
            drive_bias_compensation=None,
        )
        with self.assertRaisesRegex(ValueError, "legacy.*capture identity"):
            replace(failure, protocol=legacy_protocol)

    def test_projection_rejection_persists_exact_pre_actuation_attempt(self) -> None:
        reference = _reset()
        probe = CONTROL_RESOLUTION_PROTOCOL.probe_plan(3)
        recorded_target = DroidPose(
            (0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
        )
        commanded = CONTROL_RESOLUTION_PROTOCOL.probe_action(
            reference.pose,
            recorded_target,
            probe.requested_translation_meters,
        )
        target = reference.pose.applied(commanded)
        proposed_joints = (
            reference.joint_positions[0] + 0.3,
            *reference.joint_positions[1:],
        )
        projection_failure = ControlResolutionProjectionFailure(
            probe=probe,
            recorded_target_pose=recorded_target,
            start_reset=reference,
            commanded_action=commanded,
            target_pose=target,
            projection=SafetyProjectionAttempt(
                DroidActionScale.uniform(1.0),
                ControlGateDecision(
                    123,
                    target,
                    (ControlGateReason.JOINT_VELOCITY_VIOLATION,),
                ),
                0.3,
                proposed_joints,
            ),
            motion_period_seconds=0.5,
        )
        failure = resolution_failure_evidence(
            "resolution-attached-52600-c43",
            CONTROL_RESOLUTION_PROTOCOL,
            reference,
            tuple(_sample(index, 0.0) for index in range(3)),
            ControlResolutionProjectionRejected(projection_failure),
            ControlResolutionLoad.ATTACHED,
            _baseline(reference),
            _capture_identity(),
        )

        payload = failure.to_dict()
        restored = ControlResolutionFailureEvidence.from_dict(payload)

        self.assertEqual(restored, failure)
        self.assertEqual(
            restored.projection_failure.projection.gate.reasons,
            (ControlGateReason.JOINT_VELOCITY_VIOLATION,),
        )
        self.assertAlmostEqual(
            restored.projection_failure.projection.maximum_joint_delta_rad
            / restored.protocol.safety_limits.maximum_joint_velocity_radians_per_second,
            0.6,
        )
        runtime_rounding = failure.to_dict()
        runtime_rounding["projection_failure"]["target_pose"][0] += 2e-13
        runtime_rounding["projection_failure"]["projection"]["gate"][
            "next_pose"
        ][0] += 2e-13
        ControlResolutionFailureEvidence.from_dict(runtime_rounding)
        tampered_target = failure.to_dict()
        tampered_target["projection_failure"]["target_pose"][0] += 1e-10
        tampered_target["projection_failure"]["projection"]["gate"][
            "next_pose"
        ][0] += 1e-10
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(tampered_target)
        malformed = failure.to_dict()
        malformed["projection_failure"]["projection"][
            "maximum_joint_delta_rad"
        ] = 0.2
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(malformed)

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

        capture_failure = ControlResolutionCaptureFailureEvidence(
            identity=ControlResolutionCaptureAttemptIdentity(
                "resolution-attached-52600-c43",
                ControlResolutionCaptureSourceIdentity(
                    "contact-insertion-held-00",
                    52600,
                    43,
                ),
            ),
            failed_at_unix_seconds=123.0,
            contract=ControlResolutionCaptureBaselineContract(
                ControlResolutionBaselinePolicy(),
                CONTROL_RESOLUTION_PROTOCOL.safety_limits,
                ControlResolutionLoad.ATTACHED,
            ),
            baseline_attempt=attempt,
            error="UnstableControlResolutionBaseline: baseline did not stabilize",
        )

        capture_payload = capture_failure.to_dict()
        self.assertEqual(
            capture_payload["schema"],
            CONTROL_RESOLUTION_CAPTURE_FAILURE_SCHEMA,
        )
        self.assertEqual(
            ControlResolutionCaptureFailureEvidence.from_dict(capture_payload),
            capture_failure,
        )
        malformed = capture_failure.to_dict()
        malformed["identity"]["seed"] = True
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ControlResolutionCaptureFailureEvidence.from_dict(malformed)
        malformed = capture_failure.to_dict()
        malformed["failed_at_unix_seconds"] = "123.0"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ControlResolutionCaptureFailureEvidence.from_dict(malformed)

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
        runtime_rounding = failure.to_dict()
        runtime_rounding["completed_samples"][0]["actual_action"][3] += 3e-19
        self.assertEqual(
            ControlResolutionFailureEvidence.from_dict(runtime_rounding),
            failure,
        )
        tampered_action = failure.to_dict()
        tampered_action["completed_samples"][0]["actual_action"][3] += 1e-10
        with self.assertRaisesRegex(
            ValueError,
            "failure evidence is incomplete",
        ):
            ControlResolutionFailureEvidence.from_dict(tampered_action)
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
            baseline.reference_reset.joint_positions[0] + 1e-3,
            *baseline.reference_reset.joint_positions[1:],
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
                start_joint_positions=sample.start_reset.joint_positions,
                target_joint_positions=sample.start_reset.joint_positions,
                attempt=ControlResolutionSettlementAttempt(
                    requested_joint_motion_radians=0.0,
                    required_tracking_error_radians=5e-4,
                    tracking_errors_radians=(1e-3,)
                    * CONTROL_RESOLUTION_PROTOCOL.settlement.maximum_updates,
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
        runtime_rounding = failure.to_dict()
        runtime_rounding["settlement_failure"]["execution"]["target_pose"][0] += 3e-14
        runtime_rounding["settlement_failure"]["execution"]["projection"]["gate"][
            "next_pose"
        ][0] += 3e-14
        ControlResolutionFailureEvidence.from_dict(runtime_rounding)
        tampered_pose = failure.to_dict()
        tampered_pose["settlement_failure"]["execution"]["target_pose"][0] += 1e-10
        tampered_pose["settlement_failure"]["execution"]["projection"]["gate"][
            "next_pose"
        ][0] += 1e-10
        with self.assertRaisesRegex(ValueError, "failure evidence is incomplete"):
            ControlResolutionFailureEvidence.from_dict(tampered_pose)
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
                target_joint_positions=baseline.reference_reset.joint_positions,
                attempt=ControlResolutionSettlementAttempt(
                    requested_joint_motion_radians=0.0,
                    required_tracking_error_radians=5e-4,
                    tracking_errors_radians=(1e-3,)
                    * CONTROL_RESOLUTION_PROTOCOL.settlement.maximum_updates,
                    final_joint_positions=(
                        1e-3,
                        *baseline.reference_reset.joint_positions[1:],
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
            type("Attachment", (), {"attached": False})(),
            True,
            "insertion control resolution",
        )

        with self.assertRaisesRegex(RuntimeError, "attachment state changed"):
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
        runtime_rounding = report.to_dict()
        runtime_rounding["summary"]["responses"][1][
            "mean_realized_along_axis_meters"
        ] += 3e-19
        self.assertEqual(
            ControlResolutionReport.from_dict(runtime_rounding),
            report,
        )
        tampered_summary = report.to_dict()
        tampered_summary["summary"]["responses"][1][
            "mean_realized_along_axis_meters"
        ] += 1e-10
        with self.assertRaisesRegex(ValueError, "summary is inconsistent"):
            ControlResolutionReport.from_dict(tampered_summary)
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
        missing_probe_drive_target = report.to_dict()
        del missing_probe_drive_target["samples"][3][
            "drive_target_joint_positions"
        ]
        with self.assertRaisesRegex(ValueError, "report is incomplete"):
            ControlResolutionReport.from_dict(missing_probe_drive_target)
        tampered_probe_drive_target = report.to_dict()
        tampered_probe_drive_target["samples"][3][
            "drive_target_joint_positions"
        ][0] += 1e-6
        with self.assertRaisesRegex(ValueError, "report is incomplete"):
            ControlResolutionReport.from_dict(tampered_probe_drive_target)
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
                ControlResolutionDriveTarget.for_command(
                    reference.joint_positions,
                    0.04,
                ),
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
    @patch(
        "sim.isaac_control_resolution.stabilize_resolution_baseline",
        new_callable=AsyncMock,
    )
    async def test_capture_stabilizes_before_persisting_tracking_telemetry(
        self,
        stabilize: AsyncMock,
    ) -> None:
        command = JointCommand(np.zeros(7), 0.04)
        actual = JointCommand(
            np.asarray((2e-4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            0.039,
        )
        baseline = Mock()
        stabilize.return_value = (actual, baseline)
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(actual_command=Mock(return_value=actual)),
            attachment=SimpleNamespace(attached=True),
            sensor=object(),
        )
        capture = await stabilize_resolution_capture(
            runtime,
            command,
            ControlResolutionCaptureBaselineContract(
                ControlResolutionBaselinePolicy(),
                CONTROL_RESOLUTION_PROTOCOL.safety_limits,
                ControlResolutionLoad.UNLOADED,
            ),
        )

        self.assertEqual(capture.safety.arm_tracking_error_rad, 2e-4)
        self.assertAlmostEqual(capture.safety.gripper_tracking_error_m, 0.001)
        self.assertFalse(capture.safety.collision_detected)
        self.assertEqual(capture.safety.contact_force_newtons, 0.0)
        self.assertEqual(capture.previous_action, DroidAction((0.0,) * 7))
        stabilize.assert_awaited_once()
        self.assertFalse(stabilize.await_args.args[1].expected_attachment)
        baseline.validate.assert_called_once()

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

    async def test_rollback_tracking_cap_is_stricter_than_motion_floor(self) -> None:
        target = JointCommand(np.zeros(7), 0.04)
        actual_commands = (
            JointCommand(np.asarray((4.5e-4, *([0.0] * 6))), 0.04),
            JointCommand(np.asarray((3.5e-4, *([0.0] * 6))), 0.04),
            JointCommand(np.asarray((3e-4, *([0.0] * 6))), 0.04),
        )
        runtime = SimpleNamespace(
            actuators=SimpleNamespace(
                actual_command=Mock(side_effect=actual_commands)
            )
        )
        interlock = SimpleNamespace(observe=Mock())
        policy = TrackedErrorSettlement()

        with patch(
            "sim.isaac_control_resolution.advance_physics_updates",
            new=AsyncMock(),
        ):
            evidence = await settle_resolution_motion(
                runtime,
                target,
                target,
                interlock,
                policy,
                4e-4,
            )

        self.assertEqual(evidence.required_tracking_error_radians, 4e-4)
        self.assertEqual(evidence.updates_used, 3)
        self.assertEqual(
            evidence.passing_tracking_errors_radians,
            (3.5e-4, 3e-4),
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
            tracking_errors_radians=(1e-3,)
            * CONTROL_RESOLUTION_PROTOCOL.settlement.maximum_updates,
            final_joint_positions=tuple(start.arm_positions),
        )
        forged_gripper_gate = attempt.to_dict()
        forged_gripper_gate["gripper_settlement"] = GripperSettlementTrace(
            (1.0,) * CONTROL_RESOLUTION_PROTOCOL.settlement.maximum_updates,
            1e-3,
        ).to_dict()
        with self.assertRaisesRegex(ValueError, "unexpected gripper"):
            ControlResolutionSettlementAttempt.from_dict(forged_gripper_gate)
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
        self.assertEqual(settle.await_args.args[5], 5e-4)
        actuators.current_command.assert_called()

    def test_rollback_failure_validates_with_the_rollback_error_cap(self) -> None:
        protocol = CONTROL_RESOLUTION_PROTOCOL
        probe = protocol.probe_plan(protocol.repeats_per_magnitude)
        baseline = _baseline()
        drive_target = baseline.drive_target
        reference_reset = baseline.reference_reset
        rollback_target = probe.rollback_joint_target(
            drive_target,
            reference_reset,
        )
        start = (
            rollback_target[0] + 3e-3,
            *rollback_target[1:],
        )
        drive_command = probe.drive_command(
            protocol.safe_joint_motion_period(
                start,
                drive_target.joint_positions,
                protocol.motion_period_for(probe.requested_translation_meters),
            ),
            drive_target,
        )
        attempt = ControlResolutionSettlementAttempt(
            requested_joint_motion_radians=3e-3,
            required_tracking_error_radians=(
                protocol.settlement.rollback_tracking_error_cap_radians
            ),
            tracking_errors_radians=(3e-3,)
            * protocol.settlement.maximum_updates,
            final_joint_positions=start,
        )
        failure = ControlResolutionRollbackFailure(
            start_joint_positions=start,
            drive_command=drive_command,
            attempt=attempt,
            interlock=ControlInterlockEvidence(0.0, False),
            error="UnsettledControlResolutionTarget: bounded timeout",
        )

        failure.validate(
            protocol,
            probe,
            drive_target,
            reference_reset,
            expected_attachment=True,
            minimum_period_seconds=protocol.motion_period_for(
                probe.requested_translation_meters
            ),
        )

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
