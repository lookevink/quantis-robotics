from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation, ControlTarget, ProposedControl
from jepa_wm.control_safety import (
    ACTION_SCALES,
    ControlInterlockEvidence,
    ControlGateDecision,
    ControlGateReason,
    ORIENTATION_HOLD_ACTION_SCALES,
    SafetyProjectionAttempt,
    SimulatorSafetyLimits,
)
from jepa_wm.direct_safety import (
    ControlSafetySnapshot,
    DirectInsertionSafetyEvidence,
)
from jepa_wm.insertion_trial import (
    INSERTION_TRIAL_SETTLEMENT_MAXIMUM_UPDATES,
    InsertionTrialAuthority,
    InsertionTrialBinding,
    InsertionTrialExecutionEvidence,
    InsertionTrialExecutionRefresh,
    InsertionTrialPolicy,
    InsertionTrialRollbackFailure,
    InsertionTrialRollbackFailureReason,
    InsertionTrialSourceEvidence,
    build_insertion_trial_response,
)
from jepa_wm.insertion_contract import INSERTION_CONTROL_TARGET_POLICY
from jepa_wm.joint_settlement import JointSettlementAttempt
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.trial_equivalence import ControlTrialContext, TrialResetState
from sim.control_session import (
    ControlResult,
    ControlResultStatus,
    ControlSession,
    ControlSessionState,
)
from sim.isaac_insertion_trial import build_insertion_followup_execution_capture


_FINGERPRINT = "a" * 64
_PROPOSAL = Path("/tmp/proposal.pth")
_JOINTS = (0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0)
_ACTIONS = (DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),) * 3


def _observation(identifier: int) -> ControlObservation:
    return ControlObservation(
        identifier,
        100.0,
        Path("context.png"),
        ControlTarget(
            Path("target.png"),
            DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04)),
        ),
        _PROPOSAL,
        DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04)),
        DroidAction((0.0,) * 7),
        43,
    )


def _context(
    identifier: int,
    policy: ControlExecutionPolicy,
) -> ControlTrialContext:
    observation = _observation(identifier)
    return ControlTrialContext(
        observation,
        TrialResetState(
            observation.pose,
            _JOINTS,
            False,
            0.0,
            (0.4, 0.0, 0.5),
            True,
        ),
        policy,
        "insertion-held-00",
        52600,
        None,
    )


def _source() -> InsertionTrialSourceEvidence:
    response = ProposedControl(9, 100.1, _ACTIONS, _PROPOSAL, _FINGERPRINT)
    attempt = SafetyProjectionAttempt(
        ACTION_SCALES[0],
        ControlGateDecision(9, _observation(9).pose.applied(_ACTIONS[0]), ()),
        0.0,
        _JOINTS,
    )
    safety = DirectInsertionSafetyEvidence(
        9,
        100.2,
        _ACTIONS,
        ArtifactIdentity(_PROPOSAL, _FINGERPRINT),
        (attempt,),
        attempt.scale,
        ControlSafetySnapshot(
            _JOINTS,
            0.04,
            (0.4, 0.0, 0.5),
            0.0,
            False,
            True,
        ),
        JointDriveTarget(_JOINTS, 0.04),
    )
    return InsertionTrialSourceEvidence(
        _context(9, ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION),
        response,
        safety,
        JointDriveTarget(_JOINTS, 0.04),
    )


class InsertionTrialBindingTest(unittest.TestCase):
    def test_policy_compensates_one_bounded_loaded_drive_bias(self) -> None:
        policy = InsertionTrialPolicy()
        desired = tuple(value + 0.01 for value in _JOINTS)
        target = policy.forward_drive_target(
            desired,
            0.04,
            JointDriveTarget(_JOINTS, 0.04),
            tuple(value + 0.001 for value in _JOINTS),
            SimulatorSafetyLimits(),
        )

        for actual, expected in zip(target.joint_positions, desired):
            self.assertAlmostEqual(actual, expected - 0.001, places=7)

        with self.assertRaisesRegex(ValueError, "bias exceeds"):
            policy.forward_drive_target(
                desired,
                0.04,
                JointDriveTarget(_JOINTS, 0.04),
                tuple(value + 0.003 for value in _JOINTS),
                SimulatorSafetyLimits(),
            )

        with self.assertRaisesRegex(ValueError, "velocity gate"):
            policy.forward_drive_target(
                tuple(value + 0.2 for value in _JOINTS),
                0.04,
                JointDriveTarget(_JOINTS, 0.04),
                _JOINTS,
                SimulatorSafetyLimits(),
            )

    def test_control_result_round_trips_typed_settlement_and_rollback_failures(self) -> None:
        gate = ControlGateDecision(10, _observation(10).pose.applied(_ACTIONS[0]), ())
        attempt = SafetyProjectionAttempt(ACTION_SCALES[0], gate, 0.002, _JOINTS)
        settlement_failure = JointSettlementAttempt(
            0.002,
            0.0005,
            (0.001,) * 32,
            _JOINTS,
        )
        rollback_failure = InsertionTrialRollbackFailure(
            _JOINTS,
            _JOINTS,
            _JOINTS,
            True,
            InsertionTrialRollbackFailureReason.DRIVE_COMMAND_REJECTED,
            ControlInterlockEvidence(0.0, False),
            False,
            "RuntimeError: rollback did not settle",
            None,
            JointDriveTarget(_JOINTS, 0.04),
        )
        result = ControlResult(
            ControlResultStatus.ROLLBACK_FAILED,
            "insertion-trial",
            gate,
            (attempt,),
            ACTION_SCALES[0],
            0.1,
            0.0,
            0.0,
            0.0,
            execution_error="RuntimeError: motion and rollback failed",
            insertion_trial_rollback=rollback_failure,
            insertion_trial_settlement_failure=settlement_failure,
        )

        self.assertEqual(ControlResult.from_dict(result.to_dict()), result)
        contradictory = result.to_dict()
        contradictory["insertion_trial_rollback"]["joint_settlement"] = {}
        with self.assertRaisesRegex(ValueError, "incomplete"):
            ControlResult.from_dict(contradictory)

    def test_refreshes_only_timing_after_live_state_continuity(self) -> None:
        captured = _observation(10)
        response = ProposedControl(
            10,
            100.1,
            _ACTIONS,
            _PROPOSAL,
            _FINGERPRINT,
        )
        captured_state = ControlSafetySnapshot(
            _JOINTS,
            0.04,
            (0.4, 0.0, 0.5),
            0.0,
            False,
            True,
        )
        live_pose = DroidPose((0.4001, 0.0, 0.5, 0.0, 0.0, 0.0, 0.04))
        refresh = InsertionTrialExecutionRefresh(107.5, captured_state, live_pose)

        observation, authorized = refresh.authorize(
            captured,
            response,
            captured_state,
        )

        self.assertEqual(observation.captured_at_unix_seconds, 107.5)
        self.assertEqual(authorized.created_at_unix_seconds, 107.5)
        self.assertEqual(observation.pose, live_pose)
        self.assertEqual(authorized.actions, response.actions)
        self.assertEqual(
            InsertionTrialExecutionRefresh.from_dict(refresh.to_dict()),
            refresh,
        )
        blocked_gate = ControlGateDecision(
            captured.observation_id,
            captured.pose,
            (ControlGateReason.STALE_OBSERVATION,),
        )
        result = ControlResult(
            ControlResultStatus.BLOCKED,
            "insertion-trial",
            blocked_gate,
            (
                SafetyProjectionAttempt(
                    ACTION_SCALES[0],
                    blocked_gate,
                    0.0,
                    _JOINTS,
                ),
            ),
            None,
            0.1,
            None,
            None,
            0.0,
            insertion_trial_refresh=refresh,
        )
        self.assertEqual(ControlResult.from_dict(result.to_dict()), result)

        drifted = replace(captured_state, joint_positions=(0.01, *_JOINTS[1:]))
        with self.assertRaisesRegex(ValueError, "changed after capture"):
            InsertionTrialExecutionRefresh(107.5, drifted).authorize(
                captured,
                response,
                captured_state,
            )

    def test_rebinds_one_exact_passing_source_to_an_equivalent_reset(self) -> None:
        with self.assertRaisesRegex(ValueError, "binding is invalid"):
            build_insertion_trial_response(
                execution_session_id="insertion-trial",
                source_session_id="insertion-safety",
                execution=_context(
                    10,
                    ControlExecutionPolicy.INSERTION_RESET_TRIAL,
                ),
                source=replace(_source(), active_drive_target=None),
                created_at_unix_seconds=101.0,
            )

        binding, response = build_insertion_trial_response(
            execution_session_id="insertion-trial",
            source_session_id="insertion-safety",
            execution=_context(10, ControlExecutionPolicy.INSERTION_RESET_TRIAL),
            source=_source(),
            created_at_unix_seconds=101.0,
        )

        self.assertEqual(response.observation_id, 10)
        self.assertEqual(response.proposal_fingerprint, _FINGERPRINT)
        self.assertEqual(response.actions, _ACTIONS)
        self.assertEqual(InsertionTrialBinding.from_dict(binding.to_dict()), binding)
        self.assertFalse(binding.production_authority_granted)
        self.assertIsNotNone(binding.trial_policy)
        self.assertIs(binding.require_current_execution(), binding.trial_policy)
        self.assertEqual(
            binding.trial_policy.joint_settlement.maximum_updates,
            INSERTION_TRIAL_SETTLEMENT_MAXIMUM_UPDATES,
        )
        self.assertEqual(binding.trial_policy.control_period_seconds, 0.25)
        with self.assertRaisesRegex(ValueError, "control period"):
            replace(binding.trial_policy, control_period_seconds=0.5)

        prior_policy_payload = binding.to_dict()
        del prior_policy_payload["trial_policy"]["drive_bias_compensation"]
        prior_binding = InsertionTrialBinding.from_dict(prior_policy_payload)
        self.assertIsNone(prior_binding.trial_policy.drive_bias_compensation)
        with self.assertRaisesRegex(ValueError, "legacy insertion trial"):
            prior_binding.require_current_execution()

        for invalid_gripper_limit in (True, 0.1):
            invalid_policy = binding.to_dict()
            invalid_policy["trial_policy"][
                "rollback_gripper_error_meters"
            ] = invalid_gripper_limit
            with self.assertRaises(ValueError):
                InsertionTrialBinding.from_dict(invalid_policy)

        for invalid_bias_bound in (True, 0.003):
            invalid_policy = binding.to_dict()
            invalid_policy["trial_policy"]["drive_bias_compensation"][
                "maximum_bias_radians"
            ] = invalid_bias_bound
            with self.assertRaises(ValueError):
                InsertionTrialBinding.from_dict(invalid_policy)

        legacy_payload = binding.to_dict()
        del legacy_payload["trial_policy"]
        self.assertIsNone(
            InsertionTrialBinding.from_dict(legacy_payload).trial_policy
        )

        capped = replace(
            binding,
            source_selected_action_scale=ACTION_SCALES[1],
        )
        self.assertEqual(capped.allowed_projection_scales, ACTION_SCALES[1:])
        capped.validate_attempted_projection_scales((ACTION_SCALES[1],))
        with self.assertRaisesRegex(ValueError, "exceeded"):
            capped.validate_attempted_projection_scales((ACTION_SCALES[0],))

        held = replace(
            binding,
            source_selected_action_scale=ORIENTATION_HOLD_ACTION_SCALES[0],
        )
        self.assertEqual(
            held.allowed_projection_scales,
            ORIENTATION_HOLD_ACTION_SCALES,
        )

    def test_rebinds_a_followup_source_only_to_one_followup_execution(self) -> None:
        predecessor = "insertion-trial-predecessor"
        source = replace(
            _source(),
            context=replace(_source().context, previous_session_id=predecessor),
        )
        execution = replace(
            _context(10, ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL),
            previous_session_id=predecessor,
        )

        binding, response = build_insertion_trial_response(
            execution_session_id="insertion-followup-trial",
            source_session_id="insertion-followup-safety",
            execution=execution,
            source=source,
            created_at_unix_seconds=101.0,
        )

        self.assertEqual(
            binding.authority,
            InsertionTrialAuthority.FOLLOWUP_TRIAL_ONLY,
        )
        self.assertEqual(response.observation_id, execution.observation.observation_id)
        with self.assertRaisesRegex(ValueError, "not bound"):
            binding.validate_execution(
                source,
                replace(
                    InsertionTrialExecutionEvidence(execution, response),
                    context=replace(execution, previous_session_id="different-trial"),
                ),
            )

    def test_clones_one_no_actuation_followup_capture_without_resetting_it(self) -> None:
        source_observation = _observation(9)
        source_state = ControlSessionState(
            "insertion-followup-safety",
            "insertion-held-00",
            52600,
            "control-insertion-followup-safety",
            _JOINTS,
            False,
            0.0,
            previous_session_id="insertion-trial-predecessor",
            execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            plug_position=(0.4, 0.0, 0.5),
            plug_attached=True,
            current_gripper_width_m=0.04,
            insertion_target_policy=INSERTION_CONTROL_TARGET_POLICY,
            active_drive_target=JointDriveTarget(_JOINTS, 0.04),
        )

        observation, state = build_insertion_followup_execution_capture(
            "insertion-followup-trial",
            source_observation,
            source_state,
        )

        self.assertNotEqual(observation.observation_id, source_observation.observation_id)
        self.assertEqual(observation.pose, source_observation.pose)
        self.assertEqual(observation.context_frame, source_observation.context_frame)
        self.assertEqual(state.previous_session_id, source_state.previous_session_id)
        self.assertEqual(state.active_drive_target, source_state.active_drive_target)
        self.assertEqual(
            state.execution_policy,
            ControlExecutionPolicy.INSERTION_FOLLOWUP_TRIAL,
        )

    def test_rejects_reset_drift_and_non_trial_execution_policy(self) -> None:
        execution = _context(10, ControlExecutionPolicy.INSERTION_RESET_TRIAL)
        drifted_reset = replace(
            execution.reset,
            joint_positions=(0.01, *_JOINTS[1:]),
        )
        for invalid in (
            replace(execution, reset=drifted_reset),
            replace(execution, policy=ControlExecutionPolicy.DIRECT),
        ):
            with self.subTest(policy=invalid.policy):
                with self.assertRaises(ValueError):
                    build_insertion_trial_response(
                        execution_session_id="insertion-trial",
                        source_session_id="insertion-safety",
                        execution=invalid,
                        source=_source(),
                        created_at_unix_seconds=101.0,
                    )

    def test_rejects_tampered_production_authority(self) -> None:
        binding, _ = build_insertion_trial_response(
            execution_session_id="insertion-trial",
            source_session_id="insertion-safety",
            execution=_context(10, ControlExecutionPolicy.INSERTION_RESET_TRIAL),
            source=_source(),
            created_at_unix_seconds=101.0,
        )
        payload = binding.to_dict()
        payload["production_authority_granted"] = True

        with self.assertRaises(ValueError):
            InsertionTrialBinding.from_dict(payload)

    @patch("sim.control_session.validate_observation_target")
    def test_session_requires_source_evidence_and_equivalent_reset_before_claim(
        self,
        _validate_target,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            control_root = data_root / "control_sessions"
            recording = data_root / "recordings" / "insertion-held-00"
            recording.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps({"metadata": {"task": "reach_and_insert"}})
            )
            source = ControlSession.at(control_root, "insertion-safety")
            source_observation = _observation(9)
            source_state = ControlSessionState(
                "insertion-safety",
                "insertion-held-00",
                52600,
                "control-insertion-safety",
                _JOINTS,
                False,
                0.0,
                execution_policy=ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
                plug_position=(0.4, 0.0, 0.5),
                plug_attached=True,
                current_gripper_width_m=0.04,
                active_drive_target=JointDriveTarget(_JOINTS, 0.04),
            )
            source_evidence = _source()
            source.write_capture(source_observation, source_state)
            source.write_response(source_evidence.response)
            source.write_direct_safety(source_evidence.safety)
            (recording / "manifest.json").write_text(
                json.dumps({"metadata": {"task": "reach_and_grasp"}})
            )
            with self.assertRaisesRegex(ValueError, "insertion evidence"):
                source.load_insertion_trial_source_evidence()
            (recording / "manifest.json").write_text(
                json.dumps({"metadata": {"task": "reach_and_insert"}})
            )
            loaded_source = source.load_insertion_trial_source_evidence()

            trial = ControlSession.at(control_root, "insertion-trial")
            trial_observation = _observation(10)
            trial_state = replace(
                source_state,
                session_id="insertion-trial",
                recording="control-insertion-trial",
                execution_policy=ControlExecutionPolicy.INSERTION_RESET_TRIAL,
            )
            trial.write_capture(trial_observation, trial_state)
            execution = trial.trial_context(trial_observation, trial_state)
            binding, response = build_insertion_trial_response(
                execution_session_id=trial.session_id,
                source_session_id=source.session_id,
                execution=execution,
                source=loaded_source,
                created_at_unix_seconds=100.3,
            )
            trial.write_insertion_trial_binding(binding, loaded_source)
            trial.write_response(response)

            self.assertEqual(trial.load()[1], response)
            trial.claim_execution()
            self.assertTrue(trial.execution_path.exists())
            with self.assertRaisesRegex(ValueError, "cannot be executed"):
                source.claim_execution()


if __name__ == "__main__":
    unittest.main()
