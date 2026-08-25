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
    ControlGateDecision,
    ORIENTATION_HOLD_ACTION_SCALES,
    SafetyProjectionAttempt,
)
from jepa_wm.direct_safety import (
    ControlSafetySnapshot,
    DirectInsertionSafetyEvidence,
)
from jepa_wm.insertion_trial import (
    InsertionTrialBinding,
    InsertionTrialSourceEvidence,
    build_insertion_trial_response,
)
from jepa_wm.training_artifact import ArtifactIdentity
from jepa_wm.trial_equivalence import ControlTrialContext, TrialResetState
from sim.control_session import ControlSession, ControlSessionState


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
    )
    return InsertionTrialSourceEvidence(
        _context(9, ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION),
        response,
        safety,
    )


class InsertionTrialBindingTest(unittest.TestCase):
    def test_rebinds_one_exact_passing_source_to_an_equivalent_reset(self) -> None:
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
