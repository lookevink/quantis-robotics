from pathlib import Path
import unittest

from jepa_wm.action import DroidAction, DroidPose
from jepa_wm.control_protocol import (
    ControlObservation,
    ProposedControl,
)
from jepa_wm.control_safety import (
    ControlGateReason,
    SimulatorControlGate,
    SimulatorSafetyState,
)
from jepa_wm.control_tracking import (
    evaluate_action_tracking,
)


def _observation(**overrides) -> ControlObservation:
    values = {
        "observation_id": 1,
        "captured_at_unix_seconds": 100.0,
        "context_frame": Path("context.png"),
        "target_frame": Path("target.png"),
        "expected_proposal": Path("/tmp/proposal.pth"),
        "pose": DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
        "previous_action": DroidAction((0.0,) * 7),
        "warmup_frames": 4,
    }
    values.update(overrides)
    return ControlObservation(**values)


def _proposal(**overrides) -> ProposedControl:
    values = {
        "observation_id": 1,
        "created_at_unix_seconds": 100.1,
        "actions": (
            DroidAction((0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
            DroidAction((0.0,) * 7),
            DroidAction((0.0,) * 7),
        ),
        "proposal": Path("/tmp/proposal.pth"),
    }
    values.update(overrides)
    return ProposedControl(**values)


def _state(**overrides) -> SimulatorSafetyState:
    current = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)
    values = {
        "observed_joint_positions": current,
        "current_joint_positions": current,
        "proposed_joint_positions": (
            0.01,
            -0.5,
            0.0,
            -2.0,
            0.0,
            1.5,
            0.5,
        ),
        "control_period_seconds": 0.25,
        "contact_force_newtons": 0.0,
        "collision_detected": False,
    }
    values.update(overrides)
    return SimulatorSafetyState(**values)


class SimulatorControlGateTest(unittest.TestCase):
    def test_accepts_one_fresh_bounded_free_space_action(self) -> None:
        decision = SimulatorControlGate().evaluate(
            _observation(),
            _proposal(),
            _state(),
            now_unix_seconds=100.2,
        )

        self.assertTrue(decision.passed)
        self.assertAlmostEqual(decision.next_pose.values[0], 0.405)

    def test_rejects_a_different_checkpoint_or_pre_observation_response(self) -> None:
        decision = SimulatorControlGate().evaluate(
            _observation(),
            _proposal(
                created_at_unix_seconds=99.9,
                proposal=Path("/tmp/other.pth"),
            ),
            _state(),
            now_unix_seconds=100.2,
        )

        self.assertIn(ControlGateReason.PROPOSAL_MISMATCH, decision.reasons)
        self.assertIn(ControlGateReason.COMMAND_TIME_INVALID, decision.reasons)

    def test_rejects_joint_state_that_drifted_from_the_observation(self) -> None:
        observed = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)
        current = (0.003, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5)
        decision = SimulatorControlGate().evaluate(
            _observation(),
            _proposal(),
            _state(
                observed_joint_positions=observed,
                current_joint_positions=current,
            ),
            now_unix_seconds=100.2,
        )

        self.assertIn(ControlGateReason.OBSERVATION_STATE_DRIFT, decision.reasons)

    def test_fails_closed_across_warmup_workspace_joint_contact_and_force(self) -> None:
        decision = SimulatorControlGate().evaluate(
            _observation(
                warmup_frames=3,
                pose=DroidPose((0.849, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
            ),
            _proposal(
                observation_id=2,
                actions=(DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),)
                * 3,
            ),
            _state(
                proposed_joint_positions=(3.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5),
                collision_detected=True,
                contact_force_newtons=3.0,
            ),
            now_unix_seconds=104.0,
        )

        self.assertEqual(
            set(decision.reasons),
            {
                ControlGateReason.STALE_OBSERVATION,
                ControlGateReason.OBSERVATION_MISMATCH,
                ControlGateReason.COMMAND_TIME_INVALID,
                ControlGateReason.WARMUP_INCOMPLETE,
                ControlGateReason.WORKSPACE_VIOLATION,
                ControlGateReason.JOINT_LIMIT_VIOLATION,
                ControlGateReason.JOINT_VELOCITY_VIOLATION,
                ControlGateReason.COLLISION_DETECTED,
                ControlGateReason.FORCE_LIMIT_EXCEEDED,
            },
        )

    def test_allows_the_measured_paused_simulator_handoff_budget(self) -> None:
        decision = SimulatorControlGate().evaluate(
            _observation(),
            _proposal(created_at_unix_seconds=100.5),
            _state(),
            now_unix_seconds=102.9,
        )

        self.assertTrue(decision.passed)

    def test_round_trips_versioned_observation_and_proposal(self) -> None:
        observation = _observation()
        proposal = _proposal()

        self.assertEqual(ControlObservation.from_dict(observation.to_dict()), observation)
        self.assertEqual(ProposedControl.from_dict(proposal.to_dict()), proposal)

    def test_rejects_cartesian_motion_in_the_wrong_direction(self) -> None:
        tracking = evaluate_action_tracking(
            DroidAction((0.001, 0.0, 0.0, 0.0, 0.01, 0.0, 0.1)),
            DroidAction((-0.0002, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.1)),
        )

        self.assertFalse(tracking.passed)
        self.assertLess(tracking.translation_cosine, 0.0)
        self.assertIn("translation_direction", tracking.to_dict()["reasons"])

    def test_scales_the_complete_action_without_changing_its_direction(self) -> None:
        action = DroidAction((0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.2))

        scaled = action.scaled(0.25)

        self.assertEqual(scaled.values, tuple(value * 0.25 for value in action.values))


if __name__ == "__main__":
    unittest.main()
