from dataclasses import replace
from math import nan
from pathlib import Path
import unittest

from sim.unknown_start_shadow import (
    UnknownStartControlHandoff,
    validate_unknown_start_handoff,
)
from tests.test_unknown_start_reset import valid_evidence


class UnknownStartShadowHandoffTest(unittest.TestCase):
    def test_handoff_wire_contract_rejects_added_authority(self) -> None:
        handoff = UnknownStartControlHandoff(
            session_id="shadow-session",
            reset_recording_id="reset-recording",
            reset_seed=62604,
            reset_result_fingerprint="a" * 64,
            reset_evidence_fingerprint="b" * 64,
            reset_contract_fingerprint="c" * 64,
            reference_recording="held-00",
            reference_seed=12600,
            context_fingerprint="d" * 64,
            routing_target_fingerprint="e" * 64,
            routing_step_fingerprint="1" * 64,
            request_fingerprint="f" * 64,
            state_fingerprint="0" * 64,
        )

        self.assertEqual(UnknownStartControlHandoff.from_dict(handoff.to_dict()), handoff)
        with self.assertRaisesRegex(ValueError, "payload"):
            UnknownStartControlHandoff.from_dict(
                {**handoff.to_dict(), "apply_action": True}
            )
        with self.assertRaisesRegex(ValueError, "invalid"):
            UnknownStartControlHandoff(
                **{**handoff.__dict__, "applied_actions": 1}
            )

    def test_exact_reset_state_can_create_only_a_zero_action_handoff(self) -> None:
        evidence = valid_evidence()
        workspace = evidence.workspace

        validate_unknown_start_handoff(
            evidence,
            arm_positions=evidence.observed_arm_positions_radians,
            gripper_width_m=evidence.observed_gripper_width_m,
            connector_position_m=workspace.connector_position_m,
            socket_position_m=workspace.socket_position_m,
            gripper_frame_position_m=workspace.gripper_control_frame_position_m,
            connector_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            expected_connector_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            socket_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            expected_socket_orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            camera_offset_m=evidence.realization.camera_offset_m,
            socket_scale=workspace.socket_scale,
            light_exposure_deltas=(evidence.realization.light_exposure_delta,),
            plug_attached=False,
            collision_detected=False,
            contact_force_newtons=0.0,
        )

        source = Path("sim/isaac_unknown_start_shadow.py").read_text()
        self.assertNotIn("set_reset_state(", source)
        self.assertNotIn("apply_control_response", source)
        self.assertNotIn("move_joint_command", source)
        self.assertNotIn("timeline.play", source)
        self.assertNotIn("advance_physics", source)
        self.assertIn("pause_control_timeline(", source)
        self.assertNotIn("timeline.stop()", source)
        self.assertIn('"applied_actions": 0', source)
        self.assertIn("UnknownStartControlHandoff(", source)
        self.assertIn("zero_action = DroidAction((0.0,) * 7)", source)
        self.assertIn('"unknown_start_handoff.json"', source)
        self.assertIn("bind_existing_fixed_joint_plug", source)
        self.assertNotIn("prepare_fixed_joint_plug", source)
        runtime = Path("sim/isaac_demo_runtime.py").read_text()
        binder = runtime.split("def bind_existing_fixed_joint_plug", 1)[1]
        self.assertNotIn(".Set(", binder)
        self.assertNotIn(".Create", binder)

    def test_handoff_fails_closed_on_drift_or_contact(self) -> None:
        evidence = valid_evidence()
        workspace = evidence.workspace
        arguments = {
            "arm_positions": evidence.observed_arm_positions_radians,
            "gripper_width_m": evidence.observed_gripper_width_m,
            "connector_position_m": workspace.connector_position_m,
            "socket_position_m": workspace.socket_position_m,
            "gripper_frame_position_m": (
                workspace.gripper_control_frame_position_m
            ),
            "connector_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "expected_connector_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "socket_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "expected_socket_orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "camera_offset_m": evidence.realization.camera_offset_m,
            "socket_scale": workspace.socket_scale,
            "light_exposure_deltas": (evidence.realization.light_exposure_delta,),
            "plug_attached": False,
            "collision_detected": False,
            "contact_force_newtons": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "drifted"):
            validate_unknown_start_handoff(
                evidence,
                **{
                    **arguments,
                    "arm_positions": (
                        evidence.observed_arm_positions_radians[0] + 1.1e-4,
                        *evidence.observed_arm_positions_radians[1:],
                    ),
                },
            )
        with self.assertRaisesRegex(ValueError, "drifted"):
            validate_unknown_start_handoff(
                replace(evidence),
                **{**arguments, "contact_force_newtons": 1e-9},
            )
        with self.assertRaisesRegex(ValueError, "drifted"):
            validate_unknown_start_handoff(
                evidence,
                **{**arguments, "arm_positions": (0.0,) * 6},
            )
        with self.assertRaisesRegex(ValueError, "drifted"):
            validate_unknown_start_handoff(
                evidence,
                **{**arguments, "gripper_width_m": nan},
            )


if __name__ == "__main__":
    unittest.main()
