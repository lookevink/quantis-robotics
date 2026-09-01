from dataclasses import replace
from math import nan
from pathlib import Path
import unittest

from sim.unknown_start_shadow import (
    UnknownStartControlHandoff,
    unknown_start_handoff_failures,
    validate_unknown_start_handoff,
)
from tests.test_unknown_start_reset import valid_evidence


class UnknownStartShadowHandoffTest(unittest.TestCase):
    def test_recovery_diagnostic_is_paused_and_read_only(self) -> None:
        source = Path("sim/isaac_unknown_start_recovery.py").read_text()
        diagnostic = source.split(
            "async def diagnose_unknown_start_candidate_rollback", 1
        )[1].split("async def recover_unknown_start_candidate_rollback", 1)[0]

        self.assertIn("await pause_control_timeline", diagnostic)
        self.assertIn("get_dof_velocities()", diagnostic)
        self.assertIn("compute_forward_kinematics(", diagnostic)
        self.assertNotIn("set_reset_state(", diagnostic)
        self.assertNotIn("resume_live_simulation(", diagnostic)
        self.assertNotIn("advance_physics_updates(", diagnostic)

    def test_live_candidate_reuses_authenticated_capture_and_reauthenticates(
        self,
    ) -> None:
        capture = Path("sim/isaac_unknown_start_shadow.py").read_text()
        recovery_source = Path("sim/isaac_unknown_start_recovery.py").read_text()
        execution = Path("sim/isaac_control_execution.py").read_text()

        self.assertIn("capture_unknown_start_candidate_observation", capture)
        self.assertIn("ControlExecutionPolicy.RESET_TRIAL_CANDIDATE", capture)
        self.assertLess(
            execution.index("reauthenticate_unknown_start_shadow_session(session_id)"),
            execution.index("session.claim_execution()"),
        )
        recovery = recovery_source.index(
            "async def recover_unknown_start_candidate_rollback"
        )
        recovery_imports = recovery_source[recovery:].split(
            "session = ControlSession.at", 1
        )[0]
        self.assertIn(
            "refresh_paused_live_control_articulation,",
            recovery_imports,
        )
        self.assertLess(
            recovery_source.index("await pause_control_timeline", recovery),
            recovery_source.index("runtime = live_runtime_for", recovery),
        )
        self.assertLess(
            recovery_source.index(
                "await refresh_paused_live_control_articulation", recovery
            ),
            recovery_source.index("await rollback_control_command", recovery),
        )
        self.assertLess(
            recovery_source.index("await rollback_control_command", recovery),
            recovery_source.index(
                "runtime.actuators.set_reset_state(", recovery
            ),
        )
        self.assertIn("allowed_active_targets = (", recovery_source)
        self.assertIn("state.active_drive_target,", recovery_source)
        self.assertIn(
            "unknown-start recovery active drive target changed", recovery_source
        )
        self.assertIn(
            "drive_target=reset_drive_target",
            recovery_source,
        )
        self.assertIn(
            "await RenderingManager.render_async()",
            recovery_source,
        )
        self.assertLess(
            recovery_source.index(
                "runtime.actuators.set_reset_state(", recovery
            ),
            recovery_source.index(
                "await RenderingManager.render_async()", recovery
            ),
        )
        self.assertLess(
            recovery_source.index(
                "await RenderingManager.render_async()", recovery
            ),
            recovery_source.index(
                "reauthenticate_unknown_start_shadow_session(session_id)",
                recovery,
            ),
        )
        self.assertIn(
            "physics_simulation_time_seconds() != initialization_time",
            recovery_source,
        )
        runtime = Path("sim/isaac_demo_runtime.py").read_text()
        reset = runtime.split("def set_reset_state", 1)[1].split("\n\n\n", 1)[0]
        self.assertIn("self.articulation.set_dof_velocities(", reset)

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

        self.assertEqual(
            UnknownStartControlHandoff.from_dict(handoff.to_dict()), handoff
        )
        with self.assertRaisesRegex(ValueError, "payload"):
            UnknownStartControlHandoff.from_dict(
                {**handoff.to_dict(), "apply_action": True}
            )
        with self.assertRaisesRegex(ValueError, "invalid"):
            UnknownStartControlHandoff(**{**handoff.__dict__, "applied_actions": 1})

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
        self.assertIn("def _joint_kinematic_snapshot(", source)
        self.assertIn('"panda_hand", joints', source)
        self.assertIn('"right_gripper", joints', source)
        self.assertIn("snapshot = _joint_kinematic_snapshot(", source)
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
            "gripper_frame_position_m": (workspace.gripper_control_frame_position_m),
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
        self.assertEqual(
            unknown_start_handoff_failures(
                evidence,
                **{**arguments, "contact_force_newtons": 1e-9},
            ),
            ("contact_force_zero",),
        )


if __name__ == "__main__":
    unittest.main()
