from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from jepa_wm.contact_grasp_rotation_resolution import (
    ROLLED_BACK_SESSION_ID,
    SOURCE_SESSION_ID,
    ContactGraspRotationResolution,
    rollback_drive_target,
    runtime_fingerprint,
)


class ContactGraspRotationResolutionTest(unittest.TestCase):
    def test_authority_round_trips_without_filming(self) -> None:
        authority = ContactGraspRotationResolution(
            "unknown-start-e2e-v18-62605-grasp-01",
            "a" * 64,
            "b" * 40,
        )

        payload = authority.to_dict()

        self.assertEqual(ContactGraspRotationResolution.from_dict(payload), authority)
        self.assertEqual(payload["source_session_id"], SOURCE_SESSION_ID)
        self.assertEqual(payload["rolled_back_session_id"], ROLLED_BACK_SESSION_ID)
        self.assertTrue(payload["simulator_action_authorized"])
        self.assertFalse(payload["v17_simulator_action_applied"])
        self.assertFalse(payload["filming_authorized"])

    def test_rollback_target_uses_the_pre_action_refresh_not_capture_state(self) -> None:
        from jepa_wm.joint_drive import JointDriveTarget
        from sim.control_session import ControlSession

        captured = SimpleNamespace(
            current_joint_positions=(0.1,) * 7,
            current_gripper_width_m=0.02,
        )
        refreshed = SimpleNamespace(
            joint_positions=(0.2,) * 7,
            gripper_width_m=0.03,
        )
        session = SimpleNamespace(
            load_capture=lambda: (object(), captured),
            load_result=lambda: SimpleNamespace(
                insertion_trial_refresh=SimpleNamespace(live_state=refreshed)
            ),
        )

        with patch.object(ControlSession, "at", return_value=session):
            actual = rollback_drive_target(Path("/data"))

        expected = JointDriveTarget.for_command(
            refreshed.joint_positions,
            refreshed.gripper_width_m,
        )
        captured_target = JointDriveTarget.for_command(
            captured.current_joint_positions,
            captured.current_gripper_width_m,
        )
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, captured_target)

    def test_runtime_fingerprint_binds_every_declared_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            from jepa_wm.contact_grasp_rotation_resolution import RUNTIME_FILES

            for relative in RUNTIME_FILES:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
            first = runtime_fingerprint(root)
            (root / RUNTIME_FILES[-1]).write_text("changed")

            self.assertNotEqual(runtime_fingerprint(root), first)

    def test_runner_uses_authenticated_handoff_before_followup_and_never_resets(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "ops"
            / "run_unknown_start_rotation_resolution.sh"
        ).read_text()

        handoff = runner.index("demo.capture_contact_grasp_acquisition_handoff")
        followup = runner.index("demo.capture_followup_observation")
        self.assertLess(handoff, followup)
        self.assertNotIn("restore_contact_grasp_tracking_retry", runner)
        self.assertNotIn("reset_stage", runner)
        self.assertNotIn("record_", runner)
        self.assertIn('maximum_actions="52"', runner)

        aws = (Path(__file__).resolve().parents[1] / "ops" / "aws.sh").read_text()
        self.assertIn("jepa-wm-contact-grasp-rollback-diagnostic)", aws)
        self.assertIn(
            "demo.diagnose_contact_grasp_rollback_drive_target(${rotation_resolution})",
            aws,
        )
        self.assertIn("v12) rotation_resolution=False", aws)
        self.assertIn("v15) rotation_resolution=True", aws)

    def test_live_diagnostic_selects_the_rotation_rollback_owner(self) -> None:
        import sim.isaac_control_followup as followup

        stage = object()
        runtime = object()
        target = type(
            "Target",
            (),
            {
                "joint_positions": (0.1,) * 7,
                "gripper_width_m": 0.02,
                "to_dict": lambda self: {
                    "joint_positions": list(self.joint_positions),
                    "gripper_width_m": self.gripper_width_m,
                },
            },
        )()
        omni = ModuleType("omni")
        omni.__path__ = []
        usd = ModuleType("omni.usd")
        usd.get_context = lambda: type(
            "Context", (), {"get_stage": lambda self: stage}
        )()
        omni.usd = usd

        with (
            patch.dict(sys.modules, {"omni": omni, "omni.usd": usd}),
            patch.object(followup, "live_runtime_for", return_value=runtime) as live,
            patch.object(
                followup,
                "rotation_resolution_drive_target",
                return_value=target,
            ),
            patch.object(followup, "current_drive_target", return_value=target),
        ):
            result = followup.diagnose_contact_grasp_rollback_drive_target(True)

        live.assert_called_once_with(ROLLED_BACK_SESSION_ID, stage)
        self.assertEqual(result["runtime_owner_session_id"], ROLLED_BACK_SESSION_ID)
        self.assertEqual(result["maximum_joint_delta_rad"], 0.0)
        self.assertFalse(result["simulator_action_applied"])


if __name__ == "__main__":
    unittest.main()
