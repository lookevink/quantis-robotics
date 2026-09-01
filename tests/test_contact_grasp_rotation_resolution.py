from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from jepa_wm.contact_grasp_rotation_resolution import (
    RUNTIME_OWNER_SESSION_ID,
    SOURCE_SESSION_ID,
    V15_ROLLED_BACK_SESSION_ID,
    ContactGraspRotationResolution,
    retained_drive_target,
    runtime_fingerprint,
    v15_rollback_drive_target,
)


class ContactGraspRotationResolutionTest(unittest.TestCase):
    def test_authority_round_trips_without_filming(self) -> None:
        authority = ContactGraspRotationResolution(
            "unknown-start-e2e-v19-62605-grasp-01",
            "a" * 64,
            "b" * 40,
        )

        payload = authority.to_dict()

        self.assertEqual(ContactGraspRotationResolution.from_dict(payload), authority)
        self.assertEqual(payload["source_session_id"], SOURCE_SESSION_ID)
        self.assertEqual(payload["runtime_owner_session_id"], RUNTIME_OWNER_SESSION_ID)
        self.assertTrue(payload["simulator_action_authorized"])
        self.assertFalse(payload["source_partial_session_action_applied"])
        self.assertFalse(payload["filming_authorized"])

    def test_handoff_target_uses_the_applied_v18_drive_command(self) -> None:
        from jepa_wm.control_rollout import ControlStepSummary
        from jepa_wm.joint_drive import JointDriveTarget
        from sim.control_session import ControlSession

        session = object()
        target = JointDriveTarget((0.2,) * 7, 0.03)
        step = SimpleNamespace(contact_grasp_drive_target=lambda: target)

        with (
            patch.object(ControlSession, "at", return_value=session),
            patch.object(ControlStepSummary, "from_session", return_value=step),
        ):
            actual = retained_drive_target(Path("/data"))

        self.assertEqual(actual, target)

    def test_historical_v15_target_uses_its_pre_action_refresh(self) -> None:
        from jepa_wm.joint_drive import JointDriveTarget
        from sim.control_session import ControlSession

        refreshed = SimpleNamespace(
            joint_positions=(0.2,) * 7,
            gripper_width_m=0.03,
        )
        session = SimpleNamespace(
            load_result=lambda: SimpleNamespace(
                insertion_trial_refresh=SimpleNamespace(live_state=refreshed)
            )
        )

        with patch.object(ControlSession, "at", return_value=session):
            actual = v15_rollback_drive_target(Path("/data"))

        self.assertEqual(
            actual,
            JointDriveTarget.for_command(
                refreshed.joint_positions,
                refreshed.gripper_width_m,
            ),
        )

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
        self.assertIn('maximum_actions="50"', runner)

        aws = (Path(__file__).resolve().parents[1] / "ops" / "aws.sh").read_text()
        self.assertIn("jepa-wm-contact-grasp-rollback-diagnostic)", aws)
        self.assertIn("jepa-wm-contact-grasp-followup-diagnostic)", aws)
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
                "v15_rollback_drive_target",
                return_value=target,
            ),
            patch.object(followup, "current_drive_target", return_value=target),
        ):
            result = followup.diagnose_contact_grasp_rollback_drive_target(True)

        live.assert_called_once_with(V15_ROLLED_BACK_SESSION_ID, stage)
        self.assertEqual(
            result["runtime_owner_session_id"], V15_ROLLED_BACK_SESSION_ID
        )
        self.assertEqual(result["maximum_joint_delta_rad"], 0.0)
        self.assertFalse(result["simulator_action_applied"])


if __name__ == "__main__":
    unittest.main()
