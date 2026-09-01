from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jepa_wm.action import DroidAction
from jepa_wm.contact_grasp_acquisition_resolution import (
    RUNTIME_FILES,
    ContactGraspAcquisitionResolution,
    runtime_fingerprint,
    validate_diagnostic_evidence,
)
from jepa_wm.control_safety import contact_grasp_action_scales


class ContactGraspAcquisitionResolutionTest(unittest.TestCase):
    def _handoff(self) -> ContactGraspAcquisitionResolution:
        return ContactGraspAcquisitionResolution(
            "unknown-start-e2e-v14-62605-grasp-01",
            "a" * 64,
            "b" * 40,
        )

    def test_round_trips_frozen_authority_without_filming(self) -> None:
        handoff = self._handoff()

        self.assertEqual(
            ContactGraspAcquisitionResolution.from_dict(handoff.to_dict()),
            handoff,
        )
        self.assertTrue(handoff.to_dict()["no_actuation_diagnostic_required"])
        self.assertFalse(handoff.to_dict()["filming_authorized"])

    def test_rejects_changed_authority(self) -> None:
        payload = self._handoff().to_dict()
        payload["simulator_action_authorized"] = False

        with self.assertRaisesRegex(ValueError, "changed"):
            ContactGraspAcquisitionResolution.from_dict(payload)

    def test_runtime_fingerprint_binds_every_execution_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index, relative in enumerate(RUNTIME_FILES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"runtime {index}\n")
            before = runtime_fingerprint(root)
            (root / RUNTIME_FILES[-1]).write_text("changed\n")

            self.assertEqual(len(before), 64)
            self.assertNotEqual(runtime_fingerprint(root), before)

    def test_runner_requires_diagnostic_before_serial_motion(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "ops"
            / "run_unknown_start_acquisition_resolution.sh"
        ).read_text()

        diagnostic = runner.index(
            "demo.diagnose_contact_grasp_acquisition_resolution"
        )
        apply = runner.index("demo.apply_control_response")
        self.assertLess(diagnostic, apply)
        self.assertEqual(
            runner.count("demo.capture_contact_grasp_acquisition_handoff"), 1
        )
        self.assertEqual(runner.count("demo.capture_followup_observation"), 1)
        self.assertIn('maximum_actions="52"', runner)
        self.assertNotIn("reset_stage", runner)
        self.assertNotIn("film", runner)

    def test_diagnostic_requires_first_safe_frozen_coarse_scale(self) -> None:
        handoff = self._handoff()
        claim_fingerprint = "c" * 64
        action = DroidAction((0.003, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2))
        scales = contact_grasp_action_scales(action, coarse_acquisition=True)
        attempts = [
            {"scale": scale.to_dict(), "passed": index == 1}
            for index, scale in enumerate(scales)
        ]
        payload = {
            "schema": (
                "quantis.contact_grasp_acquisition_resolution_diagnostic.v2"
            ),
            "status": "passed_no_actuation",
            "source_session_id": "unknown-start-e2e-v10-62605-grasp-52",
            "followup_session_id": handoff.followup_session_id,
            "claim_fingerprint": claim_fingerprint,
            "action": list(action.values),
            "selected_scale": scales[1].to_dict(),
            "attempts": attempts,
            "simulator_action_applied": False,
            "runtime_owner_session_id": "unknown-start-e2e-v12-62605-grasp-01",
            "active_drive_target": {
                "joint_positions": [0.0] * 7,
                "gripper_width_m": 0.07,
            },
        }

        self.assertEqual(
            validate_diagnostic_evidence(
                payload,
                handoff,
                claim_fingerprint,
            ),
            payload,
        )
        payload["selected_scale"] = scales[0].to_dict()
        with self.assertRaisesRegex(ValueError, "did not pass"):
            validate_diagnostic_evidence(payload, handoff, claim_fingerprint)


if __name__ == "__main__":
    unittest.main()
