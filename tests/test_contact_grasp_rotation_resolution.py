from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jepa_wm.contact_grasp_rotation_resolution import (
    ROLLED_BACK_SESSION_ID,
    SOURCE_SESSION_ID,
    ContactGraspRotationResolution,
    runtime_fingerprint,
)


class ContactGraspRotationResolutionTest(unittest.TestCase):
    def test_authority_round_trips_without_filming(self) -> None:
        authority = ContactGraspRotationResolution(
            "unknown-start-e2e-v17-62605-grasp-01",
            "a" * 64,
            "b" * 40,
        )

        payload = authority.to_dict()

        self.assertEqual(ContactGraspRotationResolution.from_dict(payload), authority)
        self.assertEqual(payload["source_session_id"], SOURCE_SESSION_ID)
        self.assertEqual(payload["rolled_back_session_id"], ROLLED_BACK_SESSION_ID)
        self.assertTrue(payload["simulator_action_authorized"])
        self.assertFalse(payload["filming_authorized"])

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


if __name__ == "__main__":
    unittest.main()
