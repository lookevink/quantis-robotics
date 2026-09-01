from __future__ import annotations

from pathlib import Path
import unittest

from jepa_wm.contact_grasp_acquisition_hold import ContactGraspAcquisitionHold


class ContactGraspAcquisitionHoldTest(unittest.TestCase):
    def test_round_trips_frozen_authority_without_filming(self) -> None:
        handoff = ContactGraspAcquisitionHold(
            "unknown-start-e2e-v9-62605-grasp-01",
            "a" * 64,
            "b" * 40,
        )

        self.assertEqual(
            ContactGraspAcquisitionHold.from_dict(handoff.to_dict()), handoff
        )
        self.assertFalse(handoff.to_dict()["filming_authorized"])

    def test_rejects_changed_authority(self) -> None:
        handoff = ContactGraspAcquisitionHold("session", "a" * 64, "b" * 40)
        payload = handoff.to_dict()
        payload["production_authority_granted"] = True

        with self.assertRaisesRegex(ValueError, "changed"):
            ContactGraspAcquisitionHold.from_dict(payload)

    def test_runner_is_serial_and_does_not_reset_or_film(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "ops"
            / "run_unknown_start_acquisition_hold.sh"
        ).read_text()

        self.assertEqual(
            runner.count("demo.capture_contact_grasp_acquisition_handoff"), 1
        )
        self.assertEqual(runner.count("demo.capture_followup_observation"), 1)
        self.assertNotIn("reset_stage", runner)
        self.assertNotIn("film", runner)


if __name__ == "__main__":
    unittest.main()
