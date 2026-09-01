from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jepa_wm.contact_grasp_acquisition_handoff import (
    REPLAY_FINGERPRINT,
    RUNTIME_FILES,
    ContactGraspAcquisitionHandoff,
    runtime_fingerprint,
)


class ContactGraspAcquisitionHandoffTest(unittest.TestCase):
    def _handoff(self) -> ContactGraspAcquisitionHandoff:
        return ContactGraspAcquisitionHandoff(
            followup_session_id="unknown-start-e2e-v5-62605-grasp-01",
            runtime_fingerprint="a" * 64,
            source_revision="b" * 40,
        )

    def test_round_trips_only_the_frozen_authority(self) -> None:
        handoff = self._handoff()

        self.assertEqual(
            ContactGraspAcquisitionHandoff.from_dict(handoff.to_dict()),
            handoff,
        )
        self.assertEqual(handoff.replay_fingerprint, REPLAY_FINGERPRINT)

    def test_rejects_an_unrecognized_or_changed_field(self) -> None:
        payload = self._handoff().to_dict()
        payload["filming_authorized"] = True

        with self.assertRaisesRegex(ValueError, "changed"):
            ContactGraspAcquisitionHandoff.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "invalid"):
            replace(self._handoff(), replay_fingerprint="c" * 64)

    def test_runtime_fingerprint_binds_every_execution_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for index, relative in enumerate(RUNTIME_FILES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"runtime {index}\n")
            before = runtime_fingerprint(root)
            changed = root / RUNTIME_FILES[-1]
            changed.write_text("changed\n")

            self.assertEqual(len(before), 64)
            self.assertNotEqual(runtime_fingerprint(root), before)

    def test_runner_uses_one_dedicated_handoff_then_only_followups(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "ops"
            / "run_unknown_start_acquisition_recovery.sh"
        ).read_text()

        self.assertEqual(
            runner.count("demo.capture_contact_grasp_acquisition_handoff"),
            1,
        )
        self.assertEqual(runner.count("demo.capture_followup_observation"), 1)
        self.assertNotIn("reset_stage", runner)
        self.assertNotIn("film", runner)
        self.assertNotIn("record_candidate_demo", runner)


if __name__ == "__main__":
    unittest.main()
