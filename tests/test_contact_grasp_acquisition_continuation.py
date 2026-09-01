from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jepa_wm.contact_grasp_acquisition_continuation import (
    RUNTIME_FILES,
    ContactGraspAcquisitionContinuation,
    runtime_fingerprint,
)


class ContactGraspAcquisitionContinuationTest(unittest.TestCase):
    def _handoff(self) -> ContactGraspAcquisitionContinuation:
        return ContactGraspAcquisitionContinuation(
            followup_session_id="unknown-start-e2e-v6-62605-grasp-01",
            runtime_fingerprint="a" * 64,
            source_revision="b" * 40,
        )

    def test_round_trips_only_the_frozen_authority(self) -> None:
        handoff = self._handoff()

        self.assertEqual(
            ContactGraspAcquisitionContinuation.from_dict(handoff.to_dict()),
            handoff,
        )
        self.assertFalse(handoff.to_dict()["filming_authorized"])

    def test_rejects_changed_authority(self) -> None:
        payload = self._handoff().to_dict()
        payload["simulator_action_authorized"] = False

        with self.assertRaisesRegex(ValueError, "changed"):
            ContactGraspAcquisitionContinuation.from_dict(payload)

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

    def test_runner_is_serial_and_never_resets_or_films(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "ops"
            / "run_unknown_start_acquisition_continuation.sh"
        ).read_text()

        self.assertEqual(
            runner.count("demo.capture_contact_grasp_acquisition_handoff"), 1
        )
        self.assertEqual(runner.count("demo.capture_followup_observation"), 1)
        self.assertNotIn("reset_stage", runner)
        self.assertNotIn("film", runner)


if __name__ == "__main__":
    unittest.main()
