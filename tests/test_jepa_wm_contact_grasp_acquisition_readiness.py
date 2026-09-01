from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.contact_grasp_acquisition_readiness import (
    MAXIMUM_HOLD_GRIPPER_DELTA,
    _gripper_phase_evidence,
)
from jepa_wm.insertion_layout import CONTACT_INSERTION_LAYOUT, ContactInsertionSegment
from jepa_wm.task_windows import (
    CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW,
    proposal_window,
)


class ContactGraspAcquisitionReadinessTest(unittest.TestCase):
    def _report(self, root: Path, *, corrupt_context: int | None = None) -> Path:
        close_start = CONTACT_INSERTION_LAYOUT.start_index(
            ContactInsertionSegment.GRASP_CLOSE
        ) - 1
        attach = CONTACT_INSERTION_LAYOUT.start_index(
            ContactInsertionSegment.GRASP_ATTACH
        )
        results = []
        for context in CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW.context_indices:
            gripper = (
                0.0
                if context < close_start
                else 0.01
                if context < attach
                else 0.0
            )
            if context == corrupt_context:
                gripper = -0.03
            results.append(
                {
                    "context_index": context,
                    "proposed_actions": [[0.0] * 6 + [gripper]],
                }
            )
        report = root / "report.json"
        report.write_text(json.dumps({"results": results}))
        return report

    def test_window_covers_initial_approach_through_retained_grasp(self) -> None:
        window = CONTACT_GRASP_ACQUISITION_PROPOSAL_WINDOW

        self.assertEqual(window.start_index, 0)
        self.assertEqual(window.context_indices[-1], 125)
        self.assertEqual(proposal_window("contact-grasp-acquisition"), window)

    def test_gripper_phase_gate_requires_hold_close_and_retention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evidence = _gripper_phase_evidence(
                self._report(Path(directory))
            )

        self.assertTrue(evidence["passed"])
        self.assertEqual(
            evidence["maximum_hold_gripper_delta"],
            MAXIMUM_HOLD_GRIPPER_DELTA,
        )

    def test_gripper_phase_gate_rejects_the_v4_opening_failure(self) -> None:
        close_context = CONTACT_INSERTION_LAYOUT.start_index(
            ContactInsertionSegment.GRASP_CLOSE
        )
        with tempfile.TemporaryDirectory() as directory:
            evidence = _gripper_phase_evidence(
                self._report(Path(directory), corrupt_context=close_context)
            )

        self.assertFalse(evidence["passed"])


if __name__ == "__main__":
    unittest.main()
