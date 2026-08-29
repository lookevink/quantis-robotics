from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.grasp_recording import GraspDemonstrationEvidence
from jepa_wm.grasp_task import (
    GraspTaskStep,
    ReachAndGraspFailure,
    evaluate_reach_and_grasp,
)
from sim.exploration import DOMAIN_DATASET_ID


def _step(
    x: float,
    *,
    attached: bool,
    tracking: bool = True,
    collision: bool = False,
    force: float = 0.0,
) -> GraspTaskStep:
    return GraspTaskStep((x, 0.0, 0.0), attached, tracking, collision, force)


class ReachAndGraspGateTest(unittest.TestCase):
    def test_passes_attachment_and_retained_lift(self) -> None:
        decision = evaluate_reach_and_grasp(
            (
                _step(0.0, attached=False),
                _step(0.0, attached=True),
                _step(0.025, attached=True),
            )
        )

        self.assertTrue(decision.passed)
        self.assertEqual(decision.acquisition_index, 1)
        self.assertAlmostEqual(decision.maximum_retained_displacement_meters, 0.025)

    def test_free_space_motion_and_closure_cannot_pass(self) -> None:
        decision = evaluate_reach_and_grasp(
            (
                _step(0.0, attached=False),
                _step(0.10, attached=False),
                _step(0.20, attached=False),
            )
        )

        self.assertFalse(decision.passed)
        self.assertIn(
            ReachAndGraspFailure.NO_ATTACHMENT_TRANSITION,
            decision.failures,
        )

    def test_directional_retention_rejects_orthogonal_or_reverse_drift(self) -> None:
        for name, terminal in (
            ("orthogonal", GraspTaskStep((0.0, 0.03, 0.0), True, True, False, 0.0)),
            ("reverse", GraspTaskStep((-0.03, 0.0, 0.0), True, True, False, 0.0)),
        ):
            with self.subTest(name=name):
                decision = evaluate_reach_and_grasp(
                    (
                        _step(0.0, attached=False),
                        _step(0.0, attached=True),
                        terminal,
                    ),
                    retained_direction=(1.0, 0.0, 0.0),
                )

                self.assertFalse(decision.passed)
                self.assertEqual(
                    decision.maximum_retained_displacement_meters, 0.0
                )
                self.assertIn(
                    ReachAndGraspFailure.INSUFFICIENT_LIFT,
                    decision.failures,
                )

    def test_rejects_lost_attachment_and_unsafe_motion(self) -> None:
        decision = evaluate_reach_and_grasp(
            (
                _step(0.0, attached=False),
                _step(0.0, attached=True),
                _step(0.03, attached=False, tracking=False, collision=True, force=2.1),
            )
        )

        self.assertFalse(decision.passed)
        self.assertIn(ReachAndGraspFailure.ATTACHMENT_LOST, decision.failures)
        self.assertIn(ReachAndGraspFailure.TRACKING_FAILED, decision.failures)
        self.assertIn(ReachAndGraspFailure.COLLISION_DETECTED, decision.failures)
        self.assertIn(ReachAndGraspFailure.CONTACT_FORCE_EXCEEDED, decision.failures)


class GraspDemonstrationTest(unittest.TestCase):
    def _recording(self, root: Path, attached: tuple[bool, ...]) -> Path:
        recording = root / "grasp-recording"
        recording.mkdir()
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                    "recording_id": recording.name,
                    "fps": 4,
                    "frames": len(attached),
                    "metadata": {
                        "dataset": DOMAIN_DATASET_ID,
                        "split": "held_out",
                        "seed": 11401,
                        "task": GRASP_TASK_ID,
                    },
                }
            )
        )
        with (recording / "steps.jsonl").open("w") as output:
            for index, is_attached in enumerate(attached):
                output.write(
                    json.dumps(
                        {
                            "index": index,
                            "simulation_time_seconds": index * 0.25,
                            "plug_attached": is_attached,
                            "plug_position": [0.03 * max(0, index - 1), 0.0, 0.0],
                        }
                    )
                    + "\n"
                )
        return recording

    def test_validates_a_cadenced_retained_grasp_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = GraspDemonstrationEvidence.from_recording(
                self._recording(Path(temp_dir), (False, True, True)),
                expected_split="held_out",
            )

        self.assertEqual(evidence.acquisition_index, 1)
        self.assertAlmostEqual(evidence.retained_displacement_meters, 0.03)

    def test_rejects_a_recording_without_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "attachment transition"):
                GraspDemonstrationEvidence.from_recording(
                    self._recording(Path(temp_dir), (False, False, False)),
                    expected_split="held_out",
                )


if __name__ == "__main__":
    unittest.main()
