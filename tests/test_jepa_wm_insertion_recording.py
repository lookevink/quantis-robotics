from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.insertion_recording import InsertionDemonstrationEvidence
from sim.exploration import DOMAIN_DATASET_ID


class InsertionDemonstrationTest(unittest.TestCase):
    def _write_recording(
        self,
        root: Path,
        *,
        fps: int = 4,
        grasp_offset: float = 0.04,
        seated_observations: int = 4,
    ) -> Path:
        recording = root / "insertion-held-12402"
        recording.mkdir()
        steps = (
                (0.0, 0.04, False, "approaching_cable"),
                (0.0, 0.04, True, "cable_grasped"),
                (-0.08, -0.04, True, "aligned_with_socket"),
            ) + tuple(
                (-0.10, -0.06, True, "plug_seated")
                for _ in range(seated_observations)
            )
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                        "schema": "quantis.demo_recording.v5",
                        "recording_id": recording.name,
                        "fps": fps,
                        "frames": len(steps),
                        "metadata": {
                            "dataset": DOMAIN_DATASET_ID,
                            "split": "held_out",
                            "seed": 12402,
                            "task": INSERTION_TASK_ID,
                            "insertion_target": {
                                "socket_position": [-0.10, 0.0, 0.0],
                                "socket_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                                "insertion_axis": [-1.0, 0.0, 0.0],
                                "grasp_offset_meters": grasp_offset,
                                "evidence_mode": "kinematic_scripted_baseline",
                            },
                        },
                }
            )
        )
        with (recording / "steps.jsonl").open("w") as output:
            for index, (tip_x, hand_x, attached, stage) in enumerate(steps):
                output.write(
                    json.dumps(
                        {
                                "index": index,
                                "simulation_time_seconds": index * 0.25,
                                "stage": stage,
                                "plug_position": [tip_x, 0.0, 0.0],
                                "plug_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                                "end_effector_world_position": [hand_x, 0.0, 0.0],
                                "gripper_frame_world_position": [hand_x, 0.0, 0.0],
                                "plug_attached": attached,
                        }
                    )
                    + "\n"
                )
        return recording

    def test_reconstructs_rearward_grasp_alignment_and_seating(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(Path(temporary_directory))

            evidence = InsertionDemonstrationEvidence.from_recording(
                recording,
                expected_split="held_out",
            )

        self.assertEqual(evidence.acquisition_index, 1)
        self.assertEqual(evidence.seated_index, 3)
        self.assertEqual(evidence.seated_observations, 4)
        self.assertAlmostEqual(evidence.grasp_clearance_meters, 0.04)
        self.assertTrue(evidence.kinematic_only)

    def test_rejects_non_droid_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(Path(temporary_directory), fps=5)
            with self.assertRaisesRegex(ValueError, "kinematic insertion"):
                InsertionDemonstrationEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_noncanonical_grasp_offset_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                grasp_offset=0.031,
            )
            with self.assertRaisesRegex(ValueError, "grasp offset"):
                InsertionDemonstrationEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_fewer_than_four_seated_observations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                seated_observations=2,
            )
            with self.assertRaisesRegex(ValueError, "retain a seated plug"):
                InsertionDemonstrationEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )


if __name__ == "__main__":
    unittest.main()
