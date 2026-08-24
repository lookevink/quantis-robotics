from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.insertion_contract import INSERTION_TASK_ID
from jepa_wm.insertion_recording import (
    ContactInsertionEvidence,
    InsertionDemonstrationEvidence,
)
from sim.exploration import DOMAIN_DATASET_ID


class InsertionDemonstrationTest(unittest.TestCase):
    def _write_recording(
        self,
        root: Path,
        *,
        fps: int = 4,
        grasp_offset: float = 0.04,
        seated_observations: int = 4,
        contact_aware: bool = False,
        maximum_force: float = 0.8,
    ) -> Path:
        recording = root / "insertion-held-12402"
        recording.mkdir()
        if contact_aware:
            phases = (
                ["initial"]
                + ["pre_grasp"] * 8
                + ["grasp"] * 8
                + ["grasp_close"] * 4
                + ["grasp_attached"]
                + ["pre_insertion"] * 8
                + ["pre_insertion_settle"] * 4
                + ["pre_insertion"] * 8
                + ["pre_insertion_settle"] * 2
                + ["insert"] * 64
                + ["insert_settle"] * seated_observations
            )
            steps = tuple(
                (
                    (
                        (0.0, 0.04, False, "approaching_cable")
                        if index < 21
                        else (0.0, 0.04, True, "cable_grasped")
                        if index < 42
                        else (-0.08, -0.04, True, "aligned_with_socket")
                        if index < 108
                        else (-0.10, -0.06, True, "plug_seated")
                    )
                    + (phase,)
                )
                for index, phase in enumerate(phases)
            )
        else:
            steps = (
                    (0.0, 0.04, False, "approaching_cable", "initial"),
                    (0.0, 0.04, True, "cable_grasped", "grasp_attached"),
                    (-0.08, -0.04, True, "aligned_with_socket", "insert"),
                ) + tuple(
                    (-0.10, -0.06, True, "plug_seated", "insert_settle")
                    for _ in range(seated_observations)
                )
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                        "schema": (
                            "quantis.demo_recording.v9"
                            if contact_aware
                            else "quantis.demo_recording.v5"
                        ),
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
                                "evidence_mode": (
                                    "contact_aware_scripted_baseline"
                                    if contact_aware
                                    else "kinematic_scripted_baseline"
                                ),
                                **(
                                    {
                                        "connector_collisions_enabled": True,
                                        "contact_sensor": "connector_tip",
                                        "compliant_collision_parts": ["latch"],
                                        "attachment": "dynamic_fixed_joint",
                                        "socket_scale": 1.05,
                                        "insertion_steps": 64,
                                        "expected_frames": 112,
                                    }
                                    if contact_aware
                                    else {}
                                ),
                            },
                        },
                }
            )
        )
        with (recording / "steps.jsonl").open("w") as output:
            for index, (tip_x, hand_x, attached, stage, phase) in enumerate(steps):
                output.write(
                    json.dumps(
                        {
                                "index": index,
                                "simulation_time_seconds": index * 0.25,
                                "stage": stage,
                                "phase": phase,
                                "plug_position": [tip_x, 0.0, 0.0],
                                "plug_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                                "end_effector_world_position": [hand_x, 0.0, 0.0],
                                "gripper_frame_world_position": [hand_x, 0.0, 0.0],
                                "plug_attached": attached,
                                **(
                                    {
                                        "collision_detected": False,
                                        "contact_force_newtons": (
                                            maximum_force
                                            if stage == "plug_seated"
                                            else 0.0
                                        ),
                                        "arm_tracking_error_rad": 0.001,
                                        "gripper_tracking_error_m": 0.0002,
                                    }
                                    if contact_aware
                                    else {}
                                ),
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
            with self.assertRaisesRegex(ValueError, "expected insertion"):
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

    def test_reconstructs_contact_force_collision_and_tracking_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
            )
            evidence = ContactInsertionEvidence.from_recording(
                recording,
                expected_split="held_out",
            )

        self.assertTrue(evidence.decision.passed)
        self.assertEqual(evidence.seated_observations, 4)
        self.assertAlmostEqual(evidence.maximum_contact_force_newtons, 0.8)
        self.assertTrue(evidence.to_dict()["contact_aware"])

    def test_rejects_contact_force_above_the_insertion_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
                maximum_force=2.1,
            )
            with self.assertRaisesRegex(ValueError, "contact_force_exceeded"):
                ContactInsertionEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_tampered_contact_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
            )
            manifest_path = recording / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["metadata"]["insertion_target"]["attachment"] = "kinematic"
            manifest_path.write_text(json.dumps(manifest))

            with self.assertRaisesRegex(ValueError, "instrumentation"):
                ContactInsertionEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_tampered_contact_phase_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
            )
            steps_path = recording / "steps.jsonl"
            steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
            steps[44]["phase"] = "pre_insertion_settle"
            steps_path.write_text("\n".join(json.dumps(step) for step in steps) + "\n")

            with self.assertRaisesRegex(ValueError, "phase contract"):
                ContactInsertionEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_contact_grasp_clearance_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
            )
            steps_path = recording / "steps.jsonl"
            steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
            steps[21]["gripper_frame_world_position"] = [0.031, 0.0, 0.0]
            steps_path.write_text("\n".join(json.dumps(step) for step in steps) + "\n")

            with self.assertRaisesRegex(ValueError, "grasp offset"):
                ContactInsertionEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_delayed_contact_attachment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
            )
            steps_path = recording / "steps.jsonl"
            steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
            steps[21]["plug_attached"] = False
            steps_path.write_text("\n".join(json.dumps(step) for step in steps) + "\n")

            with self.assertRaisesRegex(ValueError, "attachment contract"):
                ContactInsertionEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_numeric_string_contact_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
            )
            steps_path = recording / "steps.jsonl"
            steps = [json.loads(line) for line in steps_path.read_text().splitlines()]
            steps[0]["contact_force_newtons"] = "0"
            steps_path.write_text("\n".join(json.dumps(step) for step in steps) + "\n")

            with self.assertRaisesRegex(ValueError, "telemetry"):
                ContactInsertionEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                )

    def test_rejects_wrong_expected_contact_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording = self._write_recording(
                Path(temporary_directory),
                contact_aware=True,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                ContactInsertionEvidence.from_recording(
                    recording,
                    expected_split="held_out",
                    expected_seed=12403,
                )


if __name__ == "__main__":
    unittest.main()
