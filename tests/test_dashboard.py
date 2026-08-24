from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from jepa.dashboard import PANEL_SIZE, render_dashboard_panels


class DashboardTest(unittest.TestCase):
    def test_renders_validated_grasp_replay_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "fps": 12,
                        "frames": 1,
                        "metadata": {
                            "grasp_demo": {
                                "schema": "quantis.jepa_wm_grasp_demo.v1",
                                "visualization_only": True,
                                "readiness_id": "grasp-readiness-v2",
                                "baseline_experiment_id": "grasp-baseline-12401-v2",
                                "rollout_id": "rollout-12401",
                                "seed": 12401,
                                "proposal": {
                                    "path": "/tmp/proposal.pth",
                                    "fingerprint": "a" * 64,
                                },
                                "source_steps": 8,
                                "task_outcome": {
                                    "passed": True,
                                    "acquisition_index": 1,
                                    "attached_observations": 8,
                                    "maximum_retained_displacement_meters": 0.0585,
                                    "failures": [],
                                },
                                "replay_tracking_passed": True,
                                "maximum_replay_joint_error_rad": 0.001,
                                "maximum_replay_gripper_error_m": 0.0002,
                                "replay_safety_passed": True,
                                "maximum_replay_contact_force_newtons": 0.0,
                                "replay_collision_detected": False,
                            }
                        },
                    }
                )
            )
            (recording / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 0,
                        "stage": "cable_grasped",
                        "phase": "pre_insertion_complete",
                        "arm_positions": [0.0] * 7,
                        "gripper_width_m": 0.01,
                        "plug_position": [0.02, -0.25, 1.32],
                        "plug_attached": True,
                    }
                )
                + "\n"
            )

            result = render_dashboard_panels(recording)

            self.assertTrue(result["grasp_demo"])
            self.assertFalse(result["candidate_demo"])
            self.assertTrue(
                (recording / "dashboard" / "panel" / "frame_000000.png").is_file()
            )

    def test_renders_candidate_search_metrics_from_recording_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir)
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "fps": 12,
                        "frames": 1,
                        "metadata": {
                            "candidate_demo": {
                                "schema": "quantis.jepa_wm_candidate_demo.v1",
                                "visualization_only": True,
                                "report_id": "candidate-proof-11401",
                                "candidate_session": "candidate-11401",
                                "source_session": "source-11401",
                                "seed": 11401,
                                "policy": "reset_trial_candidate",
                                "candidates_scored": 640,
                                "energy_improvement": 0.00014,
                                "actual_action": [0.002, 0.0, 0.0, 0.001, 0.0, 0.0, 0.03],
                                "tracking_passed": True,
                                "maximum_replay_joint_error_rad": 0.001,
                                "maximum_replay_gripper_error_m": 0.0002,
                                "maximum_replay_contact_force_newtons": 0.0,
                                "replay_collision_detected": False,
                                "selected_action_scale": {
                                    "translation": 1.0,
                                    "rotation": 1.0,
                                    "gripper": 1.0,
                                },
                                "planner": {
                                    "horizon": 3,
                                    "iterations": 5,
                                    "samples": 128,
                                    "elites": 12,
                                    "seed": 237,
                                    "minimum_standard_deviation": 0.0001,
                                },
                            }
                        },
                    }
                )
            )
            (recording / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 0,
                        "stage": "approaching_cable",
                        "phase": "ready",
                        "arm_positions": [0.0] * 7,
                        "gripper_width_m": 0.07,
                        "plug_position": [-0.02, -0.25, 1.32],
                        "plug_attached": False,
                    }
                )
                + "\n"
            )

            result = render_dashboard_panels(recording)

            self.assertTrue(result["candidate_demo"])
            self.assertTrue(
                (recording / "dashboard" / "panel" / "frame_000000.png").is_file()
            )

    def test_renders_one_synchronized_panel_per_recording_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir) / "demo-20260822T031500Z"
            report_dir = recording / "jepa" / "wrist"
            report_dir.mkdir(parents=True)
            (recording / "manifest.json").write_text(
                json.dumps({"fps": 12, "frames": 2})
            )
            steps = [
                {
                    "index": index,
                    "stage": "approaching_cable",
                    "phase": "ready",
                    "arm_positions": [0.0] * 7,
                    "gripper_width_m": 0.07,
                    "plug_position": [-0.02, -0.25, 1.32],
                    "plug_attached": False,
                }
                for index in range(2)
            ]
            (recording / "steps.jsonl").write_text(
                "".join(json.dumps(step) + "\n" for step in steps)
            )
            (report_dir / "stage_report.json").write_text(
                json.dumps(
                    {
                        "predictions": [
                            {
                                "actual": "approaching_cable",
                                "stage": "approaching_cable",
                                "similarity": 0.98,
                                "margin": 0.04,
                            }
                        ]
                    }
                )
            )

            result = render_dashboard_panels(recording)

            self.assertEqual(result["frames"], 2)
            self.assertEqual(result["jepa_stages"], ["approaching_cable"])
            self.assertEqual(result["layout"]["output_size"], [2560, 1440])
            first = recording / "dashboard" / "panel" / "frame_000000.png"
            second = recording / "dashboard" / "panel" / "frame_000001.png"
            layout = json.loads((recording / "dashboard" / "layout.json").read_text())
            self.assertEqual(layout["primary_size"], [1920, 1080])
            self.assertEqual(layout["primary_y"], 180)
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            with Image.open(first) as panel:
                self.assertEqual(panel.size, PANEL_SIZE)

    def test_rejects_manifest_telemetry_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recording = Path(temp_dir)
            (recording / "manifest.json").write_text(
                json.dumps({"fps": 12, "frames": 2})
            )
            (recording / "steps.jsonl").write_text(
                json.dumps(
                    {
                        "index": 0,
                        "stage": "approaching_cable",
                        "phase": "ready",
                        "arm_positions": [0.0] * 7,
                        "gripper_width_m": 0.07,
                        "plug_position": [-0.02, -0.25, 1.32],
                        "plug_attached": False,
                    }
                )
                + "\n"
            )

            with self.assertRaisesRegex(ValueError, "frame counts differ"):
                render_dashboard_panels(recording)


if __name__ == "__main__":
    unittest.main()
