from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from jepa.dashboard import PANEL_SIZE, render_dashboard_panels


class DashboardTest(unittest.TestCase):
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
