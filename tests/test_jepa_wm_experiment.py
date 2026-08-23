from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.experiment import summarize_experiment


class DomainExperimentTest(unittest.TestCase):
    def _recording(self, root: Path, name: str, split: str, seed: int) -> Path:
        recording = root / "recordings" / name
        recording.mkdir(parents=True)
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                    "recording_id": name,
                    "metadata": {
                        "dataset": "jepa_wm_domain_v1",
                        "split": split,
                        "seed": seed,
                    },
                }
            )
        )
        return recording

    def _adapter(self, root: Path, training: tuple[Path, ...]) -> Path:
        adapter = root / "checkpoints" / "domain-adapter.pth"
        adapter.parent.mkdir(parents=True)
        adapter.touch()
        Path(f"{adapter}.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "base_model": "jepa_wm_droid",
                        "source_revision": "revision",
                        "camera": "wrist",
                        "training_recordings": [path.name for path in training],
                        "training_steps": 500,
                    }
                }
            )
        )
        return adapter

    def _report(
        self,
        recording: Path,
        adapter: Path,
        improvements: tuple[float, ...],
    ) -> Path:
        report = recording / "jepa_wm" / "wrist_eval.json"
        report.parent.mkdir()
        report.write_text(
            json.dumps(
                {
                    "recording": str(recording),
                    "camera": "wrist",
                    "adapter": str(adapter),
                    "results": [
                        {
                            "improvement_over_zero": improvement,
                            "recorded_action_wins": improvement > 0,
                        }
                        for improvement in improvements
                    ],
                }
            )
        )
        return report

    def test_requires_every_held_out_seed_and_the_aggregate_to_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = (
                self._recording(root, "domain-train-1200", "train", 1200),
                self._recording(root, "domain-train-1201", "train", 1201),
            )
            adapter = self._adapter(root, training)
            reports = tuple(
                self._report(
                    self._recording(root, f"domain-held-{seed}", "held_out", seed),
                    adapter,
                    (0.2, 0.1, 0.3, -0.01),
                )
                for seed in (2200, 2201)
            )
            output = root / "experiment.json"

            summary = summarize_experiment(
                "domain-proof",
                training,
                reports,
                output,
            )

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["aggregate"]["rollouts"], 8)
            self.assertAlmostEqual(
                summary["aggregate"]["mean_improvement_over_zero"],
                0.1475,
            )
            self.assertEqual(
                summary["aggregate"]["recorded_action_win_rate"],
                0.75,
            )
            self.assertTrue(summary["aggregate"]["control_gate"]["passed"])
            self.assertTrue(
                all(
                    evaluation["control_gate"]["passed"]
                    for evaluation in summary["held_out_evaluations"]
                )
            )
            self.assertEqual(json.loads(output.read_text()), summary)

    def test_fails_when_one_held_out_seed_misses_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = (self._recording(root, "train", "train", 1200),)
            adapter = self._adapter(root, training)
            reports = tuple(
                self._report(
                    self._recording(root, name, "held_out", seed),
                    adapter,
                    improvements,
                )
                for name, seed, improvements in (
                    ("good", 2200, (0.2, 0.2, 0.2, 0.2)),
                    ("weak", 2201, (0.2, 0.2, -0.1, -0.1)),
                )
            )

            summary = summarize_experiment(
                "domain-proof",
                training,
                reports,
                root / "experiment.json",
            )

            self.assertFalse(summary["passed"])
            self.assertFalse(
                summary["held_out_evaluations"][1]["control_gate"]["passed"]
            )

    def test_rejects_train_held_out_leakage_and_duplicate_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = (self._recording(root, "train", "train", 1200),)
            adapter = self._adapter(root, training)
            held_out = self._recording(root, "held", "held_out", 2200)
            report = self._report(held_out, adapter, (0.2, 0.2, 0.2, 0.2))

            with self.assertRaisesRegex(
                ValueError, "held-out recordings must be unique"
            ):
                summarize_experiment(
                    "duplicate",
                    training,
                    (report, report),
                    root / "duplicate.json",
                )

            (held_out / "manifest.json").write_text(
                json.dumps(
                    {
                        "recording_id": "held",
                        "metadata": {
                            "dataset": "jepa_wm_domain_v1",
                            "split": "train",
                            "seed": 2200,
                        },
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "expected 'held_out'"):
                summarize_experiment(
                    "leaked",
                    training,
                    (report,),
                    root / "leaked.json",
                )

    def test_rejects_an_adapter_trained_on_another_recording_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = (self._recording(root, "train", "train", 1200),)
            adapter = self._adapter(root, training)
            Path(f"{adapter}.json").write_text(
                json.dumps(
                    {
                        "metadata": {
                            "base_model": "jepa_wm_droid",
                            "source_revision": "revision",
                            "camera": "wrist",
                            "training_recordings": ["another-training-run"],
                            "training_steps": 500,
                        }
                    }
                )
            )
            held_out = self._recording(root, "held", "held_out", 2200)
            report = self._report(held_out, adapter, (0.2, 0.2, 0.2, 0.2))

            with self.assertRaisesRegex(ValueError, "do not match"):
                summarize_experiment(
                    "stale-adapter",
                    training,
                    (report,),
                    root / "stale.json",
                )


if __name__ == "__main__":
    unittest.main()
