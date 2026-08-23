from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.proposal_readiness import summarize_proposal_readiness


class ProposalReadinessTest(unittest.TestCase):
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

    def _proposal(self, root: Path, training: tuple[Path, ...]) -> Path:
        proposal = root / "checkpoints" / "proposal.pth"
        proposal.parent.mkdir()
        proposal.touch()
        Path(f"{proposal}.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "base_model": "jepa_wm_droid",
                        "source_revision": "revision",
                        "camera": "wrist",
                        "training_recordings": [path.name for path in training],
                        "training_steps": 2000,
                    }
                }
            )
        )
        return proposal

    def _report(self, recording: Path, proposal: Path, *, pass_rate: float) -> Path:
        rollouts = 62
        passed = round(rollouts * pass_rate)
        recorded = [[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3
        aligned = [[0.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3
        opposed = [[-0.008, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 3
        results = []
        for index in range(rollouts):
            proposed = aligned if index < passed else opposed
            cosine = 1.0 if index < passed else -1.0
            results.append(
                {
                    "context_index": index + 4,
                    "target_index": index + 7,
                    "recorded_actions": recorded,
                    "proposed_actions": proposed,
                    "sequence_mse": sum(
                        (left - right) ** 2
                        for recorded_action, proposed_action in zip(recorded, proposed)
                        for left, right in zip(recorded_action, proposed_action)
                    )
                    / 21,
                    "first_action_cosine": cosine,
                    "first_action_gate": {
                        "passed": index < passed,
                        "recorded_action_is_active": True,
                        "cosine": cosine,
                        "reasons": [] if index < passed else ["direction_mismatch"],
                    },
                }
            )
        mean_mse = sum(result["sequence_mse"] for result in results) / rollouts
        mean_cosine = (
            sum(result["first_action_cosine"] for result in results) / rollouts
        )
        report = recording / "jepa_wm" / "proposal_eval.json"
        report.parent.mkdir(exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "schema": "quantis.jepa_wm_action_proposal_evaluation.v1",
                    "status": "evaluated",
                    "proposal": str(proposal),
                    "recording": str(recording),
                    "camera": "wrist",
                    "rollouts": rollouts,
                    "window": {"start_index": 4, "count": rollouts, "stride": 1},
                    "mean_sequence_mse": mean_mse,
                    "mean_first_action_cosine": mean_cosine,
                    "first_action_gate_pass_rate": passed / rollouts,
                    "active_first_action_direction_pass_rate": passed / rollouts,
                    "stationary_first_action_hold_rate": None,
                    "results": results,
                }
            )
        )
        return report

    def test_passes_two_disjoint_whole_seed_operational_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = tuple(
                self._recording(root, f"train-{seed}", "train", seed)
                for seed in range(100, 112)
            )
            proposal = self._proposal(root, training)
            reports = tuple(
                self._report(
                    self._recording(root, f"held-{seed}", "held_out", seed),
                    proposal,
                    pass_rate=1.0,
                )
                for seed in (200, 201)
            )
            output = root / "readiness.json"

            summary = summarize_proposal_readiness(proposal, reports, output)

            self.assertTrue(summary["passed"])
            self.assertEqual(summary["aggregate"]["rollouts"], 124)
            self.assertEqual(json.loads(output.read_text()), summary)

    def test_fails_if_one_seed_misses_the_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = tuple(
                self._recording(root, f"train-{seed}", "train", seed)
                for seed in range(100, 112)
            )
            proposal = self._proposal(root, training)
            reports = (
                self._report(
                    self._recording(root, "held-good", "held_out", 200),
                    proposal,
                    pass_rate=1.0,
                ),
                self._report(
                    self._recording(root, "held-weak", "held_out", 201),
                    proposal,
                    pass_rate=0.9,
                ),
            )

            summary = summarize_proposal_readiness(
                proposal, reports, root / "readiness.json"
            )

            self.assertFalse(summary["passed"])

    def test_fails_with_fewer_than_twelve_training_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = tuple(
                self._recording(root, f"train-{seed}", "train", seed)
                for seed in range(100, 111)
            )
            proposal = self._proposal(root, training)
            reports = tuple(
                self._report(
                    self._recording(root, f"held-{seed}", "held_out", seed),
                    proposal,
                    pass_rate=1.0,
                )
                for seed in (200, 201)
            )

            summary = summarize_proposal_readiness(
                proposal, reports, root / "readiness.json"
            )

            self.assertFalse(summary["passed"])

    def test_rejects_a_training_recording_as_held_out(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = (self._recording(root, "train", "train", 100),)
            proposal = self._proposal(root, training)
            report = self._report(training[0], proposal, pass_rate=1.0)

            with self.assertRaisesRegex(ValueError, "expected 'held_out'"):
                summarize_proposal_readiness(
                    proposal,
                    (report, report),
                    root / "readiness.json",
                )

    def test_rejects_tampered_per_rollout_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = tuple(
                self._recording(root, f"train-{seed}", "train", seed)
                for seed in range(100, 112)
            )
            proposal = self._proposal(root, training)
            held = self._recording(root, "held", "held_out", 200)
            report = self._report(held, proposal, pass_rate=1.0)
            payload = json.loads(report.read_text())
            payload["results"][0]["sequence_mse"] = 0.0
            report.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "rollout metrics"):
                summarize_proposal_readiness(
                    proposal,
                    (report,),
                    root / "readiness.json",
                )

    def test_rejects_training_and_held_out_seed_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = tuple(
                self._recording(root, f"train-{seed}", "train", seed)
                for seed in range(100, 112)
            )
            proposal = self._proposal(root, training)
            reports = (
                self._report(
                    self._recording(root, "held-overlap", "held_out", 100),
                    proposal,
                    pass_rate=1.0,
                ),
                self._report(
                    self._recording(root, "held-clean", "held_out", 200),
                    proposal,
                    pass_rate=1.0,
                ),
            )

            with self.assertRaisesRegex(ValueError, "seeds overlap"):
                summarize_proposal_readiness(
                    proposal,
                    reports,
                    root / "readiness.json",
                )

    def test_rejects_tampered_rollout_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = tuple(
                self._recording(root, f"train-{seed}", "train", seed)
                for seed in range(100, 112)
            )
            proposal = self._proposal(root, training)
            report = self._report(
                self._recording(root, "held", "held_out", 200),
                proposal,
                pass_rate=1.0,
            )
            payload = json.loads(report.read_text())
            payload["results"][1]["context_index"] = 4
            report.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ValueError, "rollout evidence"):
                summarize_proposal_readiness(
                    proposal,
                    (report,),
                    root / "readiness.json",
                )

    def test_rejects_non_droid_or_out_of_bounds_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            training = tuple(
                self._recording(root, f"train-{seed}", "train", seed)
                for seed in range(100, 112)
            )
            proposal = self._proposal(root, training)
            held = self._recording(root, "held", "held_out", 200)
            for mutation, message in (
                (lambda result: result["proposed_actions"].pop(), "rollout evidence"),
                (
                    lambda result: result["proposed_actions"][0].__setitem__(
                        0, 0.021
                    ),
                    "planner bounds",
                ),
            ):
                with self.subTest(message=message):
                    report = self._report(held, proposal, pass_rate=1.0)
                    payload = json.loads(report.read_text())
                    mutation(payload["results"][0])
                    report.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(ValueError, message):
                        summarize_proposal_readiness(
                            proposal,
                            (report,),
                            root / "readiness.json",
                        )


if __name__ == "__main__":
    unittest.main()
