from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None

from jepa_wm.action import DroidAction
from jepa_wm.insertion_proposal_readiness import (
    INSERTION_EVALUATION_BOUNDS,
    INSERTION_WINDOW,
    summarize_insertion_proposal_readiness,
    validate_insertion_proposal_identity,
    validate_insertion_training_selection,
    validate_insertion_evaluation_window,
    validate_insertion_goal_deltas,
)
from jepa_wm.insertion_corpus import InsertionCorpusRoster
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
)
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL
from jepa_wm.training_artifact import (
    TrainingArtifactMetadata,
    artifact_fingerprint,
    rollout_training_selection_fingerprint,
)

if torch is not None:
    from jepa_wm.proposal import (
        ActionProposalNetwork,
        ProposalConditioning,
        save_action_proposal,
    )
    from jepa_wm.proprioception import DroidValueNormalization, ScalarNormalization


class InsertionProposalReadinessTest(unittest.TestCase):
    def test_window_covers_every_post_attachment_rollout_through_seating(self) -> None:
        start = CONTACT_INSERTION_RECORDING.start_index(
            ContactInsertionSegment.GRASP_ATTACH
        )
        self.assertEqual(
            INSERTION_WINDOW.to_dict(),
            {
                "start_index": start,
                "count": (
                    CONTACT_INSERTION_RECORDING.frame_count
                    - DROID_ROLLOUT_PROTOCOL.context_frames
                    - DROID_ROLLOUT_PROTOCOL.action_horizon
                    + 1
                    - start
                ),
                "stride": 1,
            },
        )

    def test_rejects_an_evaluation_outside_the_complete_insertion_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            report = Path(temporary_directory) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "recording": "/tmp/recording",
                        "window": {"start_index": 44, "count": 64, "stride": 1},
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "complete insertion window"):
                validate_insertion_evaluation_window(report)

    def test_rejects_tampered_held_out_goal_deltas(self) -> None:
        goal_action = DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        results = [
            {
                "context_index": context_index,
                "target_index": context_index + 3,
                "goal_delta": list(goal_action.values),
                "recorded_actions": [list(goal_action.values)] * 3,
            }
            for context_index in INSERTION_WINDOW.context_indices
        ]
        with patch(
            "jepa_wm.task_proposal_readiness.load_rollout_at",
            side_effect=lambda *args, context_index, **kwargs: SimpleNamespace(
                context=(SimpleNamespace(index=context_index),),
                target=SimpleNamespace(index=context_index + 3),
                goal_action=goal_action,
                actions=(goal_action,) * 3,
            ),
        ):
            validate_insertion_goal_deltas(Path("/tmp/held-out"), results)
            results[-1]["goal_delta"][0] = -0.01
            with self.assertRaisesRegex(ValueError, "does not match telemetry"):
                validate_insertion_goal_deltas(Path("/tmp/held-out"), results)

    def test_rejects_reported_actions_that_do_not_match_raw_telemetry(self) -> None:
        action = DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        results = [
            {
                "context_index": context_index,
                "target_index": context_index + 3,
                "goal_delta": list(action.values),
                "recorded_actions": [list(action.values)] * 3,
            }
            for context_index in INSERTION_WINDOW.context_indices
        ]
        results[7]["recorded_actions"][0][0] = -0.01
        with patch(
            "jepa_wm.task_proposal_readiness.load_rollout_at",
            side_effect=lambda *args, context_index, **kwargs: SimpleNamespace(
                context=(SimpleNamespace(index=context_index),),
                target=SimpleNamespace(index=context_index + 3),
                goal_action=action,
                actions=(action,) * 3,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "do not match telemetry"):
                validate_insertion_goal_deltas(Path("/tmp/held-out"), results)

    @unittest.skipIf(torch is None, "PyTorch is required for checkpoint binding")
    def test_binds_complete_conditioning_to_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            proposal = Path(temporary_directory) / "insertion.pth"
            normalization = DroidValueNormalization(
                np.zeros(7, dtype=np.float32),
                np.ones(7, dtype=np.float32),
            )
            metadata = TrainingArtifactMetadata(
                "jepa_wm_droid", "revision", "wrist", ("train-00",), 3000
            )
            checkpoint = ActionProposalNetwork(
                feature_dimension=2,
                horizon=3,
                hidden_dimension=4,
                action_mean=torch.zeros((3, 7)),
                action_standard_deviation=torch.ones((3, 7)),
                conditioning=ProposalConditioning(
                    pose=normalization,
                    previous_action=normalization,
                    goal_delta=normalization,
                    task_progress=ScalarNormalization(21.0, 1.0),
                ),
            )
            selection = {
                "window": INSERTION_WINDOW.to_dict(),
                "selection_bounds": INSERTION_EVALUATION_BOUNDS.to_dict(),
                "recording_selections": [
                    {
                        "recording": "train-00",
                        "context_indices": list(INSERTION_WINDOW.context_indices),
                    }
                ],
                "rollouts": INSERTION_WINDOW.count,
            }
            selection_fingerprint = rollout_training_selection_fingerprint(selection)
            save_action_proposal(
                checkpoint,
                proposal,
                metadata,
                training_selection_fingerprint=selection_fingerprint,
            )
            proposal.with_suffix(".pth.json").write_text(
                json.dumps(
                    {
                        "proposal_fingerprint": artifact_fingerprint(proposal),
                        "metadata": metadata.to_dict(),
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": True,
                            "task_progress": True,
                        },
                        **selection,
                        "training_selection_fingerprint": selection_fingerprint,
                    }
                )
            )

            self.assertEqual(
                validate_insertion_proposal_identity(proposal).fingerprint,
                artifact_fingerprint(proposal),
            )
            sidecar = proposal.with_suffix(".pth.json")
            payload = json.loads(sidecar.read_text())
            payload["metadata"]["training_recordings"] = ["another-training-set"]
            sidecar.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_insertion_proposal_identity(proposal)
            payload["metadata"] = metadata.to_dict()
            sidecar.write_text(json.dumps(payload))
            payload["recording_selections"][0]["context_indices"][0] = 22
            sidecar.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_insertion_proposal_identity(proposal)
            payload["recording_selections"][0]["context_indices"][0] = 21
            sidecar.write_text(json.dumps(payload))
            proposal.write_bytes(proposal.read_bytes() + b"replaced")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                validate_insertion_proposal_identity(proposal)

    def test_requires_every_stationary_inclusive_training_context(self) -> None:
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 3000
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            proposal = Path(temporary_directory) / "insertion.pth"
            proposal.with_suffix(".pth.json").write_text(
                json.dumps(
                    {
                        "window": INSERTION_WINDOW.to_dict(),
                        "selection_bounds": INSERTION_EVALUATION_BOUNDS.to_dict(),
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": True,
                            "task_progress": True,
                        },
                        "rollouts": INSERTION_WINDOW.count - 1,
                        "recording_selections": [
                            {
                                "recording": "train-00",
                                "context_indices": list(
                                    INSERTION_WINDOW.context_indices[:-1]
                                ),
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "selection evidence"):
                validate_insertion_training_selection(proposal, metadata)

    def test_summary_requires_the_exact_canonical_corpus_roster(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            roster = InsertionCorpusRoster.create("insertion-v9-2600", 2600)
            roster_path = root / "roster.json"
            roster.write(roster_path)
            proposal = root / "proposal.pth"
            output = root / "readiness.json"
            reports = (root / "held-00.json", root / "held-01.json")
            policy = SimpleNamespace(
                summarize=Mock(
                    return_value={
                        "passed": False,
                        "corpus_roster": roster.to_dict(),
                    }
                ),
            )
            with patch(
                "jepa_wm.insertion_proposal_readiness.INSERTION_READINESS",
                policy,
            ):
                summary = summarize_insertion_proposal_readiness(
                    proposal,
                    reports,
                    output,
                    roster_path,
                )

            self.assertEqual(summary["corpus_roster"], roster.to_dict())
            expectations = policy.summarize.call_args.kwargs["corpus"]
            self.assertEqual(len(expectations), 14)
            self.assertEqual(expectations[0].seed, 2600)
            self.assertEqual(expectations[-1].seed, 12601)


if __name__ == "__main__":
    unittest.main()
