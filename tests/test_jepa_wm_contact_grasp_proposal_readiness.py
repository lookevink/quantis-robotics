from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.action import DroidAction
from jepa_wm.contact_grasp_proposal_readiness import (
    CONTACT_GRASP_EVALUATION_BOUNDS,
    CONTACT_GRASP_WINDOW,
    validate_contact_grasp_evaluation_window,
    validate_contact_grasp_goal_deltas,
    validate_contact_grasp_training_selection,
)
from jepa_wm.task_windows import proposal_window
from jepa_wm.training_artifact import TrainingArtifactMetadata


class ContactGraspProposalReadinessTest(unittest.TestCase):
    def test_window_covers_close_attachment_and_retained_grasp(self) -> None:
        self.assertEqual(
            CONTACT_GRASP_WINDOW.to_dict(),
            {"start_index": 18, "count": 8, "stride": 1},
        )
        self.assertEqual(proposal_window("contact-grasp"), CONTACT_GRASP_WINDOW)

    def test_training_requires_every_contact_grasp_context(self) -> None:
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid", "revision", "wrist", ("train-00",), 3000
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            proposal = Path(temporary_directory) / "contact-grasp.pth"
            proposal.with_suffix(".pth.json").write_text(
                json.dumps(
                    {
                        "window": CONTACT_GRASP_WINDOW.to_dict(),
                        "selection_bounds": CONTACT_GRASP_EVALUATION_BOUNDS.to_dict(),
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": True,
                            "task_progress": True,
                        },
                        "rollouts": CONTACT_GRASP_WINDOW.count - 1,
                        "recording_selections": [
                            {
                                "recording": "train-00",
                                "context_indices": list(
                                    CONTACT_GRASP_WINDOW.context_indices[:-1]
                                ),
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "selection evidence"):
                validate_contact_grasp_training_selection(proposal, metadata)

    def test_evaluation_requires_contact_insertion_held_out_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "recording": str(root / "held-00"),
                        "window": CONTACT_GRASP_WINDOW.to_dict(),
                        "selection_bounds": CONTACT_GRASP_EVALUATION_BOUNDS.to_dict(),
                        "conditioning": {
                            "proprioception": True,
                            "action_history": True,
                            "goal_delta": True,
                            "task_progress": True,
                        },
                        "results": [],
                    }
                )
            )
            with patch(
                "jepa_wm.contact_grasp_proposal_readiness.ContactGraspEvidence.from_recording",
                side_effect=ValueError("not contact-aware"),
            ):
                with self.assertRaisesRegex(ValueError, "not contact-aware"):
                    validate_contact_grasp_evaluation_window(report)

    def test_goal_deltas_are_bound_to_the_contact_window(self) -> None:
        action = DroidAction((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.2))
        results = [
            {
                "context_index": context_index,
                "target_index": context_index + 3,
                "goal_delta": list(action.values),
                "recorded_actions": [list(action.values)] * 3,
            }
            for context_index in CONTACT_GRASP_WINDOW.context_indices
        ]
        with patch(
            "jepa_wm.task_proposal_readiness.load_rollout_at",
            side_effect=lambda *args, context_index, **kwargs: SimpleNamespace(
                context=(SimpleNamespace(index=context_index),),
                target=SimpleNamespace(index=context_index + 3),
                goal_action=action,
                actions=(action,) * 3,
            ),
        ):
            validate_contact_grasp_goal_deltas(Path("/tmp/held-out"), results)
            results[0]["goal_delta"][-1] = 0.2
            with self.assertRaisesRegex(ValueError, "does not match telemetry"):
                validate_contact_grasp_goal_deltas(Path("/tmp/held-out"), results)


if __name__ == "__main__":
    unittest.main()
