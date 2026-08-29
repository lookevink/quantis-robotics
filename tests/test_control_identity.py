from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.training_artifact import artifact_fingerprint
from jepa_wm.control_policy import ControlExecutionPolicy
from sim.control_identity import (
    ControlProposalRef,
    requires_authenticated_control_proposal,
)


class ControlProposalRefTest(unittest.TestCase):
    def _proposal(self, root: Path, name: str = "contact-grasp-v1") -> Path:
        proposal = root / f"{name}.pth"
        proposal.write_bytes(b"frozen proposal")
        proposal.with_suffix(".pth.json").write_text(
            json.dumps({"proposal_fingerprint": artifact_fingerprint(proposal)})
        )
        return proposal

    def test_authenticates_one_logical_name_to_exact_checkpoint_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal = self._proposal(root)

            reference = ControlProposalRef.from_name("contact-grasp-v1", root=root)

            self.assertEqual(reference.name, "contact-grasp-v1")
            self.assertEqual(reference.path, proposal.resolve())
            self.assertEqual(reference.fingerprint, artifact_fingerprint(proposal))
            self.assertEqual(
                reference,
                ControlProposalRef.from_dict(reference.to_dict()),
            )

    def test_rejects_a_filename_where_a_logical_name_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._proposal(root, "contact-grasp-v1.pth")

            with self.assertRaisesRegex(ValueError, "logical proposal name"):
                ControlProposalRef.from_name("contact-grasp-v1.pth", root=root)

    def test_rejects_missing_or_mismatched_proposal_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proposal = self._proposal(root)
            proposal.with_suffix(".pth.json").write_text(
                json.dumps({"proposal_fingerprint": "0" * 64})
            )

            with self.assertRaisesRegex(ValueError, "fingerprint"):
                ControlProposalRef.from_name("contact-grasp-v1", root=root)

    def test_only_model_command_policies_require_checkpoint_evidence(self) -> None:
        self.assertTrue(
            requires_authenticated_control_proposal(ControlExecutionPolicy.DIRECT)
        )
        self.assertTrue(
            requires_authenticated_control_proposal(
                ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT
            )
        )
        self.assertFalse(
            requires_authenticated_control_proposal(
                ControlExecutionPolicy.ZERO_BASELINE
            )
        )
        self.assertFalse(
            requires_authenticated_control_proposal(
                ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
            )
        )


if __name__ == "__main__":
    unittest.main()
