from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from jepa_wm.contact_grasp_horizon_completion import (
    HANDOFF_SCHEMA,
    MAXIMUM_ACTIONS,
    PROPOSAL_FINGERPRINT,
    PROPOSAL_NAME,
    SOURCE_APPLIED_ACTIONS,
    SOURCE_CUMULATIVE_APPLIED_ACTIONS,
    SOURCE_HORIZON_ACTIONS,
    SOURCE_SESSION_COUNT,
    SOURCE_SESSION_ID,
    WORKER_FINGERPRINT,
    WORKER_IDENTITY,
    ContactGraspHorizonCompletion,
    failure,
    paths,
    rollback_drive_target,
    runtime_fingerprint,
)


class ContactGraspHorizonCompletionTest(unittest.TestCase):
    def test_freezes_the_expanded_model_worker_and_action_horizon(self) -> None:
        handoff = ContactGraspHorizonCompletion(
            "unknown-start-e2e-v27-62605-grasp-001",
            runtime_fingerprint(),
            "1" * 40,
        )

        self.assertEqual(
            ContactGraspHorizonCompletion.from_dict(handoff.to_dict()),
            handoff,
        )
        payload = handoff.to_dict()
        self.assertEqual(payload["schema"], HANDOFF_SCHEMA)
        self.assertEqual(payload["source_session_id"], SOURCE_SESSION_ID)
        self.assertEqual(payload["source_attempted_actions"], SOURCE_SESSION_COUNT)
        self.assertEqual(payload["source_applied_actions"], SOURCE_APPLIED_ACTIONS)
        self.assertEqual(
            payload["source_cumulative_applied_actions"],
            SOURCE_CUMULATIVE_APPLIED_ACTIONS,
        )
        self.assertEqual(payload["source_horizon_actions"], SOURCE_HORIZON_ACTIONS)
        self.assertEqual(payload["maximum_actions"], MAXIMUM_ACTIONS)
        self.assertEqual(payload["proposal_name"], PROPOSAL_NAME)
        self.assertEqual(payload["proposal_fingerprint"], PROPOSAL_FINGERPRINT)
        self.assertEqual(payload["worker_identity"], WORKER_IDENTITY)
        self.assertEqual(payload["worker_fingerprint"], WORKER_FINGERPRINT)
        self.assertTrue(payload["simulator_action_authorized"])
        self.assertFalse(payload["filming_authorized"])
        self.assertFalse(payload["production_authority_granted"])

    def test_rejects_any_changed_frozen_field(self) -> None:
        handoff = ContactGraspHorizonCompletion(
            "unknown-start-e2e-v27-62605-grasp-001",
            "2" * 64,
            "1" * 40,
        )
        payload = handoff.to_dict()
        payload["maximum_actions"] = 52

        with self.assertRaisesRegex(ValueError, "changed"):
            ContactGraspHorizonCompletion.from_dict(payload)

        with self.assertRaisesRegex(ValueError, "invalid"):
            replace(handoff, runtime_fingerprint="short")

    def test_failure_terminalization_is_idempotent_for_inherited_err_traps(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)
            claim_path, _, _, failure_path = paths(checkpoint_root)
            claim_path.parent.mkdir(parents=True)
            claim_path.write_text(json.dumps({"claim": "frozen"}) + "\n")

            first = failure(checkpoint_root, "status_001:exit_1")
            second = failure(checkpoint_root, "status_001:exit_1")

            self.assertEqual(second, first)
            self.assertEqual(json.loads(failure_path.read_text()), first)
            with self.assertRaisesRegex(ValueError, "invalid"):
                failure(checkpoint_root, "status_002:exit_1")

    def test_rollback_target_uses_synchronized_pre_action_refresh(self) -> None:
        refreshed = SimpleNamespace(
            joint_positions=(0.1,) * 7,
            gripper_width_m=0.03,
        )
        result = SimpleNamespace(
            insertion_trial_refresh=SimpleNamespace(live_state=refreshed),
        )

        target = rollback_drive_target(result)

        self.assertLess(
            max(
                abs(actual - expected)
                for actual, expected in zip(
                    target.joint_positions,
                    refreshed.joint_positions,
                )
            ),
            1e-8,
        )
        self.assertAlmostEqual(
            target.gripper_width_m,
            refreshed.gripper_width_m,
        )

    def test_rollback_target_requires_synchronized_pre_action_refresh(self) -> None:
        with self.assertRaisesRegex(ValueError, "rollback refresh is missing"):
            rollback_drive_target(SimpleNamespace(insertion_trial_refresh=None))

    def test_aws_workflow_gates_v27_on_the_no_actuation_diagnostic(self) -> None:
        aws = (Path(__file__).resolve().parents[1] / "ops" / "aws.sh").read_text()
        workflow = aws[aws.index("jepa-wm-contact-grasp-horizon-completion)") :]

        diagnostic = workflow.index(
            "demo.diagnose_contact_grasp_tracking_rollback("
            "'unknown-start-e2e-v25-62605-grasp-066')"
        )
        physical_run = workflow.index("run_unknown_start_horizon_completion.sh")
        self.assertLess(diagnostic, physical_run)


if __name__ == "__main__":
    unittest.main()
