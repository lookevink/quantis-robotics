import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.training_artifact import artifact_fingerprint
from jepa_wm.unknown_start_live_action import (
    PREDECESSOR_SESSION_ID,
    claim,
    paths,
    runtime_fingerprint,
)


class UnknownStartLiveActionTest(unittest.TestCase):
    @staticmethod
    def write_recovery(data_root: Path) -> str:
        path = (
            data_root
            / "control_sessions"
            / PREDECESSOR_SESSION_ID
            / "rollback_recovery.json"
        )
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "quantis.unknown_start_rollback_recovery.v1",
                    "session_id": PREDECESSOR_SESSION_ID,
                    "recovered": True,
                    "recovery_mode": "drive_then_paused_reset_initialization",
                    "applied_model_actions": 0,
                    "drive_arm_error_radians": 8e-4,
                    "drive_gripper_error_meters": 5e-6,
                    "collision_detected": False,
                    "contact_force_newtons": 0.0,
                    "plug_attached": False,
                    "timeline_playing": False,
                }
            )
        )
        return artifact_fingerprint(path)

    def test_claim_is_single_action_non_filming_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "data"
            recovery = self.write_recovery(data_root)
            fingerprint = runtime_fingerprint()
            payload = claim(root, "a" * 40, fingerprint, data_root, recovery)

            self.assertEqual(payload["maximum_model_actions"], 1)
            self.assertFalse(payload["filming_authorized"])
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim(root, "a" * 40, fingerprint, data_root, recovery)
            self.assertTrue(paths(root)[0].is_file())

    def test_claim_rejects_runtime_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                claim(
                    Path(directory),
                    "a" * 40,
                    "f" * 64,
                    Path(directory) / "data",
                    "b" * 64,
                )

    def test_workflow_preflights_before_claim_and_executes_atomically_once(
        self,
    ) -> None:
        runner = Path("ops/run_unknown_start_live_action.sh").read_text()
        facade = Path("sim/isaac_demo.py").read_text()

        self.assertLess(
            runner.index("recover_unknown_start_candidate_rollback"),
            runner.index("preflight_unknown_start_shadow"),
        )
        self.assertLess(
            runner.index("preflight_unknown_start_shadow"),
            runner.index("unknown_start_live_action claim"),
        )
        self.assertIn("prepare_experimental_candidate_source", runner)
        self.assertEqual(runner.count("execute_unknown_start_candidate_action"), 1)
        self.assertNotIn("capture_unknown_start_candidate_observation", runner)
        self.assertNotIn("persist_experimental_candidate_response", runner)
        self.assertNotIn("apply_control_response", runner)
        self.assertIn("async def execute_unknown_start_candidate_action", facade)
        self.assertIn("await _capture_unknown_start_candidate_observation", facade)
        self.assertIn("persist_experimental_candidate_response", facade)
        self.assertIn("await _apply_control_response", facade)
        self.assertIn(
            'proposal_name="contact-grasp-v10-drive-slow-2600_task12_h256_s3000"',
            runner,
        )
        self.assertNotIn("experimental_shadow_candidate", runner)
        self.assertNotIn("record_candidate_demo", runner)
