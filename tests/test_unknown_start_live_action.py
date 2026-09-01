from pathlib import Path
import tempfile
import unittest

from jepa_wm.unknown_start_live_action import claim, paths, runtime_fingerprint


class UnknownStartLiveActionTest(unittest.TestCase):
    def test_claim_is_single_action_non_filming_and_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fingerprint = runtime_fingerprint()
            payload = claim(root, "a" * 40, fingerprint)

            self.assertEqual(payload["maximum_model_actions"], 1)
            self.assertFalse(payload["filming_authorized"])
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim(root, "a" * 40, fingerprint)
            self.assertTrue(paths(root)[0].is_file())

    def test_claim_rejects_runtime_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "runtime changed"):
                claim(Path(directory), "a" * 40, "f" * 64)

    def test_workflow_preflights_before_claim_and_executes_atomically_once(
        self,
    ) -> None:
        runner = Path("ops/run_unknown_start_live_action.sh").read_text()
        facade = Path("sim/isaac_demo.py").read_text()

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
