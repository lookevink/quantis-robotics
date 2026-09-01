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

    def test_workflow_preflights_before_claim_and_applies_once(self) -> None:
        runner = Path("ops/run_unknown_start_live_action.sh").read_text()

        self.assertLess(
            runner.index("preflight_unknown_start_shadow"),
            runner.index("unknown_start_live_action claim"),
        )
        self.assertEqual(runner.count("apply_control_response"), 1)
        self.assertIn("prepare_experimental_candidate_source", runner)
        self.assertIn("persist_experimental_candidate_response", runner)
        self.assertNotIn("record_candidate_demo", runner)
