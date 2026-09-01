from hashlib import sha256
from pathlib import Path
import tempfile
import unittest

from jepa_wm.physical_shadow_replay import claim, load_experiment


class PhysicalShadowReplayTest(unittest.TestCase):
    def test_frozen_contract_is_one_offline_non_actuating_replay(self) -> None:
        experiment = load_experiment(
            Path(".scratch/jepa-physical-shadow-replay-v1/experiment-config.json")
        )

        self.assertEqual(experiment["execution"]["replays"], 1)
        self.assertFalse(experiment["execution"]["isaac"])
        self.assertFalse(experiment["execution"]["apply_action"])
        self.assertFalse(experiment["execution"]["train"])

    def test_claim_authenticates_inputs_and_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            files = {}
            for name in ("request.json", "shadow.json"):
                artifact = source / name
                artifact.write_text(name)
                files[name] = sha256(name.encode()).hexdigest()
            manifest = root / "worker.json"
            manifest.write_text("worker")
            experiment = {
                "source": {"session": str(source), "files": files},
                "worker": {
                    "manifest": str(manifest),
                    "fingerprint": sha256(b"worker").hexdigest(),
                },
                "output": str(root / "output"),
            }

            payload = claim(experiment)

            self.assertEqual(payload["replays_claimed"], 1)
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim(experiment)

    def test_runner_has_no_simulator_or_actuation_seam(self) -> None:
        runner = Path("ops/run_physical_shadow_replay.sh").read_text()

        self.assertIn("jepa_wm.control_client", runner)
        self.assertNotIn("isaac_server_call", runner)
        self.assertNotIn("apply_control_response", runner)
        self.assertNotIn("capture_and_respond_control_session", runner)


if __name__ == "__main__":
    unittest.main()
