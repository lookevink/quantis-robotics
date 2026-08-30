from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from jepa_wm.insertion_training_contract import (
    INSERTION_EPOCH_STEPS,
    INSERTION_ROLLOUTS_PER_RECORDING,
    INSERTION_TRAINING_BATCH_SIZE,
)


class InsertionTrainingContractTest(unittest.TestCase):
    def test_refreshed_corpus_epoch_covers_every_post_attachment_rollout(self) -> None:
        self.assertEqual(INSERTION_ROLLOUTS_PER_RECORDING, 168)
        self.assertEqual(INSERTION_TRAINING_BATCH_SIZE, 1)
        self.assertEqual(INSERTION_EPOCH_STEPS, 2016)

    def test_shell_epoch_default_uses_the_canonical_training_contract(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            runner = Path(temporary_directory) / "run.sh"
            runner.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                f"source '{repository}/ops/shell_helpers.sh'\n"
                f"insertion_epoch_steps '{repository}' python3\n"
            )
            result = subprocess.run(
                ["bash", str(runner)],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2016")


if __name__ == "__main__":
    unittest.main()
