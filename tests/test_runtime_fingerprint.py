from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from jepa_wm.runtime_fingerprint import (
    runtime_source_files,
    runtime_source_fingerprint,
)


class RuntimeFingerprintTest(unittest.TestCase):
    def test_discovers_critical_transitive_runtime_sources(self) -> None:
        root = Path(__file__).resolve().parents[1]

        files = runtime_source_files(root)

        self.assertIn("sim/grasp_task.py", files)
        self.assertIn("sim/isaac_demo_runtime.py", files)
        self.assertIn("jepa_wm/control_tracking.py", files)
        self.assertIn("jepa_wm/joint_settlement.py", files)
        self.assertIn("jepa_wm/insertion_trial.py", files)
        self.assertIn("scripts/validate.sh", files)

    def test_adding_a_runtime_source_invalidates_the_fingerprint(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "jepa_wm" / "existing.py"
            source.parent.mkdir(parents=True)
            source.write_text("VALUE = 1\n")
            before = runtime_source_fingerprint(root)

            future = root / "future_runtime"
            future.mkdir()
            (future / "new_dependency.py").write_text("VALUE = 2\n")

            self.assertNotEqual(runtime_source_fingerprint(root), before)


if __name__ == "__main__":
    unittest.main()
