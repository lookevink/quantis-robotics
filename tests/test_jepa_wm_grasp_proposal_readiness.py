from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.grasp_proposal_readiness import validate_grasp_evaluation_window


class GraspProposalReadinessTest(unittest.TestCase):
    def test_rejects_an_evaluation_that_mixes_exploration_with_the_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "recording": "/tmp/recording",
                        "window": {
                            "start_index": 49,
                            "count": 50,
                            "stride": 1,
                        },
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "complete task window"):
                validate_grasp_evaluation_window(report)


if __name__ == "__main__":
    unittest.main()
