from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from sim.recording_jobs import RecordingJobManager


class RecordingJobManagerTest(unittest.TestCase):
    def test_persists_success_after_returning_the_background_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RecordingJobManager(Path(temp_dir))

            async def exercise() -> dict[str, object]:
                async def record(recording_id: str) -> dict[str, str]:
                    await asyncio.sleep(0)
                    return {"recording_id": recording_id}

                started = manager.start("demo-job", record)
                path = Path(str(started["job"]))
                for _ in range(10):
                    if path.is_file():
                        return json.loads(path.read_text())
                    await asyncio.sleep(0)
                self.fail("recording job did not persist a result")

            result = asyncio.run(exercise())

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["result"], {"recording_id": "demo-job"})


if __name__ == "__main__":
    unittest.main()
