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
                self.assertEqual(json.loads(path.read_text())["status"], "running")
                for _ in range(10):
                    payload = json.loads(path.read_text())
                    if payload["status"] != "running":
                        return payload
                    await asyncio.sleep(0)
                self.fail("recording job did not persist a result")

            result = asyncio.run(exercise())

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["result"], {"recording_id": "demo-job"})

    def test_terminalizes_cancelled_background_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RecordingJobManager(Path(temp_dir))

            async def exercise() -> dict[str, object]:
                async def record(recording_id: str) -> dict[str, str]:
                    del recording_id
                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")

                started = manager.start("cancelled-job", record)
                await asyncio.sleep(0)
                task = next(iter(manager._tasks.values()))
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                return json.loads(Path(str(started["job"])).read_text())

            result = asyncio.run(exercise())

        self.assertEqual(result["status"], "error")
        self.assertIn("cancelled", str(result["error"]))

    def test_persists_progress_without_losing_the_heartbeat_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RecordingJobManager(Path(temp_dir))

            async def exercise() -> dict[str, object]:
                release = asyncio.Event()

                async def record(recording_id: str) -> dict[str, str]:
                    manager.progress(
                        recording_id,
                        phase="known_start_stabilization",
                        completed_units=1,
                        total_units=3,
                    )
                    await release.wait()
                    return {"recording_id": recording_id}

                started = manager.start("progress-job", record)
                await asyncio.sleep(0)
                payload = json.loads(Path(str(started["job"])).read_text())
                release.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                return payload

            running = asyncio.run(exercise())

        self.assertEqual(running["status"], "running")
        self.assertIsInstance(running["run_id"], str)
        self.assertEqual(running["phase"], "known_start_stabilization")
        self.assertEqual(running["completed_units"], 1)
        self.assertEqual(running["total_units"], 3)

    def test_cancels_a_background_job_by_recording_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RecordingJobManager(Path(temp_dir))

            async def exercise() -> tuple[dict[str, object], dict[str, object]]:
                async def record(recording_id: str) -> dict[str, str]:
                    del recording_id
                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")

                started = manager.start("managed-cancel", record)
                await asyncio.sleep(0)
                cancelled = manager.cancel("managed-cancel")
                for _ in range(10):
                    payload = json.loads(Path(str(started["job"])).read_text())
                    if payload["status"] != "running":
                        return cancelled, payload
                    await asyncio.sleep(0)
                self.fail("cancelled job did not terminalize")

            cancelled, terminal = asyncio.run(exercise())

        self.assertEqual(cancelled["status"], "cancelling")
        self.assertEqual(terminal["status"], "error")
        self.assertIn("cancelled", str(terminal["error"]))


if __name__ == "__main__":
    unittest.main()
