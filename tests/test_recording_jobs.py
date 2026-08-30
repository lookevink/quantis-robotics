from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path

from sim.recording_jobs import RecordingJobManager


class RecordingJobManagerTest(unittest.TestCase):
    def test_facade_capture_paths_share_the_operation_interlock(self) -> None:
        from sim import isaac_demo

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RecordingJobManager(Path(temp_dir))

            async def exercise() -> None:
                release = asyncio.Event()

                async def foreground() -> dict[str, str]:
                    await release.wait()
                    return {"status": "complete"}

                task = asyncio.create_task(
                    manager.run_exclusive("active-motion", foreground)
                )
                await asyncio.sleep(0)
                with (
                    patch.object(isaac_demo, "_RECORDING_JOBS", manager),
                    patch.object(
                        isaac_demo,
                        "_capture_followup_observation",
                        new=AsyncMock(),
                    ),
                    patch.object(
                        isaac_demo,
                        "_capture_cameras",
                        new=AsyncMock(),
                    ),
                    patch.object(
                        isaac_demo,
                        "_persist_insertion_followup_response",
                    ),
                ):
                    with self.assertRaisesRegex(
                        ValueError,
                        "another simulator operation",
                    ):
                        await isaac_demo.capture_followup_observation(
                            "next-session",
                            "previous-session",
                            "proposal",
                        )
                    with self.assertRaisesRegex(
                        ValueError,
                        "another simulator operation",
                    ):
                        await isaac_demo.capture_cameras()
                    with self.assertRaisesRegex(
                        ValueError,
                        "another simulator operation",
                    ):
                        isaac_demo.persist_insertion_followup_response(
                            "execution-session",
                            "safety-session",
                        )
                release.set()
                await task

            asyncio.run(exercise())

    def test_rejects_a_second_concurrent_simulator_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RecordingJobManager(Path(temp_dir))

            async def exercise() -> None:
                async def record(recording_id: str) -> dict[str, str]:
                    del recording_id
                    await asyncio.Event().wait()
                    raise AssertionError("unreachable")

                manager.start("first-job", record)
                await asyncio.sleep(0)
                with self.assertRaisesRegex(
                    ValueError,
                    "another simulator operation",
                ):
                    manager.start("second-job", record)

            asyncio.run(exercise())

    def test_foreground_action_interlocks_background_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RecordingJobManager(Path(temp_dir))

            async def exercise() -> None:
                release = asyncio.Event()

                async def foreground() -> dict[str, str]:
                    await release.wait()
                    return {"status": "complete"}

                async def record(recording_id: str) -> dict[str, str]:
                    return {"recording_id": recording_id}

                task = asyncio.create_task(
                    manager.run_exclusive("foreground-action", foreground)
                )
                await asyncio.sleep(0)
                self.assertEqual(manager.active_operation_id(), "foreground-action")
                with self.assertRaisesRegex(
                    ValueError,
                    "another simulator operation",
                ):
                    manager.start("blocked-recording", record)
                release.set()
                self.assertEqual(await task, {"status": "complete"})
                self.assertIsNone(manager.active_operation_id())

            asyncio.run(exercise())

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
