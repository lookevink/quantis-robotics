"""Background lifecycle and atomic status artifacts for long recordings."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import traceback
from time import time
from typing import Any, Awaitable, Callable
from uuid import uuid4

from sim.recording import validate_recording_id


RecordingFactory = Callable[[str], Awaitable[dict[str, Any]]]
RECORDING_HEARTBEAT_INTERVAL_SECONDS = 5.0
RECORDING_HEARTBEAT_TIMEOUT_SECONDS = 30.0


def running_job_is_stale(payload: dict[str, Any], *, now: float | None = None) -> bool:
    if payload.get("status") != "running":
        return False
    heartbeat = payload.get("heartbeat_unix_seconds")
    run_id = payload.get("run_id")
    return (
        not isinstance(run_id, str)
        or not run_id
        or isinstance(heartbeat, bool)
        or not isinstance(heartbeat, (int, float))
        or (now if now is not None else time()) - float(heartbeat)
        > RECORDING_HEARTBEAT_TIMEOUT_SECONDS
    )


def job_is_quarantinable(payload: dict[str, Any]) -> bool:
    return payload.get("status") == "error" or running_job_is_stale(payload)


class RecordingJobManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._run_ids: dict[str, str] = {}
        self._progress: dict[str, dict[str, Any]] = {}

    def start(
        self, recording_id: str, recording_factory: RecordingFactory
    ) -> dict[str, Any]:
        path = self.path_for(recording_id)
        if path.exists():
            raise ValueError(f"recording job already exists: {recording_id}")
        run_id = uuid4().hex
        self._run_ids[recording_id] = run_id
        self._write_running(recording_id, run_id)
        task = asyncio.ensure_future(self._run(recording_id, run_id, recording_factory))
        self._tasks[recording_id] = task
        task.add_done_callback(
            lambda completed, identity=recording_id: self._forget(identity, completed)
        )
        return {
            "status": "recording",
            "recording_id": recording_id,
            "job": str(path),
        }

    def progress(
        self,
        recording_id: str,
        *,
        phase: str,
        completed_units: int,
        total_units: int,
    ) -> None:
        """Persist bounded progress for one running background operation."""

        if (
            recording_id not in self._tasks
            or not phase
            or isinstance(completed_units, bool)
            or not isinstance(completed_units, int)
            or isinstance(total_units, bool)
            or not isinstance(total_units, int)
            or completed_units < 0
            or total_units <= 0
            or completed_units > total_units
        ):
            raise ValueError("recording job progress is invalid")
        self._progress[recording_id] = {
            "phase": phase,
            "completed_units": completed_units,
            "total_units": total_units,
            "progress_unix_seconds": time(),
        }
        self._write_running(recording_id, self._run_ids[recording_id])

    def cancel(self, recording_id: str) -> dict[str, Any]:
        """Request cancellation of one live task by its persisted identity."""

        validate_recording_id(recording_id)
        task = self._tasks.get(recording_id)
        if task is None or task.done():
            raise ValueError(f"recording job is not running: {recording_id}")
        task.cancel()
        return {"status": "cancelling", "recording_id": recording_id}

    def _forget(
        self,
        recording_id: str,
        completed: asyncio.Task[None],
    ) -> None:
        if self._tasks.get(recording_id) is completed:
            self._tasks.pop(recording_id, None)
            self._run_ids.pop(recording_id, None)
            self._progress.pop(recording_id, None)

    def path_for(self, recording_id: str) -> Path:
        validate_recording_id(recording_id)
        return self.root / f"{recording_id}.json"

    async def _run(
        self,
        recording_id: str,
        run_id: str,
        recording_factory: RecordingFactory,
    ) -> None:
        heartbeat = asyncio.ensure_future(self._heartbeat(recording_id, run_id))
        cancelled = False
        try:
            result = await recording_factory(recording_id)
            payload = {"status": "complete", "result": result}
        except asyncio.CancelledError:
            cancelled = True
            payload = {
                "status": "error",
                "error": "recording task was cancelled",
            }
        except Exception as error:
            payload = {
                "status": "error",
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        finally:
            heartbeat.cancel()
            try:
                await heartbeat
            except asyncio.CancelledError:
                pass
        self._write(
            recording_id,
            {
                "recording_id": recording_id,
                "run_id": run_id,
                **self._progress.get(recording_id, {}),
                **payload,
            },
        )
        if cancelled:
            raise asyncio.CancelledError

    async def _heartbeat(self, recording_id: str, run_id: str) -> None:
        while True:
            await asyncio.sleep(RECORDING_HEARTBEAT_INTERVAL_SECONDS)
            self._write_running(recording_id, run_id)

    def _write_running(self, recording_id: str, run_id: str) -> None:
        self._write(
            recording_id,
            {
                "status": "running",
                "recording_id": recording_id,
                "run_id": run_id,
                "heartbeat_unix_seconds": time(),
                **self._progress.get(recording_id, {}),
            },
        )

    def _write(self, recording_id: str, payload: dict[str, Any]) -> None:
        path = self.path_for(recording_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
