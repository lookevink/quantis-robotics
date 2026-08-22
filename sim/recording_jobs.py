"""Background lifecycle and atomic status artifacts for long recordings."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import traceback
from typing import Any, Awaitable, Callable

from sim.recording import validate_recording_id


RecordingFactory = Callable[[str], Awaitable[dict[str, Any]]]


class RecordingJobManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._tasks: set[asyncio.Task[None]] = set()

    def start(
        self, recording_id: str, recording_factory: RecordingFactory
    ) -> dict[str, Any]:
        path = self.path_for(recording_id)
        if path.exists():
            raise ValueError(f"recording job already exists: {recording_id}")
        task = asyncio.ensure_future(self._run(recording_id, recording_factory))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return {
            "status": "recording",
            "recording_id": recording_id,
            "job": str(path),
        }

    def path_for(self, recording_id: str) -> Path:
        validate_recording_id(recording_id)
        return self.root / f"{recording_id}.json"

    async def _run(
        self, recording_id: str, recording_factory: RecordingFactory
    ) -> None:
        try:
            result = await recording_factory(recording_id)
        except Exception as error:
            self._write(
                recording_id,
                {
                    "status": "error",
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            return
        self._write(recording_id, {"status": "complete", "result": result})

    def _write(self, recording_id: str, payload: dict[str, Any]) -> None:
        path = self.path_for(recording_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        temporary.replace(path)
