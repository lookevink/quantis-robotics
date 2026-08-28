from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from sim.isaac_demo_camera import DemoRecorder


class DemoRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def test_prepare_current_observes_every_camera_warmup_update(
        self,
    ) -> None:
        advance = AsyncMock()
        observe = Mock()
        annotator = Mock()
        annotator.get_data.side_effect = (
            np.empty((0,), dtype=np.uint8),
            np.zeros((4, 4, 4), dtype=np.uint8),
        )
        recorder = object.__new__(DemoRecorder)
        recorder._annotators = {"wrist": annotator}
        recorder._writer = Mock()
        omni = ModuleType("omni")
        kit = ModuleType("omni.kit")
        app = ModuleType("omni.kit.app")
        app.get_app = Mock(
            return_value=SimpleNamespace(next_update_async=advance)
        )
        omni.kit = kit
        kit.app = app

        with patch.dict(
            sys.modules,
            {
                "omni": omni,
                "omni.kit": kit,
                "omni.kit.app": app,
            },
        ):
            await recorder.prepare_current(observe, max_attempts=2)

        self.assertEqual(advance.await_count, 2)
        self.assertEqual(observe.call_count, 2)
        recorder._writer.frame_paths.assert_not_called()

    async def test_current_frame_capture_never_advances_for_incomplete_rgb(
        self,
    ) -> None:
        advance = AsyncMock()
        annotator = Mock()
        annotator.get_data.return_value = np.empty((0,), dtype=np.uint8)
        recorder = object.__new__(DemoRecorder)
        recorder._annotators = {"wrist": annotator}
        recorder._writer = Mock()
        omni = ModuleType("omni")
        kit = ModuleType("omni.kit")
        app = ModuleType("omni.kit.app")
        app.get_app = Mock(
            return_value=SimpleNamespace(next_update_async=advance)
        )
        omni.kit = kit
        kit.app = app

        with (
            patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.kit": kit,
                    "omni.kit.app": app,
                },
            ),
            self.assertRaisesRegex(RuntimeError, "complete RGB"),
        ):
            await recorder.capture_current(Mock())

        advance.assert_not_awaited()
        recorder._writer.frame_paths.assert_not_called()


if __name__ == "__main__":
    unittest.main()
