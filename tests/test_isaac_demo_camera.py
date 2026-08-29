from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from sim.isaac_demo_camera import DemoRecorder, JEPA_WM_CAMERA_SPECS


class DemoRecorderTest(unittest.IsolatedAsyncioTestCase):
    async def test_deferred_camera_completes_and_detaches_terminal_lifecycle(
        self,
    ) -> None:
        product = Mock()
        annotator = Mock()
        annotator.get_data.return_value = np.zeros((4, 4, 4), dtype=np.uint8)
        replicator = ModuleType("omni.replicator.core")
        replicator.create = SimpleNamespace(render_product=Mock(return_value=product))
        replicator.AnnotatorRegistry = SimpleNamespace(
            get_annotator=Mock(return_value=annotator)
        )
        replicator.orchestrator = SimpleNamespace(step_async=AsyncMock())
        advance = AsyncMock()
        omni = ModuleType("omni")
        omni_replicator = ModuleType("omni.replicator")
        kit = ModuleType("omni.kit")
        app = ModuleType("omni.kit.app")
        app.get_app = Mock(return_value=SimpleNamespace(next_update_async=advance))
        omni.replicator = omni_replicator
        omni_replicator.core = replicator
        omni.kit = kit
        kit.app = app
        writer = Mock()
        writer.metadata = {}
        writer.frame_paths.return_value = {"wrist": Mock()}
        writer.finish.return_value = Mock()

        with (
            patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.replicator": omni_replicator,
                    "omni.replicator.core": replicator,
                    "omni.kit": kit,
                    "omni.kit.app": app,
                },
            ),
            patch(
                "sim.isaac_demo_camera.RecordingWriter",
                return_value=writer,
            ),
            patch("PIL.Image.fromarray") as fromarray,
        ):
            recorder = DemoRecorder(
                "deferred-lifecycle",
                camera_specs=JEPA_WM_CAMERA_SPECS,
                defer_camera_activation=True,
            )
            await recorder.initialize()
            self.assertFalse(recorder.cameras_active)
            recorder.activate_cameras()
            await recorder.prepare_current(Mock(), max_attempts=1)
            await recorder.capture_current(Mock())
            result = recorder.finish()

        self.assertEqual(result, writer.finish.return_value)
        replicator.orchestrator.step_async.assert_awaited_once()
        fromarray.return_value.save.assert_called_once()
        writer.add_step.assert_called_once()
        annotator.detach.assert_called_once_with([product])
        product.destroy.assert_called_once()
        self.assertFalse(recorder.cameras_active)

    def test_deferred_recorder_attaches_no_render_product_until_activation(
        self,
    ) -> None:
        product = Mock()
        annotator = Mock()
        replicator = ModuleType("omni.replicator.core")
        replicator.create = SimpleNamespace(render_product=Mock(return_value=product))
        replicator.AnnotatorRegistry = SimpleNamespace(
            get_annotator=Mock(return_value=annotator)
        )
        omni = ModuleType("omni")
        omni_replicator = ModuleType("omni.replicator")
        omni.replicator = omni_replicator
        omni_replicator.core = replicator

        with (
            patch.dict(
                sys.modules,
                {
                    "omni": omni,
                    "omni.replicator": omni_replicator,
                    "omni.replicator.core": replicator,
                },
            ),
            patch("sim.isaac_demo_camera.RecordingWriter"),
        ):
            recorder = DemoRecorder(
                "deferred-camera",
                camera_specs=JEPA_WM_CAMERA_SPECS,
                defer_camera_activation=True,
            )
            replicator.create.render_product.assert_not_called()
            annotator.attach.assert_not_called()

            recorder.activate_cameras()

        replicator.create.render_product.assert_called_once()
        annotator.attach.assert_called_once_with([product])

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
        app.get_app = Mock(return_value=SimpleNamespace(next_update_async=advance))
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
        app.get_app = Mock(return_value=SimpleNamespace(next_update_async=advance))
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
