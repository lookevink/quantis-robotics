"""Camera mounting and RGB verification capture for the Isaac demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sim.isaac_demo_scene import (
    PRESENTATION_CAMERA_PATH,
    WRIST_CAMERA_PATH,
    matrix_to_wxyz,
)
from sim.recording import RecordingSnapshot, RecordingWriter


RECORDING_ROOT = "/isaac-sim/.local/share/ov/data/quantis/recordings"
CAMERA_SPECS = (
    ("presentation", PRESENTATION_CAMERA_PATH),
    ("wrist", WRIST_CAMERA_PATH),
)


def _look_at_rotation(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    camera_z = -forward
    camera_x = np.cross(up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    return np.column_stack((camera_x, camera_y, camera_z))


def configure_wrist_camera() -> dict[str, Any]:
    """Mount the wrist camera above and beside the arm, aimed at the gripper."""

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    camera = stage.GetPrimAtPath(WRIST_CAMERA_PATH)
    if not camera.IsValid():
        raise RuntimeError(f"camera prim is missing: {WRIST_CAMERA_PATH}")

    translation = np.array([0.16, 0.08, -0.20], dtype=np.float64)
    gripper_center = np.array([0.0, 0.0, 0.10], dtype=np.float64)
    orientation = matrix_to_wxyz(
        _look_at_rotation(
            translation, gripper_center, np.array([1.0, 0.0, 0.0])
        )
    )

    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*translation))
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(orientation[0], Gf.Vec3d(*orientation[1:]))
    )
    return {
        "translation": translation.tolist(),
        "orientation_wxyz": orientation.tolist(),
    }


class DemoRecorder:
    """Capture both demo cameras and synchronized robot state per update."""

    def __init__(self, recording_id: str, *, fps: int = 8) -> None:
        import omni.replicator.core as rep

        self._rep = rep
        self._writer = RecordingWriter(
            Path(RECORDING_ROOT),
            recording_id=recording_id,
            fps=fps,
            cameras=tuple(label for label, _ in CAMERA_SPECS),
        )
        self._render_products: dict[str, Any] = {}
        self._annotators: dict[str, Any] = {}
        for label, camera_path in CAMERA_SPECS:
            render_product = rep.create.render_product(camera_path, (640, 480))
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([render_product])
            self._render_products[label] = render_product
            self._annotators[label] = annotator

    @property
    def output_dir(self) -> Path:
        return self._writer.output_dir

    @property
    def frame_count(self) -> int:
        return self._writer.frame_count

    @property
    def video_paths(self) -> dict[str, Path]:
        return {
            camera: self.output_dir / f"{camera}.mp4"
            for camera in self._writer.cameras
        }

    async def initialize(self) -> None:
        """Warm Replicator before Isaac creates the physics articulation view."""

        await self._rep.orchestrator.step_async(
            rt_subframes=1,
            pause_timeline=True,
            delta_time=0.0,
        )

    async def capture(self, snapshot: RecordingSnapshot) -> None:
        import omni.kit.app
        from PIL import Image

        # The render products capture while the timeline is playing. A normal
        # Kit update keeps the physics view intact and advances their RGB data;
        # invoking Replicator's explicit step here would take over the timeline.
        await omni.kit.app.get_app().next_update_async()
        frame_paths = self._writer.frame_paths()
        for label, annotator in self._annotators.items():
            pixels = annotator.get_data()
            if pixels.size == 0:
                raise RuntimeError(f"camera produced an empty frame: {label}")
            Image.fromarray(pixels[:, :, :3]).save(frame_paths[label])
        self._writer.add_step(snapshot)

    def finish(self) -> Path:
        self._close()
        return self._writer.finish()

    def abort(self) -> None:
        """Release render products after a failed run without masking its error."""

        self._close()

    def _close(self) -> None:
        for label, annotator in self._annotators.items():
            render_product = self._render_products[label]
            annotator.detach([render_product])
            render_product.destroy()
        self._annotators.clear()
        self._render_products.clear()


async def capture_cameras(
    output_dir: str = "/isaac-sim/.local/share/ov/data/quantis/captures",
) -> dict[str, Any]:
    """Capture the presentation and wrist cameras from the current live pose."""

    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from PIL import Image

    stage = omni.usd.get_context().get_stage()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    captures = {}

    for label, camera_path in CAMERA_SPECS:
        if not stage.GetPrimAtPath(camera_path).IsValid():
            raise RuntimeError(f"camera prim is missing: {camera_path}")
        render_product = rep.create.render_product(camera_path, (640, 480))
        annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        try:
            annotator.attach([render_product])
            await rep.orchestrator.step_async(rt_subframes=4)
            pixels = annotator.get_data()
            path = destination / f"{label}.png"
            Image.fromarray(pixels[:, :, :3]).save(path)
            captures[label] = {
                "camera": camera_path,
                "path": str(path),
                "shape": list(pixels.shape),
            }
        finally:
            annotator.detach([render_product])
            render_product.destroy()

    return {
        "timeline_paused": not omni.timeline.get_timeline_interface().is_playing(),
        "captures": captures,
    }
