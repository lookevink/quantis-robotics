"""Camera mounting and RGB verification capture for the Isaac demo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jepa.contract import DEFAULT_FRAMES, ObservationStage
from sim.isaac_demo_scene import (
    PRESENTATION_CAMERA_PATH,
    WRIST_CAMERA_PATH,
    matrix_to_wxyz,
)
from sim.recording import RecordingSnapshot, RecordingWriter


RECORDING_ROOT = "/isaac-sim/.local/share/ov/data/quantis/recordings"
RECORDING_JOB_ROOT = "/isaac-sim/.local/share/ov/data/quantis/recording_jobs"
DEMO_RESOLUTION = (1920, 1080)
JEPA_WM_RESOLUTION = (512, 512)
DEMO_FPS = 12


@dataclass(frozen=True)
class CameraSpec:
    label: str
    path: str
    resolution: tuple[int, int]


CAMERA_SPECS = (
    CameraSpec("presentation", PRESENTATION_CAMERA_PATH, DEMO_RESOLUTION),
    CameraSpec("wrist", WRIST_CAMERA_PATH, DEMO_RESOLUTION),
)
JEPA_WM_CAMERA_SPECS = (CameraSpec("wrist", WRIST_CAMERA_PATH, JEPA_WM_RESOLUTION),)


def _look_at_rotation(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    camera_z = -forward
    camera_x = np.cross(up, camera_z)
    camera_x /= np.linalg.norm(camera_x)
    camera_y = np.cross(camera_z, camera_x)
    return np.column_stack((camera_x, camera_y, camera_z))


async def _wait_for_rgb(
    annotators: Mapping[str, Any],
    advance: Any,
    *,
    advance_before_first_read: bool = True,
    max_attempts: int = 8,
) -> dict[str, np.ndarray]:
    """Read all annotators after advancing until every RGB frame is complete."""

    pixels_by_camera: dict[str, np.ndarray] = {}
    last_shapes: dict[str, tuple[int, ...]] = {}
    for attempt in range(max_attempts):
        if advance_before_first_read or attempt > 0:
            await advance()
        pixels_by_camera = {
            label: np.asarray(annotator.get_data())
            for label, annotator in annotators.items()
        }
        last_shapes = {
            label: pixels.shape for label, pixels in pixels_by_camera.items()
        }
        if all(
            pixels.ndim == 3 and pixels.shape[2] >= 3 and pixels.size
            for pixels in pixels_by_camera.values()
        ):
            return pixels_by_camera
    raise RuntimeError(f"cameras did not produce complete RGB frames: {last_shapes}")


def configure_wrist_camera(
    translation_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """Mount the wrist camera above and beside the arm, aimed at the gripper."""

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    camera = stage.GetPrimAtPath(WRIST_CAMERA_PATH)
    if not camera.IsValid():
        raise RuntimeError(f"camera prim is missing: {WRIST_CAMERA_PATH}")

    offset = np.asarray(translation_offset, dtype=np.float64)
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError("wrist camera offset must contain three finite values")
    translation = np.array([0.16, 0.08, -0.20], dtype=np.float64) + offset
    gripper_center = np.array([0.0, 0.0, 0.10], dtype=np.float64)
    orientation = matrix_to_wxyz(
        _look_at_rotation(translation, gripper_center, np.array([1.0, 0.0, 0.0]))
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

    def __init__(
        self,
        recording_id: str,
        *,
        fps: int = DEMO_FPS,
        minimum_stage_frames: int = DEFAULT_FRAMES,
        camera_specs: tuple[CameraSpec, ...] = CAMERA_SPECS,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        import omni.replicator.core as rep

        self._rep = rep
        if minimum_stage_frames < 0:
            raise ValueError("minimum stage frames must be non-negative")
        self.minimum_stage_frames = minimum_stage_frames
        self._writer = RecordingWriter(
            Path(RECORDING_ROOT),
            recording_id=recording_id,
            fps=fps,
            camera_resolutions={spec.label: spec.resolution for spec in camera_specs},
            metadata=metadata,
        )
        self._render_products: dict[str, Any] = {}
        self._annotators: dict[str, Any] = {}
        for spec in camera_specs:
            render_product = rep.create.render_product(spec.path, spec.resolution)
            annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            annotator.attach([render_product])
            self._render_products[spec.label] = render_product
            self._annotators[spec.label] = annotator

    @property
    def output_dir(self) -> Path:
        return self._writer.output_dir

    @property
    def frame_count(self) -> int:
        return self._writer.frame_count

    @property
    def fps(self) -> int:
        return self._writer.fps

    def stage_frame_count(self, stage: ObservationStage) -> int:
        return self._writer.stage_frame_count(stage)

    @property
    def video_paths(self) -> dict[str, Path]:
        return {
            camera: self.output_dir / f"{camera}.mp4" for camera in self._writer.cameras
        }

    async def initialize(self) -> None:
        """Warm Replicator before Isaac creates the physics articulation view."""

        await self._rep.orchestrator.step_async(
            rt_subframes=4,
            pause_timeline=True,
            delta_time=0.0,
        )

    async def capture(
        self,
        snapshot: RecordingSnapshot,
        *,
        advance: bool = True,
    ) -> None:
        import omni.kit.app
        from PIL import Image

        # The render products capture while the timeline is playing. A normal
        # Kit update keeps the physics view intact and advances their RGB data;
        # invoking Replicator's explicit step here would take over the timeline.
        app = omni.kit.app.get_app()
        pixels_by_camera = await _wait_for_rgb(
            self._annotators,
            app.next_update_async,
            advance_before_first_read=advance,
        )

        frame_paths = self._writer.frame_paths()
        for label, pixels in pixels_by_camera.items():
            Image.fromarray(pixels[:, :, :3]).save(frame_paths[label], compress_level=1)
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

    import omni.timeline

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    captures = {}

    for spec in CAMERA_SPECS:
        captures[spec.label] = {
            "camera": spec.path,
            **await capture_camera_frame(spec, destination / f"{spec.label}.png"),
        }

    return {
        "timeline_paused": not omni.timeline.get_timeline_interface().is_playing(),
        "captures": captures,
    }


async def capture_camera_frame(spec: CameraSpec, path: Path) -> dict[str, Any]:
    """Capture one current RGB observation to an explicit shared path."""

    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image

    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(spec.path).IsValid():
        raise RuntimeError(f"camera prim is missing: {spec.path}")
    render_product = rep.create.render_product(spec.path, spec.resolution)
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    try:
        annotator.attach([render_product])
        pixels = (
            await _wait_for_rgb(
                {spec.label: annotator},
                lambda: rep.orchestrator.step_async(rt_subframes=4),
            )
        )[spec.label]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels[:, :, :3]).save(path, compress_level=1)
        return {"path": str(path), "shape": list(pixels.shape)}
    finally:
        annotator.detach([render_product])
        render_product.destroy()
