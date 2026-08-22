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

    for label, camera_path in (
        ("presentation", PRESENTATION_CAMERA_PATH),
        ("wrist", WRIST_CAMERA_PATH),
    ):
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
