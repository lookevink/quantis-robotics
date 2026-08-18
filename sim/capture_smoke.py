"""Create a Franka scene and verify synchronized RGB/action/state capture.

This is deliberately a data-pipeline smoke test, not the maintenance task. It
moves a rigid red module through an extraction-like path while a built-in
Franka is visible. Replace the synthetic module delta with the controller's
actual 7D end-effector-plus-gripper command when the task controller lands.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
# 64 frames matches the V-JEPA 2 clip length the embedder reads, so a capture
# feeds the encoder without repeated frames. See jepa/embed_episode.py.
parser.add_argument("--frames", type=int, default=64)
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom

sys.path.insert(0, "/workspace")
from sim.episode import EpisodeWriter


def main() -> None:
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")

    episode_id = datetime.now(UTC).strftime("smoke-%Y%m%dT%H%M%SZ")
    episode_dir = args.output / episode_id
    episode = EpisodeWriter(
        episode_dir,
        task="rigid-module-extraction-smoke",
        robot="Franka Panda",
        action_labels=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        fps=4.0,
    )
    rgb_dir = episode_dir / "rgb"
    rgb_dir.mkdir()

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    stage = omni.usd.get_context().get_stage()

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("Isaac Sim asset root is unavailable")
    add_reference_to_stage(
        usd_path=assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        prim_path="/World/Franka",
    )

    module = UsdGeom.Cube.Define(stage, Sdf.Path("/World/HotSwapModule"))
    module.CreateSizeAttr(1.0)
    module.AddScaleOp().Set(Gf.Vec3f(0.18, 0.06, 0.035))
    translate = module.AddTranslateOp()
    display_color = module.CreateDisplayColorAttr()
    display_color.Set([Gf.Vec3f(0.8, 0.03, 0.03)])

    rep.create.light(light_type="Dome", intensity=1100.0)
    camera = rep.create.camera(
        position=(1.7, 1.7, 1.25),
        look_at=(0.35, 0.0, 0.45),
    )
    render_product = rep.create.render_product(camera, (512, 512))
    backend = rep.backends.get("DiskBackend")
    backend.initialize(output_dir=str(rgb_dir))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(backend=backend, rgb=True)
    writer.attach(render_product)
    rep.orchestrator.set_capture_on_play(False)

    recorded: list[tuple[list[float], dict[str, object]]] = []
    last_x = 0.45
    for index in range(args.frames):
        fraction = index / (args.frames - 1)
        x = 0.45 + 0.30 * fraction
        translate.Set(Gf.Vec3d(x, 0.0, 0.50))
        delta_x = x - last_x if index else 0.0
        recorded.append(
            (
                [delta_x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                {
                    "module_position": [x, 0.0, 0.50],
                    "progress": fraction,
                    "source": "capture-smoke",
                },
            )
        )
        rep.orchestrator.step(rt_subframes=2)
        last_x = x

    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()

    image_paths = sorted(rgb_dir.rglob("*.png"))
    if len(image_paths) != len(recorded):
        raise RuntimeError(
            f"captured {len(image_paths)} RGB frames for {len(recorded)} state records"
        )

    for frame, (action, state) in zip(image_paths, recorded, strict=True):
        episode.add_step(frame=frame, action=action, state=state)
    episode.finish(
        success=True,
        extra={
            "camera": "world_camera",
            "resolution": [512, 512],
            "warning": "Smoke-test motion is not a Franka control trajectory.",
        },
    )
    print(f"Wrote episode: {episode_dir}")


try:
    main()
finally:
    simulation_app.close()
