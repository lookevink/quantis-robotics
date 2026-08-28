#!/usr/bin/env python3
"""Render the silent four-action insertion demo from authenticated replay frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1920
HEIGHT = 1080
FPS = 12
BACKGROUND = (7, 12, 20)
PANEL = (10, 18, 30, 224)
WHITE = (242, 247, 252)
MUTED = (157, 174, 194)
TEAL = (65, 220, 196)
VIOLET = (165, 132, 255)
AMBER = (255, 184, 84)
GREEN = (83, 224, 137)
GRID = (48, 62, 79)


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    raise RuntimeError("demo renderer could not find a supported font")


FONTS = {
    "eyebrow": font(24, bold=True),
    "small": font(26),
    "small_bold": font(26, bold=True),
    "body": font(32),
    "body_bold": font(32, bold=True),
    "metric": font(44, bold=True),
    "title": font(60, bold=True),
    "hero": font(84, bold=True),
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def source_frame_index(path: str) -> int:
    match = re.search(r"frame_(\d+)\.png$", path)
    if match is None:
        raise ValueError(f"invalid target frame path: {path}")
    return int(match.group(1))


def action_metrics(data_root: Path, run_id: str) -> list[dict[str, Any]]:
    actions = []
    for index in range(1, 5):
        session = data_root / "control_sessions" / f"{run_id}-action{index}"
        request = load_json(session / "request.json")
        result = load_json(session / "result.json")
        progress = result["insertion_trial_post_action"]["realized_target_progress"]
        interlock = result["execution_interlock"]
        if (
            result["status"] != "applied"
            or not result["action_tracking"]["passed"]
            or result["post_action_collision_detected"]
            or interlock["collision_detected"]
            or not result["post_action_plug_attached"]
            or result["post_action_contact_force_newtons"] != 0.0
            or interlock["maximum_contact_force_newtons"] != 0.0
            or not progress["passed"]
        ):
            raise ValueError(f"source action is not safe and applied: {session.name}")
        actions.append(
            {
                "index": index,
                "target_frame": source_frame_index(request["target"]["frame"]),
                "translation_scale": result["selected_action_scale"]["translation"],
                "rotation_scale": result["selected_action_scale"]["rotation"],
                "progress_fraction": progress["translation_error_reduction_fraction"],
                "tracking_radians": result["maximum_joint_tracking_error_rad"],
            }
        )
    return actions


def run_metrics(data_root: Path, run_id: str) -> dict[str, Any]:
    report = load_json(
        data_root
        / "control_rollouts"
        / f"{run_id}-action4"
        / "report.json"
    )
    if report.get("all_steps_applied") is not True or report.get("applied_steps") != 4:
        raise ValueError(f"demo source is not 4/4 applied: {run_id}")
    command_ages = [step["timing"]["command_age_seconds"] for step in report["steps"]]
    return {
        "run_id": run_id,
        "actions": action_metrics(data_root, run_id),
        "translation_progress_meters": report["translation_progress_meters"],
        "maximum_command_age_seconds": max(command_ages),
    }


def frame_path(recording: Path, camera: str, index: int) -> Path:
    return recording / camera / f"frame_{index:06d}.png"


def contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_ratio = size[0] / size[1]
    ratio = image.width / image.height
    if ratio > target_ratio:
        crop_width = round(image.height * target_ratio)
        left = (image.width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, image.height))
    elif ratio < target_ratio:
        crop_height = round(image.width / target_ratio)
        top = (image.height - crop_height) // 2
        image = image.crop((0, top, image.width, top + crop_height))
    return image.resize(size, Image.Resampling.LANCZOS)


def rgba_frame(color: tuple[int, int, int] = BACKGROUND) -> Image.Image:
    return Image.new("RGBA", (WIDTH, HEIGHT), (*color, 255))


def text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    style: str,
    color: tuple[int, int, int] = WHITE,
    *,
    anchor: str | None = None,
) -> None:
    draw.text(xy, value, font=FONTS[style], fill=(*color, 255), anchor=anchor)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    color: tuple[int, int, int],
) -> int:
    bbox = draw.textbbox((0, 0), value, font=FONTS["small_bold"])
    width = bbox[2] - bbox[0] + 42
    draw.rounded_rectangle(
        (xy[0], xy[1], xy[0] + width, xy[1] + 48),
        radius=24,
        fill=(*color, 38),
        outline=(*color, 210),
        width=2,
    )
    text(draw, (xy[0] + 21, xy[1] + 24), value, "small_bold", color, anchor="lm")
    return width


def action_for_frame(index: int, frames_per_action: int = 18) -> int:
    if index <= 0:
        return 0
    return min(4, math.ceil(index / frames_per_action))


def hero_frame(
    recording: Path,
    source_index: int,
    metrics: dict[str, Any],
) -> Image.Image:
    presentation = Image.open(frame_path(recording, "presentation", source_index)).convert("RGB")
    wrist = Image.open(frame_path(recording, "wrist", source_index)).convert("RGB")
    canvas = cover(presentation, (WIDTH, HEIGHT)).convert("RGBA")
    shade = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shade).rectangle((0, 0, WIDTH, 148), fill=(5, 9, 16, 205))
    ImageDraw.Draw(shade).rectangle((0, 746, WIDTH, HEIGHT), fill=(5, 9, 16, 218))
    canvas.alpha_composite(shade)
    draw = ImageDraw.Draw(canvas)
    text(draw, (64, 34), "JEPA-WM · VERIFIED AUTONOMOUS ROLLOUT", "eyebrow", TEAL)
    text(draw, (64, 72), "VISUAL REPLAY · METRICS FROM THE LIVE SOURCE RUN", "small", WHITE)
    text(
        draw,
        (1856, 58),
        "OBSERVE  →  PROPOSE  →  SAFETY  →  MOVE  →  VERIFY",
        "small_bold",
        MUTED,
        anchor="rm",
    )

    action_index = action_for_frame(source_index)
    if action_index == 0:
        text(draw, (70, 800), "RESET AUTHENTICATED", "title", WHITE)
        text(draw, (70, 882), "Fresh attached held-out state", "body", MUTED)
    else:
        action = metrics["actions"][action_index - 1]
        text(draw, (70, 782), f"ACTION {action_index} / 4", "title", WHITE)
        text(draw, (70, 862), "SAFE SCALE", "eyebrow", MUTED)
        rotation = (
            "HOLD"
            if action["rotation_scale"] == 0.0
            else f"{action['rotation_scale']:.3f}×"
        )
        text(
            draw,
            (70, 900),
            f"T {action['translation_scale']:.2f}×  ·  R {rotation}",
            "metric",
            AMBER,
        )
        text(draw, (520, 862), "TARGET ERROR", "eyebrow", MUTED)
        text(
            draw,
            (520, 900),
            f"↓ {action['progress_fraction'] * 100:.2f}%",
            "metric",
            TEAL,
        )
        text(draw, (860, 862), "JOINT TRACKING", "eyebrow", MUTED)
        text(
            draw,
            (860, 900),
            f"{action['tracking_radians'] * 1000:.3f} mrad",
            "metric",
            VIOLET,
        )
        text(
            draw,
            (70, 982),
            f"LOOKAHEAD TARGET · FRAME {action['target_frame']}",
            "small_bold",
            MUTED,
        )

    pip = cover(wrist, (480, 270))
    canvas.paste(pip, (1376, 746))
    draw.rounded_rectangle((1368, 738, 1864, 1024), radius=8, outline=(*TEAL, 255), width=4)
    text(draw, (1390, 760), "WRIST CAMERA", "eyebrow", WHITE)
    x = 70
    for label in ("0 N MEASURED", "NO COLLISION", "ATTACHED"):
        x += pill(draw, (x, 1014), label, GREEN) + 18
    return canvas


def split_frame(
    left_recording: Path,
    right_recording: Path,
    source_index: int,
) -> Image.Image:
    canvas = rgba_frame()
    left = cover(
        Image.open(frame_path(left_recording, "presentation", source_index)).convert("RGB"),
        (950, 760),
    )
    right = cover(
        Image.open(frame_path(right_recording, "presentation", source_index)).convert("RGB"),
        (950, 760),
    )
    canvas.paste(left, (0, 170))
    canvas.paste(right, (970, 170))
    draw = ImageDraw.Draw(canvas)
    text(draw, (60, 42), "REPEATABILITY", "title", WHITE)
    text(draw, (60, 112), "SAME CODE · UNCHANGED CONTROL AND SAFETY GATES", "small_bold", TEAL)
    text(draw, (38, 190), "HELD-OUT RESET A", "eyebrow", WHITE)
    text(draw, (1008, 190), "HELD-OUT RESET B", "eyebrow", WHITE)
    action = action_for_frame(source_index)
    action_label = "RESET" if action == 0 else f"ACTION {action} / 4"
    text(draw, (960, 974), action_label, "metric", WHITE, anchor="mm")
    x = 380
    for label in ("8 / 8 APPLIED", "0 N MEASURED", "NO COLLISIONS", "ATTACHED"):
        x += pill(draw, (x, 1010), label, GREEN) + 18
    return canvas


def chart_axes(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> tuple[int, int, int, int]:
    text(draw, (90, 66), title, "title", WHITE)
    text(draw, (90, 138), subtitle, "body", MUTED)
    plot = (170, 250, 1780, 880)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill=(*GRID, 255), width=3)
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill=(*GRID, 255), width=3)
    return plot


def progress_chart(left: dict[str, Any], right: dict[str, Any]) -> Image.Image:
    canvas = rgba_frame()
    draw = ImageDraw.Draw(canvas)
    plot = chart_axes(
        draw,
        "REALIZED TARGET-ERROR REDUCTION",
        "Each action uses its own freshly selected target · percentages are not summed",
    )
    maximum = 70.0
    for tick in range(0, 71, 10):
        y = plot[3] - (tick / maximum) * (plot[3] - plot[1])
        draw.line((plot[0], y, plot[2], y), fill=(*GRID, 150), width=1)
        text(draw, (plot[0] - 24, round(y)), f"{tick}%", "small", MUTED, anchor="rm")
    group_width = (plot[2] - plot[0]) / 4
    bar_width = 100
    for index in range(4):
        center = plot[0] + group_width * (index + 0.5)
        for offset, metrics, color, label in (
            (-58, left, TEAL, "A"),
            (58, right, VIOLET, "B"),
        ):
            value = metrics["actions"][index]["progress_fraction"] * 100
            height = value / maximum * (plot[3] - plot[1])
            x0 = center + offset - bar_width / 2
            y0 = plot[3] - height
            draw.rounded_rectangle(
                (x0, y0, x0 + bar_width, plot[3]),
                radius=10,
                fill=(*color, 230),
            )
            text(draw, (round(x0 + bar_width / 2), round(y0 - 18)), f"{value:.2f}%", "small_bold", color, anchor="ms")
            text(draw, (round(x0 + bar_width / 2), plot[3] + 70), label, "small_bold", color, anchor="mm")
        text(draw, (round(center), plot[3] + 125), f"ACTION {index + 1}", "body_bold", WHITE, anchor="mm")
    text(draw, (90, 1018), "A · SEED 52600", "small_bold", TEAL)
    text(draw, (360, 1018), "B · SEED 52601", "small_bold", VIOLET)
    return canvas


def tracking_chart(left: dict[str, Any], right: dict[str, Any]) -> Image.Image:
    canvas = rgba_frame()
    draw = ImageDraw.Draw(canvas)
    plot = chart_axes(
        draw,
        "TERMINAL JOINT TRACKING ERROR",
        "Maximum joint-position error at each verified action endpoint · mrad",
    )
    maximum = 0.8
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = plot[3] - (tick / maximum) * (plot[3] - plot[1])
        draw.line((plot[0], y, plot[2], y), fill=(*GRID, 150), width=1)
        text(draw, (plot[0] - 24, round(y)), f"{tick:.1f}", "small", MUTED, anchor="rm")
    group_width = (plot[2] - plot[0]) / 4
    for index in range(4):
        center = plot[0] + group_width * (index + 0.5)
        draw.line((center - 70, plot[1], center - 70, plot[3]), fill=(*GRID, 100), width=1)
        for offset, metrics, color, label in (
            (-42, left, TEAL, "A"),
            (42, right, VIOLET, "B"),
        ):
            value = metrics["actions"][index]["tracking_radians"] * 1000
            y = plot[3] - (value / maximum) * (plot[3] - plot[1])
            x = center + offset
            draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=(*color, 255))
            text(draw, (round(x), round(y - 28)), f"{value:.3f}", "small_bold", color, anchor="ms")
            text(draw, (round(x), plot[3] + 66), label, "small_bold", color, anchor="mm")
        text(draw, (round(center), plot[3] + 124), f"ACTION {index + 1}", "body_bold", WHITE, anchor="mm")
    text(draw, (90, 1018), "A · SEED 52600", "small_bold", TEAL)
    text(draw, (360, 1018), "B · SEED 52601", "small_bold", VIOLET)
    return canvas


def result_card(left: dict[str, Any], right: dict[str, Any]) -> Image.Image:
    canvas = rgba_frame()
    draw = ImageDraw.Draw(canvas)
    text(draw, (960, 118), "2 HELD-OUT RESETS · 8 / 8 ACTIONS APPLIED", "title", WHITE, anchor="mm")
    text(draw, (960, 196), "FRESH OBSERVATION AND VERIFICATION AFTER EVERY ACTION", "small_bold", TEAL, anchor="mm")
    for x, label, metrics, color in (
        (120, "RESET A · SEED 52600", left, TEAL),
        (1000, "RESET B · SEED 52601", right, VIOLET),
    ):
        draw.rounded_rectangle((x, 290, x + 800, 760), radius=28, fill=(*PANEL[:3], 255), outline=(*color, 230), width=3)
        text(draw, (x + 48, 342), label, "body_bold", color)
        text(draw, (x + 48, 428), "ROLLOUT AGGREGATE TRANSLATION PROGRESS", "eyebrow", MUTED)
        text(draw, (x + 48, 470), f"{metrics['translation_progress_meters'] * 1000:.3f} mm", "hero", WHITE)
        text(draw, (x + 48, 604), "MAXIMUM COMMAND AGE", "eyebrow", MUTED)
        text(draw, (x + 48, 646), f"{metrics['maximum_command_age_seconds'] * 1000:.3f} ms", "metric", WHITE)
    x = 420
    for label in ("8 / 8 ATTACHED", "0 N MEASURED PEAK CONTACT", "0 COLLISIONS"):
        x += pill(draw, (x, 872), label, GREEN) + 24
    text(draw, (960, 1008), "Aggregate translation progress is a rollout metric, not seating depth.", "small", MUTED, anchor="mm")
    return canvas


def boundary_card() -> Image.Image:
    canvas = rgba_frame()
    draw = ImageDraw.Draw(canvas)
    text(draw, (960, 155), "BOUNDED AUTONOMOUS INSERTION DRIVE", "title", WHITE, anchor="mm")
    text(draw, (960, 238), "DEMO-SUITABLE CLOSED-LOOP CONTROL", "body_bold", TEAL, anchor="mm")
    draw.rounded_rectangle((180, 352, 920, 820), radius=26, fill=(12, 28, 34, 255), outline=(*GREEN, 220), width=3)
    text(draw, (230, 405), "DEMONSTRATED", "body_bold", GREEN)
    for index, value in enumerate((
        "Fresh observation before every move",
        "Safety-projected drive commands",
        "Verified progress across four actions",
        "Repeatability on two held-out resets",
    )):
        text(draw, (245, 488 + index * 72), f"✓  {value}", "body", WHITE)
    draw.rounded_rectangle((1000, 352, 1740, 820), radius=26, fill=(28, 19, 24, 255), outline=(*AMBER, 220), width=3)
    text(draw, (1050, 405), "NOT YET DEMONSTRATED", "body_bold", AMBER)
    for index, value in enumerate((
        "Full seating",
        "Autonomous approach and grasp",
        "Unknown-start insertion",
        "Production operation",
    )):
        text(draw, (1065, 488 + index * 72), f"—  {value}", "body", WHITE)
    text(draw, (960, 956), "JEPA-WM · QUANTIS ROBOTICS", "metric", MUTED, anchor="mm")
    return canvas


def write_frame(output: Path, index: int, image: Image.Image) -> None:
    image.convert("RGB").save(output / f"frame_{index:06d}.png", compress_level=1)


def repeat_still(output: Path, start: int, count: int, image: Image.Image) -> int:
    for index in range(start, start + count):
        write_frame(output, index, image)
    return start + count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(args: argparse.Namespace) -> None:
    data_root = args.data_root.resolve()
    hero_recording = args.hero_recording.resolve()
    repeat_recording = args.repeat_recording.resolve()
    output_dir = args.output.resolve()
    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True)
    hero_manifest = load_json(hero_recording / "manifest.json")
    repeat_manifest = load_json(repeat_recording / "manifest.json")
    for manifest, expected_run in (
        (hero_manifest, args.hero_run),
        (repeat_manifest, args.repeat_run),
    ):
        metadata = manifest["metadata"]
        if (
            manifest["fps"] != FPS
            or manifest["frames"] != 91
            or metadata["insertion_demo"]["visualization_only"] is not True
            or metadata["insertion_demo_replay"]["replay_tracking_passed"] is not True
            or metadata["insertion_demo_replay"]["replay_safety_passed"] is not True
            or metadata["insertion_demo"]["source_rollout"] != f"{expected_run}-action4"
        ):
            raise ValueError("insertion demo replay manifest is invalid")
    hero = run_metrics(data_root, args.hero_run)
    repeat = run_metrics(data_root, args.repeat_run)

    frame = 0
    for source_index in range(91):
        write_frame(frames_dir, frame, hero_frame(hero_recording, source_index, hero))
        frame += 1
    hero_hold = hero_frame(hero_recording, 90, hero)
    hold_draw = ImageDraw.Draw(hero_hold)
    hold_draw.rounded_rectangle((515, 330, 1405, 650), radius=36, fill=(5, 10, 18, 225), outline=(*TEAL, 235), width=4)
    text(hold_draw, (960, 410), "4 / 4 ACTIONS APPLIED", "hero", WHITE, anchor="mm")
    text(hold_draw, (960, 536), "VERIFIED SOURCE RUN", "body_bold", TEAL, anchor="mm")
    text(hold_draw, (960, 592), "0 N · NO COLLISION · ATTACHED", "body", GREEN, anchor="mm")
    frame = repeat_still(frames_dir, frame, 24, hero_hold)
    for source_index in range(91):
        write_frame(
            frames_dir,
            frame,
            split_frame(hero_recording, repeat_recording, source_index),
        )
        frame += 1
    frame = repeat_still(frames_dir, frame, 60, progress_chart(hero, repeat))
    frame = repeat_still(frames_dir, frame, 60, tracking_chart(hero, repeat))
    frame = repeat_still(frames_dir, frame, 60, result_card(hero, repeat))
    frame = repeat_still(frames_dir, frame, 48, boundary_card())

    output_video = output_dir / "jepa-wm-autonomous-insertion-demo.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-r",
            "30",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        check=True,
    )
    manifest = {
        "schema": "quantis.insertion_demo_video.v1",
        "silent": True,
        "fps": 30,
        "duration_seconds": frame / FPS,
        "source_frame_rate": FPS,
        "source_runs": [args.hero_run, args.repeat_run],
        "source_recordings": [hero_recording.name, repeat_recording.name],
        "visualization_only": True,
        "metrics": [hero, repeat],
        "video": output_video.name,
        "video_sha256": sha256(output_video),
    }
    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not args.keep_frames:
        shutil.rmtree(frames_dir)
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hero-recording", type=Path, required=True)
    parser.add_argument("--repeat-recording", type=Path, required=True)
    parser.add_argument("--hero-run", required=True)
    parser.add_argument("--repeat-run", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-frames", action="store_true")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
