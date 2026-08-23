"""Render synchronized demo telemetry as a presentation-video side panel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PANEL_SIZE = (640, 1440)
PRIMARY_VIDEO_SIZE = (1920, 1080)
PANEL_DIRECTORY = Path("dashboard/panel")
LAYOUT_PATH = Path("dashboard/layout.json")
BACKGROUND = (8, 13, 21)
CARD = (17, 27, 40)
OUTLINE = (42, 61, 82)
PRIMARY = (229, 237, 245)
SECONDARY = (143, 160, 178)
ACCENT = (63, 224, 160)
WARNING = (255, 188, 92)


@dataclass(frozen=True)
class DashboardLayout:
    primary_size: tuple[int, int] = PRIMARY_VIDEO_SIZE
    panel_size: tuple[int, int] = PANEL_SIZE

    @property
    def output_size(self) -> tuple[int, int]:
        return (self.primary_size[0] + self.panel_size[0], self.panel_size[1])

    @property
    def primary_y(self) -> int:
        return (self.panel_size[1] - self.primary_size[1]) // 2

    def to_dict(self) -> dict[str, object]:
        return {
            "primary_size": list(self.primary_size),
            "panel_size": list(self.panel_size),
            "primary_y": self.primary_y,
            "output_size": list(self.output_size),
        }


@dataclass(frozen=True)
class DashboardStep:
    index: int
    stage: str
    phase: str
    arm_positions: tuple[float, ...]
    gripper_width_m: float
    plug_position: tuple[float, ...]
    plug_attached: bool

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DashboardStep":
        try:
            return cls(
                index=int(payload["index"]),
                stage=str(payload["stage"]),
                phase=str(payload["phase"]),
                arm_positions=tuple(
                    float(value) for value in payload.get("arm_positions", [])
                ),
                gripper_width_m=float(payload["gripper_width_m"]),
                plug_position=tuple(
                    float(value) for value in payload.get("plug_position", [0, 0, 0])
                ),
                plug_attached=bool(payload.get("plug_attached")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid dashboard recording step: {payload}") from error


@dataclass(frozen=True)
class DashboardPrediction:
    actual: str
    stage: str
    similarity: float
    margin: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DashboardPrediction":
        try:
            return cls(
                actual=str(payload["actual"]),
                stage=str(payload["stage"]),
                similarity=float(payload["similarity"]),
                margin=float(payload["margin"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid dashboard prediction: {payload}") from error


def _font(
    size: int, *, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


FONTS = {
    "title": _font(34, bold=True),
    "section": _font(20, bold=True),
    "stage": _font(30, bold=True),
    "value": _font(24, bold=True),
    "body": _font(20),
    "small": _font(16),
    "tiny": _font(14),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_steps(path: Path) -> list[DashboardStep]:
    steps = [
        DashboardStep.from_dict(json.loads(line))
        for line in path.read_text().splitlines()
        if line
    ]
    for index, step in enumerate(steps):
        if step.index != index:
            raise ValueError(f"recording step indices are not contiguous at {index}")
    return steps


def _prediction_map(path: Path) -> dict[str, DashboardPrediction]:
    if not path.is_file():
        return {}
    report = _read_json(path)
    predictions = [
        DashboardPrediction.from_dict(payload)
        for payload in report.get("predictions", [])
        if isinstance(payload, dict)
    ]
    return {prediction.actual: prediction for prediction in predictions}


def _humanize(value: object) -> str:
    return str(value).replace("_", " ").upper()


def _text_right(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    value: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    right, y = xy
    box = draw.textbbox((0, 0), value, font=font)
    draw.text((right - (box[2] - box[0]), y), value, font=font, fill=fill)


def _card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=CARD, outline=OUTLINE, width=2)


def _metric_row(
    draw: ImageDraw.ImageDraw,
    y: int,
    label: str,
    value: str,
    *,
    color: tuple[int, int, int] = PRIMARY,
) -> None:
    draw.text((54, y), label, font=FONTS["small"], fill=SECONDARY)
    _text_right(draw, (586, y - 4), value, font=FONTS["value"], fill=color)


def _joint_bars(draw: ImageDraw.ImageDraw, values: tuple[float, ...], y: int) -> None:
    import math

    for index in range(7):
        radians = float(values[index]) if index < len(values) else 0.0
        degrees = math.degrees(radians)
        row_y = y + index * 45
        draw.text((54, row_y), f"J{index + 1}", font=FONTS["small"], fill=SECONDARY)
        track = (108, row_y + 4, 462, row_y + 18)
        draw.rounded_rectangle(track, radius=7, fill=(32, 45, 59))
        center = (track[0] + track[2]) // 2
        endpoint = int(center + max(-1.0, min(1.0, degrees / 180.0)) * 177)
        bar = (min(center, endpoint), track[1], max(center, endpoint), track[3])
        if bar[2] > bar[0]:
            draw.rounded_rectangle(bar, radius=7, fill=ACCENT)
        draw.line((center, track[1] - 3, center, track[3] + 3), fill=PRIMARY, width=2)
        _text_right(
            draw,
            (586, row_y - 1),
            f"{degrees:+06.1f}°",
            font=FONTS["body"],
            fill=PRIMARY,
        )


def _render_panel(
    step: DashboardStep,
    prediction: DashboardPrediction | None,
    *,
    frame_count: int,
    fps: int,
) -> Image.Image:
    image = Image.new("RGB", PANEL_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)

    draw.text((44, 42), "QUANTIS ROBOTICS", font=FONTS["title"], fill=PRIMARY)
    draw.text(
        (46, 88),
        "RJ45 INSERTION / SYSTEM TELEMETRY",
        font=FONTS["small"],
        fill=SECONDARY,
    )
    draw.line((44, 124, 596, 124), fill=OUTLINE, width=2)

    index = step.index
    progress = (index + 1) / frame_count
    draw.rounded_rectangle((44, 146, 596, 158), radius=6, fill=(29, 42, 57))
    draw.rounded_rectangle(
        (44, 146, 44 + int(552 * progress), 158), radius=6, fill=ACCENT
    )
    draw.text(
        (44, 170),
        f"FRAME {index + 1:04d} / {frame_count:04d}",
        font=FONTS["tiny"],
        fill=SECONDARY,
    )
    _text_right(
        draw,
        (596, 170),
        f"{index / fps:06.2f} s",
        font=FONTS["tiny"],
        fill=SECONDARY,
    )

    _card(draw, (36, 214, 604, 420))
    draw.text((54, 236), "ACTION", font=FONTS["section"], fill=SECONDARY)
    draw.text((54, 278), _humanize(step.stage), font=FONTS["stage"], fill=ACCENT)
    draw.text((54, 326), _humanize(step.phase), font=FONTS["body"], fill=PRIMARY)
    _card(draw, (36, 444, 604, 664))
    draw.text(
        (54, 466),
        "JEPA CLIP CLASSIFIER / OFFLINE",
        font=FONTS["section"],
        fill=SECONDARY,
    )
    if prediction is None:
        draw.text((54, 516), "UNAVAILABLE", font=FONTS["stage"], fill=WARNING)
        draw.text(
            (54, 570),
            "No stage report for this recording",
            font=FONTS["small"],
            fill=SECONDARY,
        )
    else:
        predicted = _humanize(prediction.stage)
        prediction_color = ACCENT if prediction.stage == step.stage else WARNING
        draw.text((54, 510), predicted, font=FONTS["stage"], fill=prediction_color)
        _metric_row(draw, 566, "COSINE SIMILARITY", f"{prediction.similarity:.4f}")
        _metric_row(draw, 616, "CONFIDENCE MARGIN", f"{prediction.margin:.4f}")

    _card(draw, (36, 688, 604, 1082))
    draw.text((54, 710), "ARM JOINT POSITIONS", font=FONTS["section"], fill=SECONDARY)
    _joint_bars(draw, step.arm_positions, 758)

    _card(draw, (36, 1106, 604, 1354))
    draw.text(
        (54, 1128), "END EFFECTOR / CONNECTOR", font=FONTS["section"], fill=SECONDARY
    )
    _metric_row(draw, 1174, "GRIPPER WIDTH", f"{step.gripper_width_m * 1000:.1f} mm")
    plug_text = "  ".join(
        f"{axis} {value:+.3f}" for axis, value in zip("XYZ", step.plug_position)
    )
    draw.text((54, 1233), "PLUG POSITION / METERS", font=FONTS["small"], fill=SECONDARY)
    draw.text((54, 1262), plug_text, font=FONTS["body"], fill=PRIMARY)
    _metric_row(
        draw,
        1310,
        "ATTACHMENT",
        "ENGAGED" if step.plug_attached else "RELEASED",
        color=ACCENT if step.plug_attached else WARNING,
    )

    return image


def render_dashboard_panels(
    recording: Path,
    *,
    jepa_camera: str = "wrist",
) -> dict[str, Any]:
    recording = recording.resolve()
    manifest = _read_json(recording / "manifest.json")
    steps = _read_steps(recording / "steps.jsonl")
    frame_count = int(manifest["frames"])
    fps = int(manifest["fps"])
    if frame_count != len(steps):
        raise ValueError("manifest and telemetry frame counts differ")
    if frame_count == 0 or fps <= 0:
        raise ValueError("recording must contain frames at a positive FPS")

    predictions = _prediction_map(
        recording / "jepa" / jepa_camera / "stage_report.json"
    )
    layout = DashboardLayout()
    output = recording / PANEL_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("frame_*.png"):
        stale.unlink()

    for step in steps:
        panel = _render_panel(
            step,
            predictions.get(step.stage),
            frame_count=frame_count,
            fps=fps,
        )
        panel.save(output / f"frame_{step.index:06d}.png", compress_level=1)

    layout_path = recording / LAYOUT_PATH
    layout_path.write_text(json.dumps(layout.to_dict(), indent=2) + "\n")

    return {
        "recording": str(recording),
        "panel_directory": str(output),
        "frames": frame_count,
        "fps": fps,
        "jepa_camera": jepa_camera,
        "jepa_stages": sorted(predictions),
        "layout": layout.to_dict(),
        "layout_path": str(layout_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("recording", type=Path)
    parser.add_argument("--jepa-camera", default="wrist")
    args = parser.parse_args()
    print(
        json.dumps(
            render_dashboard_panels(
                args.recording,
                jepa_camera=args.jepa_camera,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
