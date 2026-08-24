"""Shared presentation-dashboard layout and drawing primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


PANEL_SIZE = (640, 1440)
PRIMARY_VIDEO_SIZE = (1920, 1080)
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
    def from_dict(cls, payload: dict[str, Any]) -> DashboardStep:
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
                    float(value)
                    for value in payload.get("plug_position", [0, 0, 0])
                ),
                plug_attached=bool(payload.get("plug_attached")),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid dashboard recording step: {payload}") from error


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


def create_panel_header(
    step: DashboardStep,
    *,
    frame_count: int,
    fps: int,
    subtitle: str,
) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", PANEL_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    draw.text((44, 42), "QUANTIS ROBOTICS", font=FONTS["title"], fill=PRIMARY)
    draw.text((46, 88), subtitle, font=FONTS["small"], fill=SECONDARY)
    draw.line((44, 124, 596, 124), fill=OUTLINE, width=2)

    progress = (step.index + 1) / frame_count
    draw.rounded_rectangle((44, 146, 596, 158), radius=6, fill=(29, 42, 57))
    draw.rounded_rectangle(
        (44, 146, 44 + int(552 * progress), 158), radius=6, fill=ACCENT
    )
    draw.text(
        (44, 170),
        f"FRAME {step.index + 1:04d} / {frame_count:04d}",
        font=FONTS["tiny"],
        fill=SECONDARY,
    )
    text_right(
        draw,
        (596, 170),
        f"{step.index / fps:06.2f} s",
        font=FONTS["tiny"],
        fill=SECONDARY,
    )
    return image, draw


def text_right(
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


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=18, fill=CARD, outline=OUTLINE, width=2)


def metric_row(
    draw: ImageDraw.ImageDraw,
    y: int,
    label: str,
    value: str,
    *,
    color: tuple[int, int, int] = PRIMARY,
) -> None:
    draw.text((54, y), label, font=FONTS["small"], fill=SECONDARY)
    text_right(draw, (586, y - 4), value, font=FONTS["value"], fill=color)


def joint_bars(
    draw: ImageDraw.ImageDraw, values: tuple[float, ...], y: int
) -> None:
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
        text_right(
            draw,
            (586, row_y - 1),
            f"{degrees:+06.1f}°",
            font=FONTS["body"],
            fill=PRIMARY,
        )
