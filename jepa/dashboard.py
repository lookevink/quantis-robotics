"""Render synchronized demo telemetry as a presentation-video side panel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image

from jepa.candidate_dashboard import render_candidate_panel
from jepa.grasp_dashboard import render_grasp_panel
from jepa.dashboard_primitives import (
    ACCENT,
    FONTS,
    PANEL_SIZE,
    PRIMARY,
    SECONDARY,
    WARNING,
    DashboardLayout,
    DashboardStep,
    card,
    create_panel_header,
    joint_bars,
    metric_row,
)
from jepa_wm.candidate_demo import CandidateDemoMetadata
from jepa_wm.grasp_demo import GraspDemoMetadata


PANEL_DIRECTORY = Path("dashboard/panel")
LAYOUT_PATH = Path("dashboard/layout.json")


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


def _render_panel(
    step: DashboardStep,
    prediction: DashboardPrediction | None,
    *,
    frame_count: int,
    fps: int,
) -> Image.Image:
    image, draw = create_panel_header(
        step,
        frame_count=frame_count,
        fps=fps,
        subtitle="RJ45 INSERTION / SYSTEM TELEMETRY",
    )

    card(draw, (36, 214, 604, 420))
    draw.text((54, 236), "ACTION", font=FONTS["section"], fill=SECONDARY)
    draw.text((54, 278), _humanize(step.stage), font=FONTS["stage"], fill=ACCENT)
    draw.text((54, 326), _humanize(step.phase), font=FONTS["body"], fill=PRIMARY)
    card(draw, (36, 444, 604, 664))
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
        metric_row(draw, 566, "COSINE SIMILARITY", f"{prediction.similarity:.4f}")
        metric_row(draw, 616, "CONFIDENCE MARGIN", f"{prediction.margin:.4f}")

    card(draw, (36, 688, 604, 1082))
    draw.text((54, 710), "ARM JOINT POSITIONS", font=FONTS["section"], fill=SECONDARY)
    joint_bars(draw, step.arm_positions, 758)

    card(draw, (36, 1106, 604, 1354))
    draw.text(
        (54, 1128), "END EFFECTOR / CONNECTOR", font=FONTS["section"], fill=SECONDARY
    )
    metric_row(draw, 1174, "GRIPPER WIDTH", f"{step.gripper_width_m * 1000:.1f} mm")
    plug_text = "  ".join(
        f"{axis} {value:+.3f}" for axis, value in zip("XYZ", step.plug_position)
    )
    draw.text((54, 1233), "PLUG POSITION / METERS", font=FONTS["small"], fill=SECONDARY)
    draw.text((54, 1262), plug_text, font=FONTS["body"], fill=PRIMARY)
    metric_row(
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
    candidate_metadata = CandidateDemoMetadata.from_manifest(manifest)
    grasp_metadata = GraspDemoMetadata.from_manifest(manifest)
    if candidate_metadata is not None and grasp_metadata is not None:
        raise ValueError("recording cannot be both candidate and grasp replay")
    layout = DashboardLayout()
    output = recording / PANEL_DIRECTORY
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("frame_*.png"):
        stale.unlink()

    for step in steps:
        if grasp_metadata is not None:
            panel = render_grasp_panel(
                step,
                grasp_metadata,
                frame_count=frame_count,
                fps=fps,
            )
        elif candidate_metadata is not None:
            panel = render_candidate_panel(
                step,
                candidate_metadata,
                frame_count=frame_count,
                fps=fps,
            )
        else:
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
        "candidate_demo": candidate_metadata is not None,
        "grasp_demo": grasp_metadata is not None,
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
