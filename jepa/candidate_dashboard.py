"""Render the realized JEPA-WM candidate telemetry panel."""

from __future__ import annotations

import math

from PIL import Image

from jepa.dashboard_primitives import (
    ACCENT,
    FONTS,
    SECONDARY,
    WARNING,
    DashboardStep,
    card,
    create_panel_header,
    joint_bars,
    metric_row,
)
from jepa_wm.candidate_demo import CandidateDemoMetadata


def render_candidate_panel(
    step: DashboardStep,
    metadata: CandidateDemoMetadata,
    *,
    frame_count: int,
    fps: int,
) -> Image.Image:
    image, draw = create_panel_header(
        step,
        frame_count=frame_count,
        fps=fps,
        subtitle="JEPA-WM / BOUNDED CANDIDATE SEARCH",
    )

    card(draw, (36, 214, 604, 446))
    draw.text((54, 236), "SEARCH RESULT", font=FONTS["section"], fill=SECONDARY)
    draw.text((54, 278), "STRICT TRIAL / REPLAY", font=FONTS["stage"], fill=ACCENT)
    metric_row(draw, 334, "CEM CANDIDATES", f"{metadata.candidates_scored:,}")
    metric_row(
        draw, 382, "ENERGY IMPROVEMENT", f"{metadata.energy_improvement:+.6f}"
    )
    metric_row(draw, 426, "ACTION SCALE", metadata.action_scale_label, color=ACCENT)

    action = metadata.actual_action.values
    translation_mm = math.sqrt(sum(value * value for value in action[:3])) * 1000.0
    rotation_deg = math.degrees(math.sqrt(sum(value * value for value in action[3:6])))
    card(draw, (36, 470, 604, 720))
    draw.text((54, 492), "REALIZED CONTROL ACTION", font=FONTS["section"], fill=SECONDARY)
    metric_row(draw, 540, "TRANSLATION", f"{translation_mm:.3f} mm")
    metric_row(draw, 590, "ROTATION", f"{rotation_deg:.3f}°")
    metric_row(draw, 640, "GRIPPER DELTA", f"{action[6]:+.4f}")
    metric_row(
        draw,
        690,
        "REPLAY TRACKING",
        (
            f"PASS / {metadata.maximum_replay_joint_error_rad * 1000.0:.2f} mrad"
            if metadata.tracking_passed
            else "FAIL"
        ),
        color=ACCENT if metadata.tracking_passed else WARNING,
    )

    card(draw, (36, 744, 604, 1136))
    draw.text((54, 766), "ARM JOINT POSITIONS", font=FONTS["section"], fill=SECONDARY)
    joint_bars(draw, step.arm_positions, 814)

    card(draw, (36, 1160, 604, 1354))
    draw.text((54, 1182), "REPLAY SAFETY / PROVENANCE", font=FONTS["section"], fill=SECONDARY)
    metric_row(
        draw,
        1228,
        "MAX CONTACT FORCE",
        f"{metadata.maximum_replay_contact_force_newtons:.2f} N",
    )
    metric_row(
        draw,
        1278,
        "COLLISION",
        "CLEAR" if not metadata.replay_collision_detected else "DETECTED",
        color=ACCENT if not metadata.replay_collision_detected else WARNING,
    )
    draw.text(
        (54, 1320),
        f"HELD-OUT SEED {metadata.seed}  ·  {metadata.report_id[-24:]}",
        font=FONTS["tiny"],
        fill=SECONDARY,
    )
    return image
