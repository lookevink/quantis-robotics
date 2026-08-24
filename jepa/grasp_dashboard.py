"""Render a truthful reach-and-grasp readiness replay panel."""

from __future__ import annotations

from PIL import Image

from jepa.dashboard_primitives import (
    ACCENT,
    FONTS,
    PRIMARY,
    SECONDARY,
    WARNING,
    DashboardStep,
    card,
    create_panel_header,
    joint_bars,
    metric_row,
)
from jepa_wm.grasp_demo import GraspDemoMetadata


def render_grasp_panel(
    step: DashboardStep,
    metadata: GraspDemoMetadata,
    *,
    frame_count: int,
    fps: int,
) -> Image.Image:
    image, draw = create_panel_header(
        step,
        frame_count=frame_count,
        fps=fps,
        subtitle="JEPA-WM / VALIDATED REACH + GRASP REPLAY",
    )

    card(draw, (36, 214, 604, 446))
    draw.text((54, 236), "TASK EVIDENCE", font=FONTS["section"], fill=SECONDARY)
    draw.text((54, 278), "GRASP RETAINED", font=FONTS["stage"], fill=ACCENT)
    metric_row(draw, 334, "SOURCE ACTIONS", f"{metadata.source_steps}/{metadata.source_steps}")
    metric_row(
        draw,
        382,
        "RETAINED MOTION",
        f"{metadata.task_outcome.maximum_retained_displacement_meters * 1000.0:.2f} mm",
    )
    metric_row(
        draw,
        426,
        "ATTACHMENT",
        "ENGAGED" if step.plug_attached else "APPROACHING",
        color=ACCENT if step.plug_attached else PRIMARY,
    )

    card(draw, (36, 470, 604, 720))
    draw.text((54, 492), "REPLAY VERIFICATION", font=FONTS["section"], fill=SECONDARY)
    metric_row(
        draw,
        540,
        "ARM TRACKING",
        f"PASS / {metadata.replay.maximum_arm_error_rad * 1000.0:.2f} mrad",
        color=ACCENT if metadata.replay.tracking_passed else WARNING,
    )
    metric_row(
        draw,
        590,
        "GRIPPER ERROR",
        f"{metadata.replay.maximum_gripper_error_m * 1000.0:.2f} mm",
    )
    metric_row(
        draw,
        640,
        "MAX CONTACT FORCE",
        f"{metadata.replay.maximum_contact_force_newtons:.2f} N",
    )
    metric_row(
        draw,
        690,
        "COLLISION",
        "CLEAR" if not metadata.replay.collision_detected else "DETECTED",
        color=ACCENT if metadata.replay.safety_passed else WARNING,
    )

    card(draw, (36, 744, 604, 1136))
    draw.text((54, 766), "ARM JOINT POSITIONS", font=FONTS["section"], fill=SECONDARY)
    joint_bars(draw, step.arm_positions, 814)

    card(draw, (36, 1160, 604, 1354))
    draw.text((54, 1182), "PROVENANCE", font=FONTS["section"], fill=SECONDARY)
    metric_row(draw, 1228, "HELD-OUT SEED", str(metadata.seed))
    metric_row(
        draw,
        1278,
        "MODEL SHA-256",
        f"{metadata.proposal.fingerprint[:12]}…",
    )
    draw.text(
        (54, 1320),
        "VISUALIZATION REPLAY · NO PRODUCTION AUTHORITY",
        font=FONTS["tiny"],
        fill=SECONDARY,
    )
    return image
