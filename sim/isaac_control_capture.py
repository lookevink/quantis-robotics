"""Capture one session-bound live observation for JEPA-WM control."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from time import time
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    DROID_FPS,
    ActionRecordingContract,
    DroidAction,
    DroidPose,
)
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.domain_recording import DomainRecording
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    RECORDING_ROOT,
    ControlCaptureResult,
    ControlSession,
    ControlSessionState,
)
from sim.demo_sequence import Phase
from sim.exploration import DatasetSplit, build_exploration_plan
from sim.isaac_control_runtime import contact_sensor, read_contact
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, DemoRecorder
from sim.isaac_demo_kinematics import solve_waypoints
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    move_joint_command,
    prepare_plug,
    recording_snapshot,
    reset_stage,
)
from sim.isaac_demo_scene import ROBOT_PATH
from sim.isaac_exploration import apply_variant
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id


PROPOSAL_ROOT = Path("/home/ubuntu/docker/jepa-wm/checkpoints")


def _observation_id(session_id: str) -> int:
    identifier = int.from_bytes(sha256(session_id.encode()).digest()[:8], "big")
    return identifier or 1


def _validated_reference(name: str, seed: int) -> DomainRecording:
    validate_recording_id(name)
    reference = DomainRecording.from_path(
        RECORDING_ROOT / name,
        expected_split=DatasetSplit.HELD_OUT,
    )
    if reference.seed != seed:
        raise ValueError(
            f"reference seed {reference.seed} does not match live variant seed {seed}"
        )
    manifest = json.loads((reference.path / "manifest.json").read_text())
    if ActionRecordingContract.from_mapping(manifest.get("action")) != ACTION_RECORDING_CONTRACT:
        raise ValueError("control reference does not use the DROID action contract")
    if manifest.get("fps") != DROID_FPS:
        raise ValueError("control reference does not use the DROID frame rate")
    cameras = manifest.get("cameras")
    if not isinstance(cameras, list) or "wrist" not in cameras:
        raise ValueError("control reference does not contain a wrist camera")
    return reference


async def capture_control_observation(
    session_id: str,
    reference_recording: str,
    seed: int,
    proposal_name: str,
) -> dict[str, Any]:
    """Replay four safe warm-up frames and persist one live wrist observation."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    validate_recording_id(proposal_name)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    if session.path.exists():
        raise ValueError(f"control session already exists: {session_id}")
    reference = _validated_reference(reference_recording, seed)
    plan = build_exploration_plan(seed, DatasetSplit.HELD_OUT)
    await reset_stage()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    apply_variant(stage, plan)
    recording_id = f"control-{session_id}"
    recorder = DemoRecorder(
        recording_id,
        fps=DROID_FPS,
        minimum_stage_frames=0,
        camera_specs=JEPA_WM_CAMERA_SPECS,
        metadata={**plan.metadata(), "control_session": session_id},
    )
    timeline = omni.timeline.get_timeline_interface()
    original_rendering_dt = RenderingManager.get_dt()
    completed = False
    try:
        await recorder.initialize()
        RenderingManager.set_dt(plan.sample_period_seconds)
        attachment = prepare_plug(stage)
        sensor = contact_sensor(stage, create=True)
        await omni.kit.app.get_app().next_update_async()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        ready = solve_waypoints()[0]
        origin = JointCommand(
            ready.arm_positions + np.asarray(plan.initial_arm_offset_radians),
            ready.waypoint.gripper_width_m,
        )
        timeline.play()
        actuators.apply(origin)
        for _ in range(16):
            await omni.kit.app.get_app().next_update_async()
        initial = recording_snapshot(
            RecordingLabel(RecordingMoment.INITIAL),
            ObservationStage.APPROACHING_CABLE,
            origin,
            attachment,
        )
        await recorder.capture(initial, advance=False)
        warmup = plan.targets[0]
        warmup_command = JointCommand(
            origin.arm_positions + np.asarray(warmup.arm_offset_radians),
            warmup.gripper_width_m,
        )
        await move_joint_command(
            actuators,
            origin,
            warmup_command,
            attachment,
            frame_count=warmup.frames,
            phase=RecordingLabel(RecordingMoment.SETTLE, Phase.READY),
            stage=ObservationStage.APPROACHING_CABLE,
            recorder=recorder,
            sample_period_seconds=plan.sample_period_seconds,
        )
        collision_detected, contact_force = read_contact(sensor)
        actual_warmup = actuators.actual_command()
        timeline.pause()
        completed = True
    except Exception:
        recorder.abort()
        raise
    finally:
        RenderingManager.set_dt(original_rendering_dt)
        if not completed:
            timeline.stop()
    output = recorder.finish()
    steps = tuple(
        json.loads(line)
        for line in (output / "steps.jsonl").read_text().splitlines()
        if line
    )
    if len(steps) != warmup.frames + 1:
        raise RuntimeError("control warm-up recording has an unexpected frame count")
    context_step = steps[warmup.frames]
    if (
        context_step.get("index") != warmup.frames
        or context_step.get("action_from_previous") is None
    ):
        raise RuntimeError("control warm-up telemetry is incomplete")
    target = reference.path / "wrist" / f"frame_{warmup.frames + 3:06d}.png"
    if not target.is_file():
        raise ValueError(f"control target frame does not exist: {target}")
    observation = ControlObservation(
        observation_id=_observation_id(session_id),
        captured_at_unix_seconds=time(),
        context_frame=(output / "wrist" / f"frame_{warmup.frames:06d}.png").relative_to(
            QUANTIS_DATA_ROOT
        ),
        target_frame=target.relative_to(QUANTIS_DATA_ROOT),
        expected_proposal=PROPOSAL_ROOT / f"{proposal_name}.pth",
        pose=DroidPose(tuple(context_step["end_effector_pose"])),
        previous_action=DroidAction(tuple(context_step["action_from_previous"])),
        warmup_frames=warmup.frames,
    )
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=reference_recording,
        seed=seed,
        recording=recording_id,
        current_joint_positions=tuple(actual_warmup.arm_positions),
        collision_detected=collision_detected,
        contact_force_newtons=contact_force,
    )
    session.write_capture(observation, state)
    return ControlCaptureResult(
        session_id,
        observation,
        session.request_path,
        contact_force,
        collision_detected,
    ).to_dict()
