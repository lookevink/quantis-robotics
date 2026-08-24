"""Capture one session-bound live observation for JEPA-WM control."""

from __future__ import annotations

import json
from time import time
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import (
    ACTION_RECORDING_CONTRACT,
    DROID_FPS,
    ActionRecordingContract,
    ActionSelectionBounds,
    DroidAction,
    DroidPose,
)
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.trajectory import load_rollout_at
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    RECORDING_ROOT,
    ControlCaptureResult,
    ControlExecutionPolicy,
    ControlSession,
    ControlSessionState,
)
from sim.control_context import load_control_context
from sim.control_identity import control_proposal_path, observation_id_for_session
from sim.demo_sequence import Phase
from sim.exploration import (
    DatasetSplit,
    build_exploration_plan,
)
from sim.isaac_control_runtime import bind_live_runtime, contact_sensor, read_contact
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, DemoRecorder
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    move_joint_command,
    prepare_plug,
    recording_snapshot,
    reset_stage,
)
from sim.isaac_demo_scene import ROBOT_PATH, world_pose
from sim.isaac_exploration import apply_variant
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id


def validated_control_reference(
    name: str,
    seed: int,
    policy: ControlExecutionPolicy,
) -> DomainRecording:
    validate_recording_id(name)
    expected_split = (
        DatasetSplit.TRAIN
        if policy is ControlExecutionPolicy.CALIBRATION_COLLECTION
        else DatasetSplit.HELD_OUT
    )
    reference = DomainRecording.from_path(
        RECORDING_ROOT / name,
        expected_split=expected_split,
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
    execution_policy: str = ControlExecutionPolicy.DIRECT.value,
    context_index: int = 4,
) -> dict[str, Any]:
    """Replay a seeded segment prefix and persist one live wrist observation."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    validate_recording_id(proposal_name)
    policy = ControlExecutionPolicy(execution_policy)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    if session.path.exists():
        raise ValueError(f"control session already exists: {session_id}")
    reference = validated_control_reference(reference_recording, seed, policy)
    plan = build_exploration_plan(seed, reference.split)
    context_steps = load_control_context(reference.path, context_index, plan)
    reference_rollout = load_rollout_at(
        reference.path,
        camera="wrist",
        context_index=context_index,
        bounds=ActionSelectionBounds(minimum_action_norm=0.0),
    )
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
        metadata={
            **plan.metadata(),
            "control_session": session_id,
            "control_context_index": context_index,
        },
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
        origin = JointCommand(
            np.asarray(context_steps[0].arm_positions),
            context_steps[0].gripper_width_m,
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
        current = origin
        for step in context_steps[1:]:
            if step.plug_attached and not attachment.attached:
                attachment.attach(world_pose(attachment.hand_prim)[0])
            elif not step.plug_attached and attachment.attached:
                raise ValueError("recorded control context loses its plug attachment")
            command = JointCommand(
                np.asarray(step.arm_positions),
                step.gripper_width_m,
            )
            await move_joint_command(
                actuators,
                current,
                command,
                attachment,
                frame_count=1,
                phase=RecordingLabel(
                    (
                        RecordingMoment.ATTACHED
                        if step.plug_attached
                        else RecordingMoment.MOTION
                    ),
                    Phase.GRASP if step.plug_attached else Phase.READY,
                ),
                stage=(
                    ObservationStage.CABLE_GRASPED
                    if step.plug_attached
                    else ObservationStage.APPROACHING_CABLE
                ),
                recorder=recorder,
                sample_period_seconds=plan.sample_period_seconds,
            )
            current = command
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
    if len(steps) != context_index + 1:
        raise RuntimeError("control warm-up recording has an unexpected frame count")
    context_step = steps[context_index]
    if (
        context_step.get("index") != context_index
        or context_step.get("action_from_previous") is None
    ):
        raise RuntimeError("control warm-up telemetry is incomplete")
    target = reference_rollout.target.path
    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=time(),
        context_frame=(output / "wrist" / f"frame_{context_index:06d}.png").relative_to(
            QUANTIS_DATA_ROOT
        ),
        target=ControlTarget(
            target.relative_to(QUANTIS_DATA_ROOT), reference_rollout.target_pose
        ),
        expected_proposal=control_proposal_path(proposal_name),
        pose=DroidPose(tuple(context_step["end_effector_pose"])),
        previous_action=DroidAction(tuple(context_step["action_from_previous"])),
        warmup_frames=context_index,
    )
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=reference_recording,
        seed=seed,
        recording=recording_id,
        current_joint_positions=tuple(actual_warmup.arm_positions),
        collision_detected=collision_detected,
        contact_force_newtons=contact_force,
        execution_policy=policy,
        plug_position=tuple(float(value) for value in world_pose(attachment.prim)[0]),
        plug_attached=attachment.attached,
    )
    session.write_capture(observation, state)
    bind_live_runtime(session_id, stage, actuators, attachment, sensor)
    return ControlCaptureResult(
        session_id,
        observation,
        session.request_path,
        contact_force,
        collision_detected,
    ).to_dict()
