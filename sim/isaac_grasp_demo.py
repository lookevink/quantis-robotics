"""High-resolution replay of one readiness-validated JEPA reach-and-grasp."""

from __future__ import annotations

from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.grasp_control_readiness import GraspControlReadinessSummary
from jepa_wm.grasp_demo import GraspDemoMetadata
from jepa_wm.trial_equivalence import TrialResetState, validate_reset_equivalence
from sim.control_session import (
    QUANTIS_DATA_ROOT,
    ControlResultStatus,
    ControlSessionState,
)
from sim.demo_sequence import Phase
from sim.exploration import DatasetSplit, build_exploration_plan
from sim.isaac_control_capture import validated_control_reference
from sim.isaac_control_runtime import contact_sensor, read_contact
from sim.isaac_demo_camera import CAMERA_SPECS, DEMO_FPS, DemoRecorder
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    prepare_plug,
    recording_snapshot,
    reset_stage,
)
from sim.isaac_demo_scene import ROBOT_PATH, world_pose
from sim.isaac_exploration import apply_variant
from sim.isaac_replay import (
    ReplayRuntime,
    gripper_width_from_closedness,
)
from sim.recording import (
    RecordingLabel,
    RecordingMoment,
    RecordingSnapshot,
    validate_recording_id,
)


def validate_grasp_replay_reset(
    observation: ControlObservation,
    state: ControlSessionState,
    snapshot: RecordingSnapshot,
    actual: JointCommand,
    *,
    collision_detected: bool,
    contact_force_newtons: float,
) -> None:
    """Apply the same strict reset-equivalence contract as realized trials."""

    validate_reset_equivalence(
        TrialResetState(
            observation.pose,
            state.current_joint_positions,
            state.collision_detected,
            state.contact_force_newtons,
            state.plug_position,
            state.plug_attached,
        ),
        TrialResetState(
            snapshot.end_effector_pose,
            tuple(float(value) for value in actual.arm_positions),
            collision_detected,
            contact_force_newtons,
            tuple(float(value) for value in snapshot.plug_position),
            snapshot.plug_attached,
        ),
    )


async def record_grasp_demo(
    readiness_id: str,
    seed: int,
    recording_id: str,
    expected_proposal_fingerprint: str,
    *,
    frames_per_action: int = 8,
    hold_frames: int = 12,
) -> dict[str, Any]:
    """Replay one exact validated rollout for presentation, without authority."""

    validate_recording_id(readiness_id)
    validate_recording_id(recording_id)
    if seed < 0 or frames_per_action <= 0 or hold_frames < 0:
        raise ValueError("grasp demo replay settings are invalid")
    readiness = GraspControlReadinessSummary.load_container_reconstruction(
        QUANTIS_DATA_ROOT,
        readiness_id,
        expected_proposal_fingerprint=expected_proposal_fingerprint,
    )
    if not readiness.filming_readiness_passed:
        raise ValueError("grasp demo readiness gate did not pass")
    matching = tuple(item for item in readiness.evidence if item.seed == seed)
    if len(matching) != 1:
        raise ValueError("grasp demo seed is not uniquely readiness validated")
    evidence = matching[0]
    rollout = evidence.report.direct
    steps = rollout.complete_steps
    if (
        not steps
        or len(steps) != len(rollout.applied_steps)
        or any(
            step.status is not ControlResultStatus.APPLIED
            or step.result.post_action is None
            for step in steps
        )
    ):
        raise ValueError("grasp demo source rollout is not fully applied")
    validated_control_reference(
        evidence.report.reference_recording,
        seed,
        ControlExecutionPolicy.DIRECT,
    )
    plan = build_exploration_plan(seed, DatasetSplit.HELD_OUT)

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    await reset_stage()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    apply_variant(stage, plan)
    recorder = DemoRecorder(
        recording_id,
        fps=DEMO_FPS,
        minimum_stage_frames=0,
        camera_specs=CAMERA_SPECS,
        metadata=plan.metadata(),
    )
    timeline = omni.timeline.get_timeline_interface()
    original_rendering_dt = RenderingManager.get_dt()
    try:
        await recorder.initialize()
        RenderingManager.set_dt(1.0 / DEMO_FPS)
        attachment = prepare_plug(stage)
        sensor = contact_sensor(stage, create=True)
        await omni.kit.app.get_app().next_update_async()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        first = steps[0]
        start = JointCommand(
            np.asarray(first.state.current_joint_positions, dtype=np.float64),
            gripper_width_from_closedness(first.observation.pose.values[-1]),
        )
        timeline.play()
        actuators.apply(start)
        for _ in range(16):
            await omni.kit.app.get_app().next_update_async()
        actual = actuators.actual_command()
        initial_label = RecordingLabel(RecordingMoment.MOTION, Phase.GRASP)
        initial_snapshot = recording_snapshot(
            initial_label,
            ObservationStage.APPROACHING_CABLE,
            actual,
            attachment,
        )
        initial_collision, initial_force = read_contact(sensor)
        validate_grasp_replay_reset(
            first.observation,
            first.state,
            initial_snapshot,
            actual,
            collision_detected=initial_collision,
            contact_force_newtons=initial_force,
        )
        replay = ReplayRuntime(
            actuators,
            attachment,
            recorder,
            sensor,
            1.0 / DEMO_FPS,
        )
        replay.observe(start)
        await recorder.capture(
            initial_snapshot,
            advance=False,
        )
        for index, step in enumerate(steps):
            post_action = step.result.post_action
            if post_action is None:
                raise AssertionError("validated applied step has no post-action state")
            target = JointCommand(
                np.asarray(post_action.joint_positions, dtype=np.float64),
                gripper_width_from_closedness(post_action.pose.values[-1]),
            )
            phase = Phase.GRASP if not attachment.attached else Phase.PRE_INSERTION
            stage_label = (
                ObservationStage.CABLE_GRASPED
                if attachment.attached
                else ObservationStage.APPROACHING_CABLE
            )
            actual = await replay.transition(
                actual,
                target,
                frame_count=frames_per_action,
                phase=RecordingLabel(RecordingMoment.MOTION, phase),
                stage=stage_label,
            )
            if post_action.plug_attached and not attachment.attached:
                attachment.attach(world_pose(attachment.hand_prim)[0])
                actual = await replay.transition(
                    actual,
                    actual,
                    frame_count=1,
                    phase=RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP),
                    stage=ObservationStage.CABLE_GRASPED,
                )
            elif attachment.attached and not post_action.plug_attached:
                raise ValueError(f"grasp demo source loses attachment at step {index}")

        if not attachment.attached:
            raise ValueError("grasp demo replay never acquired the connector")
        if hold_frames:
            actual = await replay.transition(
                actual,
                actual,
                frame_count=hold_frames,
                phase=RecordingLabel(RecordingMoment.COMPLETE, Phase.PRE_INSERTION),
                stage=ObservationStage.CABLE_GRASPED,
            )
        metadata = GraspDemoMetadata(
            readiness_id=readiness_id,
            baseline_experiment_id=evidence.report.experiment_id,
            rollout_id=rollout.rollout_id,
            seed=seed,
            proposal=evidence.proposal,
            source_steps=len(steps),
            task_outcome=evidence.direct,
            replay=replay.verification,
        )
        recorder.set_metadata("grasp_demo", metadata.to_dict())
        timeline.pause()
    except Exception:
        recorder.abort()
        raise
    finally:
        RenderingManager.set_dt(original_rendering_dt)
        timeline.pause()

    output = recorder.finish()
    return {
        "status": "recorded",
        "recording": recording_id,
        "output": str(output),
        "frames": recorder.frame_count,
        "fps": recorder.fps,
        "cameras": [spec.label for spec in CAMERA_SPECS],
        "readiness": readiness_id,
        "source_rollout": rollout.rollout_id,
        "seed": seed,
        "visualization_only": True,
    }
