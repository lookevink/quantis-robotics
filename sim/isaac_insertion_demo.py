"""High-resolution replay of one validated four-action insertion rollout."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import MAX_GRIPPER_WIDTH_M
from jepa_wm.control_rollout import ControlRolloutReport, ControlStepSummary
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
)
from jepa_wm.insertion_rollout import (
    DEMO_INSERTION_ROLLOUT,
    InsertionRolloutRoster,
)
from jepa_wm.trial_equivalence import (
    ResetEquivalenceMeasurement,
    TrialResetState,
    validate_reset_equivalence,
)
from sim.control_context import load_control_context
from sim.control_session import CONTROL_ROOT, QUANTIS_DATA_ROOT, ControlSession
from sim.demo_sequence import Phase
from sim.exploration import DatasetSplit, build_exploration_plan
from sim.isaac_control_capture import validated_control_reference
from sim.isaac_control_runtime import (
    LiveContactInterlock,
    LiveInsertionInterlock,
    control_contact_sensors,
    read_control_contact,
)
from sim.isaac_demo_camera import CAMERA_SPECS, DEMO_FPS, DemoRecorder
from sim.isaac_demo_runtime import (
    ContactReading,
    JointCommand,
    advance_physics_updates,
    create_actuators,
    recording_safety_telemetry,
    recording_snapshot,
    reset_stage,
    resume_live_simulation,
)
from sim.isaac_demo_scene import PLUG_PATH, ROBOT_PATH, world_pose
from sim.isaac_exploration import (
    ExplorationRecordingMode,
    ExplorationRecordingProfile,
    apply_variant,
)
from sim.isaac_replay import ReplayRuntime
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id


def load_insertion_demo_report(run_id: str) -> ControlRolloutReport:
    """Reconstruct one exact persisted 4/4 rollout for visualization."""

    validate_recording_id(run_id)
    session_ids = tuple(
        f"{run_id}-action{index}"
        for index in range(1, DEMO_INSERTION_ROLLOUT.maximum_steps + 1)
    )
    roster = InsertionRolloutRoster(
        session_ids,
        DEMO_INSERTION_ROLLOUT.maximum_steps,
    )
    steps = tuple(
        ControlStepSummary.from_session(ControlSession.at(CONTROL_ROOT, session_id))
        for session_id in roster.session_ids
    )
    if tuple(
        step.state.resolved_insertion_rollout_position() for step in steps
    ) != roster.positions:
        raise ValueError("insertion demo source positions are invalid")
    first = steps[0]
    report = ControlRolloutReport.from_sessions(
        QUANTIS_DATA_ROOT,
        roster.session_ids[-1],
        roster.session_ids,
        reference_recording=first.state.reference_recording,
        seed=first.state.seed,
        proposal=first.observation.expected_proposal,
        requested_steps=roster.maximum_steps,
    )
    report.require_all_steps_applied()
    persisted_path = (
        QUANTIS_DATA_ROOT
        / "control_rollouts"
        / roster.session_ids[-1]
        / "report.json"
    )
    if json.loads(persisted_path.read_text()) != report.to_dict():
        raise ValueError("insertion demo source does not match its persisted report")
    return report


def _source_metadata(report: ControlRolloutReport) -> dict[str, Any]:
    return {
        "source_rollout": report.rollout_id,
        "reference_recording": report.reference_recording,
        "seed": report.seed,
        "source_report": report.to_dict(),
        "visualization_only": True,
    }


def _validate_demo_reset(
    reference: TrialResetState,
    candidate: TrialResetState,
) -> None:
    try:
        validate_reset_equivalence(reference, candidate)
    except ValueError as error:
        measurement = ResetEquivalenceMeasurement.between(reference, candidate)
        raise ValueError(
            "insertion demo reset equivalence failed: "
            f"{json.dumps(measurement.to_dict(), sort_keys=True)}"
        ) from error


async def record_insertion_demo(
    run_id: str,
    recording_id: str,
    *,
    frames_per_action: int = 18,
    hold_frames: int = 18,
) -> dict[str, Any]:
    """Replay a proven four-action rollout without creating control authority."""

    validate_recording_id(recording_id)
    if frames_per_action <= 0 or hold_frames < 0:
        raise ValueError("insertion demo frame counts are invalid")
    report = load_insertion_demo_report(run_id)
    steps = report.applied_steps
    first = steps[0]
    context_index = first.observation.warmup_frames
    reference = validated_control_reference(
        report.reference_recording,
        report.seed,
        first.state.execution_policy,
    )
    profile = ExplorationRecordingProfile.for_mode(
        ExplorationRecordingMode.CONTACT_INSERTION
    )
    plan = profile.apply_to_plan(
        build_exploration_plan(report.seed, DatasetSplit.HELD_OUT)
    )
    context_steps = load_control_context(reference.path, context_index, plan)
    context_payloads = tuple(
        json.loads(line)
        for line in (reference.path / "steps.jsonl").read_text().splitlines()
        if line
    )
    try:
        source_plug_orientation = tuple(
            float(value)
            for value in context_payloads[context_index]["plug_orientation_wxyz"]
        )
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("insertion demo source plug orientation is invalid") from error
    if len(source_plug_orientation) != 4:
        raise ValueError("insertion demo source plug orientation is invalid")

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    await reset_stage()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    apply_variant(stage, plan)
    attachment_preparation = profile.prepare_attachment(stage)
    recorder: DemoRecorder | None = None
    timeline = omni.timeline.get_timeline_interface()
    original_rendering_dt = RenderingManager.get_dt()
    try:
        import omni.replicator.core as rep

        # Warm Replicator before the articulation physics view is created, but
        # do not attach the two 1080p render products until reset authentication
        # has passed. This keeps pre-roll deterministic and bounded.
        await rep.orchestrator.step_async(
            rt_subframes=4,
            pause_timeline=True,
            delta_time=0.0,
        )
        RenderingManager.set_dt(1.0 / DEMO_FPS)
        sensor = control_contact_sensors(
            stage,
            create=True,
            include_connector=True,
        )
        contact_interlock = LiveContactInterlock(
            sensor,
            SimulatorSafetyLimits().maximum_contact_force_newtons,
            "insertion demo visualization",
        )
        await omni.kit.app.get_app().next_update_async()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        plug_rigid_prim = RigidPrim(PLUG_PATH)
        attachment = profile.bind_attachment(
            attachment_preparation,
            plug_rigid_prim,
        )
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        source_start = JointCommand(
            np.asarray(first.state.current_joint_positions, dtype=np.float64),
            first.state.current_gripper_width_m,
        )
        if (
            first.state.plug_position is None
            or first.state.active_drive_target is None
        ):
            raise ValueError("insertion demo source reset is incomplete")
        source_drive_target = JointCommand(
            np.asarray(
                first.state.active_drive_target.joint_positions,
                dtype=np.float64,
            ),
            first.state.active_drive_target.gripper_width_m,
        )
        resume_live_simulation(timeline)
        actuators.set_reset_state(source_start)
        await advance_physics_updates(1, contact_interlock.observe)
        collision_start = CONTACT_INSERTION_RECORDING.start_index(
            ContactInsertionSegment.ALIGN
        )
        # This is explicit visualization initialization, not runtime control:
        # restore the captured plug pose, recreate its fixed attachment there,
        # then authenticate the settled result against the source state below.
        plug_rigid_prim.set_world_poses(
            positions=[first.state.plug_position],
            orientations=[source_plug_orientation],
        )
        attachment.attach(world_pose(attachment.hand_prim)[0])
        attachment.set_collisions(context_index >= collision_start)
        insertion_interlock = LiveInsertionInterlock(
            contact_interlock,
            attachment,
            True,
            "insertion demo visualization",
        )
        actuators.apply_drive_command(source_drive_target)
        await advance_physics_updates(16, insertion_interlock.observe)
        actual_start = actuators.actual_command()
        initial_contact = read_control_contact(sensor)
        initial_snapshot = recording_snapshot(
            RecordingLabel(RecordingMoment.MOTION, Phase.PRE_INSERTION),
            ObservationStage.CABLE_GRASPED,
            actual_start,
            attachment,
            safety=recording_safety_telemetry(
                source_start,
                actual_start,
                ContactReading(*initial_contact),
            ),
        )
        _validate_demo_reset(
            TrialResetState(
                first.observation.pose,
                first.state.current_joint_positions,
                first.state.collision_detected,
                first.state.contact_force_newtons,
                first.state.plug_position,
                first.state.plug_attached,
            ),
            TrialResetState(
                initial_snapshot.end_effector_pose,
                tuple(float(value) for value in actual_start.arm_positions),
                initial_snapshot.safety.collision_detected,
                initial_snapshot.safety.contact_force_newtons,
                tuple(float(value) for value in initial_snapshot.plug_position),
                initial_snapshot.plug_attached,
            ),
        )
        recorder = DemoRecorder(
            recording_id,
            fps=DEMO_FPS,
            minimum_stage_frames=0,
            camera_specs=CAMERA_SPECS,
            metadata={
                **plan.metadata(),
                "insertion_demo": _source_metadata(report),
            },
        )
        await recorder.prepare_current(insertion_interlock.observe)
        actual_start = actuators.actual_command()
        initial_contact = read_control_contact(sensor)
        initial_snapshot = recording_snapshot(
            RecordingLabel(RecordingMoment.MOTION, Phase.PRE_INSERTION),
            ObservationStage.CABLE_GRASPED,
            actual_start,
            attachment,
            safety=recording_safety_telemetry(
                source_start,
                actual_start,
                ContactReading(*initial_contact),
            ),
        )
        _validate_demo_reset(
            TrialResetState(
                first.observation.pose,
                first.state.current_joint_positions,
                first.state.collision_detected,
                first.state.contact_force_newtons,
                first.state.plug_position,
                first.state.plug_attached,
            ),
            TrialResetState(
                initial_snapshot.end_effector_pose,
                tuple(float(value) for value in actual_start.arm_positions),
                initial_snapshot.safety.collision_detected,
                initial_snapshot.safety.contact_force_newtons,
                tuple(float(value) for value in initial_snapshot.plug_position),
                initial_snapshot.plug_attached,
            ),
        )
        replay = ReplayRuntime(
            actuators,
            attachment,
            recorder,
            sensor,
            1.0 / DEMO_FPS,
        )
        actual = replay.observe(source_start)
        await recorder.capture_current(initial_snapshot)
        label = RecordingLabel(RecordingMoment.MOTION, Phase.PRE_INSERTION)
        for step in steps:
            post_action = step.result.post_action
            if post_action is None:
                raise AssertionError("validated insertion demo step has no endpoint")
            target = JointCommand(
                np.asarray(post_action.joint_positions, dtype=np.float64),
                (1.0 - post_action.pose.values[-1]) * MAX_GRIPPER_WIDTH_M,
            )
            actual = await replay.transition(
                actual,
                target,
                frame_count=frames_per_action,
                phase=label,
                stage=ObservationStage.CABLE_GRASPED,
            )
        if hold_frames:
            await replay.transition(
                actual,
                actual,
                frame_count=hold_frames,
                phase=RecordingLabel(RecordingMoment.COMPLETE, Phase.PRE_INSERTION),
                stage=ObservationStage.CABLE_GRASPED,
            )
        verification = replay.verification
        recorder.set_metadata(
            "insertion_demo_replay",
            {
                **verification.to_dict(),
                "frames_per_action": frames_per_action,
                "hold_frames": hold_frames,
                "visualization_only": True,
            },
        )
        timeline.pause()
    except Exception:
        if recorder is not None:
            recorder.abort()
        raise
    finally:
        RenderingManager.set_dt(original_rendering_dt)
        timeline.pause()

    if recorder is None:
        raise AssertionError("validated insertion demo created no recorder")
    output = recorder.finish()
    return {
        "status": "recorded",
        "recording": recording_id,
        "output": str(output),
        "frames": recorder.frame_count,
        "fps": recorder.fps,
        "cameras": [spec.label for spec in CAMERA_SPECS],
        "source_run": run_id,
        "source_rollout": report.rollout_id,
        "visualization_only": True,
    }
