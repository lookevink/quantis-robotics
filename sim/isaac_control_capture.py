"""Capture one session-bound live observation for JEPA-WM control."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any, Callable

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
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.contact_grasp_target import CONTACT_GRASP_TARGET_POLICY
from jepa_wm.domain_recording import DomainRecording
from jepa_wm.control_resolution_baseline import ControlResolutionCaptureBaselineContract
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    ContactInsertionSegment,
    INSERTION_TASK_ID,
    insertion_control_target_policy,
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.insertion_rollout import (
    InsertionRolloutPosition,
    is_insertion_rollout_policy,
)
from jepa_wm.insertion_recording import ContactInsertionEvidence
from jepa_wm.persistence import write_json_atomic
from jepa_wm.physical_observation import PhysicalRoutingObservation
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.training_artifact import artifact_fingerprint
from jepa_wm.trajectory import load_rollout_at
from sim.control_capture_schedule import (
    ControlCaptureSchedule,
    ControlCapturePhase,
    ControlKnownStart,
    ControlKnownStartAuthority,
    ControlWarmupFramePlan,
    canonical_control_fingerprint,
    control_capture_schedule,
    control_reference_fingerprint,
    control_warmup_plan,
    requires_stable_insertion_capture,
    run_control_capture_phase,
    validate_known_start_collision_configuration,
    validate_known_start_pose,
)
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    RECORDING_ROOT,
    ControlCaptureResult,
    ControlSession,
    ControlSessionState,
)
from sim.control_context import (
    ControlContextPurpose,
    load_control_context,
    recording_task,
)
from sim.control_identity import (
    ControlProposalRef,
    control_proposal_path,
    observation_id_for_session,
    requires_authenticated_control_proposal,
)
from sim.control_timeline import (
    ControlTaskTimeline,
    control_context_recording_label,
)
from sim.exploration import (
    DatasetSplit,
    build_exploration_plan,
)
from sim.isaac_control_runtime import (
    LiveContactInterlock,
    LiveControlRuntime,
    bind_live_runtime,
    control_contact_sensors,
    synchronized_control_safety_snapshot,
)
from sim.isaac_demo_camera import JEPA_WM_CAMERA_SPECS, DemoRecorder
from sim.isaac_demo_runtime import (
    ContactReading,
    JointCommand,
    advance_physics_updates,
    create_actuators,
    move_joint_command,
    physics_simulation_time_seconds,
    prepare_plug,
    recording_safety_telemetry,
    recording_snapshot,
    reset_stage,
    resume_live_simulation,
)
from sim.isaac_demo_scene import (
    PLUG_PATH,
    ROBOT_PATH,
    SOCKET_PATH,
    STAGE_PATH,
    world_pose,
)
from sim.isaac_exploration import (
    ExplorationRecordingMode,
    ExplorationRecordingProfile,
    apply_variant,
)
from sim.recording import (
    RecordingLabel,
    RecordingSafetyTelemetry,
    validate_recording_id,
)


@dataclass(frozen=True)
class StabilizedControlCapture:
    safety: RecordingSafetyTelemetry
    previous_action: DroidAction


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
    if (
        ActionRecordingContract.from_mapping(manifest.get("action"))
        != ACTION_RECORDING_CONTRACT
    ):
        raise ValueError("control reference does not use the DROID action contract")
    if manifest.get("fps") != DROID_FPS:
        raise ValueError("control reference does not use the DROID frame rate")
    cameras = manifest.get("cameras")
    if not isinstance(cameras, list) or "wrist" not in cameras:
        raise ValueError("control reference does not contain a wrist camera")
    if recording_task(reference.path) == INSERTION_TASK_ID:
        ContactInsertionEvidence.from_recording(
            reference.path,
            expected_split=expected_split.value,
        )
    return reference


async def stabilize_resolution_capture(
    runtime: LiveControlRuntime,
    command: JointCommand,
    contract: ControlResolutionCaptureBaselineContract,
) -> StabilizedControlCapture:
    """Establish one stable drive-only state before strict-current capture."""

    from sim.isaac_control_resolution import (
        ResolutionControlInterlock,
        stabilize_resolution_baseline,
    )

    interlock = ResolutionControlInterlock(
        LiveContactInterlock(
            runtime.sensor,
            contract.safety_limits.maximum_contact_force_newtons,
            "insertion control resolution capture stabilization",
        ),
        runtime.attachment,
        expected_attachment=contract.load.plug_attached,
        operation="insertion control resolution capture stabilization",
    )
    _, baseline = await stabilize_resolution_baseline(
        runtime,
        interlock,
        contract.baseline_policy,
        physics_simulation_time_seconds,
        contract.load,
        contract.safety_limits,
    )
    baseline.validate(
        contract.baseline_policy,
        contract.load,
        contract.safety_limits,
    )

    actual = runtime.actuators.actual_command()
    evidence = interlock.contact.evidence
    return StabilizedControlCapture(
        recording_safety_telemetry(
            command,
            actual,
            ContactReading(
                evidence.collision_detected,
                evidence.maximum_contact_force_newtons,
            ),
        ),
        # A qualified terminal baseline is the stationary HOLD history state.
        # Its measured physical drift remains in the reset/safety evidence and
        # must not be relabeled as a commanded action for JEPA conditioning.
        DroidAction((0.0,) * 7),
    )


def recorded_control_context(
    steps: tuple[dict[str, Any], ...],
    warmup_plan: tuple[ControlWarmupFramePlan, ...],
    stable_previous_action: DroidAction | None,
) -> tuple[int, dict[str, Any], DroidAction]:
    """Resolve the terminal task context inside a sparse live recording."""

    expected_frames = sum(frame.record_rgb for frame in warmup_plan)
    if expected_frames <= 0 or len(steps) != expected_frames:
        raise RuntimeError("control warm-up recording has an unexpected frame count")
    context_frame_index = expected_frames - 1
    context_step = steps[context_frame_index]
    if context_step.get("index") != context_frame_index:
        raise RuntimeError("control warm-up telemetry is incomplete")
    if stable_previous_action is not None:
        previous_action = stable_previous_action
    else:
        try:
            previous_action = DroidAction(tuple(context_step["action_from_previous"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("control warm-up telemetry is incomplete") from error
    return context_frame_index, context_step, previous_action


def control_physical_routing_observation(
    context_step: dict[str, Any],
    insertion_target: Any,
    previous_action: DroidAction,
    *,
    insertion_control: bool,
) -> PhysicalRoutingObservation | None:
    """Derive the live router input only for contact-insertion captures."""

    if not insertion_control:
        return None
    if not isinstance(insertion_target, dict):
        raise ValueError("insertion control capture has no insertion target")
    return PhysicalRoutingObservation.from_recorded_step(
        context_step,
        insertion_target,
        previous_action,
    )


ControlCaptureProgress = Callable[[str, int, int], None]


def _reference_target_fingerprint(reference_rollout: Any) -> str:
    return canonical_control_fingerprint(
        {
            "frame_fingerprint": artifact_fingerprint(reference_rollout.target.path),
            "target_pose": list(reference_rollout.target_pose.values),
            "actions": [list(action.values) for action in reference_rollout.actions],
        }
    )


def _report_control_capture_progress(
    progress: ControlCaptureProgress | None,
    phase: str,
    completed_units: int,
    total_units: int,
) -> None:
    if progress is not None:
        progress(phase, completed_units, total_units)


async def _capture_stable_control_frame(
    *,
    session: ControlSession,
    session_id: str,
    reference_recording: str,
    seed: int,
    context_index: int,
    policy: ControlExecutionPolicy,
    capture_purpose: ControlContextPurpose,
    stage: Any,
    actuators: Any,
    attachment: Any,
    sensor: Any,
    command: JointCommand,
    recorder: DemoRecorder,
    recording_phase: RecordingLabel,
    recording_stage: ObservationStage,
    warmup_interlock: LiveContactInterlock,
) -> DroidAction:
    from jepa_wm.control_resolution import CONTROL_RESOLUTION_PROTOCOL
    from jepa_wm.control_resolution_baseline import (
        ControlResolutionCaptureAttemptIdentity,
        ControlResolutionCaptureFailureEvidence,
        ControlResolutionCaptureSourceIdentity,
    )
    from jepa_wm.control_resolution_profile import ControlResolutionLoad
    from sim.isaac_control_resolution import UnstableControlResolutionBaseline

    if not recorder.cameras_active:
        recorder.activate_cameras()
        await recorder.prepare_current(warmup_interlock.observe)
    capture_runtime = LiveControlRuntime(
        session_id,
        stage,
        actuators,
        attachment,
        sensor,
    )
    baseline_policy = CONTROL_RESOLUTION_PROTOCOL.baseline_policy
    if baseline_policy is None:
        raise RuntimeError("control resolution capture has no baseline policy")
    capture_contract = ControlResolutionCaptureBaselineContract(
        baseline_policy,
        CONTROL_RESOLUTION_PROTOCOL.safety_limits,
        (
            ControlResolutionLoad.UNLOADED
            if capture_purpose is ControlContextPurpose.CONTACT_GRASP
            else ControlResolutionLoad.ATTACHED
        ),
    )
    try:
        stabilized_capture = await stabilize_resolution_capture(
            capture_runtime,
            command,
            capture_contract,
        )
    except UnstableControlResolutionBaseline as error:
        if policy is ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT:
            failure = ControlResolutionCaptureFailureEvidence(
                identity=ControlResolutionCaptureAttemptIdentity(
                    session_id,
                    ControlResolutionCaptureSourceIdentity(
                        reference_recording,
                        seed,
                        context_index,
                    ),
                ),
                failed_at_unix_seconds=time(),
                contract=capture_contract,
                baseline_attempt=error.attempt,
                error=f"{type(error).__name__}: {error}",
            )
            session.create()
            write_json_atomic(
                session.resolution_capture_failure_path,
                failure.to_dict(),
            )
        raise
    await recorder.capture_current(
        recording_snapshot(
            recording_phase,
            recording_stage,
            command,
            attachment,
            safety=stabilized_capture.safety,
        ),
    )
    return stabilized_capture.previous_action


async def capture_control_observation(
    session_id: str,
    reference_recording: str,
    seed: int,
    proposal_name: str,
    execution_policy: str = ControlExecutionPolicy.DIRECT.value,
    context_index: int = 4,
    insertion_rollout_maximum_steps: int | None = None,
    context_purpose: str = ControlContextPurpose.STANDARD.value,
    progress: ControlCaptureProgress | None = None,
) -> dict[str, Any]:
    """Initialize or replay a context and persist one live wrist observation."""

    validate_recording_id(proposal_name)
    policy = ControlExecutionPolicy(execution_policy)
    proposal_ref = (
        ControlProposalRef.from_name(proposal_name)
        if requires_authenticated_control_proposal(policy)
        else None
    )
    expected_proposal = (
        proposal_ref.path
        if proposal_ref is not None
        else control_proposal_path(proposal_name)
    )

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager

    capture_purpose = ControlContextPurpose(context_purpose)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    if session.path.exists():
        raise ValueError(f"control session already exists: {session_id}")
    reference = validated_control_reference(reference_recording, seed, policy)
    insertion_control = recording_task(reference.path) == INSERTION_TASK_ID
    insertion_profile = (
        ExplorationRecordingProfile.for_mode(ExplorationRecordingMode.CONTACT_INSERTION)
        if insertion_control
        else None
    )
    plan = build_exploration_plan(seed, reference.split)
    if insertion_profile is not None:
        plan = insertion_profile.apply_to_plan(plan)
    context_steps = load_control_context(
        reference.path,
        context_index,
        plan,
        capture_purpose,
    )
    schedule = control_capture_schedule(
        policy,
        insertion_control=insertion_control,
        context_index=context_index,
        context_purpose=capture_purpose,
    )
    warmup_plan = schedule.frames
    task_timeline = ControlTaskTimeline.from_context(context_steps, schedule)
    recorded_task_indices = schedule.recorded_task_indices
    target_policy = insertion_control_target_policy(policy)
    reference_rollout = (
        target_policy.select(
            reference.path,
            context_index=context_index,
        )
        if target_policy is not None
        else load_rollout_at(
            reference.path,
            camera="wrist",
            context_index=context_index,
            bounds=ActionSelectionBounds(minimum_action_norm=0.0),
        )
    )
    target_metadata = reference.manifest.get("metadata", {}).get("insertion_target")
    known_start_inputs: dict[str, Any] | None = None
    if schedule.defer_camera_activation:
        if not isinstance(target_metadata, dict):
            raise ValueError("known-start reference has no insertion target")
        known_start_inputs = {
            "socket_position": tuple(
                float(value) for value in target_metadata["socket_position"]
            ),
            "socket_orientation_wxyz": tuple(
                float(value) for value in target_metadata["socket_orientation_wxyz"]
            ),
            "reference_fingerprint": control_reference_fingerprint(reference.path),
            "stage_asset_fingerprint": artifact_fingerprint(Path(STAGE_PATH)),
            "exploration_plan_fingerprint": canonical_control_fingerprint(
                plan.metadata()
            ),
            "context_fingerprint": canonical_control_fingerprint(
                [step.to_dict() for step in context_steps]
            ),
            "target_fingerprint": _reference_target_fingerprint(reference_rollout),
        }
    total_units = schedule.progress_units
    _report_control_capture_progress(progress, "reset", 0, total_units)
    await run_control_capture_phase(
        schedule.timing_budget,
        ControlCapturePhase.RESET,
        reset_stage(),
    )
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    apply_variant(stage, plan)
    attachment_preparation = (
        insertion_profile.prepare_attachment(stage)
        if insertion_profile is not None
        else prepare_plug(stage)
    )
    recording_id = f"control-{session_id}"
    recording_metadata = {
        **plan.metadata(),
        "control_session": session_id,
        "control_context_index": context_index,
        "control_capture_schedule": schedule.to_dict(),
        "control_capture_schedule_fingerprint": schedule.fingerprint,
    }
    if proposal_ref is not None:
        recording_metadata["control_proposal_ref"] = proposal_ref.to_dict()
    if capture_purpose is ControlContextPurpose.CONTACT_GRASP:
        recording_metadata["control_recorded_task_indices"] = list(
            recorded_task_indices
        )
    recorder = DemoRecorder(
        recording_id,
        fps=DROID_FPS,
        minimum_stage_frames=0,
        camera_specs=JEPA_WM_CAMERA_SPECS,
        metadata=recording_metadata,
        defer_camera_activation=schedule.defer_camera_activation,
    )
    timeline = omni.timeline.get_timeline_interface()
    original_rendering_dt = RenderingManager.get_dt()
    completed = False
    stable_previous_action: DroidAction | None = None
    known_start: ControlKnownStart | None = None
    try:
        await recorder.initialize()
        RenderingManager.set_dt(plan.sample_period_seconds)
        sensor = control_contact_sensors(
            stage,
            create=True,
            include_connector=insertion_control,
        )

        warmup_interlock = LiveContactInterlock(
            sensor,
            SimulatorSafetyLimits().maximum_contact_force_newtons,
            "insertion control warm-up",
        )

        await omni.kit.app.get_app().next_update_async()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        attachment = (
            insertion_profile.bind_attachment(
                attachment_preparation,
                RigidPrim(PLUG_PATH),
            )
            if insertion_profile is not None
            else attachment_preparation
        )
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        initialization_step = context_steps[schedule.initialization_task_index]
        if schedule.defer_camera_activation and initialization_step.plug_attached:
            raise ValueError("known-start control capture must begin unattached")
        origin = JointCommand(
            np.asarray(initialization_step.arm_positions),
            initialization_step.gripper_width_m,
        )
        resume_live_simulation(timeline)
        _report_control_capture_progress(
            progress,
            "known_start" if schedule.defer_camera_activation else "initialization",
            1,
            total_units,
        )
        actuators.set_reset_state(origin)
        if insertion_control:
            await run_control_capture_phase(
                schedule.timing_budget,
                ControlCapturePhase.KNOWN_START,
                advance_physics_updates(16, warmup_interlock.observe),
            )
        else:
            for _ in range(16):
                await omni.kit.app.get_app().next_update_async()
        collision_start = CONTACT_INSERTION_RECORDING.start_index(
            ContactInsertionSegment.ALIGN
        )
        collision_enabled = False
        if insertion_control and initialization_step.index >= collision_start:
            attachment.set_collisions(True)
            collision_enabled = True
        if schedule.defer_camera_activation:
            if known_start_inputs is None:
                raise RuntimeError("contact grasp capture has no known-start inputs")
            collision_configuration = attachment.collision_configuration
            validate_known_start_collision_configuration(
                target_metadata,
                attachment.compliant_collision_parts,
                collision_configuration,
            )
            plug_position, plug_orientation = attachment.world_pose()
            socket_position, socket_orientation = world_pose(
                stage.GetPrimAtPath(SOCKET_PATH)
            )
            validate_known_start_pose(
                "connector",
                tuple(float(value) for value in plug_position),
                tuple(float(value) for value in plug_orientation),
                initialization_step.plug_position,
                initialization_step.plug_orientation_wxyz,
            )
            validate_known_start_pose(
                "socket",
                tuple(float(value) for value in socket_position),
                tuple(float(value) for value in socket_orientation),
                known_start_inputs["socket_position"],
                known_start_inputs["socket_orientation_wxyz"],
            )
            authority = ControlKnownStartAuthority(
                known_start_inputs["reference_fingerprint"],
                known_start_inputs["stage_asset_fingerprint"],
                known_start_inputs["exploration_plan_fingerprint"],
                known_start_inputs["context_fingerprint"],
                known_start_inputs["target_fingerprint"],
                canonical_control_fingerprint(collision_configuration),
            )
            known_start = ControlKnownStart.from_context(
                reference_recording,
                seed,
                initialization_step,
                known_start_inputs["socket_position"],
                known_start_inputs["socket_orientation_wxyz"],
                schedule,
                authority,
            )
            recorder.set_metadata("control_known_start", known_start.to_dict())
            recorder.set_metadata(
                "control_known_start_fingerprint",
                known_start.fingerprint,
            )
        initialization_state = task_timeline.initialization
        initialization_frame = initialization_state.frame_plan
        initialization_phase = initialization_state.recording_label
        initialization_stage = initialization_state.observation_stage
        if initialization_frame.stabilize:
            _report_control_capture_progress(
                progress,
                ControlCapturePhase.TERMINAL_CAMERA_AND_STABILIZATION,
                2,
                total_units,
            )
            stable_previous_action = await run_control_capture_phase(
                schedule.timing_budget,
                ControlCapturePhase.TERMINAL_CAMERA_AND_STABILIZATION,
                _capture_stable_control_frame(
                    session=session,
                    session_id=session_id,
                    reference_recording=reference_recording,
                    seed=seed,
                    context_index=context_index,
                    policy=policy,
                    capture_purpose=capture_purpose,
                    stage=stage,
                    actuators=actuators,
                    attachment=attachment,
                    sensor=sensor,
                    command=origin,
                    recorder=recorder,
                    recording_phase=initialization_phase,
                    recording_stage=initialization_stage,
                    warmup_interlock=warmup_interlock,
                ),
            )
        elif initialization_frame.record_rgb:
            await recorder.capture(
                recording_snapshot(
                    initialization_phase,
                    initialization_stage,
                    origin,
                    attachment,
                ),
                advance=False,
            )
        current = origin
        for replay_index, task_state in enumerate(task_timeline.replay, start=1):
            frame_plan = task_state.frame_plan
            step = context_steps[task_state.task_index]
            _report_control_capture_progress(
                progress,
                "replay",
                1 + replay_index,
                total_units,
            )
            if (
                insertion_control
                and not collision_enabled
                and step.index >= collision_start
            ):
                attachment.set_collisions(True)
                collision_enabled = True
            if step.plug_attached and not attachment.attached:
                attachment.attach(world_pose(attachment.hand_prim)[0])
            elif not step.plug_attached and attachment.attached:
                raise ValueError("recorded control context loses its plug attachment")
            command = JointCommand(
                np.asarray(step.arm_positions),
                step.gripper_width_m,
            )
            recording_phase = task_state.recording_label
            recording_stage = task_state.observation_stage
            strict_current_capture = frame_plan.stabilize
            await move_joint_command(
                actuators,
                current,
                command,
                attachment,
                frame_count=1,
                phase=recording_phase,
                stage=recording_stage,
                recorder=(
                    None
                    if strict_current_capture or not frame_plan.record_rgb
                    else recorder
                ),
                sample_period_seconds=plan.sample_period_seconds,
                observe_safety=(
                    warmup_interlock.observe if frame_plan.observe_safety else None
                ),
            )
            if strict_current_capture:
                stable_previous_action = await run_control_capture_phase(
                    schedule.timing_budget,
                    ControlCapturePhase.TERMINAL_CAMERA_AND_STABILIZATION,
                    _capture_stable_control_frame(
                        session=session,
                        session_id=session_id,
                        reference_recording=reference_recording,
                        seed=seed,
                        context_index=context_index,
                        policy=policy,
                        capture_purpose=capture_purpose,
                        stage=stage,
                        actuators=actuators,
                        attachment=attachment,
                        sensor=sensor,
                        command=command,
                        recorder=recorder,
                        recording_phase=recording_phase,
                        recording_stage=recording_stage,
                        warmup_interlock=warmup_interlock,
                    ),
                )
            current = command
        _report_control_capture_progress(
            progress,
            ControlCapturePhase.TERMINAL_SNAPSHOT,
            total_units - 1,
            total_units,
        )
        captured_state = await run_control_capture_phase(
            schedule.timing_budget,
            ControlCapturePhase.TERMINAL_SNAPSHOT,
            synchronized_control_safety_snapshot(
                timeline,
                actuators,
                attachment,
                sensor,
                omni.kit.app.get_app().next_update_async,
            ),
        )
        _report_control_capture_progress(
            progress,
            "complete",
            total_units,
            total_units,
        )
        completed = True
    except asyncio.CancelledError:
        recorder.abort()
        raise
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
    context_frame_index, context_step, previous_action = recorded_control_context(
        steps,
        warmup_plan,
        stable_previous_action,
    )
    target = reference_rollout.target.path
    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=time(),
        context_frame=(
            output / "wrist" / f"frame_{context_frame_index:06d}.png"
        ).relative_to(QUANTIS_DATA_ROOT),
        target=ControlTarget(
            target.relative_to(QUANTIS_DATA_ROOT), reference_rollout.target_pose
        ),
        expected_proposal=expected_proposal,
        pose=DroidPose(tuple(context_step["end_effector_pose"])),
        previous_action=previous_action,
        warmup_frames=context_index,
        physical_routing=control_physical_routing_observation(
            context_step,
            target_metadata,
            previous_action,
            insertion_control=insertion_control,
        ),
    )
    active_command = actuators.current_command() if insertion_control else None
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=reference_recording,
        seed=seed,
        recording=recording_id,
        current_joint_positions=captured_state.joint_positions,
        collision_detected=captured_state.collision_detected,
        contact_force_newtons=captured_state.contact_force_newtons,
        execution_policy=policy,
        plug_position=captured_state.plug_position,
        plug_attached=captured_state.plug_attached,
        current_gripper_width_m=captured_state.gripper_width_m,
        insertion_target_policy=target_policy,
        active_drive_target=(
            JointDriveTarget(
                tuple(float(value) for value in active_command.arm_positions),
                active_command.gripper_width_m,
            )
            if active_command is not None
            else None
        ),
        insertion_rollout_position=(
            InsertionRolloutPosition.initial(insertion_rollout_maximum_steps)
            if is_insertion_rollout_policy(policy)
            else None
        ),
        contact_grasp_target_policy=(
            CONTACT_GRASP_TARGET_POLICY
            if capture_purpose is ControlContextPurpose.CONTACT_GRASP
            else None
        ),
    )
    session.write_capture(observation, state)
    bind_live_runtime(session_id, stage, actuators, attachment, sensor)
    return ControlCaptureResult(
        session_id,
        observation,
        session.request_path,
        captured_state.contact_force_newtons,
        captured_state.collision_detected,
    ).to_dict()
