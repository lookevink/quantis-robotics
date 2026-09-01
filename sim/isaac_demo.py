"""Live execution facade for the deterministic cable plug-in demo."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DROID_FPS
from sim.demo_sequence import Phase, PlugAction
from sim.exploration import DatasetSplit
from sim.isaac_demo_camera import (
    DEMO_FPS,
    RECORDING_JOB_ROOT,
    DemoRecorder,
    capture_cameras as _capture_cameras,
)
from sim.isaac_demo_kinematics import (
    preflight_report as _preflight_report,
    solve_waypoints as _solve_waypoints,
)
from sim.isaac_control_bridge import (
    apply_control_response as _apply_control_response,
    capture_control_observation as _capture_control_observation,
    capture_followup_observation as _capture_followup_observation,
    capture_insertion_transition_observation as _capture_insertion_transition_observation,
    persist_insertion_proposal_handoff,
    restore_insertion_no_actuation_retry as _restore_insertion_no_actuation_retry,
    restore_insertion_retry as _restore_insertion_retry,
    restore_insertion_rollback_retry as _restore_insertion_rollback_retry,
    restore_grasp_transition_retry as _restore_grasp_transition_retry,
    verify_grasp_to_insertion_result,
    verify_grasp_to_insertion_source,
    verify_insertion_demo_rollout_result,
    verify_insertion_followup_source,
    verify_insertion_two_step_result,
    evaluate_direct_insertion_candidate as _evaluate_direct_insertion_candidate,
    measure_insertion_control_resolution as _measure_insertion_control_resolution,
    evaluate_shadow_candidate as _evaluate_shadow_candidate,
    persist_experimental_candidate_response,
    persist_insertion_followup_response as _persist_insertion_followup_response,
    persist_insertion_trial_response,
    prepare_experimental_candidate_source,
    prepare_insertion_trial_source,
    persist_baseline_response,
)
from sim.isaac_candidate_demo import record_candidate_demo as _record_candidate_demo
from sim.isaac_grasp_demo import record_grasp_demo as _record_grasp_demo
from sim.isaac_insertion_demo import record_insertion_demo as _record_insertion_demo
from sim.isaac_unknown_start_reset import (
    authenticate_unknown_start_reset as _authenticate_unknown_start_reset,
)
from sim.isaac_unknown_start_shadow import (
    capture_unknown_start_shadow_observation as _capture_unknown_start_shadow_observation,
)
from sim.isaac_demo_runtime import (
    Actuators,
    JointCommand,
    PlugAttachment,
    create_actuators,
    move_joint_command,
    prepare_plug,
    recording_snapshot,
    reset_stage,
)
from sim.isaac_demo_scene import (
    PLUG_PATH,
    ROBOT_PATH,
    SOCKET_PATH,
    STAGE_PATH,
    world_pose,
)
from sim.recording import (
    RecordingLabel,
    RecordingMoment,
    RecordingSnapshot,
)
from sim.recording_jobs import RecordingJobManager


_RECORDING_JOBS = RecordingJobManager(RECORDING_JOB_ROOT)


def preflight_report() -> dict[str, Any]:
    """Read the live stage only while the simulator runtime is unowned."""

    return _RECORDING_JOBS.run_exclusive_sync("preflight-report", _preflight_report)


async def authenticate_unknown_start_reset(
    recording_id: str,
    seed: int,
    source_revision: str,
    runtime_source_fingerprint: str,
) -> dict[str, Any]:
    """Run one exclusive reset-only milestone-20 authentication."""

    return await _RECORDING_JOBS.run_exclusive(
        f"unknown-start-reset-{recording_id}",
        lambda: _authenticate_unknown_start_reset(
            recording_id,
            seed,
            source_revision,
            runtime_source_fingerprint,
        ),
    )


async def capture_unknown_start_shadow_observation(
    session_id: str,
    reference_recording: str,
    reference_seed: int,
    proposal_name: str,
    reset_recording_id: str,
    reset_result_fingerprint: str,
) -> dict[str, Any]:
    """Capture one exclusive zero-actuation model request from a passed reset."""

    return await _RECORDING_JOBS.run_exclusive(
        f"unknown-start-shadow-{session_id}",
        lambda: _capture_unknown_start_shadow_observation(
            session_id,
            reference_recording,
            reference_seed,
            proposal_name,
            reset_recording_id,
            reset_result_fingerprint,
        ),
    )


async def capture_cameras(
    output_dir: str = "/isaac-sim/.local/share/ov/data/quantis/captures",
) -> dict[str, Any]:
    """Capture cameras under the shared simulator-operation interlock."""

    return await _RECORDING_JOBS.run_exclusive(
        "capture-cameras",
        lambda: _capture_cameras(output_dir),
    )


async def capture_followup_observation(
    session_id: str,
    previous_session_id: str,
    proposal_name: str,
    insertion_rollout_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Capture one follow-up without overlapping another simulator operation."""

    return await _RECORDING_JOBS.run_exclusive(
        f"control-followup-{session_id}",
        lambda: _capture_followup_observation(
            session_id,
            previous_session_id,
            proposal_name,
            insertion_rollout_maximum_steps,
        ),
    )


async def capture_insertion_transition_observation(
    session_id: str,
    previous_session_id: str,
    proposal_name: str,
    insertion_rollout_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Capture a transition without overlapping another simulator operation."""

    if insertion_rollout_maximum_steps is None:
        operation = lambda: _capture_insertion_transition_observation(
            session_id,
            previous_session_id,
            proposal_name,
        )
    else:
        operation = lambda: _capture_insertion_transition_observation(
            session_id,
            previous_session_id,
            proposal_name,
            maximum_steps=insertion_rollout_maximum_steps,
        )
    return await _RECORDING_JOBS.run_exclusive(
        f"control-transition-{session_id}",
        operation,
    )


async def evaluate_direct_insertion_candidate(session_id: str) -> dict[str, Any]:
    """Refresh live safety state under the simulator-operation interlock."""

    return await _RECORDING_JOBS.run_exclusive(
        f"insertion-safety-{session_id}",
        lambda: _evaluate_direct_insertion_candidate(session_id),
    )


async def evaluate_shadow_candidate(session_id: str) -> dict[str, Any]:
    """Read shadow safety state under the simulator-operation interlock."""

    return await _RECORDING_JOBS.run_exclusive(
        f"shadow-safety-{session_id}",
        lambda: _evaluate_shadow_candidate(session_id),
    )


def restore_insertion_no_actuation_retry(
    previous_session_id: str,
    failed_safety_session_id: str,
    next_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Restore runtime ownership under the simulator-operation interlock."""

    return _RECORDING_JOBS.run_exclusive_sync(
        f"restore-insertion-{failed_safety_session_id}",
        lambda: _restore_insertion_no_actuation_retry(
            previous_session_id,
            failed_safety_session_id,
            next_maximum_steps,
        ),
    )


def restore_insertion_rollback_retry(
    previous_session_id: str,
    rolled_back_session_id: str,
    next_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Restore rollback ownership under the simulator-operation interlock."""

    return _RECORDING_JOBS.run_exclusive_sync(
        f"restore-insertion-{rolled_back_session_id}",
        lambda: _restore_insertion_rollback_retry(
            previous_session_id,
            rolled_back_session_id,
            next_maximum_steps,
        ),
    )


def restore_insertion_retry(
    previous_session_id: str,
    failed_session_id: str,
    next_maximum_steps: int | None = None,
) -> dict[str, Any]:
    """Restore either retry kind under the simulator-operation interlock."""

    return _RECORDING_JOBS.run_exclusive_sync(
        f"restore-insertion-{failed_session_id}",
        lambda: _restore_insertion_retry(
            previous_session_id,
            failed_session_id,
            next_maximum_steps,
        ),
    )


def restore_grasp_transition_retry(
    grasp_session_id: str,
    rolled_back_session_id: str,
) -> dict[str, Any]:
    """Restore grasp ownership under the simulator-operation interlock."""

    return _RECORDING_JOBS.run_exclusive_sync(
        f"restore-grasp-{rolled_back_session_id}",
        lambda: _restore_grasp_transition_retry(
            grasp_session_id,
            rolled_back_session_id,
        ),
    )


def persist_insertion_followup_response(
    session_id: str,
    source_session_id: str,
    *,
    control_root: Path | None = None,
) -> dict[str, Any]:
    """Rebind follow-up ownership under the simulator-operation interlock."""

    if control_root is None:
        operation = lambda: _persist_insertion_followup_response(
            session_id,
            source_session_id,
        )
    else:
        operation = lambda: _persist_insertion_followup_response(
            session_id,
            source_session_id,
            control_root=control_root,
        )
    return _RECORDING_JOBS.run_exclusive_sync(
        f"bind-insertion-{session_id}",
        operation,
    )


async def apply_control_response(session_id: str) -> dict[str, Any]:
    """Apply one response under the shared simulator-operation interlock."""

    return await _RECORDING_JOBS.run_exclusive(
        f"control-apply-{session_id}",
        lambda: _apply_control_response(session_id),
    )


async def measure_insertion_control_resolution(
    session_id: str,
    load: str = "attached",
    protocol: Any | None = None,
) -> dict[str, Any]:
    """Measure drive resolution without overlapping another simulator action."""

    if protocol is None:
        operation = lambda: _measure_insertion_control_resolution(session_id, load)
    else:
        operation = lambda: _measure_insertion_control_resolution(
            session_id,
            load,
            protocol,
        )
    return await _RECORDING_JOBS.run_exclusive(
        f"control-resolution-{session_id}",
        operation,
    )


async def record_candidate_demo(
    candidate_report_id: str,
    recording_id: str,
    *,
    motion_frames: int = 36,
    hold_frames: int = 12,
) -> dict[str, Any]:
    """Render a candidate only while the simulator runtime is unowned."""

    return await _RECORDING_JOBS.run_exclusive(
        f"candidate-demo-{recording_id}",
        lambda: _record_candidate_demo(
            candidate_report_id,
            recording_id,
            motion_frames=motion_frames,
            hold_frames=hold_frames,
        ),
    )


async def record_grasp_demo(
    readiness_id: str,
    exploration_seed: int,
    recording_id: str,
    proposal_fingerprint: str,
    *,
    frames_per_action: int = 8,
    hold_frames: int = 12,
) -> dict[str, Any]:
    """Render a grasp only while the simulator runtime is unowned."""

    return await _RECORDING_JOBS.run_exclusive(
        f"grasp-demo-{recording_id}",
        lambda: _record_grasp_demo(
            readiness_id,
            exploration_seed,
            recording_id,
            proposal_fingerprint,
            frames_per_action=frames_per_action,
            hold_frames=hold_frames,
        ),
    )


async def record_insertion_demo(
    source_run_id: str,
    recording_id: str,
    *,
    frames_per_action: int = 18,
    hold_frames: int = 18,
) -> dict[str, Any]:
    """Render insertion only while the simulator runtime is unowned."""

    return await _RECORDING_JOBS.run_exclusive(
        f"insertion-demo-{recording_id}",
        lambda: _record_insertion_demo(
            source_run_id,
            recording_id,
            frames_per_action=frames_per_action,
            hold_frames=hold_frames,
        ),
    )


async def _reset_demo() -> dict[str, Any]:
    """Stop physics and reopen the saved reusable starting stage."""

    return await reset_stage()


async def reset_demo() -> dict[str, Any]:
    """Reset only while no other simulator operation owns the runtime."""

    return await _RECORDING_JOBS.run_exclusive("reset-demo", _reset_demo)


async def _settle_at_target(
    target_hand_position: np.ndarray,
    attachment: PlugAttachment,
    command: JointCommand,
    *,
    phase: RecordingLabel,
    stage: ObservationStage,
    recorder: DemoRecorder | None,
    max_updates: int = 4,
    tolerance_m: float = 0.012,
) -> float:
    import omni.kit.app
    import omni.usd

    app = omni.kit.app.get_app()
    hand = omni.usd.get_context().get_stage().GetPrimAtPath(f"{ROBOT_PATH}/panda_hand")
    error = float("inf")
    for _ in range(max_updates):
        await app.next_update_async()
        hand_position, _ = world_pose(hand)
        error = float(np.linalg.norm(hand_position - target_hand_position))
        attachment.follow(hand_position)
        if recorder is not None:
            await recorder.capture(
                recording_snapshot(phase, stage, command, attachment),
                advance=False,
            )
        if error <= tolerance_m:
            return error
    return error


async def _fill_stage_observations(
    recorder: DemoRecorder | None,
    snapshot: RecordingSnapshot,
) -> None:
    if recorder is None:
        return
    while recorder.stage_frame_count(snapshot.stage) < recorder.minimum_stage_frames:
        await recorder.capture(snapshot)


async def _run_demo(recorder: DemoRecorder | None = None) -> dict[str, Any]:
    """Execute the preflighted sequence and export its final visual state."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.simulation_manager import SimulationManager

    stage = omni.usd.get_context().get_stage()
    solved = _solve_waypoints()
    stage.SetEditTarget(stage.GetSessionLayer())
    attachment = prepare_plug(stage)

    await omni.kit.app.get_app().next_update_async()
    if SimulationManager.get_physics_sim_view() is None:
        SimulationManager.initialize_physics()
    actuators = create_actuators(stage, Articulation(ROBOT_PATH))
    current = actuators.current_command()
    observation_stage = ObservationStage.APPROACHING_CABLE

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    phase_reports = []
    completed = False
    try:
        await omni.kit.app.get_app().next_update_async()
        if recorder is not None:
            await recorder.capture(
                recording_snapshot(
                    RecordingLabel(RecordingMoment.INITIAL),
                    observation_stage,
                    current,
                    attachment,
                )
            )
        for result in solved:
            waypoint = result.waypoint
            motion_width = (
                current.gripper_width_m
                if waypoint.phase == Phase.GRASP
                else waypoint.gripper_width_m
            )
            target = JointCommand(result.arm_positions, motion_width)
            max_delta = float(
                np.max(np.abs(np.rad2deg(target.arm_positions - current.arm_positions)))
            )
            duration = min(4.0, max(0.8, max_delta / 45.0))
            authored_fps = recorder.fps if recorder is not None else 8
            await move_joint_command(
                actuators,
                current,
                target,
                attachment,
                frame_count=max(1, ceil(duration * authored_fps)),
                phase=RecordingLabel(RecordingMoment.MOTION, waypoint.phase),
                stage=observation_stage,
                recorder=recorder,
            )
            current = target

            settle_error = await _settle_at_target(
                result.hand_position,
                attachment,
                current,
                phase=RecordingLabel(RecordingMoment.SETTLE, waypoint.phase),
                stage=observation_stage,
                recorder=recorder,
            )
            if settle_error > 0.012:
                raise RuntimeError(
                    f"arm failed to settle at {waypoint.phase.value}: "
                    f"{settle_error:.4f} m hand error"
                )

            if waypoint.phase == Phase.GRASP:
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.GRASP),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
            elif waypoint.phase == Phase.PRE_INSERTION:
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.PRE_INSERTION),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
                observation_stage = ObservationStage.ALIGNED_WITH_SOCKET
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.PRE_INSERTION),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )
            elif waypoint.phase == Phase.INSERT:
                observation_stage = ObservationStage.PLUG_SEATED
                await _fill_stage_observations(
                    recorder,
                    recording_snapshot(
                        RecordingLabel(RecordingMoment.SETTLE, Phase.INSERT),
                        observation_stage,
                        current,
                        attachment,
                    ),
                )

            if waypoint.plug_action == PlugAction.ATTACH:
                closed = JointCommand(current.arm_positions, waypoint.gripper_width_m)
                authored_fps = recorder.fps if recorder is not None else 8
                await move_joint_command(
                    actuators,
                    current,
                    closed,
                    attachment,
                    frame_count=max(1, ceil(0.6 * authored_fps)),
                    phase=RecordingLabel(RecordingMoment.CLOSE, Phase.GRASP),
                    stage=observation_stage,
                    recorder=recorder,
                )
                current = closed
                hand = stage.GetPrimAtPath(f"{ROBOT_PATH}/panda_hand")
                attachment.attach(world_pose(hand)[0])
                observation_stage = ObservationStage.CABLE_GRASPED
                if recorder is not None:
                    await recorder.capture(
                        recording_snapshot(
                            RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP),
                            observation_stage,
                            current,
                            attachment,
                        )
                    )
            elif waypoint.plug_action == PlugAction.DETACH:
                socket_position, _ = world_pose(stage.GetPrimAtPath(SOCKET_PATH))
                attachment.detach_at(socket_position)
                if recorder is not None:
                    await recorder.capture(
                        recording_snapshot(
                            RecordingLabel(RecordingMoment.COMPLETE, Phase.RELEASE),
                            observation_stage,
                            current,
                            attachment,
                        )
                    )

            phase_reports.append(
                {
                    "phase": waypoint.phase.value,
                    "duration_seconds": duration,
                    "target_degrees": np.rad2deg(current.arm_positions)
                    .round(3)
                    .tolist(),
                    "gripper_width_m": current.gripper_width_m,
                    "settle_error_m": settle_error,
                }
            )
        completed = True
    finally:
        if completed:
            timeline.pause()
        else:
            timeline.stop()

    result_stage = STAGE_PATH.replace(".usda", "_sequence_result.usda")
    stage.Export(result_stage)
    return {
        "status": "complete",
        "result_stage": result_stage,
        "phases": phase_reports,
        "plug_position": world_pose(attachment.prim)[0].round(6).tolist(),
    }


async def run_demo(recorder: DemoRecorder | None = None) -> dict[str, Any]:
    """Run the scripted demo under the one-operation simulator interlock."""

    return await _RECORDING_JOBS.run_exclusive(
        "run-demo",
        lambda: _run_demo(recorder=recorder),
    )


async def _record_demo(
    recording_id: str,
    *,
    fps: int,
    minimum_stage_frames: int,
) -> dict[str, Any]:
    await _reset_demo()
    recorder = DemoRecorder(
        recording_id,
        fps=fps,
        minimum_stage_frames=minimum_stage_frames,
    )
    try:
        await recorder.initialize()
        result = await _run_demo(recorder=recorder)
    except Exception:
        recorder.abort()
        raise
    output_dir = recorder.finish()
    return {
        **result,
        "recording_id": recording_id,
        "output_directory": str(output_dir),
        "videos": {camera: str(path) for camera, path in recorder.video_paths.items()},
    }


async def record_demo(recording_id: str) -> dict[str, Any]:
    """Capture the presentation recording with complete stage windows."""

    from jepa.contract import DEFAULT_FRAMES

    return await _record_demo(
        recording_id,
        fps=DEMO_FPS,
        minimum_stage_frames=DEFAULT_FRAMES,
    )


async def record_action_trajectory(recording_id: str) -> dict[str, Any]:
    """Capture motion-only frames at JEPA-WM's four-frame-per-second rate."""

    return await _record_demo(
        recording_id,
        fps=DROID_FPS,
        minimum_stage_frames=0,
    )


def start_recording(recording_id: str) -> dict[str, Any]:
    """Start a long recording without holding the Python server connection."""

    return _RECORDING_JOBS.start(recording_id, record_demo)


def start_action_recording(recording_id: str) -> dict[str, Any]:
    """Start a DROID-action trajectory capture for offline world-model evaluation."""

    return _RECORDING_JOBS.start(recording_id, record_action_trajectory)


def start_exploration_recording(
    recording_id: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    """Start a seeded domain-exploration capture in the simulator background."""

    from sim.isaac_exploration import record_exploration_trajectory

    dataset_split = DatasetSplit(split)
    return _RECORDING_JOBS.start(
        recording_id,
        lambda job_recording_id: record_exploration_trajectory(
            job_recording_id,
            seed,
            dataset_split,
        ),
    )


def start_grasp_recording(
    recording_id: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    """Start a seeded reach-and-grasp task trajectory in the background."""

    from sim.isaac_exploration import record_grasp_trajectory

    dataset_split = DatasetSplit(split)
    return _RECORDING_JOBS.start(
        recording_id,
        lambda job_recording_id: record_grasp_trajectory(
            job_recording_id,
            seed,
            dataset_split,
        ),
    )


def start_insertion_recording(
    recording_id: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    """Start a seeded rearward-grasp and insertion trajectory in the background."""

    from sim.isaac_exploration import record_insertion_trajectory

    dataset_split = DatasetSplit(split)
    return _RECORDING_JOBS.start(
        recording_id,
        lambda job_recording_id: record_insertion_trajectory(
            job_recording_id,
            seed,
            dataset_split,
        ),
    )


def start_contact_insertion_recording(
    recording_id: str,
    seed: int,
    split: str,
) -> dict[str, Any]:
    """Start collision-enabled insertion with synchronized safety telemetry."""

    from sim.isaac_exploration import record_contact_insertion_trajectory

    dataset_split = DatasetSplit(split)
    return _RECORDING_JOBS.start(
        recording_id,
        lambda job_recording_id: record_contact_insertion_trajectory(
            job_recording_id,
            seed,
            dataset_split,
        ),
    )


def start_control_capture(
    session_id: str,
    reference_recording: str,
    seed: int,
    proposal_name: str,
    execution_policy: str,
    context_index: int,
    insertion_rollout_maximum_steps: int | None,
    context_purpose: str,
) -> dict[str, Any]:
    """Start one cancellable control capture with persisted phase progress."""

    job_id = f"control-{session_id}"

    def progress(phase: str, completed: int, total: int) -> None:
        _RECORDING_JOBS.progress(
            job_id,
            phase=phase,
            completed_units=completed,
            total_units=total,
        )

    return _RECORDING_JOBS.start(
        job_id,
        lambda _job_id: _capture_control_observation(
            session_id,
            reference_recording,
            seed,
            proposal_name,
            execution_policy,
            context_index,
            insertion_rollout_maximum_steps,
            context_purpose,
            progress,
        ),
    )


def cancel_recording_job(recording_id: str) -> dict[str, Any]:
    """Cancel one live simulator job through its owning manager."""

    return _RECORDING_JOBS.cancel(recording_id)
