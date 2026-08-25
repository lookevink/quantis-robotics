"""High-resolution playback of one validated, realized JEPA-WM candidate."""

from __future__ import annotations

from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.candidate_demo import CandidateDemoMetadata
from jepa_wm.control_policy import ControlExecutionPolicy
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    ControlResultStatus,
    ControlSession,
)
from sim.demo_sequence import Phase
from sim.exploration import DatasetSplit, build_exploration_plan
from sim.isaac_control_capture import validated_control_reference
from sim.isaac_control_runtime import contact_sensor
from sim.isaac_demo_camera import CAMERA_SPECS, DEMO_FPS, DemoRecorder
from sim.isaac_demo_runtime import (
    JointCommand,
    create_actuators,
    prepare_plug,
    recording_snapshot,
    reset_stage,
)
from sim.isaac_demo_scene import ROBOT_PATH
from sim.isaac_exploration import apply_variant
from sim.isaac_replay import (
    ReplayRuntime,
    gripper_width_from_closedness,
)
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id


async def record_candidate_demo(
    candidate_report_id: str,
    recording_id: str,
    *,
    motion_frames: int = 36,
    hold_frames: int = 12,
) -> dict[str, Any]:
    """Replay a proven reset-trial action for synchronized 1080p visualization.

    This does not create new control evidence or authority. It reloads a completed,
    strictly validated candidate session, recreates its held-out reset, verifies the
    starting joints, and films the already-realized joint transition.
    """

    validate_recording_id(recording_id)
    if motion_frames <= 0 or hold_frames < 0:
        raise ValueError("candidate demo frame counts are invalid")

    from jepa_wm.candidate_trial import CandidateTrialReport

    report = CandidateTrialReport.load_persisted(
        QUANTIS_DATA_ROOT, candidate_report_id
    )
    if not report.comparison.candidate_trial_gate_passed:
        raise ValueError("candidate demo report did not pass its strict trial gate")
    candidate_session_id = report.candidate_session_id
    candidate_session = ControlSession.at(CONTROL_ROOT, candidate_session_id)
    observation, state = candidate_session.load_capture()
    result = candidate_session.load_result()
    binding = candidate_session.load_candidate_binding()
    if (
        state.execution_policy is not ControlExecutionPolicy.RESET_TRIAL_CANDIDATE
        or result.status is not ControlResultStatus.APPLIED
        or result.post_action is None
        or not result.post_action.tracking.passed
        or result.post_action.collision_detected
    ):
        raise ValueError("candidate demo source is not a successful reset trial")
    source_session = ControlSession.at(CONTROL_ROOT, binding.source_session_id)
    shadow = source_session.load_shadow()
    validated_control_reference(
        state.reference_recording,
        state.seed,
        ControlExecutionPolicy.RESET_TRIAL_CANDIDATE,
    )
    plan = build_exploration_plan(state.seed, DatasetSplit.HELD_OUT)

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
        replay = ReplayRuntime(
            actuators,
            attachment,
            recorder,
            sensor,
            1.0 / DEMO_FPS,
        )
        start = JointCommand(
            np.asarray(state.current_joint_positions, dtype=np.float64),
            gripper_width_from_closedness(observation.pose.values[-1]),
        )
        end = JointCommand(
            np.asarray(result.post_action.joint_positions, dtype=np.float64),
            gripper_width_from_closedness(result.post_action.pose.values[-1]),
        )
        timeline.play()
        actuators.set_reset_state(start)
        for _ in range(16):
            await omni.kit.app.get_app().next_update_async()
        actual_start = replay.observe(start)

        label = RecordingLabel(RecordingMoment.MOTION, Phase.READY)
        stage_label = ObservationStage.APPROACHING_CABLE
        await recorder.capture(
            recording_snapshot(label, stage_label, actual_start, attachment),
            advance=False,
        )
        actual_end = await replay.transition(
            actual_start,
            end,
            frame_count=motion_frames,
            phase=label,
            stage=stage_label,
        )
        if hold_frames:
            await replay.transition(
                actual_end,
                actual_end,
                frame_count=hold_frames,
                phase=RecordingLabel(RecordingMoment.COMPLETE, Phase.READY),
                stage=stage_label,
            )
        demo_metadata = CandidateDemoMetadata(
            report_id=candidate_report_id,
            candidate_session=candidate_session_id,
            source_session=binding.source_session_id,
            seed=state.seed,
            policy=state.execution_policy,
            selected_action_scale=result.selected_action_scale,
            candidates_scored=shadow.candidates_scored,
            planner=shadow.config.planner,
            energy_improvement=binding.energy_improvement,
            actual_action=result.post_action.actual_action,
            replay=replay.verification,
        )
        recorder.set_metadata("candidate_demo", demo_metadata.to_dict())
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
        "candidate_session": candidate_session_id,
        "candidate_report": candidate_report_id,
        "visualization_only": True,
    }
