"""High-resolution playback of one validated, realized JEPA-WM candidate."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import MAX_GRIPPER_WIDTH_M
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
from sim.isaac_control_runtime import contact_sensor, read_contact
from sim.isaac_demo_camera import CAMERA_SPECS, DEMO_FPS, DemoRecorder
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


def _gripper_width(closedness: float) -> float:
    if not isfinite(closedness) or not 0.0 <= closedness <= 1.0:
        raise ValueError("candidate demo gripper closedness is invalid")
    return (1.0 - closedness) * MAX_GRIPPER_WIDTH_M


def command_errors(
    actual: JointCommand, expected: JointCommand
) -> tuple[float, float]:
    return (
        float(np.max(np.abs(actual.arm_positions - expected.arm_positions))),
        abs(actual.gripper_width_m - expected.gripper_width_m),
    )


@dataclass
class ReplayTrackingMonitor:
    maximum_arm_error_rad: float = 0.0
    maximum_gripper_error_m: float = 0.0
    arm_tolerance_rad: float = 0.01
    gripper_tolerance_m: float = 0.003

    def observe(self, actual: JointCommand, expected: JointCommand) -> None:
        arm_error, gripper_error = command_errors(actual, expected)
        self.maximum_arm_error_rad = max(self.maximum_arm_error_rad, arm_error)
        self.maximum_gripper_error_m = max(
            self.maximum_gripper_error_m, gripper_error
        )
        if (
            arm_error > self.arm_tolerance_rad
            or gripper_error > self.gripper_tolerance_m
        ):
            raise RuntimeError("candidate visualization failed replay tracking")


@dataclass
class ReplaySafetyMonitor:
    maximum_contact_force_newtons: float = 0.0
    collision_detected: bool = False

    def observe(self, collision: bool, contact_force_newtons: float) -> None:
        if not isfinite(contact_force_newtons) or contact_force_newtons < 0.0:
            raise RuntimeError("candidate visualization contact reading is invalid")
        self.collision_detected = self.collision_detected or collision
        self.maximum_contact_force_newtons = max(
            self.maximum_contact_force_newtons, contact_force_newtons
        )
        if collision or contact_force_newtons > 2.0:
            raise RuntimeError("candidate visualization encountered unsafe contact")


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
        start = JointCommand(
            np.asarray(state.current_joint_positions, dtype=np.float64),
            _gripper_width(observation.pose.values[-1]),
        )
        end = JointCommand(
            np.asarray(result.post_action.joint_positions, dtype=np.float64),
            _gripper_width(result.post_action.pose.values[-1]),
        )
        timeline.play()
        actuators.apply(start)
        for _ in range(16):
            await omni.kit.app.get_app().next_update_async()
        actual_start = actuators.actual_command()
        tracking = ReplayTrackingMonitor()
        tracking.observe(actual_start, start)
        start_collision, start_force = read_contact(sensor)
        safety = ReplaySafetyMonitor()
        safety.observe(start_collision, start_force)

        label = RecordingLabel(RecordingMoment.MOTION, Phase.READY)
        stage_label = ObservationStage.APPROACHING_CABLE
        await recorder.capture(
            recording_snapshot(label, stage_label, actual_start, attachment),
            advance=False,
        )
        for frame in range(1, motion_frames + 1):
            progress = frame / motion_frames
            blend = progress * progress * (3.0 - 2.0 * progress)
            target = JointCommand(
                actual_start.arm_positions
                + (end.arm_positions - actual_start.arm_positions) * blend,
                actual_start.gripper_width_m
                + (end.gripper_width_m - actual_start.gripper_width_m) * blend,
            )
            await move_joint_command(
                actuators,
                actuators.actual_command(),
                target,
                attachment,
                frame_count=1,
                phase=label,
                stage=stage_label,
                recorder=recorder,
                sample_period_seconds=1.0 / DEMO_FPS,
            )
            tracking.observe(actuators.actual_command(), target)
            collision, force = read_contact(sensor)
            safety.observe(collision, force)

        actual_end = actuators.actual_command()
        tracking.observe(actual_end, end)
        for _ in range(hold_frames):
            await move_joint_command(
                actuators,
                actual_end,
                actual_end,
                attachment,
                frame_count=1,
                phase=RecordingLabel(RecordingMoment.COMPLETE, Phase.READY),
                stage=stage_label,
                recorder=recorder,
                sample_period_seconds=1.0 / DEMO_FPS,
            )
            tracking.observe(actuators.actual_command(), actual_end)
            collision, force = read_contact(sensor)
            safety.observe(collision, force)
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
            tracking_passed=True,
            maximum_replay_joint_error_rad=tracking.maximum_arm_error_rad,
            maximum_replay_gripper_error_m=tracking.maximum_gripper_error_m,
            maximum_replay_contact_force_newtons=(
                safety.maximum_contact_force_newtons
            ),
            replay_collision_detected=safety.collision_detected,
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
