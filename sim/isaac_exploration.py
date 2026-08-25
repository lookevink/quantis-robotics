"""Isaac runtime for seeded JEPA-WM domain exploration recordings."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from jepa.contract import ObservationStage
from jepa_wm.action import DROID_FPS
from jepa_wm.grasp_contract import GRASP_TASK_ID
from jepa_wm.insertion_contract import (
    CONTACT_AWARE_INSERTION_MODE,
    CONTACT_INSERTION_RECORDING,
    ContactInsertionRecordingContract,
    ContactInsertionSegment,
    INSERTION_AXIS,
    INSERTION_TASK_ID,
    KINEMATIC_INSERTION_MODE,
    REARWARD_GRASP_OFFSET_METERS,
)
from jepa_wm.insertion_task import InsertionTaskLimits
from sim.demo_sequence import Phase
from sim.exploration import (
    DatasetSplit,
    ExplorationPlan,
    SegmentOutcome,
    build_exploration_plan,
    validate_sample_times,
)
from sim.isaac_demo_camera import (
    JEPA_WM_CAMERA_SPECS,
    DemoRecorder,
    configure_wrist_camera,
)
from sim.isaac_demo_kinematics import solve_waypoints
from sim.isaac_demo_runtime import (
    Actuators,
    JointCommand,
    PlugAttachment,
    ContactReading,
    advance_physics_updates,
    create_actuators,
    move_joint_command,
    prepare_fixed_joint_plug,
    prepare_plug,
    recording_snapshot,
)
from sim.isaac_demo_scene import PLUG_PATH, ROBOT_PATH, SOCKET_PATH, world_pose
from sim.isaac_control_runtime import (
    connector_contact_sensor,
    contact_sensor,
    read_contact,
)
from sim.recording import RecordingLabel, RecordingMoment
from sim.recording import RecordingSafetyTelemetry


class ExplorationRecordingMode(str, Enum):
    DOMAIN = "domain"
    GRASP = "grasp"
    INSERTION = "insertion"
    CONTACT_INSERTION = "contact_insertion"

    @property
    def task_id(self) -> str:
        return {
            ExplorationRecordingMode.DOMAIN: "domain_exploration",
            ExplorationRecordingMode.GRASP: GRASP_TASK_ID,
            ExplorationRecordingMode.INSERTION: INSERTION_TASK_ID,
            ExplorationRecordingMode.CONTACT_INSERTION: INSERTION_TASK_ID,
        }[self]


@dataclass(frozen=True)
class ExplorationRecordingProfile:
    task_id: str
    evidence_mode: str | None
    include_exploration_targets: bool
    include_grasp: bool
    include_insertion: bool
    physics_attachment: bool
    socket_scale: float | None = None

    @classmethod
    def for_mode(cls, mode: ExplorationRecordingMode) -> ExplorationRecordingProfile:
        return {
            ExplorationRecordingMode.DOMAIN: cls(
                mode.task_id, None, True, False, False, False
            ),
            ExplorationRecordingMode.GRASP: cls(
                mode.task_id, None, True, True, False, False
            ),
            ExplorationRecordingMode.INSERTION: cls(
                mode.task_id,
                KINEMATIC_INSERTION_MODE,
                True,
                True,
                True,
                False,
            ),
            ExplorationRecordingMode.CONTACT_INSERTION: cls(
                mode.task_id,
                CONTACT_AWARE_INSERTION_MODE,
                False,
                True,
                True,
                True,
                CONTACT_INSERTION_RECORDING.socket_scale,
            ),
        }[mode]

    def apply_to_plan(self, plan: ExplorationPlan) -> ExplorationPlan:
        if self.socket_scale is None:
            return plan
        return replace(plan, socket_scale=self.socket_scale)

    def prepare_attachment(self, stage: Any) -> Any:
        if self.physics_attachment:
            return prepare_fixed_joint_plug(stage)
        return prepare_plug(stage)

    def bind_attachment(self, preparation: Any, rigid_prim: Any) -> PlugAttachment:
        if self.physics_attachment:
            return preparation.bind_physics(rigid_prim)
        return preparation

    def metadata(
        self,
        plan: ExplorationPlan,
        stage: Any,
        attachment: Any,
    ) -> dict[str, Any]:
        metadata = {**plan.metadata(), "task": self.task_id}
        if not self.include_exploration_targets:
            metadata.update(segments=0, segment_outcomes=[])
        if self.evidence_mode is None:
            return metadata
        socket_position, socket_orientation = world_pose(
            stage.GetPrimAtPath(SOCKET_PATH)
        )
        target = {
            "socket_position": socket_position.tolist(),
            "socket_orientation_wxyz": socket_orientation.tolist(),
            "insertion_axis": list(INSERTION_AXIS),
            "grasp_offset_meters": REARWARD_GRASP_OFFSET_METERS,
            "evidence_mode": self.evidence_mode,
        }
        if self.physics_attachment:
            target.update(
                CONTACT_INSERTION_RECORDING.instrumentation_metadata(
                    attachment.compliant_collision_parts
                )
            )
        metadata["insertion_target"] = target
        return metadata


SafetyObserver = Callable[[], "ContactReading"]
CONTACT_INSERTION_LIMITS = InsertionTaskLimits()


def _contact_frames(
    contract: ContactInsertionRecordingContract | None,
    segment: ContactInsertionSegment,
    default: int,
    label: RecordingLabel,
    stage: ObservationStage,
) -> int:
    if contract is None:
        return default
    span = contract.span(segment)
    if span.phase != label.value or span.stage != stage.value:
        raise RuntimeError(f"contact insertion runtime span drifted: {segment.value}")
    return span.frames


def apply_variant(stage: Any, plan: ExplorationPlan) -> None:
    """Author seeded camera, task-geometry, and lighting changes in-session."""

    from pxr import Gf, UsdGeom

    configure_wrist_camera(plan.camera_offset_m)
    scene_offset = np.asarray(plan.scene_offset_m, dtype=np.float64)
    for path in (PLUG_PATH, SOCKET_PATH):
        prim = stage.GetPrimAtPath(path)
        translation = prim.GetAttribute("xformOp:translate")
        if not translation.IsValid():
            raise RuntimeError(f"exploration prim has no translation: {path}")
        translation.Set(Gf.Vec3d(*(np.asarray(translation.Get()) + scene_offset)))

    socket = stage.GetPrimAtPath(SOCKET_PATH)
    xformable = UsdGeom.Xformable(socket)
    scale_op = next(
        (
            operation
            for operation in xformable.GetOrderedXformOps()
            if operation.GetOpType() == UsdGeom.XformOp.TypeScale
        ),
        None,
    )
    if scale_op is None:
        scale_op = xformable.AddScaleOp(
            UsdGeom.XformOp.PrecisionDouble,
            "domainVariant",
        )
        scale_op.Set(Gf.Vec3d(plan.socket_scale))
    else:
        current_scale = scale_op.Get()
        scaled = np.asarray(current_scale, dtype=np.float64) * plan.socket_scale
        scale_op.Set(current_scale.__class__(*scaled))

    for prim in stage.Traverse():
        exposure = prim.GetAttribute("inputs:exposure")
        current = exposure.Get() if exposure.IsValid() else None
        if isinstance(current, (int, float)):
            exposure.Set(float(current) + plan.light_exposure_delta)


def _recording_label(outcome: SegmentOutcome) -> RecordingLabel:
    moment = (
        RecordingMoment.SETTLE
        if outcome == SegmentOutcome.STATIONARY
        else RecordingMoment.MOTION
    )
    return RecordingLabel(moment, Phase.READY)


async def _record_successful_grasp(
    actuators: Actuators,
    current: JointCommand,
    attachment: PlugAttachment,
    recorder: DemoRecorder,
    sample_period_seconds: float,
    observe_safety: SafetyObserver | None = None,
    contact_contract: ContactInsertionRecordingContract | None = None,
) -> tuple[JointCommand, tuple[float, ...]]:
    """Record approach, acquisition, retained retreat, and hold."""

    solved = solve_waypoints()
    pre_grasp = JointCommand(
        solved[1].arm_positions,
        solved[1].waypoint.gripper_width_m,
    )
    grasp_open = JointCommand(
        solved[2].arm_positions,
        solved[1].waypoint.gripper_width_m,
    )
    grasp_closed = JointCommand(
        solved[2].arm_positions,
        solved[2].waypoint.gripper_width_m,
    )
    retained = JointCommand(
        solved[1].arm_positions,
        solved[2].waypoint.gripper_width_m,
    )
    sample_times = []
    approach_label = RecordingLabel(RecordingMoment.MOTION, Phase.PRE_GRASP)
    grasp_label = RecordingLabel(RecordingMoment.MOTION, Phase.GRASP)
    close_label = RecordingLabel(RecordingMoment.CLOSE, Phase.GRASP)
    for command, frames, label in (
        (
            pre_grasp,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.PRE_GRASP,
                8,
                approach_label,
                ObservationStage.APPROACHING_CABLE,
            ),
            approach_label,
        ),
        (
            grasp_open,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.GRASP_OPEN,
                8,
                grasp_label,
                ObservationStage.APPROACHING_CABLE,
            ),
            grasp_label,
        ),
        (
            grasp_closed,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.GRASP_CLOSE,
                4,
                close_label,
                ObservationStage.APPROACHING_CABLE,
            ),
            close_label,
        ),
    ):
        sample_times.extend(
            await move_joint_command(
                actuators,
                current,
                command,
                attachment,
                frame_count=frames,
                phase=label,
                stage=ObservationStage.APPROACHING_CABLE,
                recorder=recorder,
                sample_period_seconds=sample_period_seconds,
                observe_safety=observe_safety,
            )
        )
        current = command
    attachment.attach(world_pose(attachment.hand_prim)[0])
    attach_label = RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP)
    retreat_label = RecordingLabel(RecordingMoment.MOTION, Phase.PRE_INSERTION)
    retreat_hold_label = RecordingLabel(
        RecordingMoment.SETTLE, Phase.PRE_INSERTION
    )
    for command, frames, label in (
        (
            current,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.GRASP_ATTACH,
                1,
                attach_label,
                ObservationStage.CABLE_GRASPED,
            ),
            attach_label,
        ),
        (
            retained,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.RETREAT,
                8,
                retreat_label,
                ObservationStage.CABLE_GRASPED,
            ),
            retreat_label,
        ),
        (
            retained,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.RETREAT_HOLD,
                4,
                retreat_hold_label,
                ObservationStage.CABLE_GRASPED,
            ),
            retreat_hold_label,
        ),
    ):
        sample_times.extend(
            await move_joint_command(
                actuators,
                current,
                command,
                attachment,
                frame_count=frames,
                phase=label,
                stage=ObservationStage.CABLE_GRASPED,
                recorder=recorder,
                sample_period_seconds=sample_period_seconds,
                observe_safety=observe_safety,
            )
        )
        current = command
    return current, tuple(sample_times)


async def _record_successful_insertion(
    actuators: Actuators,
    current: JointCommand,
    attachment: PlugAttachment,
    recorder: DemoRecorder,
    sample_period_seconds: float,
    observe_safety: SafetyObserver | None = None,
    enable_connector_collisions: bool = False,
    contact_contract: ContactInsertionRecordingContract | None = None,
) -> tuple[JointCommand, tuple[float, ...]]:
    """Record alignment, insertion, and an attached seated hold."""

    solved = solve_waypoints()
    pre_insertion = JointCommand(
        solved[3].arm_positions,
        solved[3].waypoint.gripper_width_m,
    )
    inserted = JointCommand(
        solved[4].arm_positions,
        solved[4].waypoint.gripper_width_m,
    )
    sample_times = []
    if enable_connector_collisions:
        attachment.set_collisions(True)
    align_label = RecordingLabel(RecordingMoment.MOTION, Phase.PRE_INSERTION)
    align_hold_label = RecordingLabel(
        RecordingMoment.SETTLE, Phase.PRE_INSERTION
    )
    insert_label = RecordingLabel(RecordingMoment.MOTION, Phase.INSERT)
    seated_label = RecordingLabel(RecordingMoment.SETTLE, Phase.INSERT)
    for command, frames, label, stage in (
        (
            pre_insertion,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.ALIGN,
                8,
                align_label,
                ObservationStage.CABLE_GRASPED,
            ),
            align_label,
            ObservationStage.CABLE_GRASPED,
        ),
        (
            pre_insertion,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.ALIGN_HOLD,
                2,
                align_hold_label,
                ObservationStage.ALIGNED_WITH_SOCKET,
            ),
            align_hold_label,
            ObservationStage.ALIGNED_WITH_SOCKET,
        ),
        (
            inserted,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.INSERT,
                8,
                insert_label,
                ObservationStage.ALIGNED_WITH_SOCKET,
            ),
            insert_label,
            ObservationStage.ALIGNED_WITH_SOCKET,
        ),
        (
            inserted,
            _contact_frames(
                contact_contract,
                ContactInsertionSegment.SEATED_HOLD,
                4,
                seated_label,
                ObservationStage.PLUG_SEATED,
            ),
            seated_label,
            ObservationStage.PLUG_SEATED,
        ),
    ):
        sample_times.extend(
            await move_joint_command(
                actuators,
                current,
                command,
                attachment,
                frame_count=frames,
                phase=label,
                stage=stage,
                recorder=recorder,
                sample_period_seconds=sample_period_seconds,
                observe_safety=observe_safety,
            )
        )
        current = command
    return current, tuple(sample_times)


async def record_exploration_trajectory(
    recording_id: str,
    seed: int,
    split: DatasetSplit,
    *,
    mode: ExplorationRecordingMode = ExplorationRecordingMode.DOMAIN,
) -> dict[str, Any]:
    """Capture a seeded, true-4-FPS wrist rollout for domain adaptation."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation, RigidPrim
    from isaacsim.core.rendering_manager import RenderingManager
    from isaacsim.core.simulation_manager import SimulationManager
    from sim.isaac_demo_runtime import reset_stage

    profile = ExplorationRecordingProfile.for_mode(mode)
    plan = profile.apply_to_plan(build_exploration_plan(seed, split))
    await reset_stage()
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    apply_variant(stage, plan)
    attachment_preparation = profile.prepare_attachment(stage)
    metadata = profile.metadata(plan, stage, attachment_preparation)
    recorder = DemoRecorder(
        recording_id,
        fps=DROID_FPS,
        minimum_stage_frames=0,
        camera_specs=JEPA_WM_CAMERA_SPECS,
        metadata=metadata,
    )
    timeline = omni.timeline.get_timeline_interface()
    original_rendering_dt = RenderingManager.get_dt()
    completed = False
    sample_times = []
    try:
        await recorder.initialize()
        # Isaac keeps the physics scene at its existing high-frequency dt and
        # advances the timeline by this render interval, yielding one rendered
        # observation for each DROID 4 FPS sample instead of rendering every
        # intermediate physics tick. Configure it before creating the physics
        # tensor view because timeline timing changes invalidate that view.
        RenderingManager.set_dt(plan.sample_period_seconds)
        safety_observer = None
        if profile.physics_attachment:
            hand_sensor = contact_sensor(stage, create=True)
            plug_sensor = connector_contact_sensor(stage, create=True)

            def safety_observer() -> ContactReading:
                hand_collision, hand_force = read_contact(hand_sensor)
                _, connector_force = read_contact(plug_sensor)
                force = max(hand_force, connector_force)
                if hand_collision or force > (
                    CONTACT_INSERTION_LIMITS.maximum_contact_force_newtons
                ):
                    raise RuntimeError(
                        "contact-aware insertion exceeded its live safety limit: "
                        f"collision={hand_collision}, force={force:.3f} N"
                    )
                return ContactReading(hand_collision, force)

        await omni.kit.app.get_app().next_update_async()
        if SimulationManager.get_physics_sim_view() is None:
            SimulationManager.initialize_physics()
        attachment = profile.bind_attachment(
            attachment_preparation,
            RigidPrim(PLUG_PATH),
        )
        actuators = create_actuators(stage, Articulation(ROBOT_PATH))
        ready = solve_waypoints()[0]
        origin = JointCommand(
            ready.arm_positions + np.asarray(plan.initial_arm_offset_radians),
            ready.waypoint.gripper_width_m,
        )
        current = origin
        observation_stage = ObservationStage.APPROACHING_CABLE
        timeline.play()
        await advance_physics_updates(1, safety_observer)
        actuators.set_reset_state(origin)
        initial_safety = await advance_physics_updates(
            16,
            safety_observer,
        )
        initial_actual = actuators.actual_command()
        initial = recording_snapshot(
            RecordingLabel(RecordingMoment.INITIAL),
            observation_stage,
            current,
            attachment,
            safety=RecordingSafetyTelemetry(
                collision_detected=initial_safety.collision_detected,
                contact_force_newtons=initial_safety.force_newtons,
                arm_tracking_error_rad=float(
                    np.max(
                        np.abs(initial_actual.arm_positions - current.arm_positions)
                    )
                ),
                gripper_tracking_error_m=abs(
                    initial_actual.gripper_width_m - current.gripper_width_m
                ),
            ),
        )
        await recorder.capture(initial, advance=False)
        if initial.simulation_time_seconds is not None:
            sample_times.append(initial.simulation_time_seconds)
        if profile.physics_attachment:
            _contact_frames(
                CONTACT_INSERTION_RECORDING,
                ContactInsertionSegment.INITIAL,
                1,
                RecordingLabel(RecordingMoment.INITIAL),
                observation_stage,
            )

        targets = plan.targets if profile.include_exploration_targets else ()
        for target in targets:
            command = JointCommand(
                origin.arm_positions + np.asarray(target.arm_offset_radians),
                target.gripper_width_m,
            )
            sample_times.extend(
                await move_joint_command(
                    actuators,
                    current,
                    command,
                    attachment,
                    frame_count=target.frames,
                    phase=_recording_label(target.outcome),
                    stage=observation_stage,
                    recorder=recorder,
                    sample_period_seconds=plan.sample_period_seconds,
                    observe_safety=safety_observer,
                )
            )
            current = command
        if profile.include_grasp:
            current, grasp_times = await _record_successful_grasp(
                actuators,
                current,
                attachment,
                recorder,
                plan.sample_period_seconds,
                safety_observer,
                CONTACT_INSERTION_RECORDING if profile.physics_attachment else None,
            )
            sample_times.extend(grasp_times)
        if profile.include_insertion:
            current, insertion_times = await _record_successful_insertion(
                actuators,
                current,
                attachment,
                recorder,
                plan.sample_period_seconds,
                safety_observer,
                profile.physics_attachment,
                CONTACT_INSERTION_RECORDING if profile.physics_attachment else None,
            )
            sample_times.extend(insertion_times)
        validate_sample_times(tuple(sample_times), plan.sample_period_seconds)
        completed = True
    except Exception:
        recorder.abort()
        raise
    finally:
        RenderingManager.set_dt(original_rendering_dt)
        if completed:
            timeline.pause()
        else:
            timeline.stop()

    output_dir = recorder.finish()
    return {
        "status": "complete",
        "recording_id": recording_id,
        "output_directory": str(output_dir),
        "frames": recorder.frame_count,
        "metadata": metadata,
    }


async def record_grasp_trajectory(
    recording_id: str,
    seed: int,
    split: DatasetSplit,
) -> dict[str, Any]:
    """Capture exploration plus a successful rigid-connector grasp and retention."""

    return await record_exploration_trajectory(
        recording_id,
        seed,
        split,
        mode=ExplorationRecordingMode.GRASP,
    )


async def record_insertion_trajectory(
    recording_id: str,
    seed: int,
    split: DatasetSplit,
) -> dict[str, Any]:
    """Capture exploration plus rearward grasp, alignment, and insertion."""

    return await record_exploration_trajectory(
        recording_id,
        seed,
        split,
        mode=ExplorationRecordingMode.INSERTION,
    )


async def record_contact_insertion_trajectory(
    recording_id: str,
    seed: int,
    split: DatasetSplit,
) -> dict[str, Any]:
    """Capture collision-enabled insertion with measured safety telemetry."""

    return await record_exploration_trajectory(
        recording_id,
        seed,
        split,
        mode=ExplorationRecordingMode.CONTACT_INSERTION,
    )
