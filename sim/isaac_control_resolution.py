"""Reset-repeatable live measurement of sub-millimeter insertion control."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any

from jepa.contract import ObservationStage
from jepa_wm.action import DroidAction, DroidActionScale
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ProposedControl
from jepa_wm.control_resolution import (
    CONTROL_RESOLUTION_PROTOCOL,
    ControlResolutionProtocol,
    ControlResolutionReport,
    ControlResolutionSample,
    ControlResolutionEndpoint,
    retreat_direction,
)
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.control_safety import SafetyProjectionAttempt
from jepa_wm.direct_safety import ControlSafetySnapshot
from jepa_wm.persistence import write_json_atomic
from jepa_wm.trial_equivalence import (
    ResetEquivalenceTolerances,
    TrialResetState,
    validate_reset_equivalence,
)
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from sim.control_context import recording_task
from sim.control_session import CONTROL_ROOT, RECORDING_ROOT, ControlSession
from sim.demo_sequence import Phase
from sim.isaac_control_runtime import (
    LiveContactInterlock,
    LiveControlRuntime,
    live_runtime_for,
    read_control_contact,
    synchronized_insertion_safety_snapshot,
)
from sim.isaac_control_execution import ExecutionSafetyContext, project_control_candidate
from sim.isaac_demo_runtime import (
    JointCommand,
    advance_physics_updates,
    move_joint_command,
    recording_snapshot,
)
from sim.recording import RecordingLabel, RecordingMoment


def _reset_state(
    pose: Any,
    command: JointCommand,
    runtime: LiveControlRuntime,
    interlock: LiveContactInterlock,
) -> TrialResetState:
    plug_position, _ = runtime.attachment.world_pose()
    evidence = interlock.evidence
    return TrialResetState(
        pose=pose,
        joint_positions=tuple(float(value) for value in command.arm_positions),
        collision_detected=evidence.collision_detected,
        contact_force_newtons=evidence.maximum_contact_force_newtons,
        plug_position=tuple(float(value) for value in plug_position),
        plug_attached=runtime.attachment.attached,
    )


def _capture_reset_state(
    runtime: LiveControlRuntime,
    maximum_contact_force_newtons: float,
    operation: str,
    *,
    reference: TrialResetState | None = None,
    tolerances: ResetEquivalenceTolerances | None = None,
) -> tuple[JointCommand, TrialResetState]:
    """Read and optionally verify one live reset through its interlock."""

    interlock = LiveContactInterlock(
        runtime.sensor,
        maximum_contact_force_newtons,
        operation,
    )
    interlock.observe()
    command = runtime.actuators.actual_command()
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.INITIAL),
        ObservationStage.CABLE_GRASPED,
        command,
        runtime.attachment,
    )
    reset = _reset_state(
        snapshot.end_effector_pose,
        command,
        runtime,
        interlock,
    )
    if reference is not None:
        validate_reset_equivalence(
            reference,
            reset,
            **({"tolerances": tolerances} if tolerances is not None else {}),
        )
    return command, reset


@dataclass
class AttachedControlInterlock:
    """Abort one diagnostic interval immediately if the plug detaches."""

    contact: LiveContactInterlock
    runtime: LiveControlRuntime

    def observe(self) -> Any:
        reading = self.contact.observe()
        if not self.runtime.attachment.attached:
            raise RuntimeError("insertion control resolution lost plug attachment")
        return reading


def _capture_endpoint(
    runtime: LiveControlRuntime,
) -> tuple[JointCommand, ControlResolutionEndpoint]:
    collision_detected, contact_force = read_control_contact(runtime.sensor)
    command = runtime.actuators.actual_command()
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.SETTLE, Phase.READY),
        ObservationStage.CABLE_GRASPED,
        command,
        runtime.attachment,
    )
    plug_position, _ = runtime.attachment.world_pose()
    return command, ControlResolutionEndpoint(
        snapshot.end_effector_pose,
        ControlSafetySnapshot(
            joint_positions=tuple(float(value) for value in command.arm_positions),
            gripper_width_m=command.gripper_width_m,
            plug_position=tuple(float(value) for value in plug_position),
            contact_force_newtons=contact_force,
            collision_detected=collision_detected,
            plug_attached=runtime.attachment.attached,
        ),
    )


def resolution_joint_target(
    requested_translation_meters: float,
    start: JointCommand,
    selected: Any | None,
) -> JointCommand:
    """Keep a zero probe at its exact live start; use safe IK otherwise."""

    if requested_translation_meters == 0.0:
        return start
    if selected is None:
        raise RuntimeError("nonzero resolution probe has no safe IK target")
    return JointCommand(
        selected.solved_pose.arm_positions,
        selected.solved_pose.gripper_width_m,
    )


async def measure_insertion_control_resolution(
    session_id: str,
    protocol: ControlResolutionProtocol = CONTROL_RESOLUTION_PROTOCOL,
) -> dict[str, Any]:
    """Run bounded retreat probes, returning to one verified reset after each."""

    import omni.kit.app
    import omni.timeline
    import omni.usd

    session = ControlSession.at(CONTROL_ROOT, session_id)
    observation, state = session.load_capture()
    if (
        state.execution_policy
        is not ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT
        or recording_task(RECORDING_ROOT / state.reference_recording)
        != INSERTION_TASK_ID
        or observation.target_pose is None
        or session.response_path.exists()
        or session.execution_path.exists()
        or session.result_path.exists()
    ):
        raise ValueError("control session is not a fresh resolution measurement")
    report_path = session.path / "control_resolution.json"
    started_path = session.path / "control_resolution_started.json"
    failure_path = session.path / "control_resolution_failure.json"
    if report_path.exists() or failure_path.exists():
        raise ValueError("control resolution measurement already finished")
    try:
        with started_path.open("x", encoding="utf-8") as output:
            output.write('{"diagnostic_only":true,"production_authority_granted":false}\n')
    except FileExistsError as error:
        raise ValueError("control resolution measurement already started") from error

    stage = omni.usd.get_context().get_stage()
    runtime = live_runtime_for(session_id, stage)
    if runtime is None:
        raise RuntimeError("live insertion runtime was lost before measurement")
    timeline = omni.timeline.get_timeline_interface()
    advance = omni.kit.app.get_app().next_update_async
    limits = protocol.safety_limits
    synchronized = await synchronized_insertion_safety_snapshot(
        runtime,
        timeline,
        advance,
        state.require_safety_snapshot(),
        limits,
        operation="insertion control resolution synchronization",
    )
    runtime = synchronized.runtime
    direction = retreat_direction(observation.pose, observation.target_pose)
    samples: list[ControlResolutionSample] = []
    try:
        timeline.play()
        baseline_command, reference_reset = _capture_reset_state(
            runtime,
            limits.maximum_contact_force_newtons,
            "insertion control resolution baseline",
        )
        validate_reset_equivalence(
            ControlSession.trial_context(observation, state).reset,
            reference_reset,
            tolerances=protocol.reset_tolerances,
        )

        for index, magnitude in enumerate(protocol.requested_translations):
            start_command, start_reset = _capture_reset_state(
                runtime,
                limits.maximum_contact_force_newtons,
                f"insertion control resolution sample {index}",
                reference=reference_reset,
                tolerances=protocol.reset_tolerances,
            )
            commanded = DroidAction(
                (
                    direction[0] * magnitude,
                    direction[1] * magnitude,
                    direction[2] * magnitude,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )
            sample_time = time()
            sample_observation = replace(
                observation,
                captured_at_unix_seconds=sample_time,
                pose=start_reset.pose,
            )
            proposal = ProposedControl(
                observation_id=observation.observation_id,
                created_at_unix_seconds=sample_time,
                actions=(commanded, DroidAction((0.0,) * 7), DroidAction((0.0,) * 7)),
                proposal=observation.expected_proposal,
            )
            safety = ExecutionSafetyContext(
                sample_observation,
                start_command,
                start_reset.joint_positions,
                start_reset.contact_force_newtons,
                start_reset.collision_detected,
                limits,
            )
            if magnitude == 0.0:
                gate = safety.evaluate(
                    proposal,
                    start_reset.joint_positions,
                    now_unix_seconds=sample_time,
                )
                projection = SafetyProjectionAttempt(
                    DroidActionScale.uniform(1.0),
                    gate,
                    0.0,
                    start_reset.joint_positions,
                )
                selected = None
            else:
                projection, selected = project_control_candidate(
                    safety,
                    proposal,
                    DroidActionScale.uniform(1.0),
                    now_unix_seconds=sample_time,
                )
                if selected is None:
                    raise RuntimeError(
                        "insertion control resolution probe failed safety projection: "
                        + ",".join(reason.value for reason in projection.gate.reasons)
                    )
            if not projection.gate.passed:
                raise RuntimeError(
                    "insertion control resolution probe failed safety projection: "
                    + ",".join(reason.value for reason in projection.gate.reasons)
                )
            target = resolution_joint_target(magnitude, start_command, selected)

            sample_interlock = AttachedControlInterlock(
                LiveContactInterlock(
                    runtime.sensor,
                    limits.maximum_contact_force_newtons,
                    f"insertion control resolution sample {index}",
                ),
                runtime,
            )
            await move_joint_command(
                runtime.actuators,
                start_command,
                target,
                runtime.attachment,
                frame_count=1,
                phase=RecordingLabel(RecordingMoment.MOTION, Phase.READY),
                stage=ObservationStage.CABLE_GRASPED,
                recorder=None,
                sample_period_seconds=protocol.motion_period_seconds,
                observe_safety=sample_interlock.observe,
            )
            await advance_physics_updates(
                protocol.settling_updates,
                sample_interlock.observe,
            )
            actual, endpoint = _capture_endpoint(runtime)
            sample_interlock.observe()

            rollback_interlock = AttachedControlInterlock(
                LiveContactInterlock(
                    runtime.sensor,
                    limits.maximum_contact_force_newtons,
                    f"insertion control resolution rollback {index}",
                ),
                runtime,
            )
            await move_joint_command(
                runtime.actuators,
                actual,
                baseline_command,
                runtime.attachment,
                frame_count=1,
                phase=RecordingLabel(RecordingMoment.SETTLE, Phase.READY),
                stage=ObservationStage.CABLE_GRASPED,
                recorder=None,
                sample_period_seconds=protocol.motion_period_seconds,
                observe_safety=rollback_interlock.observe,
            )
            await advance_physics_updates(
                protocol.settling_updates,
                rollback_interlock.observe,
            )
            _, rollback_reset = _capture_reset_state(
                runtime,
                limits.maximum_contact_force_newtons,
                f"insertion control resolution reset verification {index}",
                reference=reference_reset,
                tolerances=protocol.reset_tolerances,
            )
            samples.append(
                ControlResolutionSample(
                    index=index,
                    requested_translation_meters=magnitude,
                    start_reset=start_reset,
                    commanded_action=commanded,
                    target_pose=start_reset.pose.applied(commanded),
                    projection=projection,
                    endpoint=endpoint,
                    interlock=sample_interlock.contact.evidence,
                    rollback_reset=rollback_reset,
                )
            )

        report = ControlResolutionReport(
            session_id=session_id,
            reference_recording=state.reference_recording,
            seed=state.seed,
            context_index=observation.warmup_frames,
            observation_id=observation.observation_id,
            captured_pose=observation.pose,
            recorded_target_pose=observation.target_pose,
            reference_reset=reference_reset,
            samples=tuple(samples),
            protocol=protocol,
        )
        write_json_atomic(report_path, report.to_dict())
        return report.to_dict()
    except Exception as error:
        write_json_atomic(
            failure_path,
            {
                "session_id": session_id,
                "failed_at_unix_seconds": time(),
                "completed_samples": [sample.to_dict() for sample in samples],
                "error": f"{type(error).__name__}: {error}",
                "diagnostic_only": True,
                "multi_step_authority_granted": False,
                "production_authority_granted": False,
            },
        )
        raise
    finally:
        timeline.pause()
