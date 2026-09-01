"""Zero-actuation control observation from an authenticated unknown-start reset."""

from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Any

from jepa.contract import ObservationStage
from jepa_wm.action import DroidAction
from jepa_wm.contact_grasp_target import ContactGraspTargetPolicy
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.physical_observation import PhysicalRoutingObservation
from jepa_wm.training_artifact import artifact_fingerprint
from jepa_wm.persistence import write_json_atomic
from sim.isaac_control_capture import validated_control_reference
from sim.control_identity import ControlProposalRef, observation_id_for_session
from sim.control_session import (
    CONTROL_ROOT,
    QUANTIS_DATA_ROOT,
    ControlCaptureResult,
    ControlSession,
    ControlSessionState,
)
from sim.isaac_control_runtime import (
    bind_live_runtime,
    control_contact_sensors,
    pause_control_timeline,
    read_control_contact,
)
from sim.isaac_demo_camera import (
    JEPA_WM_CAMERA_SPECS,
    WRIST_CAMERA_TRANSLATION_METERS,
    capture_camera_frame,
)
from sim.isaac_demo_runtime import (
    ContactReading,
    create_actuators,
    bind_existing_fixed_joint_plug,
    recording_safety_telemetry,
    recording_snapshot,
)
from sim.isaac_demo_scene import (
    PLUG_PATH,
    ROBOT_PATH,
    SOCKET_PATH,
    STAGE_PATH,
    WRIST_CAMERA_PATH,
    world_pose,
)
from sim.recording import RecordingLabel, RecordingMoment, validate_recording_id
from sim.unknown_start_reset import UNKNOWN_START_RESET_CONTRACT, UnknownStartResetEvidence
from sim.unknown_start_shadow import (
    UnknownStartControlHandoff,
    validate_unknown_start_handoff,
)


def _load_authenticated_reset(
    reset_recording_id: str,
    reset_result_fingerprint: str,
) -> tuple[Path, dict[str, Any], UnknownStartResetEvidence, dict[str, Any]]:
    reset_recording = QUANTIS_DATA_ROOT / "recordings" / reset_recording_id
    result_path = reset_recording / "RESULT.json"
    evidence_path = reset_recording / "unknown_start_reset_evidence.json"
    steps_path = reset_recording / "steps.jsonl"
    if artifact_fingerprint(result_path) != reset_result_fingerprint:
        raise ValueError("unknown-start terminal result fingerprint changed")
    result = json.loads(result_path.read_text())
    evidence = UnknownStartResetEvidence.from_dict(json.loads(evidence_path.read_text()))
    artifacts = result.get("artifacts", {})
    if (
        result.get("passed") is not True
        or result.get("recovery_verified") is not True
        or result.get("applied_actions") != 0
        or result.get("recording_id") != reset_recording_id
        or result.get("contract_fingerprint") != UNKNOWN_START_RESET_CONTRACT.fingerprint
        or result.get("sample_fingerprint") != evidence.sample.fingerprint
        or artifacts.get("unknown_start_reset_evidence.json")
        != artifact_fingerprint(evidence_path)
        or artifacts.get("steps.jsonl") != artifact_fingerprint(steps_path)
    ):
        raise ValueError("unknown-start terminal result is inauthentic")
    lines = steps_path.read_text().splitlines()
    if len(lines) != 1:
        raise ValueError("unknown-start reset observation is inauthentic")
    reset_step = json.loads(lines[0])
    return reset_recording, result, evidence, reset_step


def _current_variant_readback(stage: Any) -> dict[str, Any]:
    """Read every model-visible variant against the untouched stage asset."""

    from pxr import Usd, UsdGeom

    camera = stage.GetPrimAtPath(WRIST_CAMERA_PATH).GetAttribute("xformOp:translate")
    if not camera.IsValid():
        raise RuntimeError("unknown-start wrist camera translation is missing")
    camera_offset = tuple(
        float(current) - float(baseline)
        for current, baseline in zip(
            camera.Get(),
            WRIST_CAMERA_TRANSLATION_METERS,
        )
    )
    scale_operations = [
        operation
        for operation in UsdGeom.Xformable(
            stage.GetPrimAtPath(SOCKET_PATH)
        ).GetOrderedXformOps()
        if operation.GetOpType() == UsdGeom.XformOp.TypeScale
    ]
    if len(scale_operations) != 1:
        raise RuntimeError("unknown-start socket scale is ambiguous")
    socket_scale = tuple(float(value) for value in scale_operations[0].Get())
    if max(socket_scale) != min(socket_scale):
        raise RuntimeError("unknown-start socket scale is not uniform")

    base_stage = Usd.Stage.Open(STAGE_PATH)
    if base_stage is None:
        raise RuntimeError("unknown-start base stage cannot be authenticated")
    current_exposures: dict[str, float] = {}
    base_exposures: dict[str, float] = {}
    for current_prim in stage.Traverse():
        exposure = current_prim.GetAttribute("inputs:exposure")
        value = exposure.Get() if exposure.IsValid() else None
        if isinstance(value, (int, float)):
            current_exposures[str(current_prim.GetPath())] = float(value)
    for base_prim in base_stage.Traverse():
        exposure = base_prim.GetAttribute("inputs:exposure")
        value = exposure.Get() if exposure.IsValid() else None
        if isinstance(value, (int, float)):
            base_exposures[str(base_prim.GetPath())] = float(value)
    if not current_exposures or set(current_exposures) != set(base_exposures):
        raise RuntimeError("unknown-start light set changed")
    light_deltas = tuple(
        current_exposures[path] - base_exposures[path]
        for path in sorted(current_exposures)
    )
    base_socket = base_stage.GetPrimAtPath(SOCKET_PATH)
    matrix = UsdGeom.Xformable(base_socket).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    rotation = matrix.ExtractRotationQuat()
    base_socket_orientation = tuple(
        float(value) for value in (rotation.GetReal(), *rotation.GetImaginary())
    )
    return {
        "camera_offset_m": camera_offset,
        "socket_scale": socket_scale[0],
        "light_exposure_deltas": light_deltas,
        "base_socket_orientation_wxyz": base_socket_orientation,
    }


def _validated_live_snapshot(
    stage: Any,
    actuators: Any,
    attachment: Any,
    sensors: Any,
    evidence: UnknownStartResetEvidence,
    reset_step: dict[str, Any],
) -> tuple[Any, Any, Any, tuple[float, ...], tuple[float, ...], dict[str, Any]]:
    actual = actuators.actual_command()
    authored = actuators.current_command()
    collision, force = read_control_contact(sensors)
    contact = ContactReading(collision, force)
    snapshot = recording_snapshot(
        RecordingLabel(RecordingMoment.INITIAL),
        ObservationStage.APPROACHING_CABLE,
        actual,
        attachment,
        safety=recording_safety_telemetry(authored, actual, contact),
    )
    socket_position, socket_orientation = world_pose(stage.GetPrimAtPath(SOCKET_PATH))
    variant = _current_variant_readback(stage)
    validate_unknown_start_handoff(
        evidence,
        arm_positions=tuple(float(value) for value in actual.arm_positions),
        gripper_width_m=float(actual.gripper_width_m),
        connector_position_m=tuple(float(value) for value in snapshot.plug_position),
        socket_position_m=tuple(float(value) for value in socket_position),
        gripper_frame_position_m=tuple(
            float(value) for value in snapshot.gripper_frame_world_position
        ),
        connector_orientation_wxyz=tuple(
            float(value) for value in snapshot.plug_orientation_wxyz
        ),
        expected_connector_orientation_wxyz=tuple(
            float(value) for value in reset_step["plug_orientation_wxyz"]
        ),
        socket_orientation_wxyz=tuple(float(value) for value in socket_orientation),
        expected_socket_orientation_wxyz=variant[
            "base_socket_orientation_wxyz"
        ],
        camera_offset_m=variant["camera_offset_m"],
        socket_scale=variant["socket_scale"],
        light_exposure_deltas=variant["light_exposure_deltas"],
        plug_attached=snapshot.plug_attached,
        collision_detected=collision,
        contact_force_newtons=force,
    )
    return (
        actual,
        authored,
        snapshot,
        tuple(float(value) for value in socket_position),
        tuple(float(value) for value in socket_orientation),
        variant,
    )


def _validate_context_image(path: Path, captured: dict[str, Any]) -> None:
    from PIL import Image

    width, height = JEPA_WM_CAMERA_SPECS[0].resolution
    shape = captured.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[:2] != [height, width]
        or shape[2] < 3
    ):
        raise RuntimeError("unknown-start context raster shape is invalid")
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (width, height):
            raise RuntimeError("unknown-start context PNG contract is invalid")
        image.verify()
    with Image.open(path) as image:
        image.load()


def reauthenticate_unknown_start_shadow_session(session_id: str) -> None:
    """Recheck a captured unknown-start session before later Isaac evaluation."""

    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    session = ControlSession.at(CONTROL_ROOT, session_id)
    handoff_path = session.path / "unknown_start_handoff.json"
    if not handoff_path.is_file():
        return
    if omni.timeline.get_timeline_interface().is_playing():
        raise RuntimeError("unknown-start shadow evaluation requires a paused timeline")
    handoff = UnknownStartControlHandoff.from_dict(
        json.loads(handoff_path.read_text())
    )
    if (
        handoff.session_id != session_id
        or artifact_fingerprint(session.path / "context.png")
        != handoff.context_fingerprint
        or artifact_fingerprint(session.path / "unknown_start_routing_target.json")
        != handoff.routing_target_fingerprint
        or artifact_fingerprint(session.path / "unknown_start_routing_step.json")
        != handoff.routing_step_fingerprint
        or artifact_fingerprint(session.request_path) != handoff.request_fingerprint
        or artifact_fingerprint(session.state_path) != handoff.state_fingerprint
    ):
        raise ValueError("unknown-start shadow session identity changed")
    _, _, evidence, reset_step = _load_authenticated_reset(
        handoff.reset_recording_id,
        handoff.reset_result_fingerprint,
    )
    stage = omni.usd.get_context().get_stage()
    actuators = create_actuators(stage, Articulation(ROBOT_PATH))
    attachment = bind_existing_fixed_joint_plug(stage).bind_physics(
        RigidPrim(PLUG_PATH)
    )
    sensors = control_contact_sensors(stage, create=False, include_connector=True)
    _validated_live_snapshot(
        stage,
        actuators,
        attachment,
        sensors,
        evidence,
        reset_step,
    )


async def preflight_unknown_start_shadow(
    reset_recording_id: str,
    reset_result_fingerprint: str,
) -> dict[str, Any]:
    """Pause Isaac and reauthenticate the reset before claiming an experiment."""

    import omni.kit.app
    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    validate_recording_id(reset_recording_id)
    timeline = omni.timeline.get_timeline_interface()
    await pause_control_timeline(
        timeline,
        omni.kit.app.get_app().next_update_async,
    )
    if timeline.is_playing():
        raise RuntimeError("unknown-start shadow preflight could not pause timeline")
    _, _, evidence, reset_step = _load_authenticated_reset(
        reset_recording_id,
        reset_result_fingerprint,
    )
    stage = omni.usd.get_context().get_stage()
    actuators = create_actuators(stage, Articulation(ROBOT_PATH))
    attachment = bind_existing_fixed_joint_plug(stage).bind_physics(
        RigidPrim(PLUG_PATH)
    )
    sensors = control_contact_sensors(stage, create=False, include_connector=True)
    _validated_live_snapshot(
        stage,
        actuators,
        attachment,
        sensors,
        evidence,
        reset_step,
    )
    return {
        "status": "ready",
        "reset_recording_id": reset_recording_id,
        "reset_result_fingerprint": reset_result_fingerprint,
        "timeline_playing": False,
        "applied_actions": 0,
    }


async def capture_unknown_start_shadow_observation(
    session_id: str,
    reference_recording: str,
    reference_seed: int,
    proposal_name: str,
    reset_recording_id: str,
    reset_result_fingerprint: str,
) -> dict[str, Any]:
    """Create one fresh model request from v5 state without resetting or moving."""

    import omni.timeline
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation, RigidPrim

    for identifier in (session_id, reference_recording, proposal_name, reset_recording_id):
        validate_recording_id(identifier)
    session = ControlSession.at(CONTROL_ROOT, session_id)
    if session.path.exists():
        raise ValueError(f"control session already exists: {session_id}")
    if omni.timeline.get_timeline_interface().is_playing():
        raise RuntimeError("unknown-start shadow capture requires a paused timeline")
    reset_recording, _, evidence, reset_step = _load_authenticated_reset(
        reset_recording_id,
        reset_result_fingerprint,
    )
    evidence_path = reset_recording / "unknown_start_reset_evidence.json"

    reference = validated_control_reference(
        reference_recording,
        reference_seed,
        ControlExecutionPolicy.DIRECT,
    )
    proposal = ControlProposalRef.from_name(proposal_name)
    stage = omni.usd.get_context().get_stage()
    actuators = create_actuators(stage, Articulation(ROBOT_PATH))
    attachment = bind_existing_fixed_joint_plug(stage).bind_physics(
        RigidPrim(PLUG_PATH)
    )
    sensors = control_contact_sensors(stage, create=False, include_connector=True)
    actual, authored, snapshot, socket_position, socket_orientation, _ = (
        _validated_live_snapshot(
            stage,
            actuators,
            attachment,
            sensors,
            evidence,
            reset_step,
        )
    )

    context_path = session.path / "context.png"

    def observe_safety() -> ContactReading:
        current_collision, current_force = read_control_contact(sensors)
        if current_collision or current_force != 0.0:
            raise RuntimeError("unknown-start handoff lost zero-contact state")
        return ContactReading(current_collision, current_force)

    captured_context = await capture_camera_frame(
        JEPA_WM_CAMERA_SPECS[0],
        context_path,
        observe_safety=observe_safety,
    )
    _validate_context_image(context_path, captured_context)
    if omni.timeline.get_timeline_interface().is_playing():
        raise RuntimeError("unknown-start shadow capture resumed the timeline")
    actual, authored, snapshot, socket_position, socket_orientation, _ = (
        _validated_live_snapshot(
            stage,
            actuators,
            attachment,
            sensors,
            evidence,
            reset_step,
        )
    )
    collision, force = read_control_contact(sensors)
    zero_action = DroidAction((0.0,) * 7)
    reference_target = reference.manifest.get("metadata", {}).get("insertion_target")
    if not isinstance(reference_target, dict):
        raise ValueError("unknown-start reference has no insertion target")
    target_metadata = {
        **reference_target,
        "socket_position": list(socket_position),
        "socket_orientation_wxyz": reference_target["socket_orientation_wxyz"],
        "geometry_source": "authenticated_unknown_start_live_state",
        "reset_recording_id": reset_recording_id,
    }
    routing_target_path = session.path / "unknown_start_routing_target.json"
    write_json_atomic(routing_target_path, target_metadata)
    routing_step = {
        field: reset_step[field]
        for field in (
            "plug_position",
            "plug_orientation_wxyz",
            "end_effector_world_position",
            "gripper_frame_world_position",
            "gripper_width_m",
            "arm_tracking_error_rad",
            "gripper_tracking_error_m",
            "contact_force_newtons",
            "plug_attached",
        )
    }
    routing_step_path = session.path / "unknown_start_routing_step.json"
    write_json_atomic(routing_step_path, routing_step)
    reference_scene_offset = reference.manifest.get("metadata", {}).get(
        "scene_offset_m"
    )
    if not isinstance(reference_scene_offset, list) or len(reference_scene_offset) != 3:
        raise ValueError("unknown-start reference scene offset is invalid")
    target_policy = ContactGraspTargetPolicy.for_scene_translation(
        tuple(
            reset_value - float(reference_value)
            for reset_value, reference_value in zip(
                evidence.sample.scene_offset_m,
                reference_scene_offset,
            )
        )
    )
    initial_target = target_policy.initial_target(
        reference.path,
        frame_root=QUANTIS_DATA_ROOT,
    )
    observation = ControlObservation(
        observation_id=observation_id_for_session(session_id),
        captured_at_unix_seconds=time(),
        context_frame=context_path.relative_to(QUANTIS_DATA_ROOT),
        target=initial_target,
        expected_proposal=proposal.path,
        pose=snapshot.end_effector_pose,
        previous_action=zero_action,
        warmup_frames=target_policy.context_index_for_target(
            initial_target.frame
        ),
        physical_routing=PhysicalRoutingObservation.from_recorded_step(
            routing_step,
            target_metadata,
            zero_action,
        ),
    )
    state = ControlSessionState(
        session_id=session_id,
        reference_recording=reference_recording,
        seed=reference_seed,
        recording=reset_recording_id,
        current_joint_positions=tuple(float(value) for value in actual.arm_positions),
        collision_detected=False,
        contact_force_newtons=0.0,
        execution_policy=ControlExecutionPolicy.DIRECT,
        plug_position=tuple(float(value) for value in snapshot.plug_position),
        plug_attached=False,
        current_gripper_width_m=float(actual.gripper_width_m),
        active_drive_target=JointDriveTarget(
            tuple(float(value) for value in authored.arm_positions),
            authored.gripper_width_m,
        ),
        contact_grasp_target_policy=target_policy,
    )
    session.write_capture(observation, state)
    binding = UnknownStartControlHandoff(
        session_id=session_id,
        reset_recording_id=reset_recording_id,
        reset_seed=evidence.sample.seed,
        reset_result_fingerprint=reset_result_fingerprint,
        reset_evidence_fingerprint=artifact_fingerprint(evidence_path),
        reset_contract_fingerprint=UNKNOWN_START_RESET_CONTRACT.fingerprint,
        reference_recording=reference_recording,
        reference_seed=reference_seed,
        context_fingerprint=artifact_fingerprint(context_path),
        routing_target_fingerprint=artifact_fingerprint(routing_target_path),
        routing_step_fingerprint=artifact_fingerprint(routing_step_path),
        request_fingerprint=artifact_fingerprint(session.request_path),
        state_fingerprint=artifact_fingerprint(session.state_path),
    ).to_dict()

    write_json_atomic(session.path / "unknown_start_handoff.json", binding)
    bind_live_runtime(session_id, stage, actuators, attachment, sensors)
    return {
        **ControlCaptureResult(
            session_id,
            observation,
            session.request_path,
            0.0,
            False,
        ).to_dict(),
        "unknown_start_handoff": binding,
    }
