"""Reload the shared Isaac runtime in strict dependency order."""

from __future__ import annotations

import importlib.util
import importlib
from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class _ActuatorsHandoff:
    articulation: Any
    arm_attributes: tuple[Any, ...]
    finger_attributes: tuple[Any, ...]


@dataclass(frozen=True)
class KinematicPlugMotionHandoff:
    prim: Any
    hand_prim: Any
    hand_to_plug_offset: Any | None
    kind: str = field(default="kinematic", init=False)


@dataclass(frozen=True)
class FixedJointPlugMotionHandoff:
    prim: Any
    hand_prim: Any
    rigid_prim: Any
    fixed_joint: Any
    hand_to_plug_offset: Any | None
    kind: str = field(default="fixed_joint", init=False)


@dataclass(frozen=True)
class _PlugAttachmentHandoff:
    motion: KinematicPlugMotionHandoff | FixedJointPlugMotionHandoff
    collision_attributes: tuple[Any, ...]
    excluded_collision_paths: frozenset[str]


@dataclass(frozen=True)
class _ContactSensorsHandoff:
    hand: Any
    connector: Any | None


@dataclass(frozen=True)
class _LiveRuntimeHandoff:
    session_id: str
    stage: Any
    actuators: _ActuatorsHandoff
    attachment: _PlugAttachmentHandoff
    sensor: _ContactSensorsHandoff


def _contact_sensor_handoff(sensor: Any) -> _ContactSensorsHandoff:
    if hasattr(sensor, "hand") or hasattr(sensor, "connector"):
        try:
            return _ContactSensorsHandoff(sensor.hand, sensor.connector)
        except AttributeError as error:
            raise RuntimeError("live control sensor handoff is malformed") from error
    return _ContactSensorsHandoff(sensor, None)


def _plug_motion_handoff(
    motion: Any,
) -> KinematicPlugMotionHandoff | FixedJointPlugMotionHandoff:
    offset = motion.hand_to_plug_offset
    copied_offset = None if offset is None else offset.copy()
    if hasattr(motion, "rigid_prim") or hasattr(motion, "fixed_joint"):
        return FixedJointPlugMotionHandoff(
            motion.prim,
            motion.hand_prim,
            motion.rigid_prim,
            motion.fixed_joint,
            copied_offset,
        )
    return KinematicPlugMotionHandoff(
        motion.prim,
        motion.hand_prim,
        copied_offset,
    )


def _resident_live_runtime_handoff(
    module: ModuleType | None,
) -> _LiveRuntimeHandoff | None:
    """Read both current and pre-handoff resident runtime generations."""

    if module is None:
        return None
    runtime = getattr(module, "_live_runtime", None)
    if runtime is None:
        return None
    try:
        actuators = runtime.actuators
        motion = runtime.attachment.motion
        collisions = runtime.attachment.collisions
        return _LiveRuntimeHandoff(
            runtime.session_id,
            runtime.stage,
            _ActuatorsHandoff(
                actuators.articulation,
                tuple(actuators.arm_attributes),
                tuple(actuators.finger_attributes),
            ),
            _PlugAttachmentHandoff(
                _plug_motion_handoff(motion),
                tuple(collisions.collision_attributes),
                frozenset(collisions.excluded_collision_paths),
            ),
            _contact_sensor_handoff(runtime.sensor),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("live control runtime handoff is malformed") from error


def _reload_project_module_from_source(module_name: str) -> ModuleType:
    """Refresh one project module from source, bypassing stale bind-mount bytecode."""

    importlib.invalidate_caches()
    module = sys.modules.get(module_name)
    spec = (
        module.__spec__ if module is not None else importlib.util.find_spec(module_name)
    )
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        raise RuntimeError(f"project module has no Python source: {module_name}")
    created = module is None
    if module is None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
    source_path = Path(spec.origin)
    code = compile(source_path.read_text(encoding="utf-8"), str(source_path), "exec")
    namespace = module.__dict__
    preserved = {
        "__name__": module_name,
        "__package__": spec.parent,
        "__loader__": spec.loader,
        "__spec__": spec,
        "__file__": spec.origin,
        "__cached__": spec.cached,
        "__builtins__": namespace.get("__builtins__", __builtins__),
    }
    namespace.clear()
    namespace.update(preserved)
    try:
        exec(code, namespace)
    except Exception:
        if created:
            sys.modules.pop(module_name, None)
        raise
    return module


def _resident_simulator_operation_id(manager: Any) -> str | None:
    """Read both current and pre-interlock resident job managers safely."""

    current = getattr(manager, "active_operation_id", None)
    if callable(current):
        try:
            return current()
        except Exception:
            return "unknown-resident-simulator-operation"
    tasks = getattr(manager, "_tasks", None)
    if not isinstance(tasks, dict):
        return "unknown-resident-simulator-operation"
    for identity, task in tasks.items():
        try:
            if not task.done():
                return str(identity)
        except Exception:
            return "unknown-resident-simulator-operation"
    return None


def reload_demo_runtime() -> ModuleType:
    """Refresh a persistent Python server without mixing class generations."""

    resident_demo = sys.modules.get("sim.isaac_demo")
    resident_jobs = getattr(resident_demo, "_RECORDING_JOBS", None)
    if (
        resident_demo is not None
        and resident_jobs is not None
        and _resident_simulator_operation_id(resident_jobs) is not None
    ):
        # Reloading the facade would orphan its asyncio task/interlock. Keep the
        # resident generation until the one simulator operation terminalizes.
        return resident_demo
    control_runtime = sys.modules.get("sim.isaac_control_runtime")
    runtime_handoff = _resident_live_runtime_handoff(control_runtime)
    # The Isaac Python server retains sys.modules across repository syncs.
    # Every owner must be refreshed before modules that import its classes.
    for module_name in (
        "jepa.contract",
        "jepa_wm.identifiers",
        "jepa_wm.persistence",
        "jepa_wm.insertion_layout",
        "jepa_wm.action",
        "jepa_wm.training_artifact",
        "jepa_wm.contact_grasp_acquisition_handoff",
        "jepa_wm.contact_grasp_acquisition_continuation",
        "jepa_wm.contact_grasp_acquisition_hold",
        "jepa_wm.contact_grasp_acquisition_resolution",
        "jepa_wm.contact_grasp_rotation_resolution",
        "jepa_wm.contact_grasp_horizon_completion",
        "jepa_wm.control_policy",
        "jepa_wm.planner",
        "jepa_wm.action_prior",
        "jepa_wm.planner_readiness",
        "jepa_wm.control_protocol",
        "jepa_wm.control_tracking",
        "jepa_wm.control_safety",
        "jepa_wm.joint_drive",
        "jepa_wm.insertion_refresh",
        "jepa_wm.insertion_rollout",
        "jepa_wm.joint_settlement",
        "jepa_wm.target_progress",
        "jepa_wm.grasp_task",
        "jepa_wm.replay_verification",
        "jepa_wm.grasp_contract",
        "jepa_wm.trajectory",
        "jepa_wm.insertion_transition",
        "jepa_wm.insertion_contract",
        "jepa_wm.task_windows",
        "jepa_wm.contact_grasp_target",
        "jepa_wm.insertion_task",
        "jepa_wm.direct_safety",
        "jepa_wm.objective_calibration",
        "jepa_wm.shadow_planning",
        "jepa_wm.shadow_safety",
        "jepa_wm.trial_equivalence",
        "jepa_wm.control_resolution_profile",
        "jepa_wm.control_resolution_baseline",
        "jepa_wm.control_resolution_drive",
        "jepa_wm.unknown_start_reset_runtime",
        "sim.demo_sequence",
        "sim.recording",
        "sim.control_identity",
        "jepa_wm.control_resolution",
        "jepa_wm.candidate_demo",
        "jepa_wm.grasp_demo",
        "jepa_wm.experimental_candidate",
        "jepa_wm.insertion_trial",
        "sim.recording_jobs",
        "sim.exploration",
        "jepa_wm.domain_recording",
        "jepa_wm.grasp_recording",
        "jepa_wm.insertion_recording",
        "sim.control_context",
        "jepa_wm.control_replay",
        "sim.isaac_demo_scene",
        "sim.isaac_demo_camera",
        "sim.isaac_demo_kinematics",
        "sim.grasp_task",
        "sim.isaac_demo_runtime",
        "sim.isaac_control_runtime",
        "sim.isaac_exploration",
        "sim.unknown_start_reset",
        "sim.isaac_unknown_start_reset",
        "sim.unknown_start_shadow",
        "sim.isaac_unknown_start_shadow",
        "sim.control_session",
        "sim.trial_source_cache",
        "jepa_wm.control_rollout",
        "jepa_wm.grasp_to_insertion",
        "jepa_wm.control_baselines",
        "jepa_wm.whole_seed_readiness",
        "jepa_wm.grasp_control_readiness",
        "jepa_wm.calibration_sessions",
        "jepa_wm.candidate_trial",
        "sim.isaac_replay",
        "sim.control_capture_schedule",
        "sim.isaac_control_capture",
        "sim.isaac_control_execution",
        "sim.isaac_unknown_start_recovery",
        "sim.isaac_control_resolution",
        "sim.isaac_control_followup",
        "sim.isaac_shadow_safety",
        "sim.isaac_insertion_safety",
        "sim.isaac_insertion_trial",
        "sim.isaac_candidate_binding",
        "sim.isaac_baseline_response",
        "sim.isaac_candidate_demo",
        "sim.isaac_grasp_demo",
        "sim.isaac_insertion_demo",
        "sim.isaac_control_bridge",
    ):
        reloaded = _reload_project_module_from_source(module_name)
        if module_name == "sim.isaac_control_runtime" and runtime_handoff is not None:
            reloaded.restore_live_runtime_handoff(runtime_handoff)
    return _reload_project_module_from_source("sim.isaac_demo")
