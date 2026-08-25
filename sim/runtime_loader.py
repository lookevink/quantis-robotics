"""Reload the shared Isaac runtime in strict dependency order."""

from __future__ import annotations

import importlib.util
import importlib
from pathlib import Path
import sys
from types import ModuleType


def _reload_project_module_from_source(module_name: str) -> ModuleType:
    """Refresh one project module from source, bypassing stale bind-mount bytecode."""

    importlib.invalidate_caches()
    module = sys.modules.get(module_name)
    spec = (
        module.__spec__
        if module is not None
        else importlib.util.find_spec(module_name)
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


def reload_demo_runtime() -> ModuleType:
    """Refresh a persistent Python server without mixing class generations."""

    # The Isaac Python server retains sys.modules across repository syncs.
    # Every owner must be refreshed before modules that import its classes.
    for module_name in (
        "jepa.contract",
        "jepa_wm.identifiers",
        "jepa_wm.action",
        "jepa_wm.training_artifact",
        "jepa_wm.control_policy",
        "jepa_wm.planner",
        "jepa_wm.action_prior",
        "jepa_wm.planner_readiness",
        "jepa_wm.control_protocol",
        "jepa_wm.control_safety",
        "jepa_wm.control_tracking",
        "jepa_wm.grasp_task",
        "jepa_wm.replay_verification",
        "jepa_wm.grasp_contract",
        "jepa_wm.trajectory",
        "jepa_wm.insertion_contract",
        "jepa_wm.insertion_task",
        "jepa_wm.direct_safety",
        "jepa_wm.objective_calibration",
        "jepa_wm.shadow_planning",
        "jepa_wm.shadow_safety",
        "jepa_wm.trial_equivalence",
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
        "sim.isaac_demo_runtime",
        "sim.isaac_control_runtime",
        "sim.grasp_task",
        "sim.isaac_exploration",
        "sim.control_session",
        "sim.trial_source_cache",
        "jepa_wm.control_rollout",
        "jepa_wm.control_baselines",
        "jepa_wm.whole_seed_readiness",
        "jepa_wm.grasp_control_readiness",
        "jepa_wm.calibration_sessions",
        "jepa_wm.candidate_trial",
        "sim.isaac_replay",
        "sim.isaac_control_capture",
        "sim.isaac_control_execution",
        "sim.isaac_control_resolution",
        "sim.isaac_control_followup",
        "sim.isaac_shadow_safety",
        "sim.isaac_insertion_safety",
        "sim.isaac_insertion_trial",
        "sim.isaac_candidate_binding",
        "sim.isaac_baseline_response",
        "sim.isaac_candidate_demo",
        "sim.isaac_grasp_demo",
        "sim.isaac_control_bridge",
    ):
        _reload_project_module_from_source(module_name)
    return _reload_project_module_from_source("sim.isaac_demo")
