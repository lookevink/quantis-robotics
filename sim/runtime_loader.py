"""Reload the shared Isaac runtime in strict dependency order."""

from __future__ import annotations

import importlib
from types import ModuleType


def _reload(module_name: str) -> ModuleType:
    """Import or refresh one module after its dependencies are current."""

    return importlib.reload(importlib.import_module(module_name))


def reload_demo_runtime() -> ModuleType:
    """Refresh a persistent Python server without mixing class generations."""

    # The Isaac Python server retains sys.modules across repository syncs.
    # Every owner must be refreshed before modules that import its classes.
    for module_name in (
        "jepa.contract",
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
        "jepa_wm.insertion_contract",
        "jepa_wm.insertion_task",
        "jepa_wm.objective_calibration",
        "jepa_wm.shadow_planning",
        "jepa_wm.shadow_safety",
        "jepa_wm.trial_equivalence",
        "sim.demo_sequence",
        "sim.recording",
        "jepa_wm.candidate_demo",
        "jepa_wm.grasp_demo",
        "jepa_wm.experimental_candidate",
        "sim.recording_jobs",
        "sim.exploration",
        "jepa_wm.domain_recording",
        "jepa_wm.grasp_recording",
        "jepa_wm.insertion_recording",
        "sim.control_context",
        "jepa_wm.trajectory",
        "jepa_wm.control_replay",
        "sim.isaac_demo_scene",
        "sim.isaac_demo_camera",
        "sim.isaac_demo_kinematics",
        "sim.isaac_demo_runtime",
        "sim.grasp_task",
        "sim.isaac_exploration",
        "sim.control_session",
        "jepa_wm.control_rollout",
        "jepa_wm.control_baselines",
        "jepa_wm.whole_seed_readiness",
        "jepa_wm.grasp_control_readiness",
        "jepa_wm.calibration_sessions",
        "jepa_wm.candidate_trial",
        "sim.isaac_control_runtime",
        "sim.isaac_replay",
        "sim.isaac_control_capture",
        "sim.isaac_control_execution",
        "sim.isaac_control_followup",
        "sim.isaac_shadow_safety",
        "sim.isaac_candidate_binding",
        "sim.isaac_baseline_response",
        "sim.isaac_candidate_demo",
        "sim.isaac_grasp_demo",
        "sim.isaac_control_bridge",
    ):
        _reload(module_name)
    return _reload("sim.isaac_demo")
