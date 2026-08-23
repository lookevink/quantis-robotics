"""Reload the shared Isaac runtime in dependency order for Python-server calls."""

from __future__ import annotations

import importlib
from types import ModuleType


def reload_demo_runtime() -> ModuleType:
    import jepa.contract as contract
    import jepa_wm.action as wm_action
    import jepa_wm.control_protocol as control_protocol
    import jepa_wm.control_safety as control_safety
    import jepa_wm.control_tracking as control_tracking
    import jepa_wm.domain_recording as domain_recording
    import jepa_wm.trajectory as trajectory
    import jepa_wm.control_replay as control_replay
    import sim.recording as recording
    import sim.recording_jobs as recording_jobs
    import sim.exploration as exploration
    import sim.isaac_demo_scene as scene
    import sim.isaac_demo_camera as camera
    import sim.isaac_demo_kinematics as kinematics
    import sim.isaac_demo_runtime as runtime
    import sim.isaac_exploration as isaac_exploration
    import sim.control_session as control_session
    import sim.isaac_control_runtime as control_runtime
    import sim.isaac_control_capture as control_capture
    import sim.isaac_control_execution as control_execution
    import sim.isaac_control_bridge as control_bridge
    import sim.isaac_demo as demo

    for module in (
        contract,
        wm_action,
        control_protocol,
        control_safety,
        control_tracking,
        trajectory,
        control_replay,
        recording,
        recording_jobs,
        exploration,
        domain_recording,
        scene,
        camera,
        kinematics,
        runtime,
        isaac_exploration,
        control_session,
        control_runtime,
        control_capture,
        control_execution,
        control_bridge,
        demo,
    ):
        importlib.reload(module)
    return demo
