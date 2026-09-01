from __future__ import annotations

import unittest
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest.mock import Mock, patch

from sim.runtime_loader import (
    _reload_project_module_from_source,
    _resident_simulator_operation_id,
    reload_demo_runtime,
)


class RuntimeLoaderTest(unittest.TestCase):
    def test_reads_active_and_idle_pre_interlock_managers(self) -> None:
        active_task = Mock(done=Mock(return_value=False))
        complete_task = Mock(done=Mock(return_value=True))
        active_legacy = Mock(spec=[])
        active_legacy._tasks = {"legacy-recording": active_task}
        idle_legacy = Mock(spec=[])
        idle_legacy._tasks = {"finished-recording": complete_task}

        self.assertEqual(
            _resident_simulator_operation_id(active_legacy),
            "legacy-recording",
        )
        self.assertIsNone(_resident_simulator_operation_id(idle_legacy))

    def test_keeps_the_resident_facade_while_a_simulator_operation_runs(self) -> None:
        resident = ModuleType("sim.isaac_demo")
        resident._RECORDING_JOBS = Mock(  # type: ignore[attr-defined]
            active_operation_id=Mock(return_value="active-recording")
        )

        with patch.dict(sys.modules, {"sim.isaac_demo": resident}):
            self.assertIs(reload_demo_runtime(), resident)

    def test_reload_discovers_a_new_source_after_a_cached_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            module_name = "quantis_new_reload_fixture"
            sys.path.insert(0, str(root))
            try:
                importlib.invalidate_caches()
                self.assertIsNone(importlib.util.find_spec(module_name))
                cached_directory_mtime = root.stat().st_mtime
                (root / f"{module_name}.py").write_text("VALUE = 'new'\n")
                os.utime(root, (cached_directory_mtime, cached_directory_mtime))

                loaded = _reload_project_module_from_source(module_name)

                self.assertEqual(loaded.VALUE, "new")
            finally:
                sys.path.remove(str(root))
                sys.modules.pop(module_name, None)

    def test_reload_uses_source_when_timestamp_bytecode_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            module_name = "quantis_stale_reload_fixture"
            source = root / f"{module_name}.py"
            source.write_text("VALUE = 'old'\nOLD = True\n")
            timestamp = source.stat().st_mtime
            sys.path.insert(0, str(root))
            try:
                imported = importlib.import_module(module_name)
                self.assertEqual(imported.VALUE, "old")
                source.write_text("VALUE = 'new'\nNEW = True\n")
                os.utime(source, (timestamp, timestamp))

                refreshed = _reload_project_module_from_source(module_name)

                self.assertEqual(refreshed.VALUE, "new")
                self.assertTrue(refreshed.NEW)
                self.assertFalse(hasattr(refreshed, "OLD"))
            finally:
                sys.path.remove(str(root))
                sys.modules.pop(module_name, None)

    def test_reloads_action_safety_and_shadow_contract_as_one_generation(self) -> None:
        source = """
import runpy
reload_demo_runtime = runpy.run_path(
    "sim/runtime_loader.py"
)["reload_demo_runtime"]
from jepa_wm import contact_grasp_acquisition_hold, contact_grasp_acquisition_resolution, contact_grasp_horizon_completion, control_resolution_baseline, control_resolution_profile, insertion_transition
from sim import isaac_control_runtime as old_control_runtime
from sim import isaac_demo_runtime as old_demo_runtime
from sim import isaac_insertion_demo as old_insertion_demo
stage = object()
articulation = object()
arm_attributes = [object() for _ in range(7)]
finger_attributes = [object() for _ in range(2)]
actuators = old_demo_runtime.Actuators(
    articulation, arm_attributes, finger_attributes
)
plug_prim = object()
hand_prim = object()
rigid_prim = object()
fixed_joint = object()
collision_attributes = [object()]
attachment = old_demo_runtime.PlugAttachment(
    old_demo_runtime.FixedJointPlugMotion(
        plug_prim,
        hand_prim,
        rigid_prim,
        fixed_joint,
    ),
    old_demo_runtime.PlugCollisionPolicy(collision_attributes),
)
hand_sensor = object()
connector_sensor = object()
sensor = old_control_runtime.ControlContactSensors(
    hand_sensor, connector_sensor
)
old_control_runtime.bind_live_runtime(
    "live-session", stage, actuators, attachment, sensor
)
del control_resolution_baseline.ControlResolutionBaselineAttempt
del control_resolution_profile.ControlResolutionLoad
del insertion_transition.resolve_insertion_followup_proposal
del contact_grasp_acquisition_hold.ContactGraspAcquisitionHold
del contact_grasp_acquisition_resolution.ContactGraspAcquisitionResolution
del contact_grasp_horizon_completion.ContactGraspHorizonCompletion
del old_insertion_demo.record_insertion_demo
reload_demo_runtime()
from jepa_wm import action, contact_grasp_acquisition_hold, contact_grasp_acquisition_resolution, contact_grasp_horizon_completion, control_resolution, control_resolution_baseline, control_resolution_profile, control_safety, direct_safety, experimental_candidate, insertion_contract, insertion_recording, insertion_refresh, insertion_transition, insertion_trial, joint_drive, objective_calibration, shadow_planning, shadow_safety, training_artifact
from sim import control_identity, control_session, demo_sequence, isaac_control_followup, isaac_control_runtime, isaac_demo, isaac_demo_kinematics, isaac_demo_runtime, isaac_exploration, isaac_insertion_demo, isaac_insertion_trial, recording, trial_source_cache
assert isaac_control_followup.ContactGraspAcquisitionHold is contact_grasp_acquisition_hold.ContactGraspAcquisitionHold
assert isaac_control_followup.ContactGraspAcquisitionResolution is contact_grasp_acquisition_resolution.ContactGraspAcquisitionResolution
assert isaac_control_followup.ContactGraspHorizonCompletion is contact_grasp_horizon_completion.ContactGraspHorizonCompletion
assert control_resolution_baseline.ControlResolutionLoad is control_resolution_profile.ControlResolutionLoad
assert control_resolution.ControlResolutionLoad is control_resolution_profile.ControlResolutionLoad
assert control_resolution.ControlResolutionBaselineAttempt is control_resolution_baseline.ControlResolutionBaselineAttempt
assert control_safety.DroidActionScale is action.DroidActionScale
assert control_safety.DroidPose is action.DroidPose
assert direct_safety.SafetyProjectionAttempt is control_safety.SafetyProjectionAttempt
assert experimental_candidate.validate_recording_id is recording.validate_recording_id
assert control_identity.validate_recording_id is recording.validate_recording_id
assert control_session.InsertionTrialBinding is insertion_trial.InsertionTrialBinding
assert control_session.JointDriveTarget is joint_drive.JointDriveTarget
assert insertion_refresh.JointDriveTarget is joint_drive.JointDriveTarget
assert insertion_transition.ArtifactIdentity is training_artifact.ArtifactIdentity
assert isaac_insertion_trial.InsertionTrialSourceEvidence is insertion_trial.InsertionTrialSourceEvidence
assert trial_source_cache.ControlSession is control_session.ControlSession
assert shadow_planning.TaskProgressObjective is objective_calibration.TaskProgressObjective
assert isaac_demo_kinematics.build_demo_sequence is demo_sequence.build_demo_sequence
assert isaac_exploration.INSERTION_TASK_ID == insertion_contract.INSERTION_TASK_ID
assert insertion_recording.RECORDING_SCHEMA == recording.RECORDING_SCHEMA
assert isaac_demo._record_insertion_demo is isaac_insertion_demo.record_insertion_demo
restored_runtime = isaac_control_runtime.live_runtime_for("live-session", stage)
assert isinstance(restored_runtime, isaac_control_runtime.LiveControlRuntime)
assert isinstance(restored_runtime.actuators, isaac_demo_runtime.Actuators)
assert restored_runtime.actuators is not actuators
assert restored_runtime.actuators.articulation is articulation
assert restored_runtime.actuators.arm_attributes == arm_attributes
assert restored_runtime.actuators.finger_attributes == finger_attributes
assert isinstance(restored_runtime.attachment, isaac_demo_runtime.PlugAttachment)
assert restored_runtime.attachment is not attachment
assert restored_runtime.attachment.motion.rigid_prim is rigid_prim
assert restored_runtime.attachment.motion.prim is plug_prim
assert restored_runtime.attachment.motion.hand_prim is hand_prim
assert restored_runtime.attachment.motion.fixed_joint is fixed_joint
assert restored_runtime.attachment.collisions.collision_attributes == collision_attributes
assert isinstance(restored_runtime.sensor, isaac_control_runtime.ControlContactSensors)
assert restored_runtime.sensor is not sensor
assert restored_runtime.sensor.hand is hand_sensor
assert restored_runtime.sensor.connector is connector_sensor
kinematic_stage = object()
kinematic_offset = __import__("numpy").asarray((0.01, 0.02, 0.03))
kinematic_motion = isaac_demo_runtime.KinematicPlugMotion(
    plug_prim,
    hand_prim,
    kinematic_offset,
)
kinematic_attachment = isaac_demo_runtime.PlugAttachment(
    kinematic_motion,
    isaac_demo_runtime.PlugCollisionPolicy(collision_attributes),
)
isaac_control_runtime.bind_live_runtime(
    "kinematic-session",
    kinematic_stage,
    restored_runtime.actuators,
    kinematic_attachment,
    restored_runtime.sensor,
)
reload_demo_runtime()
restored_kinematic = isaac_control_runtime.live_runtime_for(
    "kinematic-session", kinematic_stage
)
assert isinstance(
    restored_kinematic.attachment.motion,
    isaac_demo_runtime.KinematicPlugMotion,
)
assert restored_kinematic.attachment.motion.prim is plug_prim
assert restored_kinematic.attachment.motion.hand_prim is hand_prim
assert restored_kinematic.attachment.motion.hand_to_plug_offset is not kinematic_offset
assert tuple(restored_kinematic.attachment.motion.hand_to_plug_offset) == tuple(kinematic_offset)
scale = control_safety.ACTION_SCALES[0]
evidence = shadow_safety.ShadowSafetyEvidence(
    observation_id=1,
    evaluated_at_unix_seconds=2.0,
    counterfactual_as_of_unix_seconds=1.0,
    planned_actions=(action.DroidAction((0.0,) * 7),) * 3,
    attempts=(control_safety.SafetyProjectionAttempt(
        scale,
        control_safety.ControlGateDecision(1, action.DroidPose((0.0,) * 7), ()),
        0.0,
        (0.0,) * 7,
    ),),
    selected_action_scale=scale,
)
assert shadow_safety.ShadowSafetyEvidence.from_dict(evidence.to_dict()) == evidence
"""
        subprocess.run([sys.executable, "-c", source], check=True)


if __name__ == "__main__":
    unittest.main()
