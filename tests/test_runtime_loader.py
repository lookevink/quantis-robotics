from __future__ import annotations

import unittest
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sim.runtime_loader import _reload_project_module_from_source


class RuntimeLoaderTest(unittest.TestCase):
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
from sim.runtime_loader import reload_demo_runtime
from jepa_wm import control_resolution_baseline, control_resolution_profile
del control_resolution_baseline.ControlResolutionBaselineAttempt
del control_resolution_profile.ControlResolutionLoad
reload_demo_runtime()
from jepa_wm import action, control_resolution, control_resolution_baseline, control_resolution_profile, control_safety, direct_safety, experimental_candidate, insertion_contract, insertion_recording, insertion_trial, objective_calibration, shadow_planning, shadow_safety
from sim import control_identity, control_session, demo_sequence, isaac_demo_kinematics, isaac_exploration, isaac_insertion_trial, recording, trial_source_cache
assert control_resolution_baseline.ControlResolutionLoad is control_resolution_profile.ControlResolutionLoad
assert control_resolution.ControlResolutionLoad is control_resolution_profile.ControlResolutionLoad
assert control_resolution.ControlResolutionBaselineAttempt is control_resolution_baseline.ControlResolutionBaselineAttempt
assert control_safety.DroidActionScale is action.DroidActionScale
assert control_safety.DroidPose is action.DroidPose
assert direct_safety.SafetyProjectionAttempt is control_safety.SafetyProjectionAttempt
assert experimental_candidate.validate_recording_id is recording.validate_recording_id
assert control_identity.validate_recording_id is recording.validate_recording_id
assert control_session.InsertionTrialBinding is insertion_trial.InsertionTrialBinding
assert isaac_insertion_trial.InsertionTrialSourceEvidence is insertion_trial.InsertionTrialSourceEvidence
assert trial_source_cache.ControlSession is control_session.ControlSession
assert shadow_planning.TaskProgressObjective is objective_calibration.TaskProgressObjective
assert isaac_demo_kinematics.build_demo_sequence is demo_sequence.build_demo_sequence
assert isaac_exploration.INSERTION_TASK_ID == insertion_contract.INSERTION_TASK_ID
assert insertion_recording.RECORDING_SCHEMA == recording.RECORDING_SCHEMA
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
