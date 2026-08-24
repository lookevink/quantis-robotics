from __future__ import annotations

import unittest
import subprocess
import sys


class RuntimeLoaderTest(unittest.TestCase):
    def test_reloads_action_safety_and_shadow_contract_as_one_generation(self) -> None:
        source = """
from sim.runtime_loader import reload_demo_runtime
reload_demo_runtime()
from jepa_wm import action, control_safety, experimental_candidate, insertion_contract, insertion_recording, objective_calibration, shadow_planning, shadow_safety
from sim import demo_sequence, isaac_demo_kinematics, isaac_exploration, recording
assert control_safety.DroidActionScale is action.DroidActionScale
assert control_safety.DroidPose is action.DroidPose
assert experimental_candidate.validate_recording_id is recording.validate_recording_id
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
