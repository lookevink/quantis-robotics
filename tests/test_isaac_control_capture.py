from __future__ import annotations

import json
from math import cos, sin
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import asyncio

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DROID_FPS, DroidAction
from jepa_wm.control_policy import ControlExecutionPolicy
from jepa_wm.insertion_contract import INSERTION_TASK_ID
from sim.control_identity import observation_id_for_session
from sim.isaac_control_capture import (
    ControlKnownStart,
    ControlKnownStartAuthority,
    control_capture_schedule,
    control_context_recording_label,
    control_warmup_plan,
    control_physical_routing_observation,
    recorded_control_context,
    requires_stable_insertion_capture,
    validate_known_start_collision_configuration,
    validate_known_start_pose,
    validated_control_reference,
)
from sim.control_capture_schedule import (
    ControlCapturePhase,
    ControlCaptureTimingBudget,
    run_control_capture_phase,
)
from sim.control_context import ControlContextPurpose, RecordedControlStep
from sim.demo_sequence import Phase
from sim.recording import RecordingLabel, RecordingMoment


class ControlCaptureContractTest(unittest.TestCase):
    def test_insertion_capture_persists_the_observed_physical_router_input(self) -> None:
        step = {
            "plug_position": [0.1, 0.2, 0.3],
            "plug_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "end_effector_world_position": [0.2, 0.3, 0.4],
            "gripper_frame_world_position": [0.15, 0.25, 0.35],
            "gripper_width_m": 0.02,
            "arm_tracking_error_rad": 0.001,
            "gripper_tracking_error_m": 0.0005,
            "contact_force_newtons": 0.0,
            "plug_attached": False,
        }
        target = {
            "socket_position": [0.0, 0.0, 0.0],
            "socket_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "insertion_axis": [1.0, 0.0, 0.0],
        }

        observed = control_physical_routing_observation(
            step,
            target,
            DroidAction((0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1)),
            insertion_control=True,
        )

        self.assertIsNotNone(observed)
        assert observed is not None
        self.assertEqual(len(observed.values), 26)
        self.assertIsNone(
            control_physical_routing_observation(
                step,
                target,
                DroidAction((0.0,) * 7),
                insertion_control=False,
            )
        )

    def test_capture_phase_deadline_cancels_the_owned_operation(self) -> None:
        cancelled = False

        async def exercise() -> None:
            nonlocal cancelled

            async def blocked() -> None:
                nonlocal cancelled
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled = True

            budget = ControlCaptureTimingBudget(
                ((ControlCapturePhase.KNOWN_START, 0.001),), 1
            )
            with self.assertRaisesRegex(RuntimeError, "deadline.*known_start"):
                await run_control_capture_phase(
                    budget,
                    ControlCapturePhase.KNOWN_START,
                    blocked(),
                )

        asyncio.run(exercise())
        self.assertTrue(cancelled)

    def test_unbudgeted_capture_path_runs_without_a_deadline(self) -> None:
        async def exercise() -> str:
            async def complete() -> str:
                return "complete"

            return await run_control_capture_phase(
                None,
                ControlCapturePhase.REPLAY,
                complete(),
            )

        self.assertEqual(asyncio.run(exercise()), "complete")

    def test_unattached_known_start_retains_its_ready_task_phase(self) -> None:
        self.assertEqual(
            control_context_recording_label(False, 110),
            RecordingLabel(RecordingMoment.MOTION, Phase.READY),
        )

    def test_initialization_label_preserves_reset_and_attachment_semantics(
        self,
    ) -> None:
        self.assertEqual(
            control_context_recording_label(False, 0),
            RecordingLabel(RecordingMoment.INITIAL),
        )
        self.assertEqual(
            control_context_recording_label(True, 110),
            RecordingLabel(RecordingMoment.ATTACHED, Phase.GRASP),
        )
        with self.assertRaisesRegex(ValueError, "task index"):
            control_context_recording_label(False, -1)

    def test_known_start_timing_budget_fits_the_client_bound(self) -> None:
        schedule = control_capture_schedule(
            ControlExecutionPolicy.DIRECT,
            insertion_control=True,
            context_index=110,
            context_purpose=ControlContextPurpose.CONTACT_GRASP,
        )

        budget = schedule.timing_budget
        self.assertIsNotNone(budget)
        assert budget is not None
        self.assertEqual(budget.maximum_total_seconds, 900)
        self.assertLessEqual(
            sum(seconds for _, seconds in budget.phases),
            budget.maximum_total_seconds,
        )
        budget.validate_elapsed(
            ControlCapturePhase.TERMINAL_CAMERA_AND_STABILIZATION,
            600.0,
        )
        with self.assertRaisesRegex(ValueError, "exceeded"):
            budget.validate_elapsed(
                ControlCapturePhase.TERMINAL_CAMERA_AND_STABILIZATION,
                600.001,
            )

    def test_known_start_pose_and_collision_bounds_fail_closed(self) -> None:
        expected_position = (-0.1, 0.0, 1.0)
        expected_orientation = (1.0, 0.0, 0.0, 0.0)
        validate_known_start_pose(
            "connector",
            expected_position,
            tuple(-value for value in expected_orientation),
            expected_position,
            expected_orientation,
        )
        with self.assertRaisesRegex(RuntimeError, "position"):
            validate_known_start_pose(
                "connector",
                (-0.09998, 0.0, 1.0),
                expected_orientation,
                expected_position,
                expected_orientation,
            )
        with self.assertRaisesRegex(RuntimeError, "orientation"):
            validate_known_start_pose(
                "connector",
                expected_position,
                (cos(0.001), sin(0.001), 0.0, 0.0),
                expected_position,
                expected_orientation,
            )

        target = {
            "connector_collisions_enabled": True,
            "compliant_collision_parts": ["StrainRelief"],
        }
        configuration = (
            ("/World/RJ45_Plug/Body", True),
            ("/World/RJ45_Plug/StrainRelief", False),
        )
        validate_known_start_collision_configuration(
            target,
            ("StrainRelief",),
            configuration,
        )
        with self.assertRaisesRegex(RuntimeError, "collision configuration"):
            validate_known_start_collision_configuration(
                target,
                ("StrainRelief",),
                (("/World/RJ45_Plug/Body", False),),
            )

    def test_contact_grasp_uses_a_fingerprinted_known_start_without_replay(
        self,
    ) -> None:
        schedule = control_capture_schedule(
            ControlExecutionPolicy.DIRECT,
            insertion_control=True,
            context_index=110,
            context_purpose=ControlContextPurpose.CONTACT_GRASP,
        )

        self.assertEqual(schedule.initialization_task_index, 110)
        self.assertEqual(schedule.replay_frames, ())
        self.assertTrue(schedule.defer_camera_activation)
        self.assertEqual(schedule.progress_units, 5)
        self.assertEqual(schedule.recorded_task_indices, (110,))
        self.assertEqual(len(schedule.fingerprint), 64)
        self.assertEqual(schedule, type(schedule).from_dict(schedule.to_dict()))
        authority = ControlKnownStartAuthority(*("0" * 64,) * 6)
        step = RecordedControlStep(
            110,
            (0.0,) * 7,
            0.04,
            False,
            (-0.1, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0),
        )
        known_start = ControlKnownStart.from_context(
            "contact-held-01",
            12601,
            step,
            (-0.2, 0.0, 1.0),
            (1.0, 0.0, 0.0, 0.0),
            schedule,
            authority,
        )
        self.assertEqual(known_start.task_index, 110)
        self.assertEqual(len(known_start.fingerprint), 64)
        self.assertEqual(
            known_start.fingerprint,
            ControlKnownStart.from_context(
                "contact-held-01",
                12601,
                step,
                (-0.2, 0.0, 1.0),
                (1.0, 0.0, 0.0, 0.0),
                schedule,
                authority,
            ).fingerprint,
        )
        changed_authority = ControlKnownStartAuthority(
            "1" * 64,
            *("0" * 64,) * 5,
        )
        self.assertNotEqual(
            known_start.fingerprint,
            ControlKnownStart.from_context(
                "contact-held-01",
                12601,
                step,
                (-0.2, 0.0, 1.0),
                (1.0, 0.0, 0.0, 0.0),
                schedule,
                changed_authority,
            ).fingerprint,
        )

    def test_standard_capture_keeps_prefix_replay_and_eager_camera(self) -> None:
        schedule = control_capture_schedule(
            ControlExecutionPolicy.DIRECT,
            insertion_control=False,
            context_index=4,
            context_purpose=ControlContextPurpose.STANDARD,
        )

        self.assertEqual(schedule.initialization_task_index, 0)
        self.assertEqual(
            tuple(frame.task_index for frame in schedule.replay_frames),
            (1, 2, 3, 4),
        )
        self.assertFalse(schedule.defer_camera_activation)
        self.assertEqual(schedule.recorded_task_indices, (0, 1, 2, 3, 4))

    def test_contact_grasp_plan_preserves_context_with_terminal_rgb(
        self,
    ) -> None:
        context_index = 110

        plan = control_warmup_plan(
            ControlExecutionPolicy.DIRECT,
            insertion_control=True,
            context_index=context_index,
            context_purpose=ControlContextPurpose.CONTACT_GRASP,
        )

        self.assertEqual(
            tuple(frame.task_index for frame in plan),
            tuple(range(context_index + 1)),
        )
        self.assertEqual(
            tuple(frame.task_index for frame in plan if frame.record_rgb),
            (context_index,),
        )
        self.assertEqual(
            tuple(frame.task_index for frame in plan if frame.stabilize),
            (context_index,),
        )
        self.assertTrue(all(frame.observe_safety for frame in plan[1:]))
        stable_previous_action = DroidAction((0.0,) * 7)
        frame_index, context_step, previous_action = recorded_control_context(
            ({"index": 0, "action_from_previous": None},),
            plan,
            stable_previous_action,
        )
        self.assertEqual(frame_index, 0)
        self.assertEqual(context_step["index"], 0)
        self.assertEqual(previous_action, stable_previous_action)

    def test_standard_capture_preserves_every_warmup_rgb(self) -> None:
        context_index = 4

        plan = control_warmup_plan(
            ControlExecutionPolicy.DIRECT,
            insertion_control=False,
            context_index=context_index,
            context_purpose=ControlContextPurpose.STANDARD,
        )

        self.assertEqual(
            tuple(frame.task_index for frame in plan if frame.record_rgb),
            tuple(range(context_index + 1)),
        )
        self.assertFalse(any(frame.stabilize for frame in plan))
        self.assertFalse(any(frame.observe_safety for frame in plan))
        action = DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        frame_index, _, previous_action = recorded_control_context(
            tuple(
                {
                    "index": step_index,
                    "action_from_previous": (
                        None if step_index == 0 else list(action.values)
                    ),
                }
                for step_index in range(context_index + 1)
            ),
            plan,
            None,
        )
        self.assertEqual(frame_index, context_index)
        self.assertEqual(previous_action, action)

    def test_stabilizes_every_insertion_capture_that_can_lead_to_motion(self) -> None:
        for policy in (
            ControlExecutionPolicy.INSERTION_SAFETY_EVALUATION,
            ControlExecutionPolicy.INSERTION_RESET_TRIAL,
            ControlExecutionPolicy.INSERTION_RESOLUTION_MEASUREMENT,
        ):
            with self.subTest(policy=policy):
                self.assertTrue(
                    requires_stable_insertion_capture(
                        policy,
                        insertion_control=True,
                        step_index=43,
                        context_index=43,
                    )
                )
        self.assertFalse(
            requires_stable_insertion_capture(
                ControlExecutionPolicy.DIRECT,
                insertion_control=True,
                step_index=43,
                context_index=43,
            )
        )
        self.assertTrue(
            requires_stable_insertion_capture(
                ControlExecutionPolicy.DIRECT,
                insertion_control=True,
                step_index=18,
                context_index=18,
                context_purpose=ControlContextPurpose.CONTACT_GRASP,
            )
        )
        self.assertFalse(
            requires_stable_insertion_capture(
                ControlExecutionPolicy.INSERTION_RESET_TRIAL,
                insertion_control=False,
                step_index=43,
                context_index=43,
            )
        )
        self.assertFalse(
            requires_stable_insertion_capture(
                ControlExecutionPolicy.INSERTION_RESET_TRIAL,
                insertion_control=True,
                step_index=42,
                context_index=43,
            )
        )

    def _recording(
        self,
        root: Path,
        *,
        seed: int,
        split: str = "held_out",
        task: str | None = None,
    ) -> Path:
        recording = root / "held-reference"
        recording.mkdir()
        (recording / "manifest.json").write_text(
            json.dumps(
                {
                    "recording_id": recording.name,
                    "fps": DROID_FPS,
                    "cameras": ["wrist"],
                    "action": ACTION_RECORDING_CONTRACT.to_dict(),
                    "metadata": {
                        "dataset": "jepa_wm_domain_v1",
                        "split": split,
                        "seed": seed,
                        **({"task": task} if task is not None else {}),
                    },
                }
            )
        )
        return recording

    def test_binds_observation_identity_to_the_session(self) -> None:
        self.assertGreater(observation_id_for_session("session-a"), 0)
        self.assertNotEqual(
            observation_id_for_session("session-a"),
            observation_id_for_session("session-b"),
        )

    def test_accepts_only_the_matching_held_out_droid_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=11400)
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                reference = validated_control_reference(
                    recording.name,
                    11400,
                    ControlExecutionPolicy.DIRECT,
                )

                self.assertEqual(reference.seed, 11400)

    def test_rejects_a_reference_from_another_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=11400)
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                with self.assertRaisesRegex(ValueError, "does not match"):
                    validated_control_reference(
                        recording.name,
                        11401,
                        ControlExecutionPolicy.DIRECT,
                    )

    def test_training_references_require_the_calibration_collection_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=1400, split="train")
            with patch("sim.isaac_control_capture.RECORDING_ROOT", root):
                reference = validated_control_reference(
                    recording.name,
                    1400,
                    ControlExecutionPolicy.CALIBRATION_COLLECTION,
                )
                self.assertEqual(reference.split.value, "train")
                with self.assertRaisesRegex(ValueError, "expected 'held_out'"):
                    validated_control_reference(
                        recording.name,
                        1400,
                        ControlExecutionPolicy.DIRECT,
                    )

    def test_contact_insertion_reference_requires_strict_raw_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = self._recording(root, seed=52600, task=INSERTION_TASK_ID)
            with (
                patch("sim.isaac_control_capture.RECORDING_ROOT", root),
                patch(
                    "sim.isaac_control_capture.ContactInsertionEvidence.from_recording"
                ) as validate,
            ):
                validated_control_reference(
                    recording.name,
                    52600,
                    ControlExecutionPolicy.DIRECT,
                )

            validate.assert_called_once_with(
                recording.resolve(), expected_split="held_out"
            )


if __name__ == "__main__":
    unittest.main()
