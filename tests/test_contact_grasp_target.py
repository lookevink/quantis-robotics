from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DroidAction, DroidPose
from jepa_wm.contact_grasp_target import (
    ACQUISITION_PROGRESS_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    CONTACT_GRASP_TARGET_POLICY,
    DIRECTIONAL_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    EXACT_COARSE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    TRACKING_ROBUST_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    RESOLUTION_FLOORED_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    FULL_ACQUISITION_RESOLUTION_FLOORED_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    TRACKING_ROBUST_CLOSE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    TRACKING_SETTLEMENT_CLOSE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    AXIS_RESOLVABLE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    LOADED_DRIVE_COMPENSATED_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    OBSERVABLE_ROTATION_LOADED_DRIVE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    HORIZON_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    LEGACY_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    RESOLUTION_AWARE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    ROTATION_RESOLVABLE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    TASK_RELATIVE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    TRACKING_BOUNDED_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    ContactGraspTargetStep,
    ContactGraspTargetPolicy,
)


_TERMINAL_V3_ACTIONS = (
    DroidAction(
        (
            -0.0005394376930780709,
            0.0003137695020996034,
            -0.0005303949001245201,
            -0.0024283351376652718,
            0.001132694655098021,
            -0.0017445380799472332,
            0.010228123515844345,
        )
    ),
    DroidAction(
        (
            -0.0005975229432806373,
            -0.00005648603109875694,
            0.0009232184966094792,
            -0.0025696740485727787,
            0.0017236964777112007,
            -0.006025914568454027,
            0.019305793568491936,
        )
    ),
    DroidAction(
        (
            -0.0013235784135758877,
            -0.00016435707220807672,
            0.0006800679257139564,
            -0.0033748429268598557,
            -0.0024368693120777607,
            0.007257508113980293,
            0.005395814776420593,
        )
    ),
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.control_protocol import ControlObservation
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL
from sim.control_session import ControlSessionState


def _pose(x: float) -> DroidPose:
    return DroidPose((x, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))


class ContactGraspTargetPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.acquisition_target = (
            CONTACT_GRASP_TARGET_POLICY.acquisition_target_index
        )
        self.initial_context = (
            self.acquisition_target - DROID_ROLLOUT_PROTOCOL.action_horizon
        )
        self.transport_contexts = (
            CONTACT_GRASP_TARGET_POLICY.transport_context_indices
        )
        self.transport_targets = (
            CONTACT_GRASP_TARGET_POLICY.transport_target_indices
        )
        self.reference_poses = dict(
            zip(
                self.transport_contexts,
                map(
                    _pose,
                    (
                        0.3356,
                        0.3322,
                        0.3225,
                        0.3076,
                        0.2899,
                        *(
                            0.2899
                            for _ in range(len(self.transport_contexts) - 5)
                        ),
                    ),
                ),
            )
        )

    @staticmethod
    def _target_path(index: int) -> Path:
        return Path(f"recordings/reference/wrist/frame_{index:06d}.png")

    def test_holds_the_acquisition_target_until_attachment(self) -> None:
        target = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.334),
            plug_attached=False,
            previous_target=self._target_path(self.acquisition_target),
            reference_context_poses=self.reference_poses,
        )

        self.assertEqual(target, self.acquisition_target)

    def test_advances_from_attachment_by_reference_state_not_action_count(self) -> None:
        first_retreat = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3355),
            plug_attached=True,
            previous_target=self._target_path(self.acquisition_target),
            reference_context_poses=self.reference_poses,
        )
        held_retreat = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3340),
            plug_attached=True,
            previous_target=self._target_path(self.transport_targets[0]),
            reference_context_poses=self.reference_poses,
        )
        advanced_retreat = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3300),
            plug_attached=True,
            previous_target=self._target_path(self.transport_targets[0]),
            reference_context_poses=self.reference_poses,
        )

        self.assertEqual(first_retreat, self.transport_targets[0])
        self.assertEqual(held_retreat, self.transport_targets[0])
        self.assertEqual(advanced_retreat, self.transport_targets[1])
        self.assertEqual(
            CONTACT_GRASP_TARGET_POLICY.context_index_for_target(
                self._target_path(self.transport_targets[0])
            ),
            self.transport_contexts[0],
        )

    def test_holds_the_conditioning_context_with_the_acquisition_target(self) -> None:
        self.assertEqual(
            CONTACT_GRASP_TARGET_POLICY.context_index_for_target(
                self._target_path(self.acquisition_target)
            ),
            self.initial_context,
        )
        with self.assertRaisesRegex(ValueError, "trained window"):
            CONTACT_GRASP_TARGET_POLICY.context_index_for_target(
                self._target_path(self.acquisition_target + 1)
            )

    def test_never_regresses_or_runs_past_the_trained_window(self) -> None:
        retained = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3356),
            plug_attached=True,
            previous_target=self._target_path(self.transport_targets[-2]),
            reference_context_poses=self.reference_poses,
        )
        terminal = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.20),
            plug_attached=True,
            previous_target=self._target_path(self.transport_targets[-1]),
            reference_context_poses=self.reference_poses,
        )

        self.assertEqual(retained, self.transport_targets[-2])
        self.assertEqual(terminal, self.transport_targets[-1])

    def test_round_trips_only_the_current_policy(self) -> None:
        payload = CONTACT_GRASP_TARGET_POLICY.to_dict()

        self.assertEqual(
            ContactGraspTargetPolicy.from_dict(payload),
            CONTACT_GRASP_TARGET_POLICY,
        )
        with self.assertRaisesRegex(ValueError, "target policy"):
            ContactGraspTargetPolicy.from_dict({"schema": "legacy"})

        legacy = ContactGraspTargetPolicy.from_dict(
            {"schema": LEGACY_CONTACT_GRASP_TARGET_POLICY_SCHEMA}
        )
        directional = ContactGraspTargetPolicy.from_dict(
            {"schema": DIRECTIONAL_CONTACT_GRASP_TARGET_POLICY_SCHEMA}
        )
        self.assertFalse(legacy.requires_directional_transport_progress)
        self.assertTrue(directional.requires_directional_transport_progress)
        self.assertFalse(directional.uses_horizon_transport_action)
        self.assertEqual(
            ContactGraspTargetPolicy.from_dict(legacy.to_dict()), legacy
        )
        self.assertTrue(
            CONTACT_GRASP_TARGET_POLICY.requires_directional_transport_progress
        )
        self.assertTrue(CONTACT_GRASP_TARGET_POLICY.uses_horizon_transport_action)
        self.assertEqual(
            CONTACT_GRASP_TARGET_POLICY.schema,
            OBSERVABLE_ROTATION_LOADED_DRIVE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
        )
        self.assertTrue(
            CONTACT_GRASP_TARGET_POLICY.requires_axis_resolvable_transport
        )
        self.assertTrue(
            CONTACT_GRASP_TARGET_POLICY.uses_attached_drive_bias_compensation
        )
        self.assertFalse(
            CONTACT_GRASP_TARGET_POLICY.uses_measured_acquisition_progress
        )
        self.assertTrue(CONTACT_GRASP_TARGET_POLICY.requires_resolvable_rotation)
        loaded_drive_v19 = ContactGraspTargetPolicy(
            LOADED_DRIVE_COMPENSATED_CONTACT_GRASP_TARGET_POLICY_SCHEMA
        )
        self.assertFalse(loaded_drive_v19.requires_resolvable_rotation)
        self.assertTrue(loaded_drive_v19.requires_axis_resolvable_transport)
        self.assertTrue(loaded_drive_v19.uses_attached_drive_bias_compensation)
        reconstructed_v19 = ContactGraspTargetPolicy.from_dict(
            loaded_drive_v19.to_dict()
        )
        self.assertEqual(reconstructed_v19, loaded_drive_v19)
        self.assertEqual(
            reconstructed_v19.schema,
            LOADED_DRIVE_COMPENSATED_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
        )
        self.assertFalse(reconstructed_v19.requires_resolvable_rotation)
        self.assertFalse(
            ContactGraspTargetPolicy(
                HORIZON_CONTACT_GRASP_TARGET_POLICY_SCHEMA
            ).requires_axis_resolvable_transport
        )
        self.assertFalse(
            ContactGraspTargetPolicy(
                AXIS_RESOLVABLE_CONTACT_GRASP_TARGET_POLICY_SCHEMA
            ).uses_attached_drive_bias_compensation
        )

    def test_task_relative_policy_translates_pose_but_not_reference_frame(self) -> None:
        policy = ContactGraspTargetPolicy.for_scene_translation((0.01, -0.02, 0.03))
        pose = DroidPose((0.2, 0.3, 0.4, 0.0, 0.0, 0.0, 0.5))

        translated = policy._translated_pose(pose)
        for actual, expected in zip(
            translated.values,
            (0.21, 0.28, 0.43, 0.0, 0.0, 0.0, 0.5),
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            policy.schema,
            TRACKING_SETTLEMENT_CLOSE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
        )
        self.assertFalse(policy.requires_axis_resolvable_transport)
        self.assertTrue(policy.uses_measured_acquisition_progress)
        self.assertTrue(policy.requires_resolvable_rotation)
        self.assertTrue(policy.uses_exact_coarse_translation_projection)
        self.assertEqual(ContactGraspTargetPolicy.from_dict(policy.to_dict()), policy)

        historical = ContactGraspTargetPolicy(
            TASK_RELATIVE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
            (0.01, -0.02, 0.03),
        )
        self.assertFalse(historical.uses_measured_acquisition_progress)
        self.assertEqual(
            ContactGraspTargetPolicy.from_dict(historical.to_dict()),
            historical,
        )

        acquisition_v6 = ContactGraspTargetPolicy(
            ACQUISITION_PROGRESS_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
            (0.01, -0.02, 0.03),
        )
        self.assertFalse(
            acquisition_v6.uses_coarse_acquisition_action(
                self._target_path(96),
                plug_attached=False,
            )
        )

    def test_resolution_aware_policy_uses_coarse_motion_only_before_close(self) -> None:
        policy = ContactGraspTargetPolicy.for_scene_translation((0.0, 0.0, 0.0))

        self.assertEqual(
            policy.schema,
            TRACKING_SETTLEMENT_CLOSE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
        )
        self.assertEqual(
            policy.coarse_acquisition_maximum_translation_meters,
            0.001,
        )
        self.assertTrue(policy.uses_coarse_orientation_hold_fallback)
        self.assertEqual(
            policy.minimum_coarse_translation_command_meters,
            0.0005,
        )
        self.assertEqual(
            policy.fine_acquisition_maximum_translation_meters,
            0.0006,
        )
        self.assertTrue(
            policy.uses_coarse_acquisition_action(
                self._target_path(96),
                plug_attached=False,
            )
        )
        self.assertFalse(
            policy.uses_coarse_acquisition_action(
                self._target_path(97),
                plug_attached=False,
            )
        )
        self.assertTrue(
            policy.uses_resolution_floored_acquisition_action(
                self._target_path(97),
                plug_attached=False,
            )
        )
        self.assertFalse(
            policy.uses_resolution_floored_acquisition_action(
                self._target_path(97),
                plug_attached=True,
            )
        )

        historical = ContactGraspTargetPolicy(
            RESOLUTION_AWARE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            historical.coarse_acquisition_maximum_translation_meters,
            0.005,
        )
        self.assertEqual(
            ContactGraspTargetPolicy.from_dict(historical.to_dict()),
            historical,
        )
        tracking_bounded = ContactGraspTargetPolicy(
            TRACKING_BOUNDED_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
            (0.0, 0.0, 0.0),
        )
        self.assertFalse(tracking_bounded.requires_resolvable_rotation)
        historical_rotation = ContactGraspTargetPolicy(
            ROTATION_RESOLVABLE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(historical_rotation.transport_target_indices[-1], 120)
        self.assertFalse(
            historical_rotation.uses_exact_coarse_translation_projection
        )
        historical_exact = ContactGraspTargetPolicy(
            EXACT_COARSE_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
            (0.0, 0.0, 0.0),
        )
        self.assertEqual(
            historical_exact.coarse_acquisition_maximum_translation_meters,
            0.002,
        )
        self.assertEqual(policy.transport_target_indices[-1], 128)
        self.assertFalse(
            policy.uses_coarse_acquisition_action(
                self._target_path(96),
                plug_attached=True,
            )
        )

    def test_current_policy_binds_initial_target_to_measured_acquisition_pose(self) -> None:
        policy = ContactGraspTargetPolicy.for_scene_translation((0.0, 0.0, 0.0))
        contexts = policy.acquisition_context_indices
        reference = {index: _pose(index / 1000.0) for index in contexts}

        target = policy.initial_target_index(
            live_pose=_pose(0.0132),
            reference_context_poses=reference,
        )
        advanced = policy.next_target_index(
            live_pose=_pose(0.0162),
            plug_attached=False,
            previous_target=self._target_path(target),
            reference_context_poses={
                **reference,
                **{
                    index: _pose(index / 1000.0)
                    for index in policy.transport_context_indices
                },
            },
        )
        held = policy.next_target_index(
            live_pose=_pose(0.0100),
            plug_attached=False,
            previous_target=self._target_path(advanced),
            reference_context_poses={
                **reference,
                **{
                    index: _pose(index / 1000.0)
                    for index in policy.transport_context_indices
                },
            },
        )

        self.assertEqual(target, 16)
        self.assertEqual(advanced, 19)
        self.assertEqual(held, advanced)

    def test_current_attached_transport_composes_the_native_horizon(self) -> None:
        actions = (
            DroidAction((-0.000233, -0.000201, 0.000132, 0.0, 0.0, 0.0, 0.02)),
            DroidAction((-0.000154, -0.000068, 0.000115, 0.0, 0.0, 0.0, 0.01)),
            DroidAction((-0.000365, -0.000190, 0.000202, 0.0, 0.0, 0.0, 0.0)),
        )

        transport = CONTACT_GRASP_TARGET_POLICY.action_for_execution(
            actions,
            plug_attached=True,
        )
        acquisition = CONTACT_GRASP_TARGET_POLICY.action_for_execution(
            actions,
            plug_attached=False,
        )
        directional = ContactGraspTargetPolicy(
            DIRECTIONAL_CONTACT_GRASP_TARGET_POLICY_SCHEMA
        ).action_for_execution(actions, plug_attached=True)

        np.testing.assert_allclose(
            transport.values[:3],
            (-0.000752, -0.000459, 0.000449),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(acquisition, actions[0])
        self.assertEqual(directional, actions[0])
        with self.assertRaisesRegex(ValueError, "action horizon"):
            CONTACT_GRASP_TARGET_POLICY.action_for_execution(
                actions[:2],
                plug_attached=True,
            )

    def test_current_transport_does_not_amplify_unresolved_rotation(self) -> None:
        transport = CONTACT_GRASP_TARGET_POLICY.action_for_execution(
            _TERMINAL_V3_ACTIONS,
            plug_attached=True,
        )

        np.testing.assert_allclose(
            transport.values[:3],
            np.sum([action.values[:3] for action in _TERMINAL_V3_ACTIONS], axis=0),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            transport.values[3:],
            _TERMINAL_V3_ACTIONS[0].values[3:],
            rtol=0.0,
            atol=1e-12,
        )

    def test_v3_reconstructs_the_historical_full_horizon(self) -> None:
        historical = ContactGraspTargetPolicy(
            HORIZON_CONTACT_GRASP_TARGET_POLICY_SCHEMA
        ).action_for_execution(_TERMINAL_V3_ACTIONS, plug_attached=True)

        np.testing.assert_allclose(
            historical.values,
            (
                -0.002460539049934596,
                0.00009292639879276976,
                0.0010728915221989155,
                -0.008356807610833177,
                0.00038885656032716653,
                -0.0005254051070068032,
                0.034929731860756874,
            ),
            rtol=0.0,
            atol=1e-12,
        )

    def test_rejects_target_substitution_and_incomplete_reference_poses(self) -> None:
        with self.assertRaisesRegex(ValueError, "previous target"):
            CONTACT_GRASP_TARGET_POLICY.next_target_index(
                live_pose=_pose(0.3356),
                plug_attached=False,
                previous_target=self._target_path(self.acquisition_target + 1),
                reference_context_poses=self.reference_poses,
            )
        with self.assertRaisesRegex(ValueError, "reference poses"):
            CONTACT_GRASP_TARGET_POLICY.next_target_index(
                live_pose=_pose(0.3356),
                plug_attached=True,
                previous_target=self._target_path(self.acquisition_target),
                reference_context_poses={
                    self.transport_contexts[0]: _pose(0.3356)
                },
            )

    def test_session_state_round_trips_the_current_schedule(self) -> None:
        state = ControlSessionState(
            session_id="grasp-1",
            reference_recording="reference",
            seed=72600,
            recording="capture",
            current_joint_positions=(0.0,) * 7,
            collision_detected=False,
            contact_force_newtons=0.0,
            active_drive_target=JointDriveTarget((0.0,) * 7, 0.05),
            contact_grasp_target_policy=CONTACT_GRASP_TARGET_POLICY,
        )

        restored = ControlSessionState.from_dict(state.to_dict())

        self.assertEqual(
            restored.contact_grasp_target_policy,
            CONTACT_GRASP_TARGET_POLICY,
        )

        stripped = state.to_dict()
        del stripped["contact_grasp_target_policy"]
        with self.assertRaisesRegex(ValueError, "target contract"):
            ControlSessionState.from_dict(stripped)

        stripped.pop("schema")
        stripped.pop("target_contract")
        legacy = ControlSessionState.from_dict(stripped)
        self.assertIsNone(legacy.contact_grasp_target_policy)
        with self.assertRaisesRegex(ValueError, "current execution authority"):
            legacy.require_current_contact_grasp_policy()

    def test_selects_an_exact_reference_bound_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            recording = root / "recordings" / "reference"
            frames = recording / "wrist"
            frames.mkdir(parents=True)
            frame_count = self.transport_targets[-1] + 1
            positions = {index: 0.3356 for index in range(frame_count)}
            positions.update(
                {
                    index: pose.values[0]
                    for index, pose in self.reference_poses.items()
                }
            )
            steps = []
            for index in range(frame_count):
                frame = frames / f"frame_{index:06d}.png"
                frame.touch()
                steps.append(
                    {
                        "index": index,
                        "end_effector_pose": [
                            positions[index], 0.0, 0.5, 0.0, 0.0, 0.0, 0.5
                        ],
                        "frames": {"wrist": f"wrist/{frame.name}"},
                    }
                )
            (recording / "manifest.json").write_text(
                json.dumps(
                    {
                        "frames": len(steps),
                        "cameras": ["wrist"],
                        "action": ACTION_RECORDING_CONTRACT.to_dict(),
                    }
                )
            )
            (recording / "steps.jsonl").write_text(
                "\n".join(json.dumps(step) for step in steps) + "\n"
            )

            target = CONTACT_GRASP_TARGET_POLICY.select(
                recording,
                frame_root=root,
                live_pose=_pose(0.3355),
                plug_attached=True,
                previous_target=self._target_path(self.acquisition_target),
            )
            followup = ControlObservation(
                2,
                2.0,
                Path("recordings/capture/wrist/frame_000001.png"),
                target,
                Path("/tmp/proposal.pth"),
                _pose(0.3355),
                DroidAction((0.0,) * 7),
                self.transport_contexts[0],
            )
            CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                followup,
                recording,
                frame_root=root,
                require_initial=False,
            )
            with self.assertRaisesRegex(ValueError, "observation target"):
                CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                    replace(followup, warmup_frames=self.initial_context),
                    recording,
                    frame_root=root,
                    require_initial=False,
                )

            initial = replace(
                followup,
                observation_id=1,
                target=CONTACT_GRASP_TARGET_POLICY.initial_target(
                    recording,
                    frame_root=root,
                ),
                warmup_frames=self.initial_context,
            )
            CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                initial,
                recording,
                frame_root=root,
                require_initial=True,
            )
            with self.assertRaisesRegex(ValueError, "observation target"):
                CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                    replace(initial, warmup_frames=self.initial_context + 1),
                    recording,
                    frame_root=root,
                    require_initial=True,
                )
            CONTACT_GRASP_TARGET_POLICY.validate_reference_schedule(
                (ContactGraspTargetStep(followup, True),),
                recording,
                frame_root=root,
                previous_step=ContactGraspTargetStep(initial, False),
            )
            with self.assertRaisesRegex(ValueError, "target schedule"):
                CONTACT_GRASP_TARGET_POLICY.validate_reference_schedule(
                    (),
                    recording,
                    frame_root=root,
                    previous_step=ContactGraspTargetStep(initial, False),
                )
            with self.assertRaisesRegex(ValueError, "target schedule"):
                CONTACT_GRASP_TARGET_POLICY.validate_reference_schedule(
                    (ContactGraspTargetStep(followup, True),),
                    recording,
                    frame_root=root,
                )

        self.assertEqual(
            target.frame,
            self._target_path(self.transport_targets[0]),
        )
        self.assertEqual(target.pose, _pose(0.3076))


if __name__ == "__main__":
    unittest.main()
