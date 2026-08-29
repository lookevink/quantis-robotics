from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DroidAction, DroidPose
from jepa_wm.contact_grasp_target import (
    CONTACT_GRASP_TARGET_POLICY,
    DIRECTIONAL_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    LEGACY_CONTACT_GRASP_TARGET_POLICY_SCHEMA,
    ContactGraspTargetPolicy,
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
                map(_pose, (0.3356, 0.3322, 0.3225, 0.3076, 0.2899)),
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
                dict(
                    zip(
                        self.transport_contexts,
                        (0.3356, 0.3322, 0.3225, 0.3076, 0.2899),
                    )
                )
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

        self.assertEqual(
            target.frame,
            self._target_path(self.transport_targets[0]),
        )
        self.assertEqual(target.pose, _pose(0.3076))


if __name__ == "__main__":
    unittest.main()
