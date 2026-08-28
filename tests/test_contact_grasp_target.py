from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DroidAction, DroidPose
from jepa_wm.contact_grasp_target import (
    CONTACT_GRASP_TARGET_POLICY,
    ContactGraspTargetPolicy,
)
from jepa_wm.joint_drive import JointDriveTarget
from jepa_wm.control_protocol import ControlObservation
from sim.control_session import ControlSessionState


def _pose(x: float) -> DroidPose:
    return DroidPose((x, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))


class ContactGraspTargetPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reference_poses = {
            21: _pose(0.3356),
            22: _pose(0.3322),
            23: _pose(0.3225),
            24: _pose(0.3076),
            25: _pose(0.2899),
        }

    def test_holds_the_acquisition_target_until_attachment(self) -> None:
        target = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.334),
            plug_attached=False,
            previous_target=Path("recordings/reference/wrist/frame_000021.png"),
            reference_context_poses=self.reference_poses,
        )

        self.assertEqual(target, 21)

    def test_advances_from_attachment_by_reference_state_not_action_count(self) -> None:
        first_retreat = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3355),
            plug_attached=True,
            previous_target=Path("recordings/reference/wrist/frame_000021.png"),
            reference_context_poses=self.reference_poses,
        )
        held_retreat = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3340),
            plug_attached=True,
            previous_target=Path("recordings/reference/wrist/frame_000024.png"),
            reference_context_poses=self.reference_poses,
        )
        advanced_retreat = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3300),
            plug_attached=True,
            previous_target=Path("recordings/reference/wrist/frame_000024.png"),
            reference_context_poses=self.reference_poses,
        )

        self.assertEqual(first_retreat, 24)
        self.assertEqual(held_retreat, 24)
        self.assertEqual(advanced_retreat, 25)
        self.assertEqual(
            CONTACT_GRASP_TARGET_POLICY.context_index_for_target(
                Path("recordings/reference/wrist/frame_000024.png")
            ),
            21,
        )

    def test_holds_the_conditioning_context_with_the_acquisition_target(self) -> None:
        self.assertEqual(
            CONTACT_GRASP_TARGET_POLICY.context_index_for_target(
                Path("recordings/reference/wrist/frame_000021.png")
            ),
            18,
        )
        with self.assertRaisesRegex(ValueError, "trained window"):
            CONTACT_GRASP_TARGET_POLICY.context_index_for_target(
                Path("recordings/reference/wrist/frame_000022.png")
            )

    def test_never_regresses_or_runs_past_the_trained_window(self) -> None:
        retained = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.3356),
            plug_attached=True,
            previous_target=Path("recordings/reference/wrist/frame_000027.png"),
            reference_context_poses=self.reference_poses,
        )
        terminal = CONTACT_GRASP_TARGET_POLICY.next_target_index(
            live_pose=_pose(0.20),
            plug_attached=True,
            previous_target=Path("recordings/reference/wrist/frame_000028.png"),
            reference_context_poses=self.reference_poses,
        )

        self.assertEqual(retained, 27)
        self.assertEqual(terminal, 28)

    def test_round_trips_only_the_current_policy(self) -> None:
        payload = CONTACT_GRASP_TARGET_POLICY.to_dict()

        self.assertEqual(
            ContactGraspTargetPolicy.from_dict(payload),
            CONTACT_GRASP_TARGET_POLICY,
        )
        with self.assertRaisesRegex(ValueError, "target policy"):
            ContactGraspTargetPolicy.from_dict({"schema": "legacy"})

    def test_rejects_target_substitution_and_incomplete_reference_poses(self) -> None:
        with self.assertRaisesRegex(ValueError, "previous target"):
            CONTACT_GRASP_TARGET_POLICY.next_target_index(
                live_pose=_pose(0.3356),
                plug_attached=False,
                previous_target=Path("recordings/reference/wrist/frame_000022.png"),
                reference_context_poses=self.reference_poses,
            )
        with self.assertRaisesRegex(ValueError, "reference poses"):
            CONTACT_GRASP_TARGET_POLICY.next_target_index(
                live_pose=_pose(0.3356),
                plug_attached=True,
                previous_target=Path("recordings/reference/wrist/frame_000021.png"),
                reference_context_poses={21: _pose(0.3356)},
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
            positions = {
                **{index: 0.3356 for index in range(29)},
                22: 0.3322,
                23: 0.3225,
                24: 0.3076,
                25: 0.2899,
            }
            steps = []
            for index in range(29):
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
                previous_target=Path(
                    "recordings/reference/wrist/frame_000021.png"
                ),
            )
            followup = ControlObservation(
                2,
                2.0,
                Path("recordings/capture/wrist/frame_000001.png"),
                target,
                Path("/tmp/proposal.pth"),
                _pose(0.3355),
                DroidAction((0.0,) * 7),
                21,
            )
            CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                followup,
                recording,
                frame_root=root,
                require_initial=False,
            )
            with self.assertRaisesRegex(ValueError, "observation target"):
                CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                    replace(followup, warmup_frames=18),
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
                warmup_frames=18,
            )
            CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                initial,
                recording,
                frame_root=root,
                require_initial=True,
            )
            with self.assertRaisesRegex(ValueError, "observation target"):
                CONTACT_GRASP_TARGET_POLICY.validate_observation_target(
                    replace(initial, warmup_frames=19),
                    recording,
                    frame_root=root,
                    require_initial=True,
                )

        self.assertEqual(
            target.frame,
            Path("recordings/reference/wrist/frame_000024.png"),
        )
        self.assertEqual(target.pose, _pose(0.3076))


if __name__ == "__main__":
    unittest.main()
