from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from jepa_wm.action import ActionSelectionBounds, DroidAction, DroidPose
from jepa_wm.control_protocol import ControlObservation, ControlTarget
from jepa_wm.control_safety import ACTION_SCALES, ORIENTATION_HOLD_ACTION_SCALES
from jepa_wm.insertion_contract import (
    InsertionControlTargetPolicy,
    InsertionLiveTargetMetric,
)
from jepa_wm.trajectory import RecordedFrame, RecordedRollout


def _rollout(horizon: int, translation_meters: float) -> RecordedRollout:
    return RecordedRollout(
        context=(RecordedFrame(43, Path("recording/wrist/frame_000043.png")),),
        context_pose=DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)),
        previous_action=DroidAction((0.0,) * 7),
        target=RecordedFrame(
            43 + horizon,
            Path(f"recording/wrist/frame_{43 + horizon:06d}.png"),
        ),
        target_pose=DroidPose(
            (0.4 + translation_meters, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5)
        ),
        actions=(DroidAction((0.0,) * 7),) * horizon,
    )


class InsertionControlTargetPolicyTest(unittest.TestCase):
    def test_holds_orientation_inside_measured_resolution(self) -> None:
        policy = InsertionControlTargetPolicy()
        current = DroidPose((0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))

        held = policy.projection_scales(
            current,
            DroidPose((0.4005, 0.0, 0.5, 0.0, 0.0, 0.001, 0.5)),
        )
        active = policy.projection_scales(
            current,
            DroidPose((0.4005, 0.0, 0.5, 0.0, 0.0, 0.002, 0.5)),
        )

        self.assertEqual(held, ORIENTATION_HOLD_ACTION_SCALES)
        self.assertEqual(active, ACTION_SCALES)

        legacy_payload = policy.to_dict()
        del legacy_payload["orientation_hold_tolerance_radians"]
        legacy = InsertionControlTargetPolicy.from_dict(legacy_payload)
        self.assertIsNone(legacy.orientation_hold_tolerance_radians)
        self.assertEqual(
            legacy.projection_scales(current, target=current),
            ACTION_SCALES,
        )

    def test_rejects_unsafe_camera_identifier(self) -> None:
        for camera in (".", "..", "../wrist", "wrist/camera"):
            with self.subTest(camera=camera):
                with self.assertRaises(ValueError):
                    InsertionControlTargetPolicy(camera=camera)

        with self.assertRaisesRegex(ValueError, "policy is invalid"):
            InsertionControlTargetPolicy(target_origin="live_observation")
        with self.assertRaisesRegex(ValueError, "policy is invalid"):
            InsertionControlTargetPolicy(live_target_metric="forward_projection")

    def test_nondefault_policy_round_trips_and_validates_exact_observation(self) -> None:
        policy = InsertionControlTargetPolicy(
            minimum_translation_meters=4e-4,
            maximum_action_horizon=6,
            orientation_hold_tolerance_radians=9e-4,
            camera="overhead",
            action_bounds=ActionSelectionBounds(
                minimum_action_norm=0.0,
                maximum_pose_action_norm=0.08,
                maximum_gripper_action=0.5,
            ),
        )
        self.assertEqual(InsertionControlTargetPolicy.from_dict(policy.to_dict()), policy)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir).resolve()
            target_path = (
                root / "recordings/reference/overhead/frame_000048.png"
            )
            rollout = replace(
                _rollout(5, 5.14e-4),
                target=RecordedFrame(48, target_path),
            )
            observation = ControlObservation(
                observation_id=1,
                captured_at_unix_seconds=1.0,
                context_frame=Path(
                    "recordings/reference/overhead/frame_000043.png"
                ),
                target=ControlTarget(
                    target_path.relative_to(root),
                    rollout.target_pose,
                ),
                expected_proposal=Path("/tmp/proposal.pth"),
                pose=rollout.context_pose,
                previous_action=rollout.previous_action,
                warmup_frames=43,
            )
            with patch(
                "jepa_wm.insertion_contract.load_rollout_at",
                side_effect=(_rollout(3, 1e-4), _rollout(4, 2e-4), rollout),
            ) as load:
                policy.validate_observation(
                    observation,
                    root / "recordings/reference",
                    frame_root=root,
                )

            self.assertEqual(load.call_args.kwargs["camera"], "overhead")
            self.assertEqual(load.call_args.kwargs["bounds"], policy.action_bounds)
            with patch.object(
                InsertionControlTargetPolicy,
                "select",
                return_value=rollout,
            ):
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    policy.validate_observation(
                        replace(
                            observation,
                            target=ControlTarget(
                                Path(
                                    "recordings/reference/overhead/frame_000047.png"
                                ),
                                rollout.target_pose,
                            ),
                        ),
                        root / "recordings/reference",
                        frame_root=root,
                    )

    def test_selects_first_bounded_horizon_above_measured_resolution(self) -> None:
        policy = InsertionControlTargetPolicy()
        with patch(
            "jepa_wm.insertion_contract.load_rollout_at",
            side_effect=(
                _rollout(3, 1.89e-4),
                _rollout(4, 3.32e-4),
                _rollout(5, 5.14e-4),
            ),
        ) as load:
            selected = policy.select(
                Path("recording"),
                context_index=43,
            )

        self.assertEqual(selected.target.index, 48)
        self.assertEqual(
            [item.kwargs["protocol"].action_horizon for item in load.call_args_list],
            [3, 4, 5],
        )
        self.assertEqual(
            [item.kwargs["bounds"] for item in load.call_args_list],
            [policy.action_bounds] * 3,
        )

    def test_followup_selects_resolution_horizon_from_live_pose(self) -> None:
        policy = InsertionControlTargetPolicy().for_followup()
        live_pose = DroidPose((0.40035, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        with patch(
            "jepa_wm.insertion_contract.load_rollout_at",
            side_effect=(
                _rollout(3, 4e-4),
                _rollout(4, 6e-4),
                _rollout(5, 9e-4),
            ),
        ) as load:
            selected = policy.select(
                Path("recording"),
                context_index=44,
                current_pose=live_pose,
            )

        self.assertEqual(selected.target.index, 48)
        self.assertEqual(len(load.call_args_list), 3)
        self.assertEqual(
            InsertionControlTargetPolicy.from_dict(policy.to_dict()),
            policy,
        )

        with self.assertRaisesRegex(ValueError, "observation pose"):
            policy.select(Path("recording"), context_index=44)

    def test_followup_skips_targets_already_behind_the_live_pose(self) -> None:
        policy = InsertionControlTargetPolicy().for_followup()
        live_pose = DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))
        with patch(
            "jepa_wm.insertion_contract.load_rollout_at",
            side_effect=(
                _rollout(3, 4e-4),
                _rollout(4, 6e-4),
                _rollout(5, 9e-4),
                _rollout(6, 1.2e-3),
                _rollout(7, 1.55e-3),
            ),
        ) as load:
            selected = policy.select(
                Path("recording"),
                context_index=45,
                current_pose=live_pose,
            )

        self.assertEqual(selected.target.index, 50)
        self.assertEqual(len(load.call_args_list), 5)

    def test_legacy_followup_preserves_euclidean_target_selection(self) -> None:
        payload = InsertionControlTargetPolicy().to_dict()
        del payload["live_target_metric"]
        policy = InsertionControlTargetPolicy.from_dict(payload).for_followup()
        live_pose = DroidPose((0.401, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5))

        with patch(
            "jepa_wm.insertion_contract.load_rollout_at",
            side_effect=(_rollout(3, 4e-4),),
        ) as load:
            selected = policy.select(
                Path("recording"),
                context_index=45,
                current_pose=live_pose,
            )

        self.assertEqual(
            policy.live_target_metric,
            InsertionLiveTargetMetric.EUCLIDEAN_DISTANCE,
        )
        self.assertEqual(selected.target.index, 46)
        self.assertEqual(len(load.call_args_list), 1)

    def test_fails_when_no_bounded_horizon_is_resolvable(self) -> None:
        policy = InsertionControlTargetPolicy(maximum_action_horizon=4)
        with patch(
            "jepa_wm.insertion_contract.load_rollout_at",
            side_effect=(_rollout(3, 1e-4), _rollout(4, 2e-4)),
        ):
            with self.assertRaisesRegex(ValueError, "no resolvable"):
                policy.select(
                    Path("recording"),
                    context_index=43,
                )


if __name__ == "__main__":
    unittest.main()
