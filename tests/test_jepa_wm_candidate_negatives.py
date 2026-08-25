from __future__ import annotations

import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action_activity import DroidActionActivityThresholds
    from jepa_wm.candidate_negatives import (
        CandidateMiningConfig,
        mine_lowest_energy_candidates,
        sample_local_candidates,
    )
    from jepa_wm.planner import PlannerActionBounds


class _CandidateEnergyModel:
    @staticmethod
    def unroll(context, actions):
        return actions.cumsum(dim=0)


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class CandidateNegativesTest(unittest.TestCase):
    def test_tensor_activity_matches_scalar_activity_contract(self) -> None:
        thresholds = DroidActionActivityThresholds(
            translation_norm=1e-5,
            rotation_norm=1e-5,
            gripper_delta=0.005,
        )
        actions = torch.tensor(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001),
                (0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006),
            )
        )

        tensor_activity = thresholds.active_tensor(actions).tolist()
        scalar_activity = [thresholds.is_active(action.tolist()) for action in actions]

        self.assertEqual(tensor_activity, scalar_activity)

    def test_samples_finite_candidates_inside_planner_bounds(self) -> None:
        recorded = torch.zeros((3, 2, 7))
        candidates = sample_local_candidates(
            recorded,
            config=CandidateMiningConfig(
                candidates_per_rollout=8,
                noise_scale=2.0,
                bounds=PlannerActionBounds(),
            ),
            generator=torch.Generator().manual_seed(7),
        )

        self.assertEqual(candidates.shape, (3, 2, 8, 7))
        self.assertTrue(torch.isfinite(candidates).all())
        self.assertTrue(
            (torch.linalg.vector_norm(candidates[..., :3], dim=-1) <= 0.020001).all()
        )
        self.assertTrue(
            (torch.linalg.vector_norm(candidates[..., 3:6], dim=-1) <= 0.080001).all()
        )
        self.assertTrue((candidates[..., 6].abs() <= 0.250001).all())

    def test_selects_the_lowest_energy_candidate_per_rollout(self) -> None:
        candidates = torch.zeros((1, 2, 3, 7))
        candidates[0, 0, :, 0] = torch.tensor((0.5, 0.1, 0.3))
        candidates[0, 1, :, 0] = torch.tensor((-0.5, -0.2, -0.1))
        context = torch.zeros((2, 1, 7))
        target = torch.zeros((2, 1, 7))

        selected = mine_lowest_energy_candidates(
            _CandidateEnergyModel(),
            context,
            target,
            candidates,
            scoring_batch_size=2,
        )

        self.assertEqual(selected.shape, (1, 2, 7))
        torch.testing.assert_close(selected[0, :, 0], torch.tensor((0.1, -0.1)))

    def test_rejects_invalid_candidate_configuration(self) -> None:
        with self.assertRaises(ValueError):
            CandidateMiningConfig(candidates_per_rollout=1)
        with self.assertRaises(ValueError):
            CandidateMiningConfig(scoring_batch_size=0)
        with self.assertRaises(ValueError):
            CandidateMiningConfig(noise_scale=0.0)
        with self.assertRaises(ValueError):
            CandidateMiningConfig(minimum_goal_cosine=1.01)
        with self.assertRaises(ValueError):
            DroidActionActivityThresholds(translation_norm=-1.0)

    def test_tensor_clipping_matches_numpy_planner_bounds(self) -> None:
        import numpy as np

        bounds = PlannerActionBounds()
        values = torch.tensor(
            [[[[0.04, 0.0, 0.0, 0.0, 0.16, 0.0, 0.5]]]],
            dtype=torch.float64,
        )

        tensor_result = bounds.clip_tensor(values)
        numpy_result = bounds.clip(values.reshape(1, 1, 7).numpy())

        torch.testing.assert_close(
            tensor_result.reshape(1, 1, 7), torch.from_numpy(np.asarray(numpy_result))
        )

    def test_serialized_config_replays_identical_candidates(self) -> None:
        config = CandidateMiningConfig(
            candidates_per_rollout=3,
            noise_scale=0.4,
            minimum_goal_cosine=0.95,
            bounds=PlannerActionBounds(
                maximum_translation_norm=0.01,
                maximum_rotation_norm=0.04,
                maximum_gripper_delta=0.1,
            ),
        )
        restored = CandidateMiningConfig.from_dict(config.to_dict())
        recorded = torch.zeros((3, 1, 7))
        recorded[:, 0, 0] = 0.001
        goals = torch.tensor(((0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),))

        original = sample_local_candidates(
            recorded,
            config=config,
            generator=torch.Generator().manual_seed(19),
            goal_actions=goals,
        )
        replay = sample_local_candidates(
            recorded,
            config=restored,
            generator=torch.Generator().manual_seed(19),
            goal_actions=goals,
        )

        self.assertEqual(restored, config)
        torch.testing.assert_close(replay, original)

    def test_goal_aligned_mining_replaces_only_misaligned_first_actions(self) -> None:
        recorded = torch.zeros((3, 2, 7))
        recorded[:, :, 0] = 0.0011
        goals = torch.zeros((2, 7))
        goals[:, 0] = 0.003
        candidates = sample_local_candidates(
            recorded,
            config=CandidateMiningConfig(
                candidates_per_rollout=8,
                noise_scale=2.0,
                minimum_goal_cosine=0.95,
            ),
            generator=torch.Generator().manual_seed(7),
            goal_actions=goals,
        )

        first = candidates[0]
        cosines = torch.nn.functional.cosine_similarity(
            first,
            goals[:, None, :],
            dim=-1,
        )
        self.assertTrue((cosines >= 0.95 - 1e-6).all())
        self.assertTrue((candidates[1:] != recorded[1:, :, None, :]).any())

    def test_goal_aligned_mining_preserves_stationary_first_actions(self) -> None:
        recorded = torch.zeros((3, 2, 7))
        recorded[1:, :, 0] = 0.001
        recorded[0, 0, 6] = 0.001
        recorded[0, 1, 0] = 0.0011
        goals = torch.zeros((2, 7))
        goals[:, 0] = 0.003

        candidates = sample_local_candidates(
            recorded,
            config=CandidateMiningConfig(
                candidates_per_rollout=8,
                noise_scale=2.0,
                minimum_goal_cosine=0.95,
            ),
            generator=torch.Generator().manual_seed(7),
            goal_actions=goals,
        )

        torch.testing.assert_close(
            candidates[0, 0],
            recorded[0, 0].expand_as(candidates[0, 0]),
        )
        active_cosines = torch.nn.functional.cosine_similarity(
            candidates[0, 1],
            goals[1].expand_as(candidates[0, 1]),
            dim=-1,
        )
        self.assertTrue((active_cosines >= 0.95 - 1e-6).all())
        self.assertTrue((candidates[1:] != recorded[1:, :, None, :]).any())

    def test_goal_aligned_mining_requires_aligned_recorded_actions(self) -> None:
        recorded = torch.zeros((3, 1, 7))
        recorded[:, 0, 1] = 0.0011
        goals = torch.tensor(((0.003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),))

        with self.assertRaisesRegex(ValueError, "recorded first action"):
            sample_local_candidates(
                recorded,
                config=CandidateMiningConfig(minimum_goal_cosine=0.95),
                generator=torch.Generator().manual_seed(7),
                goal_actions=goals,
            )

    def test_goal_aligned_mining_bounds_recorded_fallback_before_replacement(self) -> None:
        recorded = torch.zeros((3, 1, 7))
        recorded[0, 0, 0] = 0.02000005
        goals = torch.tensor(((0.02000005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),))

        candidates = sample_local_candidates(
            recorded,
            config=CandidateMiningConfig(minimum_goal_cosine=0.95),
            generator=torch.Generator().manual_seed(7),
            goal_actions=goals,
        )

        self.assertTrue(
            (torch.linalg.vector_norm(candidates[..., :3], dim=-1) <= 0.02).all()
        )
        cosines = torch.nn.functional.cosine_similarity(
            candidates[0], goals[:, None, :], dim=-1
        )
        self.assertTrue((cosines >= 0.95 - 1e-6).all())

    def test_candidate_scoring_is_micro_batched_without_changing_selection(self) -> None:
        candidates = torch.zeros((1, 2, 4, 7))
        candidates[0, 0, :, 0] = torch.tensor((0.4, 0.1, 0.3, 0.2))
        candidates[0, 1, :, 0] = torch.tensor((-0.4, -0.2, -0.1, -0.3))
        context = torch.zeros((2, 1, 7))
        target = torch.zeros((2, 1, 7))
        batch_sizes = []

        def energy(_model, _context, _target, actions):
            batch_sizes.append(actions.shape[1])
            return actions.square().sum(dim=(0, 2))

        with patch("jepa_wm.candidate_negatives.score_actions", side_effect=energy):
            selected = mine_lowest_energy_candidates(
                object(),
                context,
                target,
                candidates,
                scoring_batch_size=2,
            )

        self.assertEqual(batch_sizes, [2, 2, 2, 2])
        torch.testing.assert_close(selected[0, :, 0], torch.tensor((0.1, -0.1)))


if __name__ == "__main__":
    unittest.main()
