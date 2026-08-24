from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
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
            _CandidateEnergyModel(), context, target, candidates
        )

        self.assertEqual(selected.shape, (1, 2, 7))
        torch.testing.assert_close(selected[0, :, 0], torch.tensor((0.1, -0.1)))

    def test_rejects_invalid_candidate_configuration(self) -> None:
        with self.assertRaises(ValueError):
            CandidateMiningConfig(candidates_per_rollout=1)
        with self.assertRaises(ValueError):
            CandidateMiningConfig(noise_scale=0.0)

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
            bounds=PlannerActionBounds(
                maximum_translation_norm=0.01,
                maximum_rotation_norm=0.04,
                maximum_gripper_delta=0.1,
            ),
        )
        restored = CandidateMiningConfig.from_dict(config.to_dict())
        recorded = torch.zeros((3, 1, 7))

        original = sample_local_candidates(
            recorded,
            config=config,
            generator=torch.Generator().manual_seed(19),
        )
        replay = sample_local_candidates(
            recorded,
            config=restored,
            generator=torch.Generator().manual_seed(19),
        )

        self.assertEqual(restored, config)
        torch.testing.assert_close(replay, original)


if __name__ == "__main__":
    unittest.main()
