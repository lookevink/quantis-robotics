from __future__ import annotations

import unittest
from pathlib import Path

from jepa.embed_episode import DEFAULT_FRAMES, pool_features, sample_paths

try:
    import torch
except ImportError:
    torch = None


def frames(count: int) -> list[Path]:
    return [Path(f"rgb_{index:04d}.png") for index in range(count)]


class SamplePathsTest(unittest.TestCase):
    def test_rejects_an_empty_episode(self) -> None:
        with self.assertRaisesRegex(ValueError, "no PNG frames"):
            sample_paths([], DEFAULT_FRAMES)

    def test_rejects_an_episode_shorter_than_the_clip(self) -> None:
        with self.assertRaisesRegex(ValueError, "32 frames but 64 were requested"):
            sample_paths(frames(32), 64)

    def test_keeps_every_frame_when_the_counts_match(self) -> None:
        exact = frames(DEFAULT_FRAMES)
        self.assertEqual(sample_paths(exact, DEFAULT_FRAMES), exact)

    def test_downsamples_a_long_episode_without_repeating_frames(self) -> None:
        picked = sample_paths(frames(128), 64)
        self.assertEqual(len(picked), 64)
        self.assertEqual(len(set(picked)), 64)
        self.assertEqual(picked[0], Path("rgb_0000.png"))
        self.assertEqual(picked[-1], Path("rgb_0127.png"))

    def test_allows_a_short_clip_when_asked_for_explicitly(self) -> None:
        short = frames(32)
        self.assertEqual(sample_paths(short, 32), short)


@unittest.skipIf(torch is None, "torch is not installed")
class PoolFeaturesTest(unittest.TestCase):
    def test_pools_tokens_into_one_vector_per_clip(self) -> None:
        features = torch.zeros(1, 5, 8)
        features[0, :, 0] = torch.arange(5, dtype=torch.float32)
        pooled = pool_features(features)
        self.assertEqual(tuple(pooled.shape), (1, 8))
        self.assertAlmostEqual(pooled[0, 0].item(), 2.0)

    def test_pools_spatiotemporal_features_over_every_axis_but_the_channel(self) -> None:
        pooled = pool_features(torch.ones(2, 4, 5, 8))
        self.assertEqual(tuple(pooled.shape), (2, 8))

    def test_passes_through_already_pooled_features(self) -> None:
        self.assertEqual(tuple(pool_features(torch.ones(1, 8)).shape), (1, 8))

    def test_rejects_a_one_dimensional_feature_tensor(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unexpected V-JEPA feature shape"):
            pool_features(torch.ones(8))


if __name__ == "__main__":
    unittest.main()
