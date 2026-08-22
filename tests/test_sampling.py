from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jepa.embed_episode import (
    DEFAULT_FRAMES,
    pool_features,
    sample_paths,
)
from jepa.observation_source import ObservationSource

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


class ObservationFramesTest(unittest.TestCase):
    def test_script_entrypoint_resolves_repository_package(self) -> None:
        repo_root = Path(__file__).parents[1]

        result = subprocess.run(
            [sys.executable, "-m", "jepa.embed_episode", "--help"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_resolves_legacy_episode_rgb_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            rgb = source / "rgb"
            rgb.mkdir()
            expected = [rgb / "frame_000000.png", rgb / "frame_000001.png"]
            for path in reversed(expected):
                path.touch()

            self.assertEqual(ObservationSource.open(source).frame_paths(), expected)

    def test_selects_a_camera_from_a_demo_recording(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / "manifest.json").write_text(
                json.dumps({"cameras": ["presentation", "wrist"]})
            )
            presentation = source / "presentation"
            wrist = source / "wrist"
            presentation.mkdir()
            wrist.mkdir()
            (presentation / "frame_000000.png").touch()
            expected = [wrist / "frame_000000.png", wrist / "frame_000001.png"]
            for path in reversed(expected):
                path.touch()

            self.assertEqual(
                ObservationSource.open(source).frame_paths("wrist"), expected
            )

            with self.assertRaisesRegex(ValueError, "available cameras"):
                ObservationSource.open(source).frame_paths("overhead")

    def test_names_recording_embeddings_for_the_selected_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)
            (source / "manifest.json").write_text(
                json.dumps({"cameras": ["presentation", "wrist"]})
            )

            self.assertEqual(
                ObservationSource.open(source).default_embedding_path("wrist"),
                source / "wrist_vjepa2_embedding.npy",
            )

    def test_preserves_the_legacy_episode_embedding_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir)

            self.assertEqual(
                ObservationSource.open(source).default_embedding_path("wrist"),
                source / "vjepa2_embedding.npy",
            )


class JepaShellTest(unittest.TestCase):
    def test_latest_falls_back_when_recordings_directory_is_absent(self) -> None:
        repo_root = Path(__file__).parents[1]
        with tempfile.TemporaryDirectory() as temp_dir:
            home = Path(temp_dir)
            remote_repo = home / "quantis-robotics"
            (remote_repo / "ops").mkdir(parents=True)
            (remote_repo / "data" / "episodes" / "legacy-episode").mkdir(
                parents=True
            )
            (remote_repo / "ops" / "shell_helpers.sh").symlink_to(
                repo_root / "ops" / "shell_helpers.sh"
            )
            (remote_repo / "ops" / "jepa_common.sh").symlink_to(
                repo_root / "ops" / "jepa_common.sh"
            )
            fake_python = home / ".venvs" / "quantis-jepa" / "bin" / "python"
            fake_python.parent.mkdir(parents=True)
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"${1:-}\" == '-m' && \"${2:-}\" == 'pip' ]]; then exit 0; fi\n"
                "printf '%s\\n' \"$*\"\n"
            )
            fake_python.chmod(0o755)

            result = subprocess.run(
                [str(repo_root / "ops" / "jepa_embed.sh"), "latest", "wrist"],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, "HOME": str(home)},
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(
                str(remote_repo / "data" / "episodes" / "legacy-episode"),
                result.stdout,
            )
            self.assertIn("--camera wrist", result.stdout)


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
