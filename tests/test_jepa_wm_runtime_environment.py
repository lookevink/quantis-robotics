from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from jepa_wm.runtime_environment import (
    claim_model_load_preflight,
    validate_headless_runtime,
)


class HeadlessRuntimeEnvironmentTest(unittest.TestCase):
    def _initialize_repository(self, path: Path) -> str:
        path.mkdir(parents=True, exist_ok=True)
        (path / "README.md").write_text("fixture")
        subprocess.run(("git", "init", "-q", str(path)), check=True)
        subprocess.run(
            ("git", "-C", str(path), "config", "user.email", "test@example.com"),
            check=True,
        )
        subprocess.run(
            ("git", "-C", str(path), "config", "user.name", "Test"),
            check=True,
        )
        subprocess.run(("git", "-C", str(path), "add", "."), check=True)
        subprocess.run(
            ("git", "-C", str(path), "commit", "-qm", "fixture"), check=True
        )
        return subprocess.run(
            ("git", "-C", str(path), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def test_validates_the_pinned_dino_source_and_checkpoint_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "jepa-wm"
            source = root / "source" / "jepa-wms"
            checkpoint = root / "checkpoints" / "jepa_wm_droid.pth.tar"
            dino_source = root / "source" / "dinov3" / "hubconf.py"
            dino_checkpoint = (
                root
                / "checkpoints"
                / "dinov3"
                / "dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
            )
            approved_checkpoint = dino_checkpoint.with_name(
                "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            )
            cached_checkpoint = (
                root
                / "cache"
                / "torch"
                / "hub"
                / "checkpoints"
                / approved_checkpoint.name
            )
            for path in (checkpoint, approved_checkpoint):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("evidence")
            dino_checkpoint.symlink_to(approved_checkpoint)
            cached_checkpoint.parent.mkdir(parents=True)
            cached_checkpoint.symlink_to(approved_checkpoint)
            dino_source.parent.mkdir(parents=True)
            dino_source.write_text("hub")
            jepa_revision = self._initialize_repository(source)
            dinov3_revision = self._initialize_repository(dino_source.parent)
            environment = {
                "JEPAWM_HOME": str(root / "source"),
                "JEPAWM_OSSCKPT": str(root / "checkpoints"),
                "TORCH_HOME": str(root / "cache" / "torch"),
                "JEPA_WM_REVISION": jepa_revision,
                "DINOV3_REVISION": dinov3_revision,
            }

            evidence = validate_headless_runtime(
                source,
                checkpoint,
                environment=environment,
                expected_dinov3_fingerprint=sha256(b"evidence").hexdigest(),
                expected_checkpoint_fingerprint=sha256(b"evidence").hexdigest(),
            )

            self.assertEqual(
                evidence["dinov3_source"], str(dino_source.parent.resolve())
            )
            self.assertEqual(
                evidence["dinov3_checkpoint"],
                str(
                    root.resolve()
                    / "checkpoints"
                    / "dinov3"
                    / dino_checkpoint.name
                ),
            )
            (source / "untracked.py").write_text("raise RuntimeError")
            with self.assertRaisesRegex(ValueError, "source revision changed"):
                validate_headless_runtime(
                    source,
                    checkpoint,
                    environment=environment,
                    expected_dinov3_fingerprint=sha256(b"evidence").hexdigest(),
                    expected_checkpoint_fingerprint=sha256(b"evidence").hexdigest(),
                )

    def test_rejects_a_divergent_dino_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "jepa-wm"
            source = root / "source" / "jepa-wms"
            checkpoint = root / "checkpoints" / "jepa_wm_droid.pth.tar"
            dino_source = root / "source" / "dinov3"
            approved = (
                root
                / "checkpoints"
                / "dinov3"
                / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            )
            stale = approved.with_name(
                "dinov3_vitl16_pretrain_lvd1689m-7c1da9a5.pth"
            )
            cached = root / "cache" / "torch" / "hub" / "checkpoints" / approved.name
            for path, contents in (
                (checkpoint, "base"),
                (approved, "approved"),
                (stale, "wrong"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents)
            cached.parent.mkdir(parents=True)
            cached.symlink_to(approved)
            dino_source.mkdir(parents=True)
            (dino_source / "hubconf.py").write_text("hub")
            environment = {
                "JEPAWM_HOME": str(root / "source"),
                "JEPAWM_OSSCKPT": str(root / "checkpoints"),
                "TORCH_HOME": str(root / "cache" / "torch"),
                "JEPA_WM_REVISION": self._initialize_repository(source),
                "DINOV3_REVISION": self._initialize_repository(dino_source),
            }

            with self.assertRaisesRegex(ValueError, "aliases"):
                validate_headless_runtime(
                    source,
                    checkpoint,
                    environment=environment,
                    expected_dinov3_fingerprint=sha256(b"approved").hexdigest(),
                    expected_checkpoint_fingerprint=sha256(b"base").hexdigest(),
                )

    def test_rejects_a_bare_python_environment_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "jepa-wm"
            source = root / "source" / "jepa-wms"
            checkpoint = root / "checkpoints" / "jepa_wm_droid.pth.tar"

            with self.assertRaisesRegex(ValueError, "JEPAWM_HOME"):
                validate_headless_runtime(source, checkpoint, environment={})

    def test_model_load_preflight_claim_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "runtime-preflight-v1.json"

            claim, payload = claim_model_load_preflight(output)

            self.assertTrue(claim.is_file())
            self.assertEqual(payload["output"], str(output.resolve()))
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim_model_load_preflight(output)


class JepaWmRuntimeShellTest(unittest.TestCase):
    def test_smoke_and_load_preflight_use_the_module_runtime_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            repository = home / "quantis-robotics"
            ops = repository / "ops"
            ops.mkdir(parents=True)
            shutil.copy("ops/jepa_wm.sh", ops / "jepa_wm.sh")
            shutil.copy("ops/shell_helpers.sh", ops / "shell_helpers.sh")
            runtime = home / "docker" / "jepa-wm"
            for path in (
                runtime / "source" / "jepa-wms" / "README.md",
                runtime / "source" / "dinov3" / "README.md",
                runtime / "checkpoints" / "jepa_wm_droid.pth.tar",
                runtime
                / "checkpoints"
                / "dinov3"
                / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("evidence")
            binary = home / "bin"
            binary.mkdir()
            log = home / "python.log"
            (binary / "git").write_text(
                "#!/usr/bin/env bash\n"
                "if [[ \"$2\" == *dinov3 ]]; then\n"
                "  echo 6876159a11b4df116f30f667f8c9888617df0751\n"
                "else\n"
                "  echo 13cf1d9c7e476f53c17714d2e0f1dc239a883ce0\n"
                "fi\n"
            )
            (binary / "git").chmod(0o755)
            python = home / ".venvs" / "quantis-jepa-wm" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text(
                "#!/usr/bin/env bash\n"
                f"printf '%s|%s|%s|%s\\n' \"$PWD\" \"$*\" "
                f"\"$JEPAWM_HOME\" \"$JEPAWM_OSSCKPT\" >> {log}\n"
            )
            python.chmod(0o755)
            environment = {
                **os.environ,
                "HOME": str(home),
                "PATH": f"{binary}:{os.environ['PATH']}",
            }

            smoke = subprocess.run(
                ("bash", str(ops / "jepa_wm.sh"), "smoke"),
                cwd=home,
                env=environment,
                capture_output=True,
                text=True,
            )
            preflight = subprocess.run(
                ("bash", str(ops / "jepa_wm.sh"), "model-load-preflight"),
                cwd=home,
                env=environment,
                capture_output=True,
                text=True,
            )
            held_out = subprocess.run(
                (
                    "bash",
                    str(ops / "jepa_wm.sh"),
                    "physical-state-residual-held-out",
                    "--sentinel",
                ),
                cwd=home,
                env=environment,
                capture_output=True,
                text=True,
            )
            held_out_v2 = subprocess.run(
                (
                    "bash",
                    str(ops / "jepa_wm.sh"),
                    "physical-state-residual-held-out-v2",
                    "--sentinel-v2",
                ),
                cwd=home,
                env=environment,
                capture_output=True,
                text=True,
            )

            self.assertEqual(smoke.returncode, 0, smoke.stderr)
            self.assertEqual(preflight.returncode, 0, preflight.stderr)
            self.assertEqual(held_out.returncode, 0, held_out.stderr)
            self.assertEqual(held_out_v2.returncode, 0, held_out_v2.stderr)
            calls = log.read_text().splitlines()
            self.assertEqual(len(calls), 4)
            self.assertIn(
                f"{repository}|-m jepa_wm.smoke --source",
                calls[0],
            )
            self.assertIn("--load-only", calls[1])
            self.assertIn("runtime-preflight-v1.json", calls[1])
            self.assertIn(
                "-m jepa_wm.physical_residual_held_out --sentinel",
                calls[2],
            )
            self.assertIn(
                "-m jepa_wm.physical_residual_held_out_v2 --sentinel-v2",
                calls[3],
            )
            self.assertTrue(
                all(
                    f"|{runtime / 'source'}|{runtime / 'checkpoints'}" in call
                    for call in calls
                )
            )


if __name__ == "__main__":
    unittest.main()
