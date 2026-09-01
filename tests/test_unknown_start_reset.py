from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from jepa_wm.unknown_start_reset_lifecycle import claim, finalize_recovery
from sim.unknown_start_reset import (
    UNKNOWN_START_RESET_CONTRACT,
    UnknownStartResetEvidence,
    UnknownStartResetPhase,
    UnknownStartSampleRealization,
    UnknownStartWorkspaceState,
)
from sim.exploration import DatasetSplit


class UnknownStartResetContractTest(unittest.TestCase):
    def test_live_claim_is_exclusive_and_runner_has_no_actuation_seam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "claim.json"
            payload = claim(
                path,
                "unknown-start-reset-v1-62600",
                62600,
                "a" * 40,
            )

            self.assertEqual(payload["evaluations_claimed"], 1)
            self.assertEqual(payload["applied_actions"], 0)
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim(path, "unknown-start-reset-v1-62600", 62600, "a" * 40)
        runner = Path("ops/run_unknown_start_reset.sh").read_text()
        self.assertIn("authenticate_unknown_start_reset", runner)
        self.assertNotIn("apply_control_response", runner)
        self.assertNotIn("control-worker-start", runner)

    def test_success_is_terminal_only_after_exact_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            recovery = root / "recovery"
            primary.mkdir()
            recovery.mkdir()
            source_revision = "a" * 40
            recording_id = "unknown-start-reset-v1-62600"
            sample = UNKNOWN_START_RESET_CONTRACT.draw(
                62600,
                forbidden_seeds=set(),
            )
            primary_claim = root / "claims" / "claim.json"
            recovery_claim = root / "recovery-claims" / "claim.json"
            claim(primary_claim, recording_id, sample.seed, source_revision)
            recovery_claim.parent.mkdir()
            recovery_claim.write_bytes(primary_claim.read_bytes())
            artifacts = {
                "manifest.json": "{}\n",
                "unknown_start_reset_evidence.json": "{}\n",
                "CAPTURE.json": json.dumps(
                    {
                        "status": "captured",
                        "recording_id": recording_id,
                        "source_revision": source_revision,
                        "contract_fingerprint": UNKNOWN_START_RESET_CONTRACT.fingerprint,
                        "sample_fingerprint": sample.fingerprint,
                        "applied_actions": 0,
                    }
                ),
            }
            for name, contents in artifacts.items():
                (primary / name).write_text(contents)
                (recovery / name).write_text(contents)

            payload = finalize_recovery(
                primary,
                recovery,
                primary_claim,
                recovery_claim,
                source_revision,
            )

            self.assertTrue(payload["passed"])
            self.assertTrue(payload["recovery_verified"])
            self.assertEqual(payload["applied_actions"], 0)
            self.assertEqual(
                (primary / "RESULT.json").read_bytes(),
                (recovery / "RESULT.json").read_bytes(),
            )

    def test_runtime_has_one_reset_and_no_action_application(self) -> None:
        source = Path("sim/isaac_unknown_start_reset.py").read_text()

        self.assertEqual(source.count("actuators.set_reset_state("), 1)
        self.assertNotIn("apply_control_response", source)
        self.assertNotIn("move_joint_command_over_physics_steps", source)
        self.assertNotIn("control-worker", source)

    def test_reserved_seed_draw_is_deterministic_and_bounded(self) -> None:
        sample = UNKNOWN_START_RESET_CONTRACT.draw(62600, forbidden_seeds={12600, 12601})

        self.assertEqual(
            sample.initial_arm_offset_radians,
            (0.003102, -0.004129, -0.002304, 0.002482, 0.006534, -0.009672, 0.00963),
        )
        self.assertEqual(sample.camera_offset_m, (0.00019, -0.002879, 0.010381))
        self.assertEqual(sample.scene_offset_m, (0.0, 0.002437, 0.003126))
        self.assertEqual(sample.socket_scale, 1.05)
        self.assertEqual(sample.light_exposure_delta, 0.370041)
        self.assertIs(sample.split, DatasetSplit.HELD_OUT)
        self.assertEqual(sample.seed, 62600)

    def test_draw_rejects_unreserved_or_previously_used_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved seed"):
            UNKNOWN_START_RESET_CONTRACT.draw(12600, forbidden_seeds=set())
        with self.assertRaisesRegex(ValueError, "already used"):
            UNKNOWN_START_RESET_CONTRACT.draw(62600, forbidden_seeds={62600})

    def test_reset_evidence_requires_exact_safe_unattached_realization(self) -> None:
        sample = UNKNOWN_START_RESET_CONTRACT.draw(62600, forbidden_seeds=set())
        evidence = UnknownStartResetEvidence(
            sample=sample,
            workspace=UnknownStartWorkspaceState(
                connector_position_m=(-0.0256, -0.247813, 1.323126),
                socket_position_m=(-0.071, -0.247563, 1.323126),
                end_effector_position_m=(0.25, -0.247813, 1.48),
                socket_scale=1.05,
            ),
            realization=UnknownStartSampleRealization(
                initial_arm_offset_radians=sample.initial_arm_offset_radians,
                camera_offset_m=sample.camera_offset_m,
                light_exposure_delta=sample.light_exposure_delta,
            ),
            realized_sample_fingerprint=sample.fingerprint,
            plug_attached=False,
            collision_detected=False,
            contact_force_newtons=0.0,
            direct_state_setting_count=1,
            prefix_replay_frames=0,
            applied_actions=0,
            phase=UnknownStartResetPhase.RESET_AUTHENTICATION,
        )

        evidence.validate(UNKNOWN_START_RESET_CONTRACT)
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(evidence, plug_attached=True).validate(UNKNOWN_START_RESET_CONTRACT)
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(evidence, prefix_replay_frames=1).validate(
                UNKNOWN_START_RESET_CONTRACT
            )
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(evidence, realized_sample_fingerprint="f" * 64).validate(
                UNKNOWN_START_RESET_CONTRACT
            )
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(evidence, direct_state_setting_count=2).validate(
                UNKNOWN_START_RESET_CONTRACT
            )
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(evidence, applied_actions=1).validate(
                UNKNOWN_START_RESET_CONTRACT
            )
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(
                evidence,
                workspace=replace(
                    evidence.workspace,
                    connector_position_m=(-0.0256, -0.247, 1.323126),
                ),
            ).validate(UNKNOWN_START_RESET_CONTRACT)
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(
                evidence,
                workspace=replace(evidence.workspace, socket_scale=1.0),
            ).validate(UNKNOWN_START_RESET_CONTRACT)
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(
                evidence,
                realization=replace(
                    evidence.realization,
                    initial_arm_offset_radians=(0.0,) * 7,
                ),
            ).validate(UNKNOWN_START_RESET_CONTRACT)
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(
                evidence,
                realization=replace(
                    evidence.realization,
                    camera_offset_m=(0.0,) * 3,
                ),
            ).validate(UNKNOWN_START_RESET_CONTRACT)
        with self.assertRaisesRegex(ValueError, "unsafe or inauthentic"):
            replace(
                evidence,
                realization=replace(
                    evidence.realization,
                    light_exposure_delta=0.0,
                ),
            ).validate(UNKNOWN_START_RESET_CONTRACT)

    def test_contract_freezes_distribution_and_authority_boundary(self) -> None:
        self.assertEqual(
            UNKNOWN_START_RESET_CONTRACT.to_dict(),
            {
                "schema": "quantis.unknown_start_reset_contract.v1",
                "seed_namespace": {"minimum": 62600, "maximum": 62699},
                "split": "held_out",
                "sampler": "build_exploration_plan.v1",
                "sampler_source_fingerprint": (
                    "0ec746dbf12fbed61c66b3c64dee6717fa15f015be4aad23e0f40e5b47a5228d"
                ),
                "bounds": {
                    "initial_arm_offset_radians": [-0.01, 0.01],
                    "camera_offset_m": [-0.012, 0.012],
                    "scene_offset_m": {
                        "x": [0.0, 0.0],
                        "y": [-0.025, 0.025],
                        "z": [-0.015, 0.015],
                    },
                    "socket_scale": [1.05, 1.05],
                    "light_exposure_delta": [-0.4, 0.4],
                },
                "sample_realization_tolerances": {
                    "initial_arm_offset_radians": 1e-5,
                    "camera_offset_m": 1e-6,
                    "light_exposure_delta": 1e-6,
                },
                "workspace_bounds_m": {
                    "baseline_m": {
                        "connector": [-0.0256, -0.25025, 1.32],
                        "socket": [-0.071, -0.25, 1.32],
                    },
                    "connector": {
                        "x": [-0.0256, -0.0256],
                        "y": [-0.27525, -0.22525],
                        "z": [1.305, 1.335],
                    },
                    "socket": {
                        "x": [-0.071, -0.071],
                        "y": [-0.275, -0.225],
                        "z": [1.305, 1.335],
                    },
                    "initial_end_effector": {
                        "x": [0.22, 0.28],
                        "y": [-0.3, -0.2],
                        "z": [1.43, 1.53],
                    },
                    "realization_tolerances": {
                        "position_m": 1e-5,
                        "socket_scale": 1e-9,
                    },
                },
                "initialization": "direct_state_setting_once",
                "direct_state_setting_count": 1,
                "prefix_replay_frames": 0,
                "runtime_motion": "drive_only",
                "maximum_initial_contact_force_newtons": 0.0,
                "require_unattached": True,
                "require_collision_free": True,
                "authority": {
                    "reset_authentication_only": True,
                    "apply_actions": False,
                    "train": False,
                    "film": False,
                },
            },
        )


if __name__ == "__main__":
    unittest.main()
