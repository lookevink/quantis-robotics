from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from jepa_wm.unknown_start_reset_lifecycle import (
    claim,
    failure,
    finalize_recovery,
    terminal_paths,
)
from jepa_wm.unknown_start_reset_runtime import (
    authenticate_runtime_source,
    runtime_source_fingerprint,
)
from sim.unknown_start_reset import (
    UNKNOWN_START_RESET_CONTRACT,
    UNKNOWN_START_RESET_EVIDENCE_SCHEMA,
    UnknownStartResetEvidence,
    UnknownStartResetPhase,
    UnknownStartSampleRealization,
    UnknownStartWorkspaceState,
)
from sim.exploration import DatasetSplit


def valid_evidence() -> UnknownStartResetEvidence:
    sample = UNKNOWN_START_RESET_CONTRACT.draw(
        62603,
        forbidden_seeds={62600, 62601, 62602},
    )
    connector_position = tuple(
        baseline + offset
        for baseline, offset in zip(
            UNKNOWN_START_RESET_CONTRACT.workspace.connector_baseline_m,
            sample.scene_offset_m,
        )
    )
    socket_position = tuple(
        baseline + offset
        for baseline, offset in zip(
            UNKNOWN_START_RESET_CONTRACT.workspace.socket_baseline_m,
            sample.scene_offset_m,
        )
    )
    return UnknownStartResetEvidence(
        sample=sample,
        workspace=UnknownStartWorkspaceState(
            connector_position_m=connector_position,
            socket_position_m=socket_position,
            gripper_control_frame_position_m=(
                0.25,
                connector_position[1],
                1.48,
            ),
            socket_scale=1.05,
        ),
        realization=UnknownStartSampleRealization(
            initial_arm_offset_radians=sample.initial_arm_offset_radians,
            camera_offset_m=sample.camera_offset_m,
            light_exposure_delta=sample.light_exposure_delta,
        ),
        observed_arm_positions_radians=(0.1,) * 7,
        observed_gripper_width_m=0.08,
        realized_sample_fingerprint=sample.fingerprint,
        plug_attached=False,
        collision_detected=False,
        contact_force_newtons=0.0,
        direct_state_setting_count=1,
        prefix_replay_frames=0,
        applied_actions=0,
        phase=UnknownStartResetPhase.RESET_AUTHENTICATION,
    )


def rgb_png(width: int, height: int) -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = b"".join(b"\x00" + bytes(width * 3) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


class UnknownStartResetContractTest(unittest.TestCase):
    def test_live_claim_is_exclusive_and_runner_has_no_actuation_seam(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger_root = Path(directory) / "ledger"
            payload = claim(
                ledger_root,
                "unknown-start-reset-v4-62603",
                62603,
                "a" * 40,
                "b" * 64,
            )

            self.assertEqual(payload["evaluations_claimed"], 1)
            self.assertEqual(payload["applied_actions"], 0)
            with self.assertRaisesRegex(ValueError, "already claimed"):
                claim(
                    ledger_root,
                    "unknown-start-reset-v4-62603",
                    62603,
                    "a" * 40,
                    "b" * 64,
                )
            with self.assertRaisesRegex(ValueError, "frozen run"):
                claim(
                    Path(directory) / "different-ledger",
                    "unknown-start-reset-v1-62600",
                    62600,
                    "a" * 40,
                    "b" * 64,
                )
        runner = Path("ops/run_unknown_start_reset.sh").read_text()
        self.assertIn("authenticate_unknown_start_reset", runner)
        self.assertNotIn("apply_control_response", runner)
        self.assertNotIn("control-worker-start", runner)
        self.assertIn("300 true", runner)
        self.assertIn("sudo install -d", runner)
        self.assertIn("sudo chown -R", runner)

    def test_success_is_terminal_only_after_exact_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            recovery = root / "recovery"
            primary.mkdir()
            recovery.mkdir()
            source_revision = "a" * 40
            runtime_fingerprint = "b" * 64
            recording_id = "unknown-start-reset-v4-62603"
            evidence = valid_evidence()
            sample = evidence.sample
            primary_ledger = root / "claims"
            recovery_ledger = root / "recovery-claims"
            claim(
                primary_ledger,
                recording_id,
                sample.seed,
                source_revision,
                runtime_fingerprint,
            )
            primary_claim, _ = terminal_paths(primary_ledger)
            recovery_claim, _ = terminal_paths(recovery_ledger)
            recovery_claim.parent.mkdir(parents=True)
            recovery_claim.write_bytes(primary_claim.read_bytes())
            evidence_contents = json.dumps(evidence.to_dict(), sort_keys=True) + "\n"
            evidence_fingerprint = sha256(evidence_contents.encode()).hexdigest()
            artifacts = {
                "manifest.json": json.dumps(
                    {
                        "schema": "quantis.demo_recording.v9",
                        "recording_id": recording_id,
                        "fps": 4,
                        "frames": 1,
                        "stage_frames": {"approaching_cable": 1},
                        "cameras": ["wrist"],
                        "resolutions": {"wrist": [512, 512]},
                        "metadata": {
                            "task": "unknown_start_reset_authentication",
                            "contract_fingerprint": UNKNOWN_START_RESET_CONTRACT.fingerprint,
                            "sample": sample.to_dict(),
                            "sample_fingerprint": sample.fingerprint,
                            "source_revision": source_revision,
                            "runtime_source_fingerprint": runtime_fingerprint,
                            "applied_actions": 0,
                            "prefix_replay_frames": 0,
                        },
                    }
                ),
                "steps.jsonl": json.dumps(
                    {
                        "index": 0,
                        "phase": "initial",
                        "stage": "approaching_cable",
                        "frames": {"wrist": "wrist/frame_000000.png"},
                        "action_from_previous": None,
                        "plug_attached": False,
                        "collision_detected": False,
                        "contact_force_newtons": 0.0,
                        "arm_positions": list(evidence.observed_arm_positions_radians),
                        "gripper_width_m": evidence.observed_gripper_width_m,
                        "plug_position": list(evidence.workspace.connector_position_m),
                        "gripper_frame_world_position": list(
                            evidence.workspace.gripper_control_frame_position_m
                        ),
                    }
                )
                + "\n",
                "unknown_start_reset_evidence.json": evidence_contents,
                "CAPTURE.json": json.dumps(
                    {
                        "status": "captured",
                        "recording_id": recording_id,
                        "source_revision": source_revision,
                        "runtime_source_fingerprint": runtime_fingerprint,
                        "contract_fingerprint": UNKNOWN_START_RESET_CONTRACT.fingerprint,
                        "sample_fingerprint": sample.fingerprint,
                        "evidence_fingerprint": evidence_fingerprint,
                        "applied_actions": 0,
                    }
                ),
            }
            for name, contents in artifacts.items():
                (primary / name).parent.mkdir(parents=True, exist_ok=True)
                (recovery / name).parent.mkdir(parents=True, exist_ok=True)
                (primary / name).write_text(contents)
                (recovery / name).write_text(contents)
            for recording in (primary, recovery):
                frame = recording / "wrist/frame_000000.png"
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.write_bytes(rgb_png(512, 512))

            payload = finalize_recovery(
                primary,
                recovery,
                primary_claim,
                recovery_claim,
                source_revision,
                runtime_fingerprint,
            )

            self.assertTrue(payload["passed"])
            self.assertTrue(payload["recovery_verified"])
            self.assertEqual(payload["applied_actions"], 0)
            self.assertTrue((primary / "RESULT.json").is_file())
            recovery_payload = json.loads(
                (recovery / "RECOVERY_VERIFIED.json").read_text()
            )
            self.assertEqual(recovery_payload["status"], "recovery_verified")
            self.assertFalse(recovery_payload["passed"])
            self.assertFalse((recovery / "RESULT.json").exists())

    def test_failure_authenticates_and_binds_detailed_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            ledger = data_root / "unknown_start_reset_v4_claims"
            source_revision = "a" * 40
            runtime_fingerprint = "b" * 64
            claim_payload = claim(
                ledger,
                "unknown-start-reset-v4-62603",
                62603,
                source_revision,
                runtime_fingerprint,
            )
            recording = (
                data_root / "recordings" / "unknown-start-reset-v4-62603"
            )
            frame = recording / "wrist/frame_000000.png"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"captured-frame")
            evidence = replace(valid_evidence(), contact_force_newtons=0.01)
            negative = {
                "schema": "quantis.unknown_start_reset_negative.v1",
                "recording_id": claim_payload["recording_id"],
                "source_revision": source_revision,
                "runtime_source_fingerprint": runtime_fingerprint,
                "contract_fingerprint": claim_payload["contract_fingerprint"],
                "sample_fingerprint": claim_payload["sample_fingerprint"],
                "captured_frame": {
                    "path": "wrist/frame_000000.png",
                    "fingerprint": sha256(frame.read_bytes()).hexdigest(),
                },
                "validation_failures": ["contact_force_zero"],
                "evidence": evidence.to_dict(),
            }
            negative_path = recording / "UNKNOWN_START_RESET_NEGATIVE.json"
            negative["source_revision"] = "c" * 40
            negative_path.write_text(json.dumps(negative, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "negative evidence"):
                failure(ledger, "simulator_reset:exit_1")
            self.assertFalse(terminal_paths(ledger)[1].exists())
            negative["source_revision"] = source_revision
            negative["evidence"]["contact_force_newtons"] = 0.0
            negative_path.write_text(json.dumps(negative, sort_keys=True) + "\n")
            with self.assertRaisesRegex(ValueError, "negative evidence"):
                failure(ledger, "simulator_reset:exit_1")
            self.assertFalse(terminal_paths(ledger)[1].exists())
            negative["evidence"]["contact_force_newtons"] = 0.01
            negative_path.write_text(json.dumps(negative, sort_keys=True) + "\n")

            payload = failure(ledger, "simulator_reset:exit_1")

            self.assertEqual(
                payload["negative_evidence_fingerprint"],
                sha256(negative_path.read_bytes()).hexdigest(),
            )

    def test_recovery_rejects_semantically_invalid_evidence(self) -> None:
        source = Path("jepa_wm/unknown_start_reset_lifecycle.py").read_text()

        self.assertIn("UnknownStartResetEvidence.from_dict", source)
        self.assertIn('"steps.jsonl"', source)
        self.assertIn('"wrist/frame_000000.png"', source)
        self.assertIn("_validate_rgb_png", source)
        with self.assertRaises(ValueError):
            UnknownStartResetEvidence.from_dict({})

    def test_frame_specific_evidence_uses_a_distinct_schema(self) -> None:
        payload = valid_evidence().to_dict()

        self.assertEqual(payload["schema"], UNKNOWN_START_RESET_EVIDENCE_SCHEMA)
        self.assertEqual(
            payload["workspace"]["gripper_control_frame"],
            "right_gripper_control_frame",
        )
        payload["schema"] = "quantis.unknown_start_reset_evidence.v1"
        with self.assertRaisesRegex(ValueError, "evidence payload is invalid"):
            UnknownStartResetEvidence.from_dict(payload)

    def test_validation_identifies_the_exact_rejected_invariant(self) -> None:
        evidence = replace(valid_evidence(), contact_force_newtons=0.01)

        self.assertEqual(
            evidence.validation_failures(UNKNOWN_START_RESET_CONTRACT),
            ("contact_force_zero",),
        )
        with self.assertRaisesRegex(ValueError, "contact_force_zero"):
            evidence.validate(UNKNOWN_START_RESET_CONTRACT)

    def test_runtime_source_roster_authenticates_exact_bytes(self) -> None:
        fingerprint = runtime_source_fingerprint()

        self.assertEqual(authenticate_runtime_source(fingerprint), fingerprint)
        with self.assertRaisesRegex(ValueError, "source changed"):
            authenticate_runtime_source("f" * 64)

    def test_runtime_has_one_reset_and_no_action_application(self) -> None:
        source = Path("sim/isaac_unknown_start_reset.py").read_text()

        self.assertEqual(source.count("actuators.set_reset_state("), 1)
        self.assertNotIn("apply_control_response", source)
        self.assertNotIn("move_joint_command_over_physics_steps", source)
        self.assertNotIn("control-worker", source)
        self.assertIn("hand_collision or plug_collision", source)
        self.assertIn("realization_tolerances.light_exposure_delta", source)
        self.assertIn("workspace.realization_scale_tolerance", source)
        self.assertIn("authored = actuators.current_command()", source)
        self.assertIn("snapshot.gripper_frame_world_position", source)
        self.assertIn('"UNKNOWN_START_RESET_NEGATIVE.json"', source)
        self.assertNotIn("plan.light_exposure_delta) > 1e-9", source)
        self.assertNotIn("atol=1e-12", source)

    def test_versioned_run_descriptor_is_the_single_shell_identity_source(self) -> None:
        lifecycle = Path("jepa_wm/unknown_start_reset_lifecycle.py").read_text()
        runner = Path("ops/run_unknown_start_reset.sh").read_text()
        aws = Path("ops/aws.sh").read_text()

        self.assertIn("UNKNOWN_START_RESET_LEDGER_NAME", lifecycle)
        self.assertIn("describe --field recording-id", runner)
        self.assertIn("describe --field ledger-name", runner)
        self.assertIn("describe --field recording-id", aws)
        self.assertIn("describe --field claim-name", aws)
        self.assertNotIn("unknown-start-reset-v4-62603", runner)

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
        evidence = valid_evidence()

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
                "schema": "quantis.unknown_start_reset_contract.v2",
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
                    "initial_gripper_control_frame": {
                        "frame": "right_gripper_control_frame",
                        "bounds": {
                            "x": [0.22, 0.28],
                            "y": [-0.3, -0.2],
                            "z": [1.43, 1.53],
                        },
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
