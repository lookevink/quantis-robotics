from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sim.demo_behavior import current_demo_behavioral_contract
from sim.demo_run import (
    DemoArtifactBinding,
    DemoArtifactRole,
    DemoCorpusEntry,
    DemoRunSpec,
    DemoRunSelection,
    DemoWorkerBinding,
    DemoWorkerRole,
    validate_demo_run_spec,
)
from sim.exploration import DatasetSplit
from jepa_wm.worker_artifacts import ControlWorkerArtifacts
from jepa_wm.action import ACTION_RECORDING_CONTRACT
from jepa_wm.insertion_contract import (
    CONTACT_INSERTION_RECORDING,
    INSERTION_TASK_ID,
)
from sim.exploration import DOMAIN_DATASET_ID


def build_demo_corpus_entry(
    root: Path,
    recording: str,
    split: DatasetSplit,
    seed: int,
) -> DemoCorpusEntry:
    path = root / recording
    path.mkdir()
    wrist = path / "wrist"
    wrist.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "quantis.demo_recording.v9",
                "recording_id": recording,
                "fps": 4,
                "frames": CONTACT_INSERTION_RECORDING.frame_count,
                "cameras": ["wrist"],
                "action": ACTION_RECORDING_CONTRACT.to_dict(),
                "metadata": {
                    "dataset": DOMAIN_DATASET_ID,
                    "split": split.value,
                    "seed": seed,
                    "task": INSERTION_TASK_ID,
                    "insertion_target": {
                        "socket_position": [-0.10, 0.0, 0.0],
                        "socket_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "insertion_axis": [-1.0, 0.0, 0.0],
                        "grasp_offset_meters": 0.04,
                        "evidence_mode": "contact_aware_scripted_baseline",
                        **CONTACT_INSERTION_RECORDING.instrumentation_metadata(
                            ("latch",)
                        ),
                    },
                },
            }
        )
    )
    step_lines = []
    for index, (phase, stage, attached) in enumerate(
        zip(
            CONTACT_INSERTION_RECORDING.phase_roster,
            CONTACT_INSERTION_RECORDING.stage_roster,
            CONTACT_INSERTION_RECORDING.attachment_roster,
        )
    ):
        tip_x = -0.10 if stage == "plug_seated" else -0.08 if attached else 0.0
        hand_x = tip_x + 0.04
        frame = wrist / f"frame_{index:06d}.png"
        frame.write_bytes(f"frame-{seed}-{index}".encode())
        step_lines.append(
            json.dumps(
                {
                    "index": index,
                    "simulation_time_seconds": index * 0.25,
                    "phase": phase,
                    "stage": stage,
                    "frames": {"wrist": f"wrist/{frame.name}"},
                    "plug_position": [tip_x, 0.0, 0.0],
                    "plug_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "end_effector_world_position": [hand_x, 0.0, 0.0],
                    "gripper_frame_world_position": [hand_x, 0.0, 0.0],
                    "plug_attached": attached,
                    "collision_detected": False,
                    "contact_force_newtons": (0.5 if stage == "plug_seated" else 0.0),
                    "arm_tracking_error_rad": 0.001,
                    "gripper_tracking_error_m": 0.0002,
                }
            )
        )
    (path / "steps.jsonl").write_text("\n".join(step_lines) + "\n")
    return DemoCorpusEntry.from_recording(path)


def build_demo_run_spec(root: Path) -> DemoRunSpec:
    proposal = root / "contact-grasp-v1.pth"
    proposal.write_bytes(b"frozen proposal")
    proposal.with_suffix(".pth.json").write_text(
        json.dumps({"proposal_fingerprint": sha256(proposal.read_bytes()).hexdigest()})
    )
    adapter = root / "world-model-adapter.pth"
    adapter.write_bytes(b"frozen world model adapter")
    stage = root / "demo.usda"
    stage.write_text("frozen stage")
    corpus = tuple(
        build_demo_corpus_entry(
            root,
            f"contact-train-{index:02d}",
            DatasetSplit.TRAIN,
            2600 + index,
        )
        for index in range(12)
    ) + tuple(
        build_demo_corpus_entry(
            root,
            f"contact-held-{index:02d}",
            DatasetSplit.HELD_OUT,
            12600 + index,
        )
        for index in range(2)
    )
    grasp_manifest = root / "grasp-worker.worker.json"
    insertion_manifest = root / "insertion-worker.worker.json"
    ControlWorkerArtifacts(proposal.resolve(), adapter.resolve()).write(grasp_manifest)
    ControlWorkerArtifacts(proposal.resolve(), adapter.resolve()).write(
        insertion_manifest
    )
    return DemoRunSpec(
        source_revision="1" * 40,
        container_image_digest=f"sha256:{'2' * 64}",
        corpus=corpus,
        artifacts=(
            DemoArtifactBinding.from_artifact(DemoArtifactRole.STAGE_ASSET, stage),
        ),
        workers=(
            DemoWorkerBinding.from_manifest(
                DemoWorkerRole.GRASP, "grasp-worker", grasp_manifest
            ),
            DemoWorkerBinding.from_manifest(
                DemoWorkerRole.INSERTION, "insertion-worker", insertion_manifest
            ),
        ),
        selection=DemoRunSelection.from_reference(
            root / corpus[12].recording,
            corpus[12].seed,
        ),
        behavior=current_demo_behavioral_contract(),
    )


class DemoRunSpecTest(unittest.TestCase):
    def _spec(self, root: Path) -> DemoRunSpec:
        return build_demo_run_spec(root)

    def test_freezes_the_complete_authenticated_demo_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self._spec(Path(temp_dir))

            self.assertEqual(len(spec.corpus), 14)
            self.assertEqual(
                spec.fingerprint, DemoRunSpec.from_dict(spec.to_dict()).fingerprint
            )
            self.assertEqual(spec.to_dict()["schema"], "quantis.demo_run_spec.v1")
            self.assertEqual(spec.terminal_contract.grasp_actions, 192)
            self.assertEqual(spec.terminal_contract.insertion_actions, 168)
            self.assertEqual(spec.action_cap, 360)
            self.assertTrue(spec.terminal_contract.require_seated_hold)
            spec_path = Path(temp_dir) / "demo-runs" / "frozen.json"
            spec.persist(spec_path)
            self.assertEqual(
                DemoRunSpec.from_dict(json.loads(spec_path.read_text())),
                spec,
            )

    def test_any_contract_change_changes_the_run_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self._spec(Path(temp_dir))

            self.assertNotEqual(
                spec.fingerprint,
                replace(
                    spec,
                    behavior=replace(
                        spec.behavior,
                        camera_configuration_fingerprint="8" * 64,
                    ),
                ).fingerprint,
            )

    def test_behavioral_contract_binds_ik_precision(self) -> None:
        original = current_demo_behavioral_contract()

        with patch(
            "sim.demo_behavior.IK_ACTIVE_ROTATION_TOLERANCE_RADIANS",
            0.0005,
        ):
            changed = current_demo_behavioral_contract()

        self.assertNotEqual(
            original.safety_limits_fingerprint,
            changed.safety_limits_fingerprint,
        )
        self.assertEqual(
            original.control_policy_fingerprint,
            changed.control_policy_fingerprint,
        )

        with patch(
            "sim.demo_behavior.IK_ORIENTATION_HOLD_TOLERANCE_RADIANS",
            0.00075,
        ):
            changed_hold = current_demo_behavioral_contract()

        self.assertNotEqual(
            original.safety_limits_fingerprint,
            changed_hold.safety_limits_fingerprint,
        )

    def test_rejects_an_incomplete_or_overlapping_corpus_roster(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self._spec(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "12 TRAIN and two HELD_OUT"):
                replace(spec, corpus=spec.corpus[:-1])
            with self.assertRaisesRegex(ValueError, "corpus roster"):
                replace(spec, corpus=spec.corpus[:-1] + (spec.corpus[-2],))

    def test_rejects_artifact_bytes_changed_after_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            spec = self._spec(Path(temp_dir))
            payload = spec.to_dict()
            spec.artifacts[0].identity.path.write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
                DemoRunSpec.from_dict(payload)

    def test_reauthenticates_every_corpus_recording_before_a_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._spec(root)
            spec.authenticate_corpus(root)
            (root / spec.corpus[0].recording / "steps.jsonl").write_text("changed\n")

            with self.assertRaisesRegex(ValueError, "corpus fingerprint"):
                spec.authenticate_corpus(root)

    def test_rejects_changed_selected_context_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._spec(root)
            selected = root / spec.selection.reference_recording
            (selected / "wrist" / "frame_000110.png").write_bytes(b"changed RGB")

            with self.assertRaisesRegex(ValueError, "corpus fingerprint"):
                spec.authenticate_corpus(root)

    def test_rejects_a_worker_dependency_changed_after_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._spec(root)
            payload = spec.to_dict()
            spec.workers[0].adapter.path.write_bytes(b"changed adapter")

            with self.assertRaisesRegex(ValueError, "worker binding"):
                DemoRunSpec.from_dict(payload)

    def test_live_preflight_requires_the_exact_frozen_spec_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._spec(root)
            spec_path = root / "demo-spec.json"
            spec_path.write_text(json.dumps(spec.to_dict()))
            required_artifacts = {
                DemoArtifactRole.STAGE_ASSET: spec.artifacts[0].identity.path,
            }
            required_workers = {
                binding.role: (binding.identity, binding.manifest.path)
                for binding in spec.workers
            }

            self.assertEqual(
                validate_demo_run_spec(
                    spec_path,
                    expected_fingerprint=spec.fingerprint,
                    recording_root=root,
                    source_revision=spec.source_revision,
                    container_image_digest=spec.container_image_digest,
                    required_artifacts=required_artifacts,
                    required_workers=required_workers,
                    reference_recording=spec.selection.reference_recording,
                    exploration_seed=spec.selection.exploration_seed,
                    grasp_actions=192,
                    insertion_actions=168,
                ),
                spec,
            )
            with self.assertRaisesRegex(ValueError, "frozen fingerprint"):
                validate_demo_run_spec(
                    spec_path,
                    expected_fingerprint="f" * 64,
                    recording_root=root,
                    source_revision=spec.source_revision,
                    container_image_digest=spec.container_image_digest,
                    required_artifacts=required_artifacts,
                    required_workers=required_workers,
                    reference_recording=spec.selection.reference_recording,
                    exploration_seed=spec.selection.exploration_seed,
                    grasp_actions=192,
                    insertion_actions=168,
                )

    def test_live_preflight_rejects_stale_behavior_or_action_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = self._spec(root)
            stale = replace(
                spec,
                behavior=replace(spec.behavior, safety_limits_fingerprint="f" * 64),
            )
            spec_path = root / "stale-demo-spec.json"
            spec_path.write_text(json.dumps(stale.to_dict()))
            required_artifacts = {
                DemoArtifactRole.STAGE_ASSET: stale.artifacts[0].identity.path,
            }
            required_workers = {
                binding.role: (binding.identity, binding.manifest.path)
                for binding in stale.workers
            }

            with self.assertRaisesRegex(ValueError, "behavioral contract"):
                validate_demo_run_spec(
                    spec_path,
                    expected_fingerprint=stale.fingerprint,
                    recording_root=root,
                    source_revision=stale.source_revision,
                    container_image_digest=stale.container_image_digest,
                    required_artifacts=required_artifacts,
                    required_workers=required_workers,
                    reference_recording=stale.selection.reference_recording,
                    exploration_seed=stale.selection.exploration_seed,
                    grasp_actions=192,
                    insertion_actions=168,
                )

            spec_path.write_text(json.dumps(spec.to_dict()))
            with self.assertRaisesRegex(ValueError, "action allocation"):
                validate_demo_run_spec(
                    spec_path,
                    expected_fingerprint=spec.fingerprint,
                    recording_root=root,
                    source_revision=spec.source_revision,
                    container_image_digest=spec.container_image_digest,
                    required_artifacts=required_artifacts,
                    required_workers=required_workers,
                    reference_recording=spec.selection.reference_recording,
                    exploration_seed=spec.selection.exploration_seed,
                    grasp_actions=8,
                    insertion_actions=168,
                )

            with self.assertRaisesRegex(ValueError, "selection"):
                validate_demo_run_spec(
                    spec_path,
                    expected_fingerprint=spec.fingerprint,
                    recording_root=root,
                    source_revision=spec.source_revision,
                    container_image_digest=spec.container_image_digest,
                    required_artifacts=required_artifacts,
                    required_workers=required_workers,
                    reference_recording=spec.corpus[-1].recording,
                    exploration_seed=spec.corpus[-1].seed,
                    grasp_actions=192,
                    insertion_actions=168,
                )


if __name__ == "__main__":
    unittest.main()
