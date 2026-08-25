from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:
    torch = None

from jepa_wm.action import ACTION_RECORDING_CONTRACT, DroidAction
from jepa_wm.contract import MODEL_ID
from jepa_wm.training_artifact import (
    TrainingArtifactMetadata,
    artifact_fingerprint,
    rollout_training_selection_fingerprint,
    training_configuration_fingerprint,
)
from jepa_wm.trajectory import DROID_ROLLOUT_PROTOCOL

if torch is not None:
    from jepa_wm.adapter import (
        LEGACY_ADAPTER_SCHEMA,
        ActionAdapterContract,
        save_action_adapter,
    )
    from jepa_wm.insertion_adapter_profile import InsertionAdapterProfile
    from jepa_wm.insertion_wm_readiness import (
        INSERTION_BOUNDS,
        INSERTION_WINDOW,
        validate_insertion_adapter,
        validate_insertion_adapter_evaluation,
    )


@unittest.skipIf(torch is None, "PyTorch is required for adapter binding")
class InsertionWorldModelReadinessTest(unittest.TestCase):
    @staticmethod
    def _adapter(
        root: Path,
        *,
        minimum_goal_cosine: float | None = None,
        noise_policy: dict | None = None,
        legacy: bool = False,
    ):
        adapter = root / "insertion-adapter.pth"
        recordings = tuple(f"insertion-train-{index:02d}" for index in range(12))
        metadata = TrainingArtifactMetadata(
            "jepa_wm_droid",
            "revision",
            "wrist",
            recordings,
            500,
        )
        model = SimpleNamespace(
            model=SimpleNamespace(
                predictor=SimpleNamespace(
                    action_encoder=torch.nn.Linear(7, 4, bias=False)
                )
            )
        )
        selection = {
            "window": INSERTION_WINDOW.to_dict(),
            "selection_bounds": INSERTION_BOUNDS.to_dict(),
            "recording_selections": [
                {
                    "recording": recording,
                    "context_indices": list(INSERTION_WINDOW.context_indices),
                }
                for recording in recordings
            ],
            "rollouts": 12 * INSERTION_WINDOW.count,
        }
        selection_fingerprint = rollout_training_selection_fingerprint(selection)
        candidate_mining = {
            "candidates_per_rollout": 4,
            "scoring_batch_size": 2,
            "noise_scale": 0.25,
            "bounds": {
                "maximum_translation_norm": 0.02,
                "maximum_rotation_norm": 0.08,
                "maximum_gripper_delta": 0.25,
            },
            "minimum_goal_cosine": minimum_goal_cosine,
            "first_action_activity": {
                "translation_norm": (
                    1e-5 if minimum_goal_cosine is not None else 0.001
                ),
                "rotation_norm": (
                    1e-5 if minimum_goal_cosine is not None else 0.005
                ),
                "gripper_delta": (
                    0.005 if minimum_goal_cosine is not None else 0.02
                ),
            },
            **({"noise_policy": noise_policy} if noise_policy else {}),
        }
        config = {"candidate_mining": candidate_mining}
        config_fingerprint = training_configuration_fingerprint(config)
        if legacy:
            adapter.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": LEGACY_ADAPTER_SCHEMA,
                    "metadata": metadata.to_dict(),
                    "training_selection_fingerprint": selection_fingerprint,
                    "action_encoder": model.model.predictor.action_encoder.state_dict(),
                },
                adapter,
            )
        else:
            save_action_adapter(
                model,
                adapter,
                ActionAdapterContract.current(
                    metadata,
                    training_selection_fingerprint=selection_fingerprint,
                    training_config_fingerprint=config_fingerprint,
                ),
            )
        Path(f"{adapter}.json").write_text(
            json.dumps(
                {
                    "adapter_fingerprint": artifact_fingerprint(adapter),
                    "metadata": metadata.to_dict(),
                    "config": config,
                    **(
                        {}
                        if legacy
                        else {"training_config_fingerprint": config_fingerprint}
                    ),
                    **selection,
                    "training_selection_fingerprint": selection_fingerprint,
                }
            )
        )
        return adapter

    def test_accepts_legacy_adapter_only_for_generic_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter = self._adapter(Path(temporary_directory), legacy=True)

            evidence = validate_insertion_adapter(
                adapter,
                expected_profile=InsertionAdapterProfile.GENERIC,
            )

            self.assertEqual(evidence.contract.schema, LEGACY_ADAPTER_SCHEMA)
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_insertion_adapter(
                    adapter,
                    expected_profile=InsertionAdapterProfile.GOAL_ALIGNED,
                )

    def test_binds_exact_window_selection_to_adapter_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            adapter = self._adapter(Path(temporary_directory))

            evidence = validate_insertion_adapter(adapter)

            self.assertEqual(evidence.identity.fingerprint, artifact_fingerprint(adapter))
            payload = json.loads(Path(f"{adapter}.json").read_text())
            payload["recording_selections"][0]["context_indices"][0] = 22
            Path(f"{adapter}.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_insertion_adapter(adapter)

    def test_binds_goal_aligned_profile_to_training_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = self._adapter(root)

            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_insertion_adapter(
                    adapter,
                    expected_profile=InsertionAdapterProfile.GOAL_ALIGNED,
                )

            aligned_adapter = self._adapter(
                root / "aligned",
                minimum_goal_cosine=0.95,
            )
            evidence = validate_insertion_adapter(
                aligned_adapter,
                expected_profile=InsertionAdapterProfile.GOAL_ALIGNED,
            )
            self.assertEqual(evidence.candidate_mining.minimum_goal_cosine, 0.95)
            self.assertEqual(
                evidence.candidate_mining.noise_policy.reference.value,
                "planner_bounds",
            )

            relative_adapter = self._adapter(
                root / "relative",
                minimum_goal_cosine=0.95,
                noise_policy={
                    "reference": "recorded_action",
                    "floors": {
                        "translation": 1e-5,
                        "rotation": 1e-5,
                        "gripper": 0.005,
                    },
                },
            )
            relative_evidence = validate_insertion_adapter(
                relative_adapter,
                expected_profile=InsertionAdapterProfile.GOAL_ALIGNED_RELATIVE,
            )
            self.assertEqual(
                relative_evidence.candidate_mining.noise_policy.reference.value,
                "recorded_action",
            )

            report_path = Path(f"{aligned_adapter}.json")
            payload = json.loads(report_path.read_text())
            payload["config"]["candidate_mining"]["noise_scale"] = 0.5
            report_path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "disagree"):
                validate_insertion_adapter(
                    aligned_adapter,
                    expected_profile=InsertionAdapterProfile.GOAL_ALIGNED,
                )

    def test_reconstructs_actions_and_energy_aggregates_from_held_out_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            adapter = validate_insertion_adapter(self._adapter(root))
            recording = root / "insertion-held-00"
            action = DroidAction((0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            results = [
                {
                    "context_indices": [context_index],
                    "target_index": context_index + 3,
                    "actions": [list(action.values)] * 3,
                    "recorded_action_energy": 1.0,
                    "zero_action_energy": 2.0,
                    "improvement_over_zero": 1.0,
                    "recorded_action_wins": True,
                }
                for context_index in INSERTION_WINDOW.context_indices
            ]
            report = root / "report.json"
            report.write_text(
                json.dumps(
                    {
                        "model": MODEL_ID,
                        "source_revision": adapter.contract.metadata.source_revision,
                        "recording": str(recording),
                        "camera": "wrist",
                        "adapter": str(adapter.identity.path),
                        "adapter_fingerprint": adapter.identity.fingerprint,
                        "rollouts": INSERTION_WINDOW.count,
                        "rollout_window": INSERTION_WINDOW.to_dict(),
                        "action_selection": INSERTION_BOUNDS.to_dict(),
                        "rollout_protocol": DROID_ROLLOUT_PROTOCOL.to_dict(),
                        "action_format": ACTION_RECORDING_CONTRACT.format,
                        "objective": "terminal_latent_l2",
                        "mean_improvement_over_zero": 1.0,
                        "recorded_action_win_rate": 1.0,
                        "control_gate": {
                            "passed": True,
                            "minimum_win_rate": 0.75,
                            "requires_positive_mean_improvement": True,
                            "reasons": [],
                        },
                        "results": results,
                    }
                )
            )
            with (
                patch(
                    "jepa_wm.insertion_wm_readiness.ContactInsertionEvidence.from_recording"
                ),
                patch(
                    "jepa_wm.insertion_wm_readiness.load_rollout_at",
                    side_effect=lambda *args, context_index, **kwargs: SimpleNamespace(
                        context=(SimpleNamespace(index=context_index),),
                        target=SimpleNamespace(index=context_index + 3),
                        actions=(action,) * 3,
                    ),
                ),
                patch(
                    "jepa_wm.insertion_wm_readiness.HeldOutEvaluation.from_payload",
                    return_value=SimpleNamespace(recording=SimpleNamespace(path=recording)),
                ),
            ):
                evidence = validate_insertion_adapter_evaluation(
                    report,
                    adapter,
                    expected_recording="insertion-held-00",
                    expected_seed=12600,
                )
                self.assertEqual(evidence.recording.path, recording)
                results[0]["actions"][0][0] = -0.001
                report_payload = json.loads(report.read_text())
                report_payload["results"] = results
                report.write_text(json.dumps(report_payload))
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    validate_insertion_adapter_evaluation(
                        report,
                        adapter,
                        expected_recording="insertion-held-00",
                        expected_seed=12600,
                    )

                report_payload = json.loads(report.read_text())
                report_payload["results"] = [
                    {
                        **result,
                        "recorded_action_energy": "1.0",
                    }
                    if index == 0
                    else result
                    for index, result in enumerate(results)
                ]
                report.write_text(json.dumps(report_payload))
                with self.assertRaisesRegex(ValueError, "native numbers"):
                    validate_insertion_adapter_evaluation(
                        report,
                        adapter,
                        expected_recording="insertion-held-00",
                        expected_seed=12600,
                    )

                report_payload["results"][0]["recorded_action_energy"] = -1.0
                report.write_text(json.dumps(report_payload))
                with self.assertRaisesRegex(ValueError, "inconsistent"):
                    validate_insertion_adapter_evaluation(
                        report,
                        adapter,
                        expected_recording="insertion-held-00",
                        expected_seed=12600,
                    )

                report_payload["results"][0]["recorded_action_energy"] = 1.0
                report_payload["source_revision"] = "different-revision"
                report.write_text(json.dumps(report_payload))
                with self.assertRaisesRegex(ValueError, "identity"):
                    validate_insertion_adapter_evaluation(
                        report,
                        adapter,
                        expected_recording="insertion-held-00",
                        expected_seed=12600,
                    )


if __name__ == "__main__":
    unittest.main()
