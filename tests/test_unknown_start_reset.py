from dataclasses import replace
import unittest

from sim.unknown_start_reset import (
    UNKNOWN_START_RESET_CONTRACT,
    UnknownStartResetEvidence,
)


class UnknownStartResetContractTest(unittest.TestCase):
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
        self.assertEqual(sample.split, "held_out")
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
            realized_sample_fingerprint=sample.fingerprint,
            plug_attached=False,
            collision_detected=False,
            contact_force_newtons=0.0,
            prefix_replay_frames=0,
            runtime_motion="drive_only",
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

    def test_contract_freezes_distribution_and_authority_boundary(self) -> None:
        self.assertEqual(
            UNKNOWN_START_RESET_CONTRACT.to_dict(),
            {
                "schema": "quantis.unknown_start_reset_contract.v1",
                "seed_namespace": {"minimum": 62600, "maximum": 62699},
                "split": "held_out",
                "sampler": "build_exploration_plan.v1",
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
                "initialization": "direct_state_setting_once",
                "prefix_replay_frames": 0,
                "runtime_motion": "drive_only",
                "maximum_initial_contact_force_newtons": 0.0,
                "require_unattached": True,
                "require_collision_free": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
