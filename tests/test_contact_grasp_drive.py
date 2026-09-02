from __future__ import annotations

import unittest

from jepa_wm.contact_grasp_drive import ContactGraspDrivePolicy
from jepa_wm.control_safety import SimulatorSafetyLimits
from jepa_wm.joint_drive import JointDriveTarget


class ContactGraspDrivePolicyTest(unittest.TestCase):
    def test_compensates_the_exact_terminal_loaded_bias(self) -> None:
        desired = (
            0.7885459129407968,
            -1.4137833805179416,
            -1.156716298844269,
            -2.6375719380288416,
            0.5996050145467479,
            3.479950376882293,
            -0.945397023219124,
        )
        active = JointDriveTarget(
            (
                0.7907184934651541,
                -1.414219232613105,
                -1.1576313847511315,
                -2.636477547537189,
                0.597839517569356,
                3.4804569991170013,
                -0.9452613320482888,
            ),
            0.02094929665327072,
        )
        stable = (
            0.7907772660255432,
            -1.4150390625,
            -1.1586326360702515,
            -2.636824607849121,
            0.5978310108184814,
            3.4805612564086914,
            -0.9453051090240479,
        )

        target = ContactGraspDrivePolicy().forward_drive_target(
            desired,
            active.gripper_width_m,
            active,
            stable,
            SimulatorSafetyLimits(),
        )

        expected = tuple(
            desired_value + drive_value - stable_value
            for desired_value, drive_value, stable_value in zip(
                desired,
                active.joint_positions,
                stable,
            )
        )
        self.assertEqual(
            target,
            JointDriveTarget.for_command(expected, active.gripper_width_m),
        )

    def test_rejects_bias_or_motion_outside_existing_bounds(self) -> None:
        policy = ContactGraspDrivePolicy()
        active = JointDriveTarget((0.0,) * 7, 0.02)
        limits = SimulatorSafetyLimits()

        with self.assertRaisesRegex(ValueError, "bias exceeds"):
            policy.forward_drive_target(
                (0.0,) * 7,
                0.02,
                active,
                (0.003,) * 7,
                limits,
            )
        stable = (0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0)
        active = JointDriveTarget(stable, 0.02)
        with self.assertRaisesRegex(ValueError, "velocity gate"):
            policy.forward_drive_target(
                (0.2, *stable[1:]),
                0.02,
                active,
                stable,
                limits,
            )


if __name__ == "__main__":
    unittest.main()
