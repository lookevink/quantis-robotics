from __future__ import annotations

import unittest

from sim.demo_sequence import DemoGeometry, Phase, PlugAction, build_demo_sequence


class DemoSequenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = DemoGeometry(
            plug_position=(-0.0256, -0.25025, 1.32),
            socket_position=(-0.071, -0.25, 1.32),
            ready_position=(0.25, -0.25, 1.48),
        )

    def test_builds_the_ordered_demo_sequence(self) -> None:
        sequence = build_demo_sequence(self.geometry)

        self.assertEqual(
            [waypoint.phase for waypoint in sequence],
            [
                Phase.READY,
                Phase.PRE_GRASP,
                Phase.GRASP,
                Phase.PRE_INSERTION,
                Phase.INSERT,
                Phase.RELEASE,
            ],
        )
        self.assertEqual(sequence[2].plug_action, PlugAction.ATTACH)
        self.assertEqual(sequence[-1].plug_action, PlugAction.DETACH)
        self.assertGreater(sequence[0].gripper_width_m, sequence[2].gripper_width_m)
        self.assertEqual(sequence[-1].gripper_width_m, sequence[0].gripper_width_m)

    def test_preserves_the_plug_to_socket_displacement(self) -> None:
        sequence = build_demo_sequence(self.geometry)
        grasp = sequence[2].target_position
        insert = sequence[4].target_position

        hand_displacement = tuple(insert[i] - grasp[i] for i in range(3))
        plug_displacement = tuple(
            self.geometry.socket_position[i] - self.geometry.plug_position[i]
            for i in range(3)
        )
        self.assertEqual(hand_displacement, plug_displacement)

    def test_approaches_from_the_positive_x_side(self) -> None:
        sequence = build_demo_sequence(
            self.geometry,
            approach_clearance_m=0.10,
            insertion_clearance_m=0.03,
        )

        pre_grasp, grasp = sequence[1:3]
        pre_insert, insert = sequence[3:5]
        self.assertAlmostEqual(pre_grasp.target_position[0] - grasp.target_position[0], 0.10)
        self.assertAlmostEqual(pre_insert.target_position[0] - insert.target_position[0], 0.03)
        self.assertEqual(pre_grasp.target_position[1:], grasp.target_position[1:])
        self.assertEqual(pre_insert.target_position[1:], insert.target_position[1:])

    def test_rejects_a_socket_on_the_wrong_side_of_the_plug(self) -> None:
        geometry = DemoGeometry(
            plug_position=(0.0, 0.0, 1.0),
            socket_position=(0.02, 0.0, 1.0),
            ready_position=(0.2, 0.0, 1.2),
        )

        with self.assertRaisesRegex(ValueError, "negative X"):
            build_demo_sequence(geometry)


if __name__ == "__main__":
    unittest.main()
