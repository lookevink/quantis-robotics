from __future__ import annotations

import unittest
from types import ModuleType
from unittest.mock import patch

import numpy as np

from sim.isaac_demo_runtime import (
    FixedJointPlugMotion,
    ContactReading,
    PlugAttachment,
    PlugCollisionPolicy,
    _advance_sample,
)


class _RigidPrim:
    def __init__(self) -> None:
        self.positions = None

    def set_world_poses(self, *, positions) -> None:
        self.positions = positions

    def get_world_poses(self):
        return (
            np.asarray(self.positions, dtype=np.float64),
            np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        )


class PlugAttachmentTest(unittest.TestCase):
    def test_refreshes_fixed_joint_tensor_handle_without_losing_attachment(self) -> None:
        original_rigid = _RigidPrim()
        refreshed_rigid = _RigidPrim()
        offset = np.array([0.04, 0.0, 0.0])
        attachment = PlugAttachment(
            motion=FixedJointPlugMotion(
                prim=object(),
                hand_prim=object(),
                rigid_prim=original_rigid,
                fixed_joint=object(),
                hand_to_plug_offset=offset,
            ),
            collisions=PlugCollisionPolicy([]),
        )

        refreshed = attachment.with_refreshed_physics(refreshed_rigid)

        self.assertIs(refreshed.motion.rigid_prim, refreshed_rigid)
        self.assertTrue(refreshed.attached)
        np.testing.assert_array_equal(refreshed.motion.hand_to_plug_offset, offset)
        self.assertIsNot(refreshed.motion.hand_to_plug_offset, offset)

    def test_follow_updates_the_bound_physics_body(self) -> None:
        rigid = _RigidPrim()
        attachment = PlugAttachment(
            motion=FixedJointPlugMotion(
                prim=object(),
                hand_prim=object(),
                rigid_prim=rigid,
                fixed_joint=object(),
                hand_to_plug_offset=np.array([0.04, 0.0, 0.0]),
            ),
            collisions=PlugCollisionPolicy([]),
        )

        attachment.follow(np.array([0.1, -0.2, 1.3]))

        self.assertIsNone(rigid.positions)
        rigid.positions = [[0.14, -0.2, 1.3]]
        position, orientation = attachment.world_pose()
        np.testing.assert_allclose(position, [0.14, -0.2, 1.3])
        np.testing.assert_allclose(orientation, [1.0, 0.0, 0.0, 0.0])


class SafetyInterlockTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _omni_modules(app, timeline):
        omni = ModuleType("omni")
        kit = ModuleType("omni.kit")
        kit_app = ModuleType("omni.kit.app")
        timeline_module = ModuleType("omni.timeline")
        kit_app.get_app = lambda: app
        timeline_module.get_timeline_interface = lambda: timeline
        omni.kit = kit
        omni.timeline = timeline_module
        kit.app = kit_app
        return {
            "omni": omni,
            "omni.kit": kit,
            "omni.kit.app": kit_app,
            "omni.timeline": timeline_module,
        }

    async def test_polls_during_sample_and_stops_on_intermediate_force(self) -> None:
        class _Timeline:
            current = 0.0

            def get_current_time(self) -> float:
                return self.current

        timeline = _Timeline()

        class _App:
            updates = 0

            async def next_update_async(self) -> None:
                self.updates += 1
                timeline.current += 0.01

        app = _App()
        modules = self._omni_modules(app, timeline)

        def observe() -> ContactReading:
            if app.updates == 2:
                raise RuntimeError("force exceeded")
            return ContactReading()

        with patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(RuntimeError, "force exceeded"):
                await _advance_sample(0.25, observe)

        self.assertEqual(app.updates, 2)

    def test_rejects_invalid_contact_reading_before_peak_aggregation(self) -> None:
        for value in (float("nan"), -0.1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "contact reading"):
                    ContactReading(False, value)

    async def test_preserves_transient_force_until_the_recorded_frame(self) -> None:
        class _Timeline:
            current = 0.0

            def get_current_time(self) -> float:
                return self.current

        timeline = _Timeline()

        class _App:
            updates = 0

            async def next_update_async(self) -> None:
                self.updates += 1
                timeline.current += 0.05

        app = _App()

        def observe() -> ContactReading:
            return ContactReading(False, 1.5 if app.updates == 2 else 0.0)

        with patch.dict("sys.modules", self._omni_modules(app, timeline)):
            reading = await _advance_sample(0.25, observe)

        self.assertFalse(reading.collision_detected)
        self.assertEqual(reading.force_newtons, 1.5)


if __name__ == "__main__":
    unittest.main()
