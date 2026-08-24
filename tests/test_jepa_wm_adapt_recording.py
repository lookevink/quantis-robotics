from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.action import DroidAction
    from jepa_wm.adapt_recording import mismatched_negative_candidates
    from jepa_wm.control_protocol import TaskContextIndex


@unittest.skipIf(torch is None, "PyTorch is not installed in the local test runtime")
class MismatchedNegativeCandidatesTest(unittest.TestCase):
    @staticmethod
    def _rollout(context_index: int, translation: float):
        action = DroidAction((translation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        return SimpleNamespace(
            task_context_index=TaskContextIndex(context_index),
            actions=(action, action, action),
        )

    def test_excludes_same_context_and_identical_action_sequences(self) -> None:
        rollouts = (
            self._rollout(69, 0.01),
            self._rollout(69, 0.02),
            self._rollout(70, 0.01),
            self._rollout(71, 0.03),
        )

        candidates = mismatched_negative_candidates(rollouts)

        self.assertEqual(candidates[0], (3,))
        self.assertEqual(candidates[1], (2, 3))

    def test_rejects_a_rollout_without_a_meaningful_negative(self) -> None:
        rollouts = (
            self._rollout(69, 0.01),
            self._rollout(70, 0.01),
        )

        with self.assertRaisesRegex(ValueError, "different-context"):
            mismatched_negative_candidates(rollouts)


if __name__ == "__main__":
    unittest.main()
