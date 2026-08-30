from __future__ import annotations

import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from jepa_wm.insertion_latent_diagnostic import (
        phase_centroid_probe,
        terminal_token_l2_energy,
        x_axis_counterfactuals,
    )


@unittest.skipIf(torch is None, "PyTorch is required for latent diagnostics")
class InsertionLatentDiagnosticTest(unittest.TestCase):
    def test_x_axis_counterfactuals_change_only_translation_x(self) -> None:
        actions = torch.tensor(
            [
                [
                    [-0.2, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    [0.3, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                ]
            ]
        )

        candidates = x_axis_counterfactuals(actions)

        self.assertEqual(
            tuple(candidates),
            ("recorded", "all_zero", "x_negative", "x_zero", "x_positive"),
        )
        torch.testing.assert_close(candidates["recorded"], actions)
        torch.testing.assert_close(candidates["all_zero"], torch.zeros_like(actions))
        torch.testing.assert_close(
            candidates["x_negative"][..., 0], torch.tensor([[-0.2, -0.3]])
        )
        torch.testing.assert_close(
            candidates["x_zero"][..., 0], torch.tensor([[0.0, 0.0]])
        )
        torch.testing.assert_close(
            candidates["x_positive"][..., 0], torch.tensor([[0.2, 0.3]])
        )
        for name in ("x_negative", "x_zero", "x_positive"):
            torch.testing.assert_close(candidates[name][..., 1:], actions[..., 1:])

    def test_terminal_token_energy_preserves_token_axis(self) -> None:
        prediction = torch.zeros((2, 1, 3, 2))
        prediction[-1, 0, 1] = torch.tensor((2.0, 0.0))
        target = torch.zeros((1, 1, 3, 2))

        energy = terminal_token_l2_energy(prediction, target)

        torch.testing.assert_close(energy, torch.tensor([[0.0, 2.0, 0.0]]))

    def test_phase_centroid_probe_holds_out_each_recording(self) -> None:
        embeddings = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ]
        )
        phases = ("retreat", "retreat", "align", "align") * 2
        recordings = ("held-00",) * 4 + ("held-01",) * 4

        result = phase_centroid_probe(embeddings, phases, recordings)

        self.assertEqual(result["examples"], 8)
        self.assertEqual(result["folds"], 2)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertGreater(result["mean_cosine_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
