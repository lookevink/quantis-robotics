from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jepa_wm.action_model_artifact import apply_action_model_artifact


class ActionModelArtifactTest(unittest.TestCase):
    def test_dispatches_the_physical_conditioning_schema_without_adapter_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "physical.pth"
            torch.save(
                {"schema": "quantis.jepa_wm_action_conditioning.v1"},
                artifact,
            )
            with (
                patch(
                    "jepa_wm.action_model_artifact.apply_action_conditioning"
                ) as conditioning,
                patch("jepa_wm.action_model_artifact.apply_action_adapter") as adapter,
            ):
                apply_action_model_artifact(
                    object(), artifact, expected_source_revision="revision"
                )

            conditioning.assert_called_once_with(
                unittest.mock.ANY,
                artifact,
                expected_source_revision="revision",
            )
            adapter.assert_not_called()
