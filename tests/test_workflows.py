from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zerodefect_ai.config import DetectorConfig, ImageConfig, ProjectConfig
from zerodefect_ai.workflows import run_smoke


class WorkflowTest(unittest.TestCase):
    def test_end_to_end_smoke(self) -> None:
        config = ProjectConfig(
            image=ImageConfig(width=48, height=48),
            detector=DetectorConfig(
                minimum_training_images=3,
                minimum_std=0.02,
                image_score_quantile=0.99,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            report = run_smoke(Path(directory) / "smoke", config)
            self.assertEqual(report["status"], "passed")
            metrics = report["evaluation"]["image_level"]
            self.assertEqual(metrics["roc_auc"], 1.0)
            self.assertGreater(metrics["true_positives"], 0)
            self.assertTrue((Path(directory) / "smoke" / "smoke_report.json").is_file())
            self.assertTrue((Path(directory) / "smoke" / "prediction" / "overlay.png").is_file())


if __name__ == "__main__":
    unittest.main()
