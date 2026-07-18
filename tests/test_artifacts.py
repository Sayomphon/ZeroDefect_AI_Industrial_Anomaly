from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from zerodefect_ai.artifacts import load_artifact, save_artifact
from zerodefect_ai.config import DetectorConfig, ImageConfig, ProjectConfig
from zerodefect_ai.detectors.statistical import StatisticalDetector
from zerodefect_ai.errors import ArtifactError


class ArtifactTest(unittest.TestCase):
    def _fitted_detector(self) -> tuple[ProjectConfig, StatisticalDetector]:
        config = ProjectConfig(
            image=ImageConfig(width=24, height=24),
            detector=DetectorConfig(minimum_training_images=3),
        )
        detector = StatisticalDetector(config.image, config.detector)
        rng = np.random.default_rng(7)
        detector.fit(
            [
                np.clip(rng.normal(0.5, 0.01, size=(24, 24, 3)), 0, 1).astype(np.float32)
                for _ in range(4)
            ]
        )
        return config, detector

    def test_round_trip_preserves_prediction(self) -> None:
        config, detector = self._fitted_detector()
        image = np.full((24, 24, 3), 0.5, dtype=np.float32)
        expected = detector.predict(image).score
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifact"
            save_artifact(
                artifact_dir,
                detector,
                config,
                2.5,
                training_summary={"training_images": 4},
            )
            loaded = load_artifact(artifact_dir)
            self.assertAlmostEqual(loaded.detector.predict(image).score, expected, places=6)
            self.assertEqual(loaded.threshold, 2.5)

    def test_checksum_detects_tampering(self) -> None:
        config, detector = self._fitted_detector()
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory) / "artifact"
            save_artifact(
                artifact_dir,
                detector,
                config,
                2.5,
                training_summary={"training_images": 4},
            )
            with (artifact_dir / "metadata.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaises(ArtifactError):
                load_artifact(artifact_dir)


if __name__ == "__main__":
    unittest.main()
