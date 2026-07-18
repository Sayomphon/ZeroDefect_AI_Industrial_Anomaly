from __future__ import annotations

import unittest

import numpy as np

from zerodefect_ai.config import DetectorConfig, ImageConfig
from zerodefect_ai.detectors.statistical import StatisticalDetector
from zerodefect_ai.errors import DataValidationError, ModelNotFittedError


class StatisticalDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image_config = ImageConfig(width=32, height=32)
        self.detector_config = DetectorConfig(
            minimum_training_images=3,
            minimum_std=0.02,
            image_score_quantile=0.99,
        )

    def _normal_images(self) -> list[np.ndarray]:
        rng = np.random.default_rng(42)
        return [
            np.clip(rng.normal(0.5, 0.005, size=(32, 32, 3)), 0, 1).astype(np.float32)
            for _ in range(6)
        ]

    def test_requires_fit(self) -> None:
        detector = StatisticalDetector(self.image_config, self.detector_config)
        with self.assertRaises(ModelNotFittedError):
            detector.predict(self._normal_images()[0])

    def test_defect_scores_higher_than_normal(self) -> None:
        detector = StatisticalDetector(self.image_config, self.detector_config)
        normal_images = self._normal_images()
        detector.fit(normal_images)
        normal_score = detector.predict(normal_images[0]).score
        defect = normal_images[0].copy()
        defect[10:18, 10:18] = 1.0
        defect_prediction = detector.predict(defect)
        self.assertGreater(defect_prediction.score, normal_score)
        self.assertEqual(defect_prediction.anomaly_map.shape, (32, 32))

    def test_rejects_wrong_shape(self) -> None:
        detector = StatisticalDetector(self.image_config, self.detector_config)
        with self.assertRaises(DataValidationError):
            detector.fit([np.zeros((31, 32, 3), dtype=np.float32)] * 3)


if __name__ == "__main__":
    unittest.main()
