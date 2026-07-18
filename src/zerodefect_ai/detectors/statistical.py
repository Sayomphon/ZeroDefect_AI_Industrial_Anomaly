"""Interpretable normal-only detector based on streaming pixel statistics."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from zerodefect_ai.config import DetectorConfig, ImageConfig
from zerodefect_ai.domain import ModelPrediction
from zerodefect_ai.errors import DataValidationError, ModelNotFittedError


class StatisticalDetector:
    """Detect deviations from a learned, aligned normal-image distribution.

    The detector uses Welford's online algorithm, so training memory remains bounded
    by image resolution instead of dataset size. It is intentionally a transparent
    baseline; it is not invariant to large camera, pose, or lighting changes.
    """

    model_type = "statistical_pixel_zscore"

    def __init__(self, image_config: ImageConfig, detector_config: DetectorConfig) -> None:
        self.image_config = image_config
        self.detector_config = detector_config
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.training_count = 0

    @property
    def expected_shape(self) -> tuple[int, int, int]:
        return (self.image_config.height, self.image_config.width, 3)

    def _validated_image(self, image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise DataValidationError("Detector input must be a NumPy array")
        if image.shape != self.expected_shape:
            raise DataValidationError(
                f"Detector input shape {image.shape} does not match {self.expected_shape}"
            )
        if not np.issubdtype(image.dtype, np.number):
            raise DataValidationError("Detector input must contain numeric pixels")
        value = np.asarray(image, dtype=np.float32)
        if not np.all(np.isfinite(value)):
            raise DataValidationError("Detector input contains NaN or infinite values")
        minimum = float(np.min(value))
        maximum = float(np.max(value))
        if minimum < 0.0 or maximum > 1.0:
            raise DataValidationError("Detector input must be normalized to [0, 1]")
        return value

    def fit(self, images: Iterable[np.ndarray]) -> int:
        count = 0
        mean = np.zeros(self.expected_shape, dtype=np.float64)
        m2 = np.zeros(self.expected_shape, dtype=np.float64)

        for raw_image in images:
            image = self._validated_image(raw_image).astype(np.float64, copy=False)
            count += 1
            delta = image - mean
            mean += delta / count
            delta_after = image - mean
            m2 += delta * delta_after

        if count < self.detector_config.minimum_training_images:
            raise DataValidationError(
                "Insufficient normal images: "
                f"{count} < {self.detector_config.minimum_training_images}"
            )

        variance = m2 / (count - 1)
        minimum_variance = self.detector_config.minimum_std**2
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(np.sqrt(np.maximum(variance, minimum_variance)), dtype=np.float32)
        self.training_count = count
        return count

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.std is None or self.training_count <= 0:
            raise ModelNotFittedError("Statistical detector has not been fitted")
        return self.mean, self.std

    def predict(self, image: np.ndarray) -> ModelPrediction:
        mean, std = self._require_fitted()
        value = self._validated_image(image)
        channel_deviation = np.abs(value - mean) / std
        anomaly_map = np.mean(channel_deviation, axis=2, dtype=np.float32)
        score = float(
            np.quantile(
                anomaly_map,
                self.detector_config.image_score_quantile,
                method="higher",
            )
        )
        if not np.isfinite(score):
            raise DataValidationError("Detector produced a non-finite anomaly score")
        return ModelPrediction(score=score, anomaly_map=anomaly_map)

    def to_arrays(self) -> Mapping[str, np.ndarray]:
        mean, std = self._require_fitted()
        return {
            "mean": np.asarray(mean, dtype=np.float32),
            "std": np.asarray(std, dtype=np.float32),
            "training_count": np.asarray([self.training_count], dtype=np.int64),
        }

    @classmethod
    def from_arrays(
        cls,
        image_config: ImageConfig,
        detector_config: DetectorConfig,
        arrays: Mapping[str, np.ndarray],
    ) -> "StatisticalDetector":
        required = {"mean", "std", "training_count"}
        if set(arrays) != required:
            raise DataValidationError(
                f"Model arrays must be exactly {sorted(required)}, got {sorted(arrays)}"
            )
        detector = cls(image_config, detector_config)
        mean = np.asarray(arrays["mean"])
        std = np.asarray(arrays["std"])
        count_array = np.asarray(arrays["training_count"])
        if mean.shape != detector.expected_shape or std.shape != detector.expected_shape:
            raise DataValidationError(
                "Artifact model arrays do not match configured image dimensions"
            )
        if mean.dtype.kind != "f" or std.dtype.kind != "f":
            raise DataValidationError("Artifact mean/std arrays must use a floating dtype")
        if count_array.shape != (1,) or count_array.dtype.kind not in {"i", "u"}:
            raise DataValidationError("Artifact training_count must be a single integer")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise DataValidationError("Artifact contains NaN or infinite model values")
        if np.any(std <= 0.0):
            raise DataValidationError("Artifact standard deviation must be positive")
        training_count = int(count_array[0])
        if training_count < detector_config.minimum_training_images:
            raise DataValidationError("Artifact training_count violates detector configuration")

        detector.mean = np.asarray(mean, dtype=np.float32)
        detector.std = np.asarray(std, dtype=np.float32)
        detector.training_count = training_count
        return detector
