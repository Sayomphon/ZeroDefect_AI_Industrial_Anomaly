"""Reusable inspection service independent from CLI and UI frameworks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from zerodefect_ai.artifacts import ArtifactBundle, load_artifact
from zerodefect_ai.domain import InspectionResult
from zerodefect_ai.errors import DataValidationError
from zerodefect_ai.image_io import load_image, preprocess_array
from zerodefect_ai.visualization import make_overlay


class InspectionService:
    """Load one immutable artifact and inspect images consistently."""

    def __init__(self, bundle: ArtifactBundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_artifact(cls, artifact_dir: Path | str) -> "InspectionService":
        return cls(load_artifact(artifact_dir))

    @property
    def default_threshold(self) -> float:
        return self.bundle.threshold

    def _inspect_preprocessed(
        self, image: np.ndarray, *, threshold: float | None = None
    ) -> InspectionResult:
        operating_threshold = self.default_threshold if threshold is None else float(threshold)
        if not np.isfinite(operating_threshold) or operating_threshold < 0.0:
            raise DataValidationError("Inspection threshold must be a finite non-negative value")
        prediction = self.bundle.detector.predict(image)
        return InspectionResult(
            score=prediction.score,
            threshold=operating_threshold,
            is_anomaly=prediction.score >= operating_threshold,
            anomaly_map=prediction.anomaly_map,
            preprocessed_image=image,
            overlay=make_overlay(image, prediction.anomaly_map),
            model_type=self.bundle.detector.model_type,
            artifact_schema_version=self.bundle.schema_version,
        )

    def inspect_path(
        self, image_path: Path | str, *, threshold: float | None = None
    ) -> InspectionResult:
        image = load_image(image_path, self.bundle.config.image)
        return self._inspect_preprocessed(image, threshold=threshold)

    def inspect_array(
        self, image_array: np.ndarray, *, threshold: float | None = None
    ) -> InspectionResult:
        image = preprocess_array(image_array, self.bundle.config.image)
        return self._inspect_preprocessed(image, threshold=threshold)
