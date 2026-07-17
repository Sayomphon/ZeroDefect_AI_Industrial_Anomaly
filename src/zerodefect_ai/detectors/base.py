"""Detector protocol shared by training, artifact, and service layers."""

from __future__ import annotations

from typing import Iterable, Mapping, Protocol

import numpy as np

from zerodefect_ai.domain import ModelPrediction


class AnomalyDetector(Protocol):
    """Minimal interface required from an anomaly detector implementation."""

    model_type: str

    def fit(self, images: Iterable[np.ndarray]) -> int:
        """Fit using normal images and return the number of consumed images."""

    def predict(self, image: np.ndarray) -> ModelPrediction:
        """Return an image-level score and pixel-level anomaly map."""

    def to_arrays(self) -> Mapping[str, np.ndarray]:
        """Return safe, non-object arrays suitable for NPZ serialization."""
