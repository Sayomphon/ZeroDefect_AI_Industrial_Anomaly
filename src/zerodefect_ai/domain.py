"""Typed values exchanged between model, service, and presentation layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ModelPrediction:
    """Raw detector output before a business decision is attached."""

    score: float
    anomaly_map: np.ndarray


@dataclass(frozen=True)
class InspectionResult:
    """End-to-end inspection result returned by the service layer."""

    score: float
    threshold: float
    is_anomaly: bool
    anomaly_map: np.ndarray
    preprocessed_image: np.ndarray
    overlay: Image.Image
    model_type: str
    artifact_schema_version: int
