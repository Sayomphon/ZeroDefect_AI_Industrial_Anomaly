"""Framework-free heatmap rendering for CLI and UI use."""

from __future__ import annotations

import numpy as np
from PIL import Image

from zerodefect_ai.errors import DataValidationError


def normalize_anomaly_map(anomaly_map: np.ndarray) -> np.ndarray:
    """Robustly normalize an anomaly map to [0, 1] for presentation only."""

    values = np.asarray(anomaly_map, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise DataValidationError("Anomaly map must be a finite two-dimensional array")
    low, high = np.quantile(values, [0.02, 0.98])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def make_overlay(image: np.ndarray, anomaly_map: np.ndarray, *, alpha: float = 0.45) -> Image.Image:
    """Blend an RGB input with a blue-to-yellow-to-red anomaly heatmap."""

    if not 0.0 <= alpha <= 1.0:
        raise DataValidationError("Overlay alpha must be in [0, 1]")
    rgb = np.asarray(image, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or anomaly_map.shape != rgb.shape[:2]:
        raise DataValidationError("Image and anomaly map shapes are incompatible")
    if not np.all(np.isfinite(rgb)) or float(np.min(rgb)) < 0.0 or float(np.max(rgb)) > 1.0:
        raise DataValidationError("Overlay image must be finite RGB values in [0, 1]")

    normalized = normalize_anomaly_map(anomaly_map)
    red = np.clip(2.0 * normalized, 0.0, 1.0)
    blue = np.clip(2.0 * (1.0 - normalized), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(2.0 * normalized - 1.0) * 1.5, 0.0, 1.0)
    heatmap = np.stack([red, green, blue], axis=2)
    blended = (1.0 - alpha) * rgb + alpha * heatmap
    return Image.fromarray(np.asarray(np.clip(blended * 255.0, 0, 255), dtype=np.uint8), mode="RGB")
