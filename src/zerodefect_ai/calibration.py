"""Threshold calibration policies for model and business objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from zerodefect_ai.errors import DataValidationError


@dataclass(frozen=True)
class CalibrationResult:
    """Selected operating point and the evidence used to choose it."""

    method: str
    threshold: float
    objective_value: float
    false_positives: int | None = None
    false_negatives: int | None = None


def _scores(values: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise DataValidationError("Calibration scores must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(result)):
        raise DataValidationError("Calibration scores contain NaN or infinite values")
    return result


def _labels(values: Iterable[int], expected_size: int) -> np.ndarray:
    result = np.asarray(list(values), dtype=np.int8)
    if result.ndim != 1 or result.size != expected_size:
        raise DataValidationError("Calibration labels must match the number of scores")
    if not np.all(np.isin(result, [0, 1])):
        raise DataValidationError("Calibration labels must contain only 0 (normal) and 1 (defect)")
    return result


def calibrate_normal_quantile(
    normal_scores: Iterable[float], quantile: float
) -> CalibrationResult:
    """Choose a threshold without defect labels, preserving a pure cold-start setting."""

    scores = _scores(normal_scores)
    if not 0.5 <= quantile <= 1.0:
        raise DataValidationError("Normal calibration quantile must be in [0.5, 1.0]")
    threshold = float(np.quantile(scores, quantile, method="higher"))
    false_reject_rate = float(np.mean(scores >= threshold))
    return CalibrationResult(
        method="normal_quantile",
        threshold=threshold,
        objective_value=false_reject_rate,
        false_positives=int(np.sum(scores >= threshold)),
        false_negatives=None,
    )


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    none_anomalous = np.nextafter(unique[-1], np.inf)
    return np.concatenate((unique, np.asarray([none_anomalous], dtype=np.float64)))


def calibrate_f1(scores_input: Iterable[float], labels_input: Iterable[int]) -> CalibrationResult:
    """Maximize anomaly-class F1 on a labeled validation set."""

    scores = _scores(scores_input)
    labels = _labels(labels_input, scores.size)
    if not np.any(labels == 1):
        raise DataValidationError("F1 calibration requires at least one defect label")

    best: tuple[float, int, float, int, int] | None = None
    for threshold in _candidate_thresholds(scores):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (labels == 1)))
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        denominator = 2 * tp + fp + fn
        f1 = 0.0 if denominator == 0 else (2.0 * tp) / denominator
        candidate = (f1, -fp, float(threshold), fp, fn)
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    assert best is not None
    return CalibrationResult(
        method="f1",
        threshold=best[2],
        objective_value=best[0],
        false_positives=best[3],
        false_negatives=best[4],
    )


def calibrate_business_cost(
    scores_input: Iterable[float],
    labels_input: Iterable[int],
    *,
    false_reject_cost: float,
    defect_escape_cost: float,
) -> CalibrationResult:
    """Minimize expected validation cost under an explicit QC cost model."""

    scores = _scores(scores_input)
    labels = _labels(labels_input, scores.size)
    if false_reject_cost < 0.0 or defect_escape_cost < 0.0:
        raise DataValidationError("Business costs must be non-negative")
    if false_reject_cost == 0.0 and defect_escape_cost == 0.0:
        raise DataValidationError("At least one business cost must be greater than zero")
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise DataValidationError("Business-cost calibration requires normal and defect labels")

    best: tuple[float, int, int, float] | None = None
    for threshold in _candidate_thresholds(scores):
        predicted = scores >= threshold
        fp = int(np.sum(predicted & (labels == 0)))
        fn = int(np.sum(~predicted & (labels == 1)))
        expected_cost = (fp * false_reject_cost + fn * defect_escape_cost) / scores.size
        candidate = (float(expected_cost), fn, fp, -float(threshold))
        if best is None or candidate < best:
            best = candidate

    assert best is not None
    return CalibrationResult(
        method="business_cost",
        threshold=-best[3],
        objective_value=best[0],
        false_positives=best[2],
        false_negatives=best[1],
    )
