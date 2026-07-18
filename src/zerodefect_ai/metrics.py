"""Dependency-light image and sampled pixel-level anomaly metrics."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from zerodefect_ai.errors import DataValidationError


def _validated_binary_inputs(
    scores_input: Iterable[float], labels_input: Iterable[int]
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(list(scores_input), dtype=np.float64)
    labels = np.asarray(list(labels_input), dtype=np.int8)
    if scores.ndim != 1 or scores.size == 0:
        raise DataValidationError("Metric scores must be a non-empty one-dimensional sequence")
    if labels.shape != scores.shape or not np.all(np.isin(labels, [0, 1])):
        raise DataValidationError("Metric labels must match scores and contain only 0 or 1")
    if not np.all(np.isfinite(scores)):
        raise DataValidationError("Metric scores contain NaN or infinite values")
    return scores, labels


def roc_auc(scores_input: Iterable[float], labels_input: Iterable[int]) -> float | None:
    """Calculate tie-aware ROC AUC using average ranks."""

    scores, labels = _validated_binary_inputs(scores_input, labels_input)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(np.sum(ranks[labels == 1]))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def average_precision(scores_input: Iterable[float], labels_input: Iterable[int]) -> float | None:
    """Calculate threshold-grouped average precision for the anomaly class."""

    scores, labels = _validated_binary_inputs(scores_input, labels_input)
    positives = int(np.sum(labels == 1))
    if positives == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_tp = 0
    cumulative_total = 0
    ap = 0.0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        group_labels = sorted_labels[start:end]
        group_tp = int(np.sum(group_labels == 1))
        cumulative_tp += group_tp
        cumulative_total += end - start
        precision = cumulative_tp / cumulative_total
        ap += (group_tp / positives) * precision
        start = end
    return float(ap)


def binary_metrics(
    scores_input: Iterable[float],
    labels_input: Iterable[int],
    threshold: float,
    *,
    false_reject_cost: float = 1.0,
    defect_escape_cost: float = 1.0,
) -> dict[str, float | int | None]:
    """Return ranking, decision, and business-oriented image-level metrics."""

    scores, labels = _validated_binary_inputs(scores_input, labels_input)
    if not np.isfinite(threshold):
        raise DataValidationError("Metric threshold must be finite")
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fp = int(np.sum(predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    expected_cost = (fp * false_reject_cost + fn * defect_escape_cost) / scores.size
    return {
        "sample_count": int(scores.size),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "threshold": float(threshold),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "accuracy": float((tp + tn) / scores.size),
        "false_reject_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "defect_escape_rate": float(fn / (tp + fn)) if tp + fn else 0.0,
        "expected_cost_per_image": float(expected_cost),
    }


def sampled_pixel_metrics(
    anomaly_maps: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    max_points: int = 1_000_000,
) -> dict[str, float | int | None]:
    """Estimate pixel ranking metrics with a deterministic memory bound."""

    if len(anomaly_maps) != len(masks) or not anomaly_maps:
        raise DataValidationError("Pixel maps and masks must be non-empty and have equal length")
    if max_points < len(anomaly_maps):
        raise DataValidationError("max_points must allow at least one pixel per image")
    per_image_budget = max(1, max_points // len(anomaly_maps))
    score_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for anomaly_map, mask in zip(anomaly_maps, masks):
        if anomaly_map.shape != mask.shape or anomaly_map.ndim != 2:
            raise DataValidationError("Each anomaly map and mask must be matching 2D arrays")
        flat_scores = np.asarray(anomaly_map, dtype=np.float64).reshape(-1)
        flat_labels = np.asarray(mask > 0, dtype=np.int8).reshape(-1)
        if not np.all(np.isfinite(flat_scores)):
            raise DataValidationError("Pixel anomaly map contains NaN or infinite values")
        if flat_scores.size > per_image_budget:
            indices = np.linspace(0, flat_scores.size - 1, per_image_budget, dtype=np.int64)
            flat_scores = flat_scores[indices]
            flat_labels = flat_labels[indices]
        score_parts.append(flat_scores)
        label_parts.append(flat_labels)

    scores = np.concatenate(score_parts)
    labels = np.concatenate(label_parts)
    return {
        "sampled_points": int(scores.size),
        "positive_fraction": float(np.mean(labels == 1)),
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
    }
