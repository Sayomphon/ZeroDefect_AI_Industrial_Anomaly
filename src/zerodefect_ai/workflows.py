"""Application workflows composed from the domain modules."""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from zerodefect_ai.artifacts import atomic_write_json, load_artifact, save_artifact
from zerodefect_ai.calibration import calibrate_normal_quantile
from zerodefect_ai.config import ProjectConfig
from zerodefect_ai.datasets import discover_images, mvtec_test_samples
from zerodefect_ai.detectors.statistical import StatisticalDetector
from zerodefect_ai.errors import ConfigurationError, DataValidationError
from zerodefect_ai.image_io import load_image, load_mask
from zerodefect_ai.metrics import binary_metrics, sampled_pixel_metrics
from zerodefect_ai.service import InspectionService
from zerodefect_ai.synthetic import generate_synthetic_mvtec


@dataclass(frozen=True)
class TrainingOutcome:
    """Operator-facing summary of a completed training workflow."""

    artifact_dir: Path
    training_images: int
    calibration_images: int
    threshold: float
    score_min: float
    score_median: float
    score_max: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_dir": str(self.artifact_dir),
            "training_images": self.training_images,
            "calibration_images": self.calibration_images,
            "threshold": self.threshold,
            "calibration_score_min": self.score_min,
            "calibration_score_median": self.score_median,
            "calibration_score_max": self.score_max,
        }


def _score_paths(
    detector: StatisticalDetector, paths: list[Path], config: ProjectConfig
) -> list[float]:
    return [detector.predict(load_image(path, config.image)).score for path in paths]


def train_statistical(
    normal_dir: Path | str,
    artifact_dir: Path | str,
    config: ProjectConfig,
    *,
    calibration_normal_dir: Path | str | None = None,
    overwrite: bool = False,
) -> TrainingOutcome:
    """Fit, calibrate, and save the normal-only statistical detector."""

    if config.calibration.method != "normal_quantile":
        raise ConfigurationError(
            "The normal-only training workflow requires calibration.method='normal_quantile'. "
            "Use labeled calibration APIs only with a separate validation set."
        )
    training_paths = discover_images(normal_dir, config.image)
    detector = StatisticalDetector(config.image, config.detector)
    training_count = detector.fit(load_image(path, config.image) for path in training_paths)

    if calibration_normal_dir is None:
        calibration_paths = training_paths
        calibration_source = "training_set_reuse"
    else:
        calibration_paths = discover_images(calibration_normal_dir, config.image)
        calibration_source = "separate_normal_validation"
    scores = _score_paths(detector, calibration_paths, config)
    calibration = calibrate_normal_quantile(scores, config.calibration.normal_quantile)

    score_array = np.asarray(scores, dtype=np.float64)
    training_summary: dict[str, Any] = {
        "training_images": training_count,
        "calibration_images": len(calibration_paths),
        "calibration_source": calibration_source,
        "calibration_method": calibration.method,
        "calibration_objective_false_reject_rate": calibration.objective_value,
        "calibration_score_min": float(np.min(score_array)),
        "calibration_score_median": float(np.median(score_array)),
        "calibration_score_max": float(np.max(score_array)),
        "source_identifier": Path(normal_dir).name,
    }
    saved_dir = save_artifact(
        artifact_dir,
        detector,
        config,
        calibration.threshold,
        training_summary=training_summary,
        overwrite=overwrite,
    )
    return TrainingOutcome(
        artifact_dir=saved_dir,
        training_images=training_count,
        calibration_images=len(calibration_paths),
        threshold=calibration.threshold,
        score_min=float(np.min(score_array)),
        score_median=float(np.median(score_array)),
        score_max=float(np.max(score_array)),
    )


def _score_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def evaluate_mvtec(
    artifact_dir: Path | str,
    dataset_root: Path | str,
    category: str,
) -> dict[str, Any]:
    """Evaluate one immutable artifact on one MVTec-compatible category."""

    bundle = load_artifact(artifact_dir)
    samples = mvtec_test_samples(dataset_root, category, bundle.config.image)
    scores: list[float] = []
    labels: list[int] = []
    defect_types: list[str] = []
    anomaly_maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    latencies_ms: list[float] = []
    record_names: list[str] = []

    for sample in samples:
        image = load_image(sample.image_path, bundle.config.image)
        started = time.perf_counter()
        prediction = bundle.detector.predict(image)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        scores.append(prediction.score)
        labels.append(sample.label)
        defect_types.append(sample.defect_type)
        record_names.append(f"{sample.defect_type}/{sample.image_path.name}")
        if sample.mask_path is not None:
            anomaly_maps.append(prediction.anomaly_map)
            masks.append(load_mask(sample.mask_path, bundle.config.image))

    decision_metrics = binary_metrics(
        scores,
        labels,
        bundle.threshold,
        false_reject_cost=bundle.config.calibration.false_reject_cost,
        defect_escape_cost=bundle.config.calibration.defect_escape_cost,
    )
    slice_metrics: dict[str, Mapping[str, float | int | None]] = {}
    normal_indices = [index for index, value in enumerate(defect_types) if value == "good"]
    for defect_type in sorted(set(defect_types) - {"good"}):
        indices = normal_indices + [
            index for index, value in enumerate(defect_types) if value == defect_type
        ]
        slice_metrics[defect_type] = binary_metrics(
            [scores[index] for index in indices],
            [labels[index] for index in indices],
            bundle.threshold,
            false_reject_cost=bundle.config.calibration.false_reject_cost,
            defect_escape_cost=bundle.config.calibration.defect_escape_cost,
        )

    predicted = np.asarray(scores) >= bundle.threshold
    false_positive_indices = [
        index for index, value in enumerate(labels) if value == 0 and bool(predicted[index])
    ]
    false_negative_indices = [
        index for index, value in enumerate(labels) if value == 1 and not bool(predicted[index])
    ]
    false_positive_indices.sort(key=lambda index: scores[index], reverse=True)
    false_negative_indices.sort(key=lambda index: scores[index])

    report: dict[str, Any] = {
        "schema_version": 1,
        "category": category,
        "model_type": bundle.detector.model_type,
        "artifact_schema_version": bundle.schema_version,
        "operating_threshold": bundle.threshold,
        "image_level": decision_metrics,
        "defect_type_slices": slice_metrics,
        "score_distribution": {
            "normal": _score_summary(
                [score for score, label in zip(scores, labels) if label == 0]
            ),
            "defect": _score_summary(
                [score for score, label in zip(scores, labels) if label == 1]
            ),
        },
        "latency_ms": {
            "count": len(latencies_ms),
            "mean": float(statistics.fmean(latencies_ms)),
            "p50": float(np.quantile(latencies_ms, 0.50)),
            "p95": float(np.quantile(latencies_ms, 0.95)),
            "scope": "detector_only_excludes_decode_and_preprocess",
        },
        "pixel_level": (
            sampled_pixel_metrics(anomaly_maps, masks) if anomaly_maps else {"available": False}
        ),
        "ground_truth_masks": {
            "available": len(masks),
            "missing_for_defects": int(sum(labels) - len(masks)),
        },
        "error_cases": {
            "false_positives": [record_names[index] for index in false_positive_indices[:10]],
            "false_negatives": [record_names[index] for index in false_negative_indices[:10]],
        },
        "evaluation_note": (
            "The artifact threshold is evaluated without recalibration on this test set. "
            "Do not tune on these results and report them as holdout performance."
        ),
    }
    return report


def predict_to_directory(
    artifact_dir: Path | str,
    image_path: Path | str,
    output_dir: Path | str,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Inspect one image and save a heatmap overlay plus machine-readable result."""

    service = InspectionService.from_artifact(artifact_dir)
    result = service.inspect_path(image_path, threshold=threshold)
    destination = Path(output_dir)
    if destination.is_symlink():
        raise DataValidationError(f"Refusing symlinked output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    overlay_path = destination / "overlay.png"
    result.overlay.save(overlay_path, format="PNG")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "input_filename": Path(image_path).name,
        "score": result.score,
        "threshold": result.threshold,
        "decision": "anomaly" if result.is_anomaly else "normal",
        "model_type": result.model_type,
        "artifact_schema_version": result.artifact_schema_version,
        "overlay": overlay_path.name,
    }
    atomic_write_json(destination / "prediction.json", payload)
    return payload


def run_smoke(output_dir: Path | str, config: ProjectConfig) -> dict[str, Any]:
    """Run synthetic generation, training, evaluation, and prediction end to end."""

    root = Path(output_dir)
    if root.is_symlink():
        raise DataValidationError(f"Refusing symlinked smoke output directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    dataset_root = root / "synthetic-mvtec"
    category = "widget"
    category_root = generate_synthetic_mvtec(dataset_root, config.image, category=category)
    artifact_dir = root / "artifact"
    training = train_statistical(
        category_root / "train" / "good",
        artifact_dir,
        config,
        calibration_normal_dir=category_root / "validation" / "good",
        overwrite=True,
    )
    evaluation = evaluate_mvtec(artifact_dir, dataset_root, category)
    defect_image = sorted((category_root / "test" / "scratch").glob("*.png"))[0]
    prediction = predict_to_directory(artifact_dir, defect_image, root / "prediction")
    report = {
        "schema_version": 1,
        "status": "passed",
        "training": training.to_dict(),
        "evaluation": evaluation,
        "sample_prediction": prediction,
        "warning": "Synthetic smoke results verify plumbing only and are not model-quality claims.",
    }
    atomic_write_json(root / "smoke_report.json", report)
    return report


def report_as_json(report: Mapping[str, Any]) -> str:
    """Serialize an operator report consistently for stdout."""

    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
