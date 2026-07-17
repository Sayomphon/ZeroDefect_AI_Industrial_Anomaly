"""Deterministic dataset discovery and MVTec AD layout mapping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from zerodefect_ai.config import ImageConfig
from zerodefect_ai.errors import DataValidationError
from zerodefect_ai.image_io import allowed_extensions

SAFE_CATEGORY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class MVTecSample:
    """One image-level test record and its optional pixel mask."""

    image_path: Path
    label: int
    defect_type: str
    mask_path: Path | None


def discover_images(
    root: Path | str,
    config: ImageConfig,
    *,
    recursive: bool = True,
    max_files: int = 100_000,
) -> list[Path]:
    """Find allowed image files without following symlinks."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise DataValidationError(f"Dataset directory not found: {root_path}")
    if root_path.is_symlink():
        raise DataValidationError(f"Refusing symlinked dataset root: {root_path}")

    extensions = allowed_extensions(config)
    candidates: Iterable[Path] = root_path.rglob("*") if recursive else root_path.iterdir()
    files: list[Path] = []
    for candidate in candidates:
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if candidate.suffix.lower() not in extensions:
            continue
        files.append(candidate)
        if len(files) > max_files:
            raise DataValidationError(f"Dataset exceeds the {max_files} file safety limit")
    files.sort(key=lambda item: item.as_posix())
    if not files:
        raise DataValidationError(f"No supported images found under: {root_path}")
    return files


def _safe_category_root(dataset_root: Path, category: str) -> Path:
    if not SAFE_CATEGORY.fullmatch(category):
        raise DataValidationError(f"Unsafe MVTec category name: {category!r}")
    if not dataset_root.is_dir() or dataset_root.is_symlink():
        raise DataValidationError(f"Invalid dataset root: {dataset_root}")
    category_root = dataset_root / category
    if not category_root.is_dir() or category_root.is_symlink():
        raise DataValidationError(f"MVTec category not found: {category_root}")
    return category_root


def mvtec_train_images(
    dataset_root: Path | str, category: str, config: ImageConfig
) -> list[Path]:
    """Return normal training images for one MVTec category."""

    root = _safe_category_root(Path(dataset_root), category)
    return discover_images(root / "train" / "good", config)


def mvtec_test_samples(
    dataset_root: Path | str, category: str, config: ImageConfig
) -> Sequence[MVTecSample]:
    """Map MVTec test folders to image labels and optional ground-truth masks."""

    root = _safe_category_root(Path(dataset_root), category)
    test_root = root / "test"
    if not test_root.is_dir() or test_root.is_symlink():
        raise DataValidationError(f"MVTec test directory not found: {test_root}")

    samples: list[MVTecSample] = []
    for defect_dir in sorted(test_root.iterdir(), key=lambda path: path.name):
        if defect_dir.is_symlink() or not defect_dir.is_dir():
            continue
        defect_type = defect_dir.name
        if not SAFE_CATEGORY.fullmatch(defect_type):
            raise DataValidationError(f"Unsafe defect type name: {defect_type!r}")
        for image_path in discover_images(defect_dir, config, recursive=False):
            mask_path: Path | None = None
            label = 0 if defect_type == "good" else 1
            if label == 1:
                expected_mask = root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                if expected_mask.is_file() and not expected_mask.is_symlink():
                    mask_path = expected_mask
            samples.append(
                MVTecSample(
                    image_path=image_path,
                    label=label,
                    defect_type=defect_type,
                    mask_path=mask_path,
                )
            )
    if not samples:
        raise DataValidationError(f"No MVTec test samples found for category: {category}")
    return samples
