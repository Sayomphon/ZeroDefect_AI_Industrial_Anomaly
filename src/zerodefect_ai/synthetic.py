"""Deterministic synthetic MVTec-like data for smoke testing only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from zerodefect_ai.config import ImageConfig


def _normal_image(config: ImageConfig, rng: np.random.Generator) -> np.ndarray:
    height, width = config.height, config.width
    y = np.linspace(-0.08, 0.08, height, dtype=np.float32)[:, None]
    x = np.linspace(-0.05, 0.05, width, dtype=np.float32)[None, :]
    base = np.clip(0.55 + y + x, 0.0, 1.0)
    noise = rng.normal(0.0, 0.008, size=(height, width)).astype(np.float32)
    gray = np.clip(base + noise, 0.0, 1.0)
    rgb = np.stack([gray, gray * 0.98, gray * 0.96], axis=2)
    return np.asarray(np.clip(rgb * 255.0, 0, 255), dtype=np.uint8)


def generate_synthetic_mvtec(
    dataset_root: Path | str,
    config: ImageConfig,
    *,
    category: str = "widget",
    seed: int = 20260717,
) -> Path:
    """Generate small normal/defect sets that verify plumbing, not model quality."""

    category_root = Path(dataset_root) / category
    train_good = category_root / "train" / "good"
    validation_good = category_root / "validation" / "good"
    test_good = category_root / "test" / "good"
    test_defect = category_root / "test" / "scratch"
    masks = category_root / "ground_truth" / "scratch"
    for directory in (train_good, validation_good, test_good, test_defect, masks):
        directory.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    for index in range(12):
        Image.fromarray(_normal_image(config, rng), mode="RGB").save(
            train_good / f"normal_{index:03d}.png"
        )
    for index in range(12):
        Image.fromarray(_normal_image(config, rng), mode="RGB").save(
            validation_good / f"validation_{index:03d}.png"
        )
    for index in range(4):
        Image.fromarray(_normal_image(config, rng), mode="RGB").save(
            test_good / f"good_{index:03d}.png"
        )
    for index in range(4):
        image = _normal_image(config, rng)
        mask = np.zeros((config.height, config.width), dtype=np.uint8)
        size = max(6, min(config.height, config.width) // 8)
        top = config.height // 3 + index
        left = config.width // 2 - size // 2 + index
        image[top : top + size, left : left + size, :] = np.asarray([245, 25, 25], dtype=np.uint8)
        mask[top : top + size, left : left + size] = 255
        Image.fromarray(image, mode="RGB").save(test_defect / f"defect_{index:03d}.png")
        Image.fromarray(mask, mode="L").save(masks / f"defect_{index:03d}_mask.png")
    return category_root
