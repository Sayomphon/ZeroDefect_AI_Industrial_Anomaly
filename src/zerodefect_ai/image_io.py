"""Hardened image decoding and deterministic preprocessing."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from zerodefect_ai.config import ImageConfig
from zerodefect_ai.errors import DataValidationError

FORMAT_EXTENSIONS: Final[dict[str, set[str]]] = {
    "PNG": {".png"},
    "JPEG": {".jpg", ".jpeg"},
    "BMP": {".bmp"},
}


def allowed_extensions(config: ImageConfig) -> set[str]:
    """Return the extension allowlist implied by decoded image formats."""

    return {extension for fmt in config.allowed_formats for extension in FORMAT_EXTENSIONS[fmt]}


def _validate_source_path(path: Path, config: ImageConfig) -> None:
    if not path.is_file():
        raise DataValidationError(f"Image file not found: {path}")
    if path.is_symlink():
        raise DataValidationError(f"Refusing symlinked image: {path}")
    if path.suffix.lower() not in allowed_extensions(config):
        raise DataValidationError(f"Image extension is not allowed: {path.suffix}")
    size = path.stat().st_size
    if size <= 0:
        raise DataValidationError(f"Image file is empty: {path}")
    if size > config.max_file_bytes:
        raise DataValidationError(
            f"Image exceeds max_file_bytes ({size} > {config.max_file_bytes}): {path.name}"
        )


def _open_verified(path: Path, config: ImageConfig) -> Image.Image:
    _validate_source_path(path, config)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path, formats=list(config.allowed_formats)) as candidate:
                decoded_format = candidate.format
                if decoded_format not in config.allowed_formats:
                    raise DataValidationError(
                        f"Decoded format {decoded_format!r} is not allowed for {path.name}"
                    )
                pixels = candidate.width * candidate.height
                if pixels > config.max_pixels:
                    raise DataValidationError(
                        f"Decoded image exceeds max_pixels ({pixels} > {config.max_pixels})"
                    )
                candidate.verify()

            with Image.open(path, formats=list(config.allowed_formats)) as verified:
                verified.load()
                return verified.copy()
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise DataValidationError(
            f"Potential decompression-bomb image rejected: {path.name}"
        ) from exc
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DataValidationError(f"Cannot decode image {path.name}: {exc}") from exc


def preprocess_pil(
    image: Image.Image, config: ImageConfig, *, is_mask: bool = False
) -> np.ndarray:
    """Convert a decoded image into the configured model shape.

    Masks use nearest-neighbor interpolation to preserve discrete annotation values.
    """

    mode = "L" if is_mask else "RGB"
    converted = image.convert(mode)
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.LANCZOS
    target_size = (config.width, config.height)
    if config.resize_policy == "center_crop":
        resized = ImageOps.fit(converted, target_size, method=resample, centering=(0.5, 0.5))
    else:
        resized = converted.resize(target_size, resample=resample)

    array = np.asarray(resized)
    if is_mask:
        return np.asarray(array > 127, dtype=np.uint8)
    result = np.asarray(array, dtype=np.float32) / np.float32(255.0)
    return np.ascontiguousarray(result)


def load_image(path: Path | str, config: ImageConfig) -> np.ndarray:
    """Safely decode and preprocess an RGB image from disk."""

    return preprocess_pil(_open_verified(Path(path), config), config)


def load_mask(path: Path | str, config: ImageConfig) -> np.ndarray:
    """Safely decode and resize a binary segmentation mask."""

    return preprocess_pil(_open_verified(Path(path), config), config, is_mask=True)


def preprocess_array(array: np.ndarray, config: ImageConfig) -> np.ndarray:
    """Validate an in-memory UI image and apply the same preprocessing contract."""

    if not isinstance(array, np.ndarray):
        raise DataValidationError("Uploaded image must be a NumPy array")
    if array.ndim not in {2, 3}:
        raise DataValidationError("Uploaded image must have 2 or 3 dimensions")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise DataValidationError("Uploaded image dimensions must be non-zero")
    if array.shape[0] * array.shape[1] > config.max_pixels:
        raise DataValidationError("Uploaded image exceeds the decoded pixel limit")
    if array.ndim == 3 and array.shape[2] not in {1, 3, 4}:
        raise DataValidationError("Uploaded image must have 1, 3, or 4 channels")
    if not np.issubdtype(array.dtype, np.number):
        raise DataValidationError("Uploaded image must contain numeric pixels")
    if not np.all(np.isfinite(array)):
        raise DataValidationError("Uploaded image contains NaN or infinite values")

    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if np.issubdtype(array.dtype, np.floating):
        if minimum < 0.0 or maximum > 255.0:
            raise DataValidationError("Floating image values must be in [0, 1] or [0, 255]")
        scaled = array * 255.0 if maximum <= 1.0 else array
    else:
        if minimum < 0.0 or maximum > 255.0:
            raise DataValidationError("Integer image values must be in [0, 255]")
        scaled = array
    uint8_array = np.clip(scaled, 0, 255).astype(np.uint8)
    if uint8_array.ndim == 3 and uint8_array.shape[2] == 1:
        uint8_array = np.squeeze(uint8_array, axis=2)
    return preprocess_pil(Image.fromarray(uint8_array), config)
