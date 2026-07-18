"""Strict, dependency-light configuration loading from TOML."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Set

from zerodefect_ai.errors import ConfigurationError

MAX_CONFIG_BYTES = 1_048_576


def _expect_keys(section: str, values: Mapping[str, Any], allowed: Set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ConfigurationError(f"Unknown keys in [{section}]: {', '.join(unknown)}")


def _require_int(name: str, value: Any, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return value


def _require_float(name: str, value: Any, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{name} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise ConfigurationError(f"{name} must be between {minimum} and {maximum}")
    return result


@dataclass(frozen=True)
class ImageConfig:
    """Image decoding and deterministic preprocessing limits."""

    width: int = 128
    height: int = 128
    resize_policy: str = "center_crop"
    max_file_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 16 * 1024 * 1024
    allowed_formats: tuple[str, ...] = ("PNG", "JPEG", "BMP")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ImageConfig":
        _expect_keys(
            "image",
            values,
            {
                "width",
                "height",
                "resize_policy",
                "max_file_bytes",
                "max_pixels",
                "allowed_formats",
            },
        )
        resize_policy = values.get("resize_policy", "center_crop")
        if resize_policy not in {"center_crop", "stretch"}:
            raise ConfigurationError("image.resize_policy must be 'center_crop' or 'stretch'")

        raw_formats = values.get("allowed_formats", ["PNG", "JPEG", "BMP"])
        if not isinstance(raw_formats, list) or not raw_formats:
            raise ConfigurationError("image.allowed_formats must be a non-empty list")
        formats: list[str] = []
        supported = {"PNG", "JPEG", "BMP"}
        for raw_format in raw_formats:
            if not isinstance(raw_format, str):
                raise ConfigurationError("Every image.allowed_formats value must be a string")
            image_format = raw_format.upper()
            if image_format not in supported:
                raise ConfigurationError(
                    f"Unsupported image format {image_format!r}; supported: {sorted(supported)}"
                )
            if image_format not in formats:
                formats.append(image_format)

        return cls(
            width=_require_int("image.width", values.get("width", 128), 16, 4096),
            height=_require_int("image.height", values.get("height", 128), 16, 4096),
            resize_policy=resize_policy,
            max_file_bytes=_require_int(
                "image.max_file_bytes", values.get("max_file_bytes", 20 * 1024 * 1024), 1, 1 << 30
            ),
            max_pixels=_require_int(
                "image.max_pixels", values.get("max_pixels", 16 * 1024 * 1024), 256, 1 << 30
            ),
            allowed_formats=tuple(formats),
        )


@dataclass(frozen=True)
class DetectorConfig:
    """Parameters for the normal-only statistical detector."""

    minimum_training_images: int = 3
    minimum_std: float = 0.02
    image_score_quantile: float = 0.995

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "DetectorConfig":
        _expect_keys(
            "detector", values, {"minimum_training_images", "minimum_std", "image_score_quantile"}
        )
        return cls(
            minimum_training_images=_require_int(
                "detector.minimum_training_images",
                values.get("minimum_training_images", 3),
                2,
                1_000_000,
            ),
            minimum_std=_require_float(
                "detector.minimum_std", values.get("minimum_std", 0.02), 1e-6, 1.0
            ),
            image_score_quantile=_require_float(
                "detector.image_score_quantile",
                values.get("image_score_quantile", 0.995),
                0.5,
                1.0,
            ),
        )


@dataclass(frozen=True)
class CalibrationConfig:
    """Decision-policy parameters kept separate from the detector score."""

    method: str = "normal_quantile"
    normal_quantile: float = 0.995
    false_reject_cost: float = 1.0
    defect_escape_cost: float = 25.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CalibrationConfig":
        _expect_keys(
            "calibration",
            values,
            {"method", "normal_quantile", "false_reject_cost", "defect_escape_cost"},
        )
        method = values.get("method", "normal_quantile")
        if method not in {"normal_quantile", "f1", "business_cost"}:
            raise ConfigurationError(
                "calibration.method must be 'normal_quantile', 'f1', or 'business_cost'"
            )
        false_reject_cost = _require_float(
            "calibration.false_reject_cost", values.get("false_reject_cost", 1.0), 0.0, 1e12
        )
        defect_escape_cost = _require_float(
            "calibration.defect_escape_cost", values.get("defect_escape_cost", 25.0), 0.0, 1e12
        )
        if false_reject_cost == 0.0 and defect_escape_cost == 0.0:
            raise ConfigurationError("At least one business cost must be greater than zero")
        return cls(
            method=method,
            normal_quantile=_require_float(
                "calibration.normal_quantile", values.get("normal_quantile", 0.995), 0.5, 1.0
            ),
            false_reject_cost=false_reject_cost,
            defect_escape_cost=defect_escape_cost,
        )


@dataclass(frozen=True)
class ArtifactConfig:
    """Resource limit for model artifact loading."""

    max_model_bytes: int = 256 * 1024 * 1024

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ArtifactConfig":
        _expect_keys("artifact", values, {"max_model_bytes"})
        return cls(
            max_model_bytes=_require_int(
                "artifact.max_model_bytes",
                values.get("max_model_bytes", 256 * 1024 * 1024),
                1024,
                4 * 1024 * 1024 * 1024,
            )
        )


@dataclass(frozen=True)
class ProjectConfig:
    """Versionable configuration for training and inference."""

    image: ImageConfig = ImageConfig()
    detector: DetectorConfig = DetectorConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    artifact: ArtifactConfig = ArtifactConfig()

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ProjectConfig":
        _expect_keys("root", values, {"image", "detector", "calibration", "artifact"})
        sections: Dict[str, Mapping[str, Any]] = {}
        for section in ("image", "detector", "calibration", "artifact"):
            raw = values.get(section, {})
            if not isinstance(raw, Mapping):
                raise ConfigurationError(f"[{section}] must be a TOML table")
            sections[section] = raw
        return cls(
            image=ImageConfig.from_mapping(sections["image"]),
            detector=DetectorConfig.from_mapping(sections["detector"]),
            calibration=CalibrationConfig.from_mapping(sections["calibration"]),
            artifact=ArtifactConfig.from_mapping(sections["artifact"]),
        )

    @classmethod
    def from_toml(cls, path: Path | str) -> "ProjectConfig":
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        if config_path.is_symlink():
            raise ConfigurationError(f"Refusing symlinked configuration: {config_path}")
        if config_path.stat().st_size > MAX_CONFIG_BYTES:
            raise ConfigurationError("Configuration file exceeds the 1 MiB safety limit")
        try:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"Cannot parse configuration: {exc}") from exc
        return cls.from_mapping(raw)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation for artifact metadata."""

        values = asdict(self)
        values["image"]["allowed_formats"] = list(self.image.allowed_formats)
        return values
