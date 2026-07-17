"""Versioned, integrity-checked, non-pickle detector artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from zerodefect_ai import __version__
from zerodefect_ai.config import ProjectConfig
from zerodefect_ai.detectors.statistical import StatisticalDetector
from zerodefect_ai.errors import ArtifactError, DataValidationError

ARTIFACT_SCHEMA_VERSION = 1
MODEL_FILENAME = "model.npz"
METADATA_FILENAME = "metadata.json"
MANIFEST_FILENAME = "manifest.json"
MAX_JSON_BYTES = 1_048_576


@dataclass(frozen=True)
class ArtifactBundle:
    """Validated detector and its immutable decision metadata."""

    detector: StatisticalDetector
    config: ProjectConfig
    threshold: float
    metadata: Mapping[str, Any]
    schema_version: int


def _canonical_json_bytes(values: Mapping[str, Any]) -> bytes:
    return (json.dumps(values, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def atomic_write_json(path: Path, values: Mapping[str, Any]) -> None:
    """Write JSON atomically so interrupted jobs do not leave a partial report."""

    _atomic_write_bytes(path, _canonical_json_bytes(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_name = handle.name
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def save_artifact(
    artifact_dir: Path | str,
    detector: StatisticalDetector,
    config: ProjectConfig,
    threshold: float,
    *,
    training_summary: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Persist a fitted detector with config, decision threshold, and integrity manifest."""

    root = Path(artifact_dir)
    if root.is_symlink():
        raise ArtifactError(f"Refusing symlinked artifact directory: {root}")
    if root.exists() and not root.is_dir():
        raise ArtifactError(f"Artifact path is not a directory: {root}")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ArtifactError(f"Artifact directory is not empty; pass --overwrite to replace: {root}")
    root.mkdir(parents=True, exist_ok=True)

    if not np.isfinite(threshold):
        raise ArtifactError("Artifact threshold must be finite")
    arrays = detector.to_arrays()
    for name, array in arrays.items():
        if np.asarray(array).dtype.hasobject:
            raise ArtifactError(f"Object array is forbidden in artifacts: {name}")
        if np.asarray(array).nbytes > config.artifact.max_model_bytes:
            raise ArtifactError(f"Model array exceeds artifact limit: {name}")

    model_path = root / MODEL_FILENAME
    metadata_path = root / METADATA_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    _write_npz_atomic(model_path, arrays)
    if model_path.stat().st_size > config.artifact.max_model_bytes:
        model_path.unlink(missing_ok=True)
        raise ArtifactError("Compressed model exceeds configured artifact limit")

    metadata: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "package_version": __version__,
        "model_type": detector.model_type,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "config": config.to_dict(),
        "training_summary": dict(training_summary),
    }
    _atomic_write_bytes(metadata_path, _canonical_json_bytes(metadata))
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "algorithm": "sha256",
        "files": {
            MODEL_FILENAME: _sha256(model_path),
            METADATA_FILENAME: _sha256(metadata_path),
        },
    }
    _atomic_write_bytes(manifest_path, _canonical_json_bytes(manifest))
    return root


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ArtifactError(f"Artifact JSON file missing or unsafe: {path.name}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ArtifactError(f"Artifact JSON exceeds safety limit: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"Cannot parse artifact JSON {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ArtifactError(f"Artifact JSON must contain an object: {path.name}")
    return value


def _verify_manifest(root: Path) -> None:
    manifest = _load_json(root / MANIFEST_FILENAME)
    if set(manifest) != {"schema_version", "algorithm", "files"}:
        raise ArtifactError("Artifact manifest has unexpected fields")
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError("Unsupported artifact manifest schema version")
    if manifest.get("algorithm") != "sha256":
        raise ArtifactError("Unsupported artifact digest algorithm")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {MODEL_FILENAME, METADATA_FILENAME}:
        raise ArtifactError("Artifact manifest file list is invalid")
    for filename, expected_digest in files.items():
        if not isinstance(expected_digest, str) or len(expected_digest) != 64:
            raise ArtifactError(f"Invalid checksum for artifact file: {filename}")
        artifact_file = root / filename
        if not artifact_file.is_file() or artifact_file.is_symlink():
            raise ArtifactError(f"Artifact file missing or unsafe: {filename}")
        if not hmac.compare_digest(_sha256(artifact_file), expected_digest):
            raise ArtifactError(f"Artifact checksum mismatch: {filename}")


def _preflight_npz(path: Path, config: ProjectConfig) -> None:
    if path.stat().st_size > config.artifact.max_model_bytes:
        raise ArtifactError("Model artifact exceeds compressed-size limit")
    expected_members = {"mean.npy", "std.npy", "training_count.npy"}
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if {member.filename for member in members} != expected_members:
                raise ArtifactError("Model NPZ contains unexpected members")
            expanded_size = sum(member.file_size for member in members)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArtifactError(f"Invalid model NPZ archive: {exc}") from exc

    expected_array_bytes = config.image.height * config.image.width * 3 * 4 * 2 + 8
    if expanded_size > expected_array_bytes + 64 * 1024:
        raise ArtifactError("Model NPZ expanded size exceeds the configured model shape")


def load_artifact(artifact_dir: Path | str) -> ArtifactBundle:
    """Load and validate a supported detector artifact without pickle execution."""

    root = Path(artifact_dir)
    if not root.is_dir() or root.is_symlink():
        raise ArtifactError(f"Artifact directory not found or unsafe: {root}")
    _verify_manifest(root)
    metadata = _load_json(root / METADATA_FILENAME)
    required_metadata = {
        "schema_version",
        "package_version",
        "model_type",
        "created_at_utc",
        "threshold",
        "config",
        "training_summary",
    }
    if set(metadata) != required_metadata:
        raise ArtifactError("Artifact metadata has missing or unexpected fields")
    if metadata.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError("Unsupported artifact metadata schema version")
    if metadata.get("model_type") != StatisticalDetector.model_type:
        raise ArtifactError(f"Unsupported detector model type: {metadata.get('model_type')!r}")
    config_value = metadata.get("config")
    if not isinstance(config_value, Mapping):
        raise ArtifactError("Artifact config must be an object")
    try:
        config = ProjectConfig.from_mapping(config_value)
    except Exception as exc:
        raise ArtifactError(f"Artifact configuration is invalid: {exc}") from exc
    threshold_value = metadata.get("threshold")
    if isinstance(threshold_value, bool) or not isinstance(threshold_value, (int, float)):
        raise ArtifactError("Artifact threshold must be numeric")
    threshold = float(threshold_value)
    if not np.isfinite(threshold):
        raise ArtifactError("Artifact threshold must be finite")

    model_path = root / MODEL_FILENAME
    _preflight_npz(model_path, config)
    try:
        with np.load(model_path, allow_pickle=False, max_header_size=10_000) as archive:
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
        detector = StatisticalDetector.from_arrays(config.image, config.detector, arrays)
    except (OSError, ValueError, KeyError, DataValidationError) as exc:
        raise ArtifactError(f"Model artifact arrays are invalid: {exc}") from exc

    return ArtifactBundle(
        detector=detector,
        config=config,
        threshold=threshold,
        metadata=metadata,
        schema_version=ARTIFACT_SCHEMA_VERSION,
    )
