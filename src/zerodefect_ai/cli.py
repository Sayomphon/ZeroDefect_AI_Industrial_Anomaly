"""Command-line interface for reproducible local and CI workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from zerodefect_ai.artifacts import atomic_write_json
from zerodefect_ai.config import ProjectConfig
from zerodefect_ai.errors import ZeroDefectError
from zerodefect_ai.workflows import (
    evaluate_mvtec,
    predict_to_directory,
    report_as_json,
    run_smoke,
    train_statistical,
)

DEFAULT_CONFIG = Path("configs/base.toml")


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="TOML configuration path (default: configs/base.toml)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zerodefect",
        description="Cold-start industrial visual anomaly detection workflows",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Fit and calibrate a normal-only baseline")
    train.add_argument("--normal-dir", type=Path, required=True)
    train.add_argument("--calibration-normal-dir", type=Path)
    train.add_argument("--artifact-dir", type=Path, required=True)
    train.add_argument("--overwrite", action="store_true")
    _add_config_argument(train)

    predict = subparsers.add_parser("predict", help="Inspect one image")
    predict.add_argument("--artifact-dir", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--threshold", type=float)

    evaluate = subparsers.add_parser("evaluate-mvtec", help="Evaluate on one MVTec category")
    evaluate.add_argument("--artifact-dir", type=Path, required=True)
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--category", required=True)
    evaluate.add_argument("--output-json", type=Path, required=True)

    smoke = subparsers.add_parser("smoke", help="Run the complete pipeline on synthetic data")
    smoke.add_argument("--output-dir", type=Path, required=True)
    _add_config_argument(smoke)
    return parser


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "train":
        config = ProjectConfig.from_toml(args.config)
        return train_statistical(
            args.normal_dir,
            args.artifact_dir,
            config,
            calibration_normal_dir=args.calibration_normal_dir,
            overwrite=args.overwrite,
        ).to_dict()
    if args.command == "predict":
        return predict_to_directory(
            args.artifact_dir,
            args.image,
            args.output_dir,
            threshold=args.threshold,
        )
    if args.command == "evaluate-mvtec":
        report = evaluate_mvtec(args.artifact_dir, args.dataset_root, args.category)
        atomic_write_json(args.output_json, report)
        return report
    if args.command == "smoke":
        config = ProjectConfig.from_toml(args.config)
        return run_smoke(args.output_dir, config)
    raise RuntimeError(f"Unhandled CLI command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = _run(args)
    except ZeroDefectError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(report_as_json(report))
    return 0
