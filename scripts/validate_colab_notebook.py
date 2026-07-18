"""Validate and optionally execute the Colab quickstart without notebook-only tooling."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REQUIRED_SNIPPETS = (
    "REPOSITORY_BRANCH = \"main\"",
    "pip\", \"install\", \".[demo]\"",
    "run_smoke(output_dir, config)",
    "ENABLE_PUBLIC_GRADIO = False",
    "confirm_public_share=True",
)
CANONICAL_NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks" / "00_colab_quickstart.ipynb"
)


def _load_notebook(path: Path) -> dict[str, Any]:
    try:
        raw_notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read notebook {path}: {exc}") from exc
    if not isinstance(raw_notebook, dict):
        raise ValueError("Colab notebook root must be an object")
    notebook: dict[str, Any] = raw_notebook
    if notebook.get("nbformat") != 4:
        raise ValueError("Colab notebook must use nbformat 4")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("Colab notebook must contain cells")
    return notebook


def validate_notebook(path: Path) -> list[dict[str, Any]]:
    """Return code cells after checking clean, deterministic notebook structure."""

    notebook = _load_notebook(path)
    code_cells: list[dict[str, Any]] = []
    combined_source: list[str] = []
    for index, cell in enumerate(notebook["cells"]):
        if not isinstance(cell, dict):
            raise ValueError(f"Cell {index} is not an object")
        source = cell.get("source", [])
        if not isinstance(source, list) or not all(isinstance(line, str) for line in source):
            raise ValueError(f"Cell {index} source must be a list of strings")
        combined_source.extend(source)
        if cell.get("cell_type") == "code":
            if cell.get("execution_count") is not None:
                raise ValueError(f"Code cell {index} contains an execution count")
            if cell.get("outputs") != []:
                raise ValueError(f"Code cell {index} contains saved output")
            compile("".join(source), f"{path}:cell-{index}", "exec")
            code_cells.append(cell)

    full_source = "".join(combined_source)
    missing = [snippet for snippet in REQUIRED_SNIPPETS if snippet not in full_source]
    if missing:
        raise ValueError(f"Notebook is missing required workflow snippets: {missing}")
    return code_cells


def execute_notebook(path: Path) -> None:
    """Execute clean code cells in one namespace while skipping dependency installation."""

    if path.resolve() != CANONICAL_NOTEBOOK:
        raise ValueError("Execution is restricted to the repository's canonical Colab notebook")
    code_cells = validate_notebook(path)
    original_directory = Path.cwd()
    previous_skip_install = os.environ.get("ZERO_DEFECT_SKIP_INSTALL")
    os.environ["ZERO_DEFECT_SKIP_INSTALL"] = "1"
    namespace: dict[str, Any] = {"__name__": "__main__"}
    try:
        for index, cell in enumerate(code_cells):
            source = "".join(cell["source"])
            # The path is fixed to a reviewed repository file above;
            # arbitrary notebooks are refused.
            exec(  # noqa: S102  # nosec B102
                compile(source, f"{path}:cell-{index}", "exec"), namespace
            )
    finally:
        os.chdir(original_directory)
        if previous_skip_install is None:
            os.environ.pop("ZERO_DEFECT_SKIP_INSTALL", None)
        else:
            os.environ["ZERO_DEFECT_SKIP_INSTALL"] = previous_skip_install


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute code cells with dependency installation disabled",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    notebook = args.notebook.resolve()
    validate_notebook(notebook)
    if args.execute:
        execute_notebook(notebook)
    print(f"Colab notebook validation passed: {notebook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
