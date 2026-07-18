"""Optional local Gradio dashboard backed by the framework-free service layer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from zerodefect_ai.errors import ZeroDefectError
from zerodefect_ai.service import InspectionService


def build_app(artifact_dir: Path | str) -> Any:
    """Create a Gradio app without importing Gradio into the production core."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError('Gradio is optional; install with: pip install -e ".[demo]"') from exc

    service = InspectionService.from_artifact(artifact_dir)
    default_threshold = service.default_threshold

    def inspect(image: np.ndarray | None, threshold: float) -> tuple[Any, str, str]:
        if image is None:
            return None, "No image", "{}"
        try:
            result = service.inspect_array(image, threshold=threshold)
        except ZeroDefectError as exc:
            return None, f"Rejected input: {exc}", "{}"
        status = "ANOMALY" if result.is_anomaly else "NORMAL"
        payload = {
            "score": result.score,
            "threshold": result.threshold,
            "decision": status.lower(),
            "model_type": result.model_type,
        }
        return (
            result.overlay,
            f"{status} — score={result.score:.4f}",
            json.dumps(payload, indent=2),
        )

    maximum_threshold = max(10.0, default_threshold * 3.0)
    with gr.Blocks(title="ZeroDefect AI") as demo:
        gr.Markdown(
            "# ZeroDefect AI\n"
            "Cold-start visual anomaly inspection. Heatmaps indicate deviation, not causal proof."
        )
        with gr.Row():
            image_input = gr.Image(type="numpy", image_mode="RGB", label="Inspection image")
            overlay_output = gr.Image(type="pil", label="Anomaly heatmap overlay")
        threshold_input = gr.Slider(
            minimum=0.0,
            maximum=maximum_threshold,
            value=default_threshold,
            step=max(maximum_threshold / 1000.0, 0.001),
            label="QC decision threshold",
        )
        inspect_button = gr.Button("Inspect", variant="primary")
        decision_output = gr.Textbox(label="Decision", interactive=False)
        json_output = gr.Code(label="Prediction JSON", language="json")
        inspect_button.click(
            inspect,
            inputs=[image_input, threshold_input],
            outputs=[overlay_output, decision_output, json_output],
        )
    return demo


def main() -> None:
    artifact_dir = os.environ.get("ZERO_DEFECT_ARTIFACT_DIR")
    if not artifact_dir:
        raise SystemExit("Set ZERO_DEFECT_ARTIFACT_DIR to a validated artifact directory")
    server_name = os.environ.get("ZERO_DEFECT_SERVER_NAME", "127.0.0.1")
    try:
        server_port = int(os.environ.get("ZERO_DEFECT_SERVER_PORT", "7860"))
    except ValueError as exc:
        raise SystemExit("ZERO_DEFECT_SERVER_PORT must be an integer") from exc
    if not 1024 <= server_port <= 65535:
        raise SystemExit("ZERO_DEFECT_SERVER_PORT must be between 1024 and 65535")
    build_app(artifact_dir).launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        show_error=False,
    )


if __name__ == "__main__":
    main()
