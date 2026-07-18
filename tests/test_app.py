from __future__ import annotations

import gc
import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from app.app import build_app, launch_colab_app
from zerodefect_ai.config import DetectorConfig, ImageConfig, ProjectConfig
from zerodefect_ai.workflows import run_smoke


class _FakeDemo:
    def __init__(self) -> None:
        self.launch_options: dict[str, object] | None = None

    def launch(self, **kwargs: object) -> dict[str, object]:
        self.launch_options = kwargs
        return kwargs


class ColabLaunchTest(unittest.TestCase):
    def test_public_share_requires_explicit_confirmation(self) -> None:
        with self.assertRaisesRegex(ValueError, "public Gradio share URL"):
            launch_colab_app(
                "artifact", username="operator", password="long-password"  # noqa: S106
            )

    def test_public_share_requires_stronger_password(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 12 characters"):
            launch_colab_app(
                "artifact",
                username="operator",
                password="too-short",  # noqa: S106
                confirm_public_share=True,
            )

    def test_colab_launch_is_authenticated_and_resource_bounded(self) -> None:
        fake_demo = _FakeDemo()
        with patch("app.app.build_app", return_value=fake_demo):
            result = launch_colab_app(
                "artifact",
                username=" operator ",
                password="correct-horse-battery",  # noqa: S106
                confirm_public_share=True,
            )

        self.assertEqual(result["auth"], ("operator", "correct-horse-battery"))
        self.assertIs(result["share"], True)
        self.assertIs(result["inline"], True)
        self.assertIs(result["debug"], True)
        self.assertEqual(result["max_threads"], 4)
        self.assertIs(result["show_error"], False)

    @unittest.skipUnless(importlib.util.find_spec("gradio"), "Gradio optional dependency absent")
    def test_real_gradio_app_builds_from_valid_artifact(self) -> None:
        config = ProjectConfig(
            image=ImageConfig(width=32, height=32),
            detector=DetectorConfig(
                minimum_training_images=3,
                minimum_std=0.02,
                image_score_quantile=0.99,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "smoke"
            run_smoke(output_dir, config)
            # Gradio 6.20 may leave internal asyncio loops for garbage collection after
            # a build-only test. The production loopback smoke launches and closes a real
            # server separately; this test only verifies Blocks construction.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="unclosed event loop.*", category=ResourceWarning
                )
                demo = build_app(output_dir / "artifact")
                try:
                    self.assertTrue(callable(demo.launch))
                finally:
                    demo.close()
                    gc.collect()


if __name__ == "__main__":
    unittest.main()
