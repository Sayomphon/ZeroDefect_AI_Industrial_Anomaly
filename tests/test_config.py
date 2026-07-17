from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from zerodefect_ai.config import ProjectConfig
from zerodefect_ai.errors import ConfigurationError


class ProjectConfigTest(unittest.TestCase):
    def test_loads_repository_config(self) -> None:
        config = ProjectConfig.from_toml(Path("configs/base.toml"))
        self.assertEqual(config.image.width, 128)
        self.assertEqual(config.image.allowed_formats, ("PNG", "JPEG", "BMP"))
        self.assertEqual(config.calibration.method, "normal_quantile")

    def test_rejects_unknown_keys(self) -> None:
        with self.assertRaises(ConfigurationError):
            ProjectConfig.from_mapping({"image": {"widht": 128}})

    def test_rejects_symlinked_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "config.toml"
            target.write_text("[image]\nwidth = 128\n", encoding="utf-8")
            link = root / "linked.toml"
            link.symlink_to(target)
            with self.assertRaises(ConfigurationError):
                ProjectConfig.from_toml(link)


if __name__ == "__main__":
    unittest.main()
