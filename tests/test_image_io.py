from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from zerodefect_ai.config import ImageConfig
from zerodefect_ai.errors import DataValidationError
from zerodefect_ai.image_io import load_image, preprocess_array


class ImageIoTest(unittest.TestCase):
    def test_loads_and_resizes_rgb(self) -> None:
        config = ImageConfig(width=20, height=16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.png"
            Image.fromarray(np.full((30, 40, 3), 128, dtype=np.uint8), mode="RGB").save(path)
            image = load_image(path, config)
            self.assertEqual(image.shape, (16, 20, 3))
            self.assertEqual(image.dtype, np.float32)

    def test_rejects_symlinked_image(self) -> None:
        config = ImageConfig(width=20, height=16)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.png"
            Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8), mode="RGB").save(target)
            link = root / "link.png"
            link.symlink_to(target)
            with self.assertRaises(DataValidationError):
                load_image(link, config)

    def test_rejects_non_finite_array(self) -> None:
        array = np.zeros((10, 10, 3), dtype=np.float32)
        array[0, 0, 0] = np.nan
        with self.assertRaises(DataValidationError):
            preprocess_array(array, ImageConfig(width=20, height=16))

    def test_accepts_single_channel_array(self) -> None:
        array = np.full((10, 10, 1), 128, dtype=np.uint8)
        image = preprocess_array(array, ImageConfig(width=20, height=16))
        self.assertEqual(image.shape, (16, 20, 3))

    def test_rejects_integer_values_above_byte_range(self) -> None:
        array = np.full((10, 10, 3), 1024, dtype=np.uint16)
        with self.assertRaises(DataValidationError):
            preprocess_array(array, ImageConfig(width=20, height=16))


if __name__ == "__main__":
    unittest.main()
