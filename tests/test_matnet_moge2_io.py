from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from Material_net.dpt import MaterialNet
from myutils.camera_utils import GeoCalibResult, write_materialist_camera_json
from test_matnet_infer_moge2 import (
    _write_rgb_png,
    load_input_rgb,
    linear_to_srgb_standard,
    srgb_to_linear_standard,
)


class TargetColorIOTest(unittest.TestCase):
    def test_srgb_linear_png_round_trip(self) -> None:
        rgb = np.array(
            [[[0, 64, 255], [12, 128, 200]]],
            dtype=np.uint8,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            output = Path(directory) / "output.png"
            self.assertTrue(
                cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            )
            linear, moge_u8, info = load_input_rgb(source)
            expected = srgb_to_linear_standard(rgb.astype(np.float32) / 255.0)
            np.testing.assert_allclose(linear, expected, atol=1e-7)
            np.testing.assert_array_equal(moge_u8, rgb)
            self.assertEqual(info["source_color_space"], "srgb")

            saved = _write_rgb_png(
                output,
                np.clip(linear_to_srgb_standard(linear), 0.0, 1.0),
            )
            np.testing.assert_array_equal(saved, rgb)

    def test_uint16_png_uses_dtype_range_not_255(self) -> None:
        rgb = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source16.png"
            self.assertTrue(
                cv2.imwrite(str(source), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            )
            linear, moge_u8, _ = load_input_rgb(source)
            expected_srgb = rgb.astype(np.float32) / 65535.0
            np.testing.assert_allclose(
                linear,
                srgb_to_linear_standard(expected_srgb),
                atol=1e-6,
            )
            np.testing.assert_array_equal(moge_u8, [[[0, 128, 255]]])

    def test_scalar_preview_round_trips_through_blender_srgb(self) -> None:
        linear = np.array([0.07, 0.18, 0.5, 1.0], dtype=np.float32)
        png_codes = np.clip(linear_to_srgb_standard(linear), 0.0, 1.0)
        png_codes = np.round(png_codes * 255.0).astype(np.uint8)
        decoded = srgb_to_linear_standard(png_codes.astype(np.float32) / 255.0)
        np.testing.assert_allclose(decoded, linear, atol=0.004)

    def test_transparent_input_is_not_silently_discarded(self) -> None:
        bgra = np.array([[[0, 0, 255, 128]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "transparent.png"
            self.assertTrue(cv2.imwrite(str(source), bgra))
            with self.assertRaisesRegex(ValueError, "non-opaque alpha"):
                load_input_rgb(source)


class _DummyMaterialNet(torch.nn.Module):
    infer_image_scaled = MaterialNet.infer_image_scaled

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen_input = None

    def forward(self, image):
        self.seen_input = image.detach().cpu()
        batch, _, height, width = image.shape
        one = torch.ones((batch, 1, height, width), device=image.device)
        normal = torch.zeros((batch, 3, height, width), device=image.device)
        normal[:, 2] = 1.0
        return {
            "depth": one,
            "albedo": one.repeat(1, 3, 1, 1),
            "roughness": one,
            "metallic": one,
            "normal": normal,
        }


class MatNetFloatInputTest(unittest.TestCase):
    def test_float_hdr_is_not_divided_by_255(self) -> None:
        model = _DummyMaterialNet().eval()
        raw = np.full((2, 3, 3), 20.0, dtype=np.float32)
        with self.assertWarnsRegex(UserWarning, "HDR values above 10"):
            _, work, _ = model.infer_image_scaled(raw, scale=1.0)
        self.assertEqual(float(work.max()), 20.0)
        self.assertEqual(float(model.seen_input.max()), 20.0)


class CameraMetadataTest(unittest.TestCase):
    def test_moge_source_is_not_labeled_geocalib(self) -> None:
        camera = GeoCalibResult(
            width=4,
            height=2,
            K=np.array([[4, 0, 2], [0, 4, 1], [0, 0, 1]], dtype=np.float32),
            extra={"source": "moge2"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.json"
            write_materialist_camera_json(
                path,
                K_work=camera.K,
                work_hw=(2, 4),
                geocalib_result=camera,
            )
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(document["schema"], "materialist_intrinsics_v2")
            self.assertEqual(document["camera_source"], "moge2_intrinsics_only")
            self.assertFalse(document["pose_used_from_estimator"])
            self.assertNotIn("pose_used_from_geocalib", document)
            self.assertNotIn("geocalib_reported_roll_pitch_ignored", document)


if __name__ == "__main__":
    unittest.main()
