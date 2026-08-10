from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hybrid_light.coordinate import (
    ile_to_materialist_points,
    project_materialist_points,
    unproject_materialist_pixels,
)
from hybrid_light.io import load_ile_lights


class HybridCoordinateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.K = np.array(
            [[300.0, 0.0, 160.0], [0.0, 300.0, 120.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def test_center_projects_to_principal_point(self) -> None:
        uv = project_materialist_points(np.array([[0.0, 0.0, -3.0]]), self.K)
        np.testing.assert_allclose(uv[0], [160.0, 120.0], atol=1e-5)

    def test_positive_x_moves_right_and_positive_y_moves_up(self) -> None:
        points = np.array([[0.5, 0.0, -3.0], [0.0, 0.5, -3.0]])
        uv = project_materialist_points(points, self.K)
        self.assertGreater(uv[0, 0], 160.0)
        self.assertLess(uv[1, 1], 120.0)

    def test_projection_round_trip(self) -> None:
        points = np.array([[0.4, 0.2, -2.0], [-0.5, -0.1, -4.0]], dtype=np.float32)
        pixels = project_materialist_points(points, self.K)
        reconstructed = unproject_materialist_pixels(pixels, -points[:, 2], self.K)
        np.testing.assert_allclose(reconstructed, points, atol=1e-5)

    def test_scale_applies_to_all_coordinates(self) -> None:
        point = np.array([1.0, 2.0, -3.0], dtype=np.float32)
        np.testing.assert_allclose(
            ile_to_materialist_points(point, geometry_scale=2.5),
            point * 2.5,
        )


class ILELoaderTest(unittest.TestCase):
    def test_current_export_shape_and_depth_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "lamp.obj").write_text(
                "v 0 0 -1\nv 1 0 -1\nv 0 1 -1\nf 1 2 3\n",
                encoding="utf-8",
            )
            document = {
                "schema_version": 1,
                "camera": {"depth_normalized": True, "depth_scale": 1.5},
                "inputs": {},
                "lights": {
                    "visible_lamps": [
                        {
                            "id": 0,
                            "center": [[0.0, 0.0, -2.0]],
                            "src": [[3.0, 2.0, 1.0]],
                            "geometry_file": "lamp.obj",
                        }
                    ]
                },
            }
            json_path = root / "light_predictions.json"
            json_path.write_text(json.dumps(document), encoding="utf-8")

            result = load_ile_lights(json_path, include_invisible=False)
            self.assertEqual(len(result.lights), 1)
            np.testing.assert_allclose(result.lights[0].scaled_center, [0.0, 0.0, -3.0])
            np.testing.assert_allclose(result.lights[0].rgb, [3.0, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
