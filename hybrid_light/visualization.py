"""Small projection overlay used before trusting a hybrid render."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .coordinate import project_materialist_points
from .light_types import MeshAreaLight


def _obj_vertices(path: Path) -> np.ndarray | None:
    if path.suffix.lower() != ".obj":
        return None
    vertices = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            values = line.split()
            if values and values[0] == "v" and len(values) >= 4:
                vertices.append([float(value) for value in values[1:4]])
    return np.asarray(vertices, dtype=np.float32) if vertices else None


def render_projection_debug(
    image_path: str | Path,
    lights: list[MeshAreaLight],
    K: np.ndarray,
    film_wh: tuple[int, int],
    output_path: str | Path,
    *,
    visible_offset: float = 0.005,
) -> list[dict]:
    """Draw projected lamp centers/geometry and return projection statistics."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    width, height = [int(value) for value in film_wh]
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    report = []

    for light in lights:
        color = (40, 60, 255) if light.visible else (0, 160, 255)
        if light.mask_path is not None and light.mask_path.is_file():
            mask = cv2.imread(str(light.mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                selected = mask > 127
                overlay[selected] = (
                    0.65 * overlay[selected] + 0.35 * np.array([0, 255, 255])
                ).astype(np.uint8)

        center = light.scaled_center.copy()
        if light.visible and visible_offset:
            norm = float(np.linalg.norm(center))
            if norm > 1e-8:
                center += -center / norm * visible_offset
        uv = project_materialist_points(center[None], K)[0]
        inside = bool(
            np.isfinite(uv).all() and 0 <= uv[0] < width and 0 <= uv[1] < height
        )

        vertices = _obj_vertices(light.geometry_path)
        if vertices is not None:
            vertices *= light.geometry_scale
            if light.visible and visible_offset:
                norm = float(np.linalg.norm(light.scaled_center))
                if norm > 1e-8:
                    vertices += -light.scaled_center / norm * visible_offset
            projected = project_materialist_points(vertices, K)
            valid = np.isfinite(projected).all(axis=1)
            if np.count_nonzero(valid) >= 3:
                hull = cv2.convexHull(projected[valid].astype(np.float32)).astype(np.int32)
                cv2.polylines(overlay, [hull], True, color, 1, cv2.LINE_AA)

        if inside:
            point = tuple(np.rint(uv).astype(int))
            cv2.drawMarker(overlay, point, color, cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
            cv2.putText(
                overlay,
                light.name,
                (min(point[0] + 6, max(width - 150, 0)), max(point[1] - 7, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        report.append(
            {
                "name": light.name,
                "center_world": center.tolist(),
                "projected_uv": uv.tolist(),
                "inside_image": inside,
                "rgb": light.rgb.tolist(),
                "geometry": str(light.geometry_path),
            }
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Could not write {output_path}")
    return report
