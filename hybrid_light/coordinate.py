"""Coordinate conversion and projection for the ILE/Materialist bridge.

Both current pipelines use camera-centered graphics coordinates after
Materialist rotates its OpenCV depth mesh by 180 degrees around X:

    +X right, +Y up, -Z forward.

The explicit conversion functions keep that assumption testable instead of
spreading implicit sign flips throughout the renderer.
"""

from __future__ import annotations

import numpy as np


def _points_array(points: np.ndarray) -> np.ndarray:
    array = np.asarray(points, dtype=np.float32)
    if array.shape[-1:] != (3,):
        raise ValueError(f"Expected points with final dimension 3, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Points must be finite")
    return array


def ile_to_materialist_points(
    points: np.ndarray,
    *,
    geometry_scale: float = 1.0,
) -> np.ndarray:
    """Convert ILE camera points into the current Materialist world frame."""
    if not np.isfinite(geometry_scale) or geometry_scale <= 0:
        raise ValueError("geometry_scale must be finite and positive")
    return _points_array(points) * np.float32(geometry_scale)


def project_materialist_points(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project +X-right/+Y-up/-Z-forward points to pixel ``(u, v)``."""
    points = _points_array(points)
    original_shape = points.shape[:-1]
    flat = points.reshape(-1, 3)
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("K must contain finite positive focal lengths")

    # Materialist world -> OpenCV camera: (x, y, z) -> (x, -y, -z).
    camera = flat * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    projected = (K @ camera.T).T
    valid = camera[:, 2] > 1e-8
    uv = np.full((len(flat), 2), np.nan, dtype=np.float32)
    uv[valid] = projected[valid, :2] / projected[valid, 2:3]
    return uv.reshape(*original_shape, 2)


def unproject_materialist_pixels(
    pixels: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Unproject pixel ``(u, v)`` and positive depth into Materialist space."""
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.shape[-1:] != (2,):
        raise ValueError(f"Expected pixels with final dimension 2, got {pixels.shape}")
    depth = np.asarray(depth, dtype=np.float32)
    if depth.shape != pixels.shape[:-1]:
        raise ValueError("depth shape must match pixels without its final dimension")
    if not np.isfinite(pixels).all() or not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Pixels must be finite and depth must be finite and positive")

    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float32).reshape(3, 3))
    homogeneous = np.concatenate(
        [pixels.reshape(-1, 2), np.ones((pixels.size // 2, 1), dtype=np.float32)],
        axis=1,
    )
    camera = (K_inv @ homogeneous.T).T * depth.reshape(-1, 1)
    internal = camera * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    return internal.reshape(*pixels.shape[:-1], 3)
