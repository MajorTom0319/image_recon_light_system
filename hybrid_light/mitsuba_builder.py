"""Build a Materialist scene with ILE mesh area emitters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .ile_window_emitter import register_ile_window_emitter
from .light_types import MeshAreaLight


def _scalar(value: Any, default: float = 0.0) -> float:
    array = np.asarray(value).reshape(-1)
    return float(array[0]) if array.size else default


def build_sensor_dict(mi, camera_meta: dict[str, Any]) -> dict[str, Any]:
    width, height = [int(value) for value in camera_meta["film.size"]]
    to_world = np.asarray(camera_meta["to_world"], dtype=np.float32)
    while to_world.ndim > 2:
        to_world = to_world[0]
    if to_world.shape != (4, 4):
        raise ValueError(f"Expected camera to_world 4x4, got {to_world.shape}")

    sensor = {
        "type": "perspective",
        "fov": float(camera_meta["y_fov"][0]),
        "fov_axis": "y",
        "near_clip": float(camera_meta.get("near_clip", 0.01)),
        "far_clip": float(camera_meta.get("far_clip", 10000.0)),
        "to_world": mi.ScalarTransform4f(to_world),
        "film": {
            "type": "hdrfilm",
            "width": width,
            "height": height,
            "pixel_format": "rgb",
            "rfilter": {"type": "box"},
        },
    }
    offset_x = _scalar(camera_meta.get("principal_point_offset_x"), 0.0)
    offset_y = _scalar(camera_meta.get("principal_point_offset_y"), 0.0)
    if offset_x:
        sensor["principal_point_offset_x"] = offset_x
    if offset_y:
        sensor["principal_point_offset_y"] = offset_y
    return sensor


def _light_transform(mi, light: MeshAreaLight, visible_offset: float):
    scale = mi.ScalarTransform4f.scale([light.geometry_scale] * 3)
    if not light.visible or visible_offset == 0:
        return scale
    center = light.scaled_center
    distance = float(np.linalg.norm(center))
    if distance <= 1e-8:
        return scale
    toward_camera = -center / distance
    return mi.ScalarTransform4f.translate(toward_camera * visible_offset) @ scale


def build_hybrid_scene_dict(
    mi,
    *,
    mesh_path: str | Path,
    camera_meta_path: str | Path,
    camera_meta: dict[str, Any],
    lights: Sequence[MeshAreaLight],
    mode: str = "local",
    envmap_path: str | Path | None = None,
    radiance_scale: float = 1.0,
    visible_offset: float = 0.005,
    use_mesh_normal: bool = True,
    max_depth: int = 8,
) -> dict[str, Any]:
    """Create a Mitsuba dictionary for local, env, or combined lighting."""
    if mode not in {"local", "env", "combined"}:
        raise ValueError(f"Unsupported rendering mode: {mode}")
    if not np.isfinite(radiance_scale) or radiance_scale <= 0:
        raise ValueError("radiance_scale must be finite and positive")
    if not np.isfinite(visible_offset) or visible_offset < 0:
        raise ValueError("visible_offset must be finite and non-negative")
    if mode in {"env", "combined"} and envmap_path is None:
        raise ValueError(f"mode={mode!r} requires envmap_path")
    if mode == "local" and not lights:
        raise ValueError("Local-only rendering requires at least one lamp")
    if any(light.is_window for light in lights):
        register_ile_window_emitter(mi)

    scene: dict[str, Any] = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": int(max_depth)},
        "sensor": build_sensor_dict(mi, camera_meta),
        "materialist_mesh": {
            "type": "ply",
            "filename": str(Path(mesh_path).resolve()),
            "bsdf": {
                "type": "MatDiffBSDF",
                "cam_meta": str(Path(camera_meta_path).resolve()),
                "use_mesh_normal": bool(use_mesh_normal),
            },
        },
    }

    if mode in {"local", "combined"}:
        for light in lights:
            suffix = light.geometry_path.suffix.lower()
            if suffix not in {".obj", ".ply"}:
                raise ValueError(f"Unsupported light mesh format: {light.geometry_path}")
            rgb = np.maximum(light.rgb * np.float32(radiance_scale), 0.0)
            if light.is_window:
                lobes = light.window_lobes.copy()
                lobes[:, :3] *= np.float32(radiance_scale)
                emitter = {"type": "ile_window"}
                for index, name in enumerate(("sun", "sky", "ground")):
                    emitter[f"{name}_rgb"] = lobes[index, :3].tolist()
                    emitter[f"{name}_direction"] = lobes[index, 3:6].tolist()
                    emitter[f"{name}_concentration"] = float(lobes[index, 6])
            else:
                emitter = {
                    "type": "area",
                    "radiance": {"type": "rgb", "value": rgb.tolist()},
                }
            shape = {
                "type": suffix[1:],
                "filename": str(light.geometry_path),
                "to_world": _light_transform(mi, light, visible_offset),
                "emitter": emitter,
            }
            if light.is_window:
                # Treat the reconstructed window mesh as a transparent light
                # aperture. The emitter remains attached to the finite shape,
                # while camera and indirect rays can continue to the scene
                # geometry behind it instead of hitting Mitsuba's default
                # zero-reflectance opaque BSDF.
                shape["bsdf"] = {"type": "null"}
            scene[f"ile_{light.name}"] = shape

    if mode in {"env", "combined"}:
        scene["far_field_env"] = {
            "type": "envmap",
            "filename": str(Path(envmap_path).resolve()),
        }
    return scene
