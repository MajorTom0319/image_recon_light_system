"""Minimal GeoCalib -> Materialist intrinsics adapter.

GeoCalib supplies only image-specific pinhole focal length/FOV. Materialist's
original camera pose is retained; roll, pitch, gravity alignment, translation,
and pose optimization are deliberately not used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


_MODEL_CACHE: dict[str, Any] = {}

# Materialist's original default sensor-to-world transform. Keeping this fixed
# preserves the original camera convention while only replacing intrinsics.
_SENSOR_TO_WORLD = np.diag([-1.0, 1.0, -1.0, 1.0]).astype(np.float32)


def _as_numpy(value: Any, *, dtype=np.float32) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array.astype(dtype, copy=False)


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        array = _as_numpy(value, dtype=np.float64).reshape(-1)
        if array.size:
            number = float(array[0])
            if math.isfinite(number):
                return number
    except Exception:
        pass
    return default


@dataclass(slots=True)
class GeoCalibResult:
    width: int
    height: int
    K: np.ndarray
    gravity_camera: np.ndarray | None = None
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    hfov_deg: float = 0.0
    vfov_deg: float = 0.0
    focal_uncertainty_px: float | None = None
    roll_uncertainty_deg: float | None = None
    pitch_uncertainty_deg: float | None = None
    vfov_uncertainty_deg: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.K = np.asarray(self.K, dtype=np.float32).reshape(3, 3)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("GeoCalib image dimensions must be positive")


def _get_geocalib_model(device: str):
    cache_key = str(device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    try:
        from geocalib import GeoCalib
    except ImportError as exc:
        raise ImportError(
            "GeoCalib is not importable. Install the official repository in "
            "the Materialist environment, e.g. `pip install -e /path/to/GeoCalib`."
        ) from exc

    # Official GeoCalib inference API. The camera model is selected in calibrate().
    model = GeoCalib()

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model = model.eval()

    _MODEL_CACHE[cache_key] = model
    return model


def _load_geocalib_image(model: Any, image_path: str | Path, device: str) -> torch.Tensor:
    if hasattr(model, "load_image"):
        image = model.load_image(str(image_path))
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        return image.to(device)

    # Compatibility fallback for GeoCalib versions without load_image().
    from PIL import Image, ImageOps

    with Image.open(image_path) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
        array = np.asarray(pil_image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def _fovs_from_K(K: np.ndarray, *, width: int, height: int) -> tuple[float, float]:
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    if fx <= 0 or fy <= 0:
        raise ValueError(f"Invalid focal lengths in K:\n{K}")
    hfov = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
    vfov = math.degrees(2.0 * math.atan(height / (2.0 * fy)))
    return hfov, vfov


def estimate_camera_geocalib(
    image_path: str | Path,
    *,
    device: str = "cuda",
    weights: str = "pinhole",
    camera_model: str = "pinhole",
) -> GeoCalibResult:
    """Estimate only image-specific pinhole intrinsics from the original image."""
    del weights  # Kept for call-site compatibility.
    if camera_model != "pinhole":
        raise ValueError("This minimal integration supports pinhole cameras only")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    model = _get_geocalib_model(device)
    image = _load_geocalib_image(model, image_path, device)
    height, width = int(image.shape[-2]), int(image.shape[-1])

    with torch.no_grad():
        try:
            result = model.calibrate(image, camera_model="pinhole")
        except TypeError:
            result = model.calibrate(image)

    camera = result["camera"]

    fx = fy = None

    # The official GeoCalib demo exposes camera.f with shape [B, 2] in pixels.
    if hasattr(camera, "f"):
        try:
            focal = _as_numpy(camera.f).reshape(-1)
            if focal.size >= 2:
                fx, fy = float(focal[-2]), float(focal[-1])
            elif focal.size == 1:
                fx = fy = float(focal[0])
        except Exception:
            fx = fy = None

    # Compatibility fallback for versions exposing only an explicit K.
    if (fx is None or fy is None) and hasattr(camera, "K"):
        raw_K = camera.K() if callable(camera.K) else camera.K
        try:
            K_est = _as_numpy(raw_K).reshape(-1, 3, 3)[0]
            fx_candidate = float(K_est[0, 0])
            fy_candidate = float(K_est[1, 1])
            if fx_candidate > 0 and fy_candidate > 0:
                fx, fy = fx_candidate, fy_candidate
        except Exception:
            pass

    # Last-resort compatibility path through vertical FoV.
    if (fx is None or fy is None) and hasattr(camera, "vfov"):
        vfov_rad = _scalar(camera.vfov, default=float("nan"))
        if math.isfinite(vfov_rad) and 0.0 < vfov_rad < math.pi:
            fy = height / (2.0 * math.tan(vfov_rad * 0.5))
            fx = fy

    if fx is None or fy is None or fx <= 0 or fy <= 0:
        raise RuntimeError("Could not extract a valid pixel-space focal length from GeoCalib")

    # The integration intentionally uses only focal length/FOV. Principal point
    # is fixed to the image center so Materialist's depth and screen projection
    # remain the same centered-pinhole model.
    K = np.array(
        [
            [fx, 0.0, width * 0.5],
            [0.0, fy, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    hfov_deg, vfov_deg = _fovs_from_K(K, width=width, height=height)

    focal_uncertainty = None
    if isinstance(result, Mapping):
        raw_uncertainty = result.get("focal_uncertainty")
        if raw_uncertainty is not None:
            focal_uncertainty = _scalar(raw_uncertainty, default=float("nan"))
            if not math.isfinite(focal_uncertainty):
                focal_uncertainty = None

    return GeoCalibResult(
        width=width,
        height=height,
        K=K,
        gravity_camera=None,
        roll_deg=0.0,
        pitch_deg=0.0,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        focal_uncertainty_px=focal_uncertainty,
        extra={"pose_used": False, "camera_model": "pinhole", "source": "geocalib"},
    )


def scale_intrinsics(
    K: np.ndarray,
    *,
    source_hw: tuple[int, int],
    target_hw: tuple[int, int],
    preprocess_meta: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Map original-image K into the exact Materialist working image."""
    K = np.asarray(K, dtype=np.float32).reshape(3, 3)
    if preprocess_meta is not None and "pixel_transform" in preprocess_meta:
        transform = np.asarray(preprocess_meta["pixel_transform"], dtype=np.float32)
        if transform.shape != (3, 3):
            raise ValueError("preprocess_meta['pixel_transform'] must be 3x3")
        scaled = transform @ K
    else:
        source_h, source_w = source_hw
        target_h, target_w = target_hw
        if min(source_h, source_w, target_h, target_w) <= 0:
            raise ValueError("Image dimensions must be positive")
        scaled = np.array(
            [
                [target_w / float(source_w), 0.0, 0.0],
                [0.0, target_h / float(source_h), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ) @ K
    return scaled.astype(np.float32)


def make_mitsuba_compatible_K(K: np.ndarray) -> np.ndarray:
    """Validate the centered pinhole K used by both mesh and renderer."""
    K = np.asarray(K, dtype=np.float32).reshape(3, 3).copy()
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError(f"Invalid camera intrinsics:\n{K}")
    K[0, 1] = 0.0
    K[1, 0] = 0.0
    K[2] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return K


def make_fixed35_camera(width: int, height: int) -> GeoCalibResult:
    """Compatibility fallback: original Materialist 35-degree horizontal FoV."""
    width = int(width)
    height = int(height)
    hfov_rad = math.radians(35.0)
    focal = width / (2.0 * math.tan(hfov_rad * 0.5))
    K = np.array(
        [
            [focal, 0.0, width * 0.5],
            [0.0, focal, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    hfov_deg, vfov_deg = _fovs_from_K(K, width=width, height=height)
    return GeoCalibResult(
        width=width,
        height=height,
        K=K,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        extra={"pose_used": False, "camera_model": "pinhole", "source": "fixed35"},
    )


def write_materialist_camera_json(
    output_path: str | Path,
    *,
    K_work: np.ndarray,
    work_hw: tuple[int, int],
    geocalib_result: GeoCalibResult,
    preprocess_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one per-image camera file shared by all Materialist stages."""
    work_h, work_w = int(work_hw[0]), int(work_hw[1])
    K_work = make_mitsuba_compatible_K(K_work)
    hfov_deg, vfov_deg = _fovs_from_K(K_work, width=work_w, height=work_h)

    camera_source = str(geocalib_result.extra.get("source", "geocalib"))
    meta: dict[str, Any] = {
        "schema": "materialist_intrinsics_v2",
        "camera_model": "pinhole",
        "camera_source": f"{camera_source}_intrinsics_only",
        "pose_source": "materialist_fixed",
        "pose_used_from_estimator": False,
        "film.size": [work_w, work_h],
        "film.crop_size": [work_w, work_h],
        "film.crop_offset": [0, 0],
        "K": K_work.tolist(),
        "K_original": np.asarray(geocalib_result.K, dtype=np.float32).tolist(),
        "x_fov": [float(hfov_deg)],
        "y_fov": [float(vfov_deg)],
        "near_clip": 0.01,
        "far_clip": 10000.0,
        "principal_point_offset_x": [0.5 - float(K_work[0, 2]) / work_w],
        "principal_point_offset_y": [0.5 - float(K_work[1, 2]) / work_h],
        "to_world": [_SENSOR_TO_WORLD.tolist()],
        "roll_deg": 0.0,
        "pitch_deg": 0.0,
        "reported_roll_pitch_ignored": True,
        "original_size": [int(geocalib_result.width), int(geocalib_result.height)],
        "work_size": [work_w, work_h],
    }
    if geocalib_result.focal_uncertainty_px is not None:
        meta["focal_uncertainty_px"] = float(geocalib_result.focal_uncertainty_px)
    if preprocess_meta is not None:
        meta["preprocess"] = dict(preprocess_meta)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2, ensure_ascii=False)
    return meta
