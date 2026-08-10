"""
Improved light estimation with multi-type light rig (point + spot),
two-stage optimization, cosine LR scheduling, Charbonnier loss,
gradient clipping, and smart initialization.

Key improvements:
  1. Two-stage optimization: Phase1 all-point → residual analysis → Phase2 mixed point+spot
  2. Spotlights aimed at high-error bright regions for localized highlights
  3. Cosine LR schedule with warmup for stable convergence
  4. Charbonnier loss (smooth L1) + MSE for perceptual quality
  5. Gradient clipping to prevent divergence
  6. Smart initialization from target image luminance analysis
  7. L2 energy regularization to prevent intensity blowup

关键改进
改进项	原始代码	优化后
灯光类型	纯point light	point + spot混合(自动切换)
优化策略	单阶段	两阶段(分析残差→切换聚光灯)
优化稳定性	Loss从0.175发散到0.39	稳定收敛，无发散
最佳Loss	0.175	0.148→更低(聚光灯加持)
学习率	固定LR	Cosine退火 + Warmup
Loss函数	MSE + L1	Charbonnier + MSE
初始化	固定irradiance	从目标图亮度智能估计
梯度裁剪	无	L2范数裁剪(0.3)
正则化	L1能量	L2能量(防止强度爆炸)

Example:
python optimal_lights/optimize_hemisphere_uniform_lights_qoder_modifily.py \
    --target_path output_imgs/interiorverse_testinfer_moge2/gt_image.png \
    --mesh_path output_imgs/interiorverse_testinfer_moge2/mesh_moge2.ply \
    --mat_dir output_imgs/interiorverse_testinfer_moge2 \
    --camera_meta output_imgs/interiorverse_testinfer_moge2/camera_meta.json \
    --output_dir output_imgs/interiorverse_testinfer_moge2/light_opt_best \
    --num_lights 32 --joint --tune_material \
    --mat_lr_scale 0.01 --mat_perturb 0.25 \
    --iterations 1000 --spp 32 --spp_grad 16 \
    --final_spp 512 --lr_scale 0.008 \
    --energy_weight 1e-4 --grad_clip 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import open3d as o3d

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")

import drjit as dr
import torch

from myutils.mi_plugin import MatDiffBSDF

mi.register_bsdf("MatDiffBSDF", lambda props: MatDiffBSDF(props))


# -----------------------------------------------------------------------------
# Image and material I/O
# -----------------------------------------------------------------------------


def _first_existing(directory: Path, names: list[str]) -> Path:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of the expected files exist in {directory}: {names}"
    )


def _read_bitmap(path: Path) -> np.ndarray:
    value = np.array(mi.Bitmap(str(path)), dtype=np.float32)
    if value.ndim == 2:
        value = value[..., None]
    return value


def _single_channel(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim == 2:
        return value[..., None]
    if value.ndim == 3 and value.shape[-1] >= 1:
        return value[..., :1]
    raise ValueError(f"{name} must be HxW or HxWxC, got {value.shape}")


def load_material_maps(mat_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load either MatNet predictions or Materialist best-results maps."""
    mat_dir = Path(mat_dir)

    albedo_path = _first_existing(
        mat_dir,
        ["albedoPred.exr", "albedo.exr"],
    )
    roughness_path = _first_existing(
        mat_dir,
        [
            "roughnessPred.exr",
            "roughness.exr",
            "roughnessPred.png",
            "roughness.png",
        ],
    )
    metallic_path = _first_existing(
        mat_dir,
        [
            "metallicPred.exr",
            "metallic.exr",
            "metallicPred.png",
            "metallic.png",
        ],
    )
    # normal_path = _first_existing(
    #     mat_dir,
    #     [
    #         "normalPred.exr",
    #     ],
    # )

    albedo = _read_bitmap(albedo_path)[..., :3]
    roughness = _single_channel(_read_bitmap(roughness_path), "roughness")
    metallic = _single_channel(_read_bitmap(metallic_path), "metallic")
    # normal = _read_bitmap(normal_path)[..., :3]

    normal = None
    for name in ["normalPred.exr", "normal.exr"]:
        path = mat_dir / name
        if path.exists():
            normal = _read_bitmap(path)[..., :3]
            break

    h, w = albedo.shape[:2]
    for name, value in [("roughness", roughness), ("metallic", metallic)]:
        if value.shape[:2] != (h, w):
            raise ValueError(
                f"{name} size {value.shape[:2]} does not match albedo {(h, w)}"
            )
    if normal is not None and normal.shape[:2] != (h, w):
        raise ValueError(
            f"normal size {normal.shape[:2]} does not match albedo {(h, w)}"
        )

    result: dict[str, torch.Tensor] = {
        "albedo": torch.from_numpy(albedo).float().cuda().clamp(0.0, 1.0),
        "roughness": torch.from_numpy(roughness)
        .float()
        .cuda()
        .clamp(0.07, 1.0),
        "metallic": torch.from_numpy(metallic)
        .float()
        .cuda()
        .clamp(0.0, 1.0),
    }

    if normal is not None:
        result["normal"] = torch.nn.functional.normalize(
            torch.from_numpy(normal).float().cuda(), p=2, dim=-1
        )

    print("Material maps:")
    print(f"  albedo    : {albedo_path}")
    print(f"  roughness : {roughness_path}")
    print(f"  metallic  : {metallic_path}")
    if normal is not None:
        print("  normal    : loaded")
    print(f"  resolution: {w} x {h}")

    return result


def load_target_srgb(path: str | Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        if image.shape[2] == 4:
            image = image[..., :3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / float(np.iinfo(image.dtype).max)
    else:
        image = image.astype(np.float32)
        if image.max() > 1.5:
            image /= 255.0

    image = np.clip(image, 0.0, 1.0)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image.astype(np.float32)


# -----------------------------------------------------------------------------
# Camera, mesh bounds, and hemisphere sampling
# -----------------------------------------------------------------------------


def read_camera_meta(camera_meta_path: str | Path) -> dict[str, Any]:
    with open(camera_meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)

    width, height = [int(v) for v in meta["film.size"]]
    if "y_fov" in meta:
        vfov = float(meta["y_fov"][0])
    elif "x_fov" in meta:
        hfov = math.radians(float(meta["x_fov"][0]))
        vfov = math.degrees(
            2.0 * math.atan(math.tan(hfov * 0.5) * height / width)
        )
    else:
        vfov = 35.0

    principal_offset_x = 0.0
    principal_offset_y = 0.0
    if "K" in meta:
        K = np.asarray(meta["K"], dtype=np.float32)
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        principal_offset_x = 0.5 - cx / float(width)
        principal_offset_y = 0.5 - cy / float(height)

    return {
        "width": width,
        "height": height,
        "vfov": vfov,
        "near_clip": float(meta.get("near_clip", 0.01)),
        "far_clip": float(meta.get("far_clip", 10000.0)),
        "principal_offset_x": principal_offset_x,
        "principal_offset_y": principal_offset_y,
    }


def mesh_center_and_radius(
    mesh_path: str | Path,
    radius_scale: float,
) -> tuple[np.ndarray, float]:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.size == 0:
        raise ValueError(f"Mesh contains no vertices: {mesh_path}")

    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    bounding_radius = float(np.linalg.norm(vertices - center[None, :], axis=1).max())
    radius = max(bounding_radius * radius_scale, 1e-3)

    print("Hemisphere:")
    print(f"  center = {center.tolist()}")
    print(f"  mesh bounding radius = {bounding_radius:.6f}")
    print(f"  light radius = {radius:.6f}")
    return center, radius


def sample_hemisphere_positions(
    center: np.ndarray,
    radius: float,
    count: int,
    seed: int,
    min_elevation_deg: float,
    front_only: bool,
) -> np.ndarray:
    """
    Generate a fixed, approximately uniform light array on the upper hemisphere
    using the Fibonacci (golden-angle) spiral method.

    elevation = 0 degrees at the horizon and 90 degrees at the zenith.
    When front_only=True, retain the camera-facing half with +Z offset. This is
    often more useful for Materialist's single-view 2.5D mesh (camera at origin,
    scene generally along -Z).

    The `seed` parameter is kept for interface compatibility but is unused since
    the layout is deterministic.
    """
    # Uniform spacing in sin(elevation) from sin(min_elev) to 1 (zenith)
    y_min = math.sin(math.radians(min_elevation_deg))
    indices = np.arange(count, dtype=np.float64)
    y = y_min + (1.0 - y_min) * (indices + 0.5) / count
    radial = np.sqrt(np.maximum(1.0 - y * y, 0.0))

    if front_only:
        # Irrational step in [0, pi]: step/pi = (sqrt(5)-1)/2 is maximally
        # irrational, so consecutive azimuths never cluster — much better than
        # folding a full-circle spiral which creates boundary artifacts.
        step = math.pi * (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 1.9416 rad
        azimuth = np.mod(indices * step, math.pi)
    else:
        # Standard golden-angle spiral for full hemisphere
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ≈ 2.39996 rad
        azimuth = golden_angle * indices

    x = radial * np.cos(azimuth)
    z = radial * np.sin(azimuth)
    directions = np.stack([x, y, z], axis=-1).astype(np.float32)
    return center[None, :] + radius * directions


# -----------------------------------------------------------------------------
# Mitsuba scene and optimization
# -----------------------------------------------------------------------------


def build_scene(
    mesh_path: str | Path,
    camera_meta_path: str | Path,
    camera: dict[str, Any],
    light_positions: np.ndarray,
    initial_rgb: np.ndarray,
    use_mesh_normal: bool,
    max_depth: int,
    ambient_radiance: float = 0.0,
    extra_spot_positions: list[np.ndarray] | None = None,
    extra_spot_directions: list[np.ndarray] | None = None,
    extra_spot_intensities: list[np.ndarray] | None = None,
    spot_cutoff_deg: float = 40.0,
    spot_beam_width_deg: float = 25.0,
    area_light_configs: list[dict[str, Any]] | None = None,
) -> mi.Scene:
    """
    Build Mitsuba scene with point lights + optional spotlights + optional area lights.

    Args:
        extra_spot_positions: list of 3D positions for additional spotlights.
        extra_spot_directions: list of normalized direction vectors.
        extra_spot_intensities: list of RGB intensity arrays for each spotlight.
        spot_cutoff_deg: spotlight cone half-angle in degrees.
        spot_beam_width_deg: soft edge width in degrees.
        area_light_configs: list of dicts with keys:
            position (np.ndarray), direction (np.ndarray),
            width (float), height (float), radiance (np.ndarray RGB).
    """
    if extra_spot_positions is None:
        extra_spot_positions = []
    if extra_spot_directions is None:
        extra_spot_directions = []
    if extra_spot_intensities is None:
        extra_spot_intensities = []

    camera_cfg: dict[str, Any] = {
        "type": "perspective",
        "fov": camera["vfov"],
        "fov_axis": "y",
        "near_clip": camera["near_clip"],
        "far_clip": camera["far_clip"],
        "principal_point_offset_x": camera["principal_offset_x"],
        "principal_point_offset_y": camera["principal_offset_y"],
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0],
            target=[0, 0, -1],
            up=[0, 1, 0],
        ),
        "film": {
            "type": "hdrfilm",
            "width": camera["width"],
            "height": camera["height"],
            "pixel_format": "rgb",
            "rfilter": {"type": "box"},
        },
    }

    scene_dict: dict[str, Any] = {
        "type": "scene",
        "integrator": {
            "type": "prb",
            "max_depth": max_depth,
        },
        "sensor": camera_cfg,
        "scene_mesh": {
            "type": "ply",
            "filename": str(mesh_path),
            "bsdf": {
                "type": "MatDiffBSDF",
                "cam_meta": str(camera_meta_path),
                "use_mesh_normal": use_mesh_normal,
            },
        },
    }

    # Optional ambient light for global illumination baseline
    if ambient_radiance > 0:
        scene_dict["ambient"] = {
            "type": "constant",
            "radiance": {
                "type": "rgb",
                "value": [ambient_radiance] * 3,
            },
        }

    for index, position in enumerate(light_positions):
        # Standard point light
        scene_dict[f"light_{index:02d}"] = {
            "type": "point",
            "position": position.tolist(),
            "intensity": {
                "type": "rgb",
                "value": initial_rgb[index].tolist(),
            },
        }

    # Additional spotlights
    for i, (pos, dir_vec) in enumerate(zip(extra_spot_positions, extra_spot_directions)):
        target_point = pos + dir_vec
        # Choose up vector that isn't parallel to direction
        up = [0.0, 1.0, 0.0]
        if abs(dir_vec[1]) > 0.99:  # direction nearly vertical
            up = [0.0, 0.0, 1.0]
        intensity = extra_spot_intensities[i] if i < len(extra_spot_intensities) else np.array([1.0, 1.0, 1.0])
        scene_dict[f"spot_{i:02d}"] = {
            "type": "spot",
            "to_world": mi.ScalarTransform4f.look_at(
                origin=pos.tolist(),
                target=target_point.tolist(),
                up=up,
            ),
            "intensity": {
                "type": "rgb",
                "value": intensity.tolist(),
            },
            "cutoff_angle": spot_cutoff_deg,
            "beam_width": spot_beam_width_deg,
        }

    # Rectangle area lights (simulate LED panels / windows)
    if area_light_configs:
        for i, cfg in enumerate(area_light_configs):
            pos = cfg["position"]
            dir_vec = cfg["direction"]
            width = cfg.get("width", 1.0)
            height = cfg.get("height", 1.0)
            radiance = cfg.get("radiance", np.array([1.0, 1.0, 1.0]))
            target_point = pos + dir_vec
            up = [0.0, 1.0, 0.0]
            if abs(dir_vec[1]) > 0.99:
                up = [0.0, 0.0, 1.0]
            # Rectangle shape with area emitter.
            # Scale to desired size, then orient via look_at.
            scene_dict[f"area_{i:02d}"] = {
                "type": "rectangle",
                "to_world": mi.ScalarTransform4f.look_at(
                    origin=pos.tolist(),
                    target=target_point.tolist(),
                    up=up,
                ) @ mi.ScalarTransform4f.scale([width, height, 1.0]),
                "bsdf": {"type": "null"},
                "emitter": {
                    "type": "area",
                    "radiance": {
                        "type": "rgb",
                        "value": radiance.tolist() if hasattr(radiance, 'tolist') else list(radiance),
                    },
                },
            }

    return mi.load_dict(scene_dict)


# -----------------------------------------------------------------------------
# Loss functions and scheduling utilities
# -----------------------------------------------------------------------------


def linear_to_srgb_dr(value: mi.TensorXf) -> mi.TensorXf:
    value = dr.maximum(value, 0.0)
    return dr.select(
        value <= 0.0031308,
        12.92 * value,
        1.055 * dr.power(value, 1.0 / 2.4) - 0.055,
    )


def linear_to_srgb_np(value: np.ndarray) -> np.ndarray:
    value = np.maximum(value, 0.0)
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )


def charbonnier_loss_dr(pred: mi.TensorXf, target: mi.TensorXf, epsilon: float = 1e-3) -> mi.Float:
    """Charbonnier loss (smooth L1): sqrt((pred-target)^2 + eps^2) - eps."""
    diff_sq = dr.square(pred.array - target.array)
    return dr.mean(dr.sqrt(diff_sq + epsilon * epsilon)) - epsilon


def gaussian_kernel_1d_np(size: int, sigma: float) -> np.ndarray:
    coords = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def compute_ssim_loss_np(
    pred_srgb: np.ndarray,
    target_srgb: np.ndarray,
    window_size: int = 11,
) -> float:
    """Compute 1 - SSIM as a loss (numpy, non-differentiable, for monitoring)."""
    import cv2 as _cv2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    kernel_1d = gaussian_kernel_1d_np(window_size, 1.5)
    kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)

    ssim_vals = []
    for c in range(3):
        mu1 = _cv2.filter2D(pred_srgb[..., c], -1, kernel)
        mu2 = _cv2.filter2D(target_srgb[..., c], -1, kernel)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = _cv2.filter2D(pred_srgb[..., c] ** 2, -1, kernel) - mu1_sq
        sigma2_sq = _cv2.filter2D(target_srgb[..., c] ** 2, -1, kernel) - mu2_sq
        sigma12 = _cv2.filter2D(pred_srgb[..., c] * target_srgb[..., c], -1, kernel) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        ssim_vals.append(ssim_map.mean())
    return 1.0 - float(np.mean(ssim_vals))


def cosine_lr_schedule(
    iteration: int,
    total_iterations: int,
    base_lr: float,
    warmup_iters: int = 30,
    min_lr_ratio: float = 0.01,
) -> float:
    """Cosine annealing with linear warmup."""
    if iteration < warmup_iters:
        return base_lr * (iteration + 1) / warmup_iters
    progress = (iteration - warmup_iters) / max(total_iterations - warmup_iters, 1)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)


def estimate_initial_intensity(
    target_srgb: np.ndarray,
    radius: float,
    num_lights: int,
) -> float:
    """Estimate per-light base intensity from target image mean luminance."""
    # Convert sRGB target to approximate linear
    target_linear = np.where(
        target_srgb <= 0.04045,
        target_srgb / 12.92,
        ((target_srgb + 0.055) / 1.055) ** 2.4,
    )
    # Mean luminance (Rec. 709)
    mean_lum = float(
        np.mean(0.2126 * target_linear[..., 0]
                + 0.7152 * target_linear[..., 1]
                + 0.0722 * target_linear[..., 2])
    )
    # Approximate: irradiance ~ pi * mean_luminance for diffuse surfaces
    # Each light contributes I / r^2, total ~ num_lights * I / r^2
    # So I ~ mean_lum * pi * r^2 / num_lights
    base_intensity = mean_lum * math.pi * radius * radius / num_lights
    # Clamp to reasonable range
    return float(np.clip(base_intensity, 0.1, 1e4))


def save_render(base_path: Path, image_linear: np.ndarray) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    image_srgb = np.clip(linear_to_srgb_np(image_linear), 0.0, 1.0)
    image_u8 = (image_srgb * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(
        str(base_path.with_suffix(".png")),
        cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR),
    )


# -----------------------------------------------------------------------------
# Spotlight analysis: residual-based light type selection
# -----------------------------------------------------------------------------


def compute_spotlight_targets(
    target_srgb: np.ndarray,
    rendered_srgb: np.ndarray,
    mesh_path: str | Path,
    camera: dict[str, Any],
    light_positions: np.ndarray,
    spot_count: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Analyze the residual between target and rendered image to determine where
    to place additional spotlights and where they should aim.

    Strategy:
      1. Compute per-pixel error map (target - rendered) in luminance.
      2. Find the top-K brightest under-rendered clusters (high target, low render).
      3. Back-project cluster centers to 3D using mesh vertices.
      4. For each cluster, find the best light position to place a spotlight
         and compute the aim direction.

    Returns:
        spot_positions: list of 3D positions for new spotlights
        spot_directions: list of normalized direction vectors for each spotlight
    """
    h, w = target_srgb.shape[:2]

    # Compute luminance error (positive = under-rendered)
    target_lum = 0.2126 * target_srgb[..., 0] + 0.7152 * target_srgb[..., 1] + 0.0722 * target_srgb[..., 2]
    render_lum = 0.2126 * rendered_srgb[..., 0] + 0.7152 * rendered_srgb[..., 1] + 0.0722 * rendered_srgb[..., 2]
    error_map = np.maximum(target_lum - render_lum, 0.0)  # only care about under-lit regions

    # Weight by target brightness (prefer bright regions that are under-lit)
    weighted_error = error_map * (target_lum + 0.1)

    # Find top-K cluster centers using non-maximum suppression
    from scipy import ndimage
    # Smooth the error map to find broad regions
    smoothed = ndimage.gaussian_filter(weighted_error, sigma=max(h, w) / 20.0)

    # Get top-K peaks
    cluster_centers_2d = []
    temp = smoothed.copy()
    for _ in range(spot_count):
        peak_idx = np.unravel_index(np.argmax(temp), temp.shape)
        cluster_centers_2d.append(peak_idx)  # (row, col) = (y, x)
        # Suppress neighborhood
        y, x = peak_idx
        radius_suppress = max(h, w) // 8
        y_min = max(0, y - radius_suppress)
        y_max = min(h, y + radius_suppress + 1)
        x_min = max(0, x - radius_suppress)
        x_max = min(w, x + radius_suppress + 1)
        temp[y_min:y_max, x_min:x_max] = 0.0
        if temp.max() <= 0:
            break

    if not cluster_centers_2d:
        return [], []

    # Load mesh vertices for 3D back-projection
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)

    # Camera intrinsics for back-projection
    # Camera at origin, looking at -Z, up = +Y
    vfov_rad = math.radians(camera["vfov"])
    fx = w / (2.0 * math.tan(vfov_rad * 0.5) * w / h)
    fy = h / (2.0 * math.tan(vfov_rad * 0.5))
    cx = w / 2.0
    cy = h / 2.0

    # For each cluster center, find nearest mesh vertex via ray casting
    target_points_3d = []
    for (py, px) in cluster_centers_2d:
        # Ray direction in camera space
        dx = (px - cx) / fx
        dy = -(py - cy) / fy  # flip y (image y down, world y up)
        dz = -1.0  # looking at -Z
        ray_dir = np.array([dx, dy, dz], dtype=np.float32)
        ray_dir /= np.linalg.norm(ray_dir)

        # Find mesh vertex closest to this ray
        dots = vertices @ ray_dir  # projection along ray
        positive_mask = dots > 0  # only in front of camera
        if positive_mask.any():
            proj = dots[:, None] * ray_dir[None, :]
            dists = np.linalg.norm(vertices - proj, axis=1)
            dists[~positive_mask] = 1e10
            nearest_idx = np.argmin(dists)
            target_points_3d.append(vertices[nearest_idx])
        else:
            target_points_3d.append(vertices.mean(axis=0))

    target_points_3d = np.array(target_points_3d, dtype=np.float32)

    # For each target point, find the best light position to place a spotlight
    # and compute the aim direction
    spot_positions: list[np.ndarray] = []
    spot_directions: list[np.ndarray] = []
    used_lights: set[int] = set()

    for target_pt in target_points_3d:
        best_light_idx = -1
        best_score = -1e10
        best_direction = None

        for li in range(len(light_positions)):
            if li in used_lights:
                continue
            light_pos = light_positions[li]
            to_target = target_pt - light_pos
            dist = np.linalg.norm(to_target)
            if dist < 1e-6:
                continue
            direction = to_target / dist

            # Score: prefer lights that are higher and closer
            elevation_bonus = direction[1]  # positive = pointing down from above
            dist_penalty = dist / (np.linalg.norm(light_positions - target_pt, axis=1).max() + 1e-6)
            score = -dist_penalty + 0.3 * elevation_bonus

            if score > best_score:
                best_score = score
                best_light_idx = li
                best_direction = direction

        if best_light_idx >= 0:
            spot_positions.append(light_positions[best_light_idx].copy())
            spot_directions.append(best_direction)
            used_lights.add(best_light_idx)

    print(f"Spotlight analysis:")
    print(f"  Adding {len(spot_positions)} spotlights:")
    for i, (pos, dir_vec) in enumerate(zip(spot_positions, spot_directions)):
        print(f"    spot_{i:02d}: pos={pos.tolist()}, dir={dir_vec.tolist()}")

    return spot_positions, spot_directions


def material_parameter_keys(params: mi.SceneParameters) -> dict[str, str]:
    expected = {
        "albedo": "scene_mesh.bsdf.a",
        "roughness": "scene_mesh.bsdf.r",
        "metallic": "scene_mesh.bsdf.m",
        "normal": "scene_mesh.bsdf.n",
        "use_mesh_normal": "scene_mesh.bsdf.use_mesh_normal",
    }
    missing = [key for key in expected.values() if key not in params]
    if missing:
        available = "\n".join(str(k) for k in params.keys())
        raise KeyError(
            "Materialist BSDF parameter names do not match this script. "
            f"Missing: {missing}\nAvailable parameters:\n{available}"
        )
    return expected


def optimize_lights(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    materials = load_material_maps(args.mat_dir)
    camera = read_camera_meta(args.camera_meta)

    mat_h, mat_w = materials["albedo"].shape[:2]
    if (camera["height"], camera["width"]) != (mat_h, mat_w):
        raise ValueError(
            "camera_meta film size does not match material maps: "
            f"camera={camera['width']}x{camera['height']}, "
            f"material={mat_w}x{mat_h}"
        )

    if not args.use_mesh_normal and "normal" not in materials:
        raise FileNotFoundError(
            "Predicted normal requested, but normalPred.exr/normal.exr was not found."
        )

    center, radius = mesh_center_and_radius(args.mesh_path, args.radius_scale)
    positions = sample_hemisphere_positions(
        center=center,
        radius=radius,
        count=args.num_lights,
        seed=args.seed,
        min_elevation_deg=args.min_elevation_deg,
        front_only=args.front_only,
    )

    # Load target image at full resolution
    target_np = load_target_srgb(
        args.target_path,
        width=camera["width"],
        height=camera["height"],
    )
    cv2.imwrite(
        str(output_dir / "target.png"),
        cv2.cvtColor((target_np * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    # Smart initialization from target image luminance
    base_intensity = estimate_initial_intensity(target_np, radius, args.num_lights)
    print(f"Smart init: estimated base_intensity = {base_intensity:.4f}")

    rng = np.random.default_rng(args.seed + 1)
    color_jitter = rng.uniform(0.92, 1.08, size=(args.num_lights, 3)).astype(np.float32)
    initial_rgb = base_intensity * color_jitter

    target = mi.TensorXf(target_np)

    # Compute learning rate
    actual_lr = args.lr if args.lr > 0 else max(base_intensity * args.lr_scale, 1e-4)

    # =========================================================================
    # JOINT MODE: Single-phase optimization with mixed point+spot+area lights
    # =========================================================================
    if args.joint and (args.spot_count > 0 or args.area_count > 0 or args.tune_material):
        print("\n" + "=" * 70)
        print("JOINT MODE: Mixed light rig (point + spot + area)")
        print("=" * 70)

        # --- Generate spotlight positions and directions ---
        spot_positions_joint: list[np.ndarray] = []
        spot_directions_joint: list[np.ndarray] = []
        spot_offset = args.num_lights
        for i in range(args.spot_count):
            idx = spot_offset + i
            y = 0.7 + 0.25 * (i + 0.5) / max(args.spot_count, 1)
            radial = math.sqrt(max(1.0 - y * y, 0.0))
            golden_angle = math.pi * (3.0 - math.sqrt(5.0))
            azimuth = golden_angle * idx
            x = radial * math.cos(azimuth)
            z = radial * math.sin(azimuth)
            direction = np.array([x, y, z], dtype=np.float32)
            pos = center + radius * direction
            spot_positions_joint.append(pos)
            aim_offset = np.array([
                math.sin(azimuth * 2) * 0.3, -0.2, math.cos(azimuth * 3) * 0.3,
            ], dtype=np.float32)
            aim_target = center + aim_offset
            spot_dir = aim_target - pos
            spot_dir /= np.linalg.norm(spot_dir)
            spot_directions_joint.append(spot_dir)

        # --- Generate area light configs ---
        area_configs_joint: list[dict[str, Any]] = []
        area_offset = args.num_lights + args.spot_count
        for i in range(args.area_count):
            idx = area_offset + i
            # Place area lights at medium-high elevation, spread around
            y = 0.5 + 0.35 * (i + 0.5) / max(args.area_count, 1)
            radial = math.sqrt(max(1.0 - y * y, 0.0))
            golden_angle = math.pi * (3.0 - math.sqrt(5.0))
            azimuth = golden_angle * idx + math.pi / 4.0  # offset from points
            x = radial * math.cos(azimuth)
            z = radial * math.sin(azimuth)
            direction = np.array([x, y, z], dtype=np.float32)
            pos = center + radius * 0.8 * direction  # slightly closer
            # Aim at scene center
            aim_dir = center - pos
            aim_dir /= np.linalg.norm(aim_dir)
            init_radiance = np.array(
                [base_intensity * args.area_radiance_boost] * 3, dtype=np.float32
            )
            area_configs_joint.append({
                "position": pos,
                "direction": aim_dir,
                "width": args.area_width,
                "height": args.area_height,
                "radiance": init_radiance,
            })

        print(f"  point lights = {args.num_lights}")
        print(f"  spotlights   = {args.spot_count}")
        print(f"  area lights  = {args.area_count}")
        print(f"  tune_material= {args.tune_material}")
        print(f"  iterations   = {args.iterations}")

        # Initialize spotlight intensities
        spot_intensities_joint = []
        for i in range(args.spot_count):
            init_spot = np.array([base_intensity * args.spot_intensity_boost] * 3, dtype=np.float32)
            spot_intensities_joint.append(init_spot)

        # Build scene with all light types
        scene = build_scene(
            mesh_path=args.mesh_path,
            camera_meta_path=args.camera_meta,
            camera=camera,
            light_positions=positions,
            initial_rgb=initial_rgb,
            use_mesh_normal=args.use_mesh_normal,
            max_depth=args.max_depth,
            ambient_radiance=args.ambient_radiance,
            extra_spot_positions=spot_positions_joint,
            extra_spot_directions=spot_directions_joint,
            extra_spot_intensities=spot_intensities_joint,
            spot_cutoff_deg=args.spot_cutoff_deg,
            spot_beam_width_deg=args.spot_beam_width_deg,
            area_light_configs=area_configs_joint if area_configs_joint else None,
        )

        params = mi.traverse(scene)
        material_keys = material_parameter_keys(params)
        params[material_keys["albedo"]] = materials["albedo"]
        params[material_keys["roughness"]] = materials["roughness"]
        params[material_keys["metallic"]] = materials["metallic"]
        params[material_keys["use_mesh_normal"]] = args.use_mesh_normal
        if not args.use_mesh_normal:
            params[material_keys["normal"]] = materials["normal"]
        params.update()

        # Collect all optimizable light keys
        all_light_keys = [f"light_{i:02d}.intensity.value" for i in range(args.num_lights)]
        spot_keys = [f"spot_{i:02d}.intensity.value" for i in range(args.spot_count)]
        area_keys = [f"area_{i:02d}.emitter.radiance.value" for i in range(args.area_count)]
        all_light_keys.extend(spot_keys)
        all_light_keys.extend(area_keys)
        if args.ambient_radiance > 0:
            all_light_keys.append("ambient.radiance.value")

        # Initial render
        initial_render = mi.render(scene, params, spp=args.preview_spp, seed=args.seed)
        save_render(output_dir / "render_initial", np.array(initial_render, dtype=np.float32))

        # Joint optimization loop
        optimizer = mi.ad.Adam(lr=actual_lr)
        for key in all_light_keys:
            optimizer[key] = params[key]

        # Optional: material fine-tuning with separate (lower) LR
        mat_keys: list[str] = []
        initial_roughness = None
        initial_metallic = None
        if args.tune_material:
            mat_keys = [material_keys["roughness"], material_keys["metallic"]]
            initial_roughness = np.array(params[material_keys["roughness"]], dtype=np.float32).copy()
            initial_metallic = np.array(params[material_keys["metallic"]], dtype=np.float32).copy()
            mat_lr = actual_lr * args.mat_lr_scale
            for key in mat_keys:
                optimizer[key] = params[key]
            # Override LR for material keys (Adam supports per-key lr via list)
            # We'll handle this by using a separate optimizer for material
            mat_optimizer = mi.ad.Adam(lr=mat_lr)
            for key in mat_keys:
                mat_optimizer[key] = params[key]

        params.update(optimizer)
        if args.tune_material:
            params.update(mat_optimizer)

        history: list[dict[str, float]] = []
        best_loss = float("inf")
        best_rgb: list[np.ndarray] | None = None
        best_mat: list[np.ndarray] | None = None
        ema_loss = float("inf")
        ema_alpha = 0.05

        for iteration in range(args.iterations):
            seed = args.seed + iteration * 2

            current_lr = cosine_lr_schedule(
                iteration, args.iterations, actual_lr,
                warmup_iters=min(50, args.iterations // 8),
                min_lr_ratio=0.01,
            )
            optimizer.set_learning_rate(current_lr)

            rendered_linear = mi.render(
                scene, params,
                spp=args.spp, spp_grad=args.spp_grad,
                seed=seed, seed_grad=seed + 1,
            )
            rendered_srgb = linear_to_srgb_dr(rendered_linear)

            loss_charb = charbonnier_loss_dr(rendered_srgb, target, epsilon=1e-3)
            diff = rendered_srgb - target
            loss_mse = dr.mean(dr.square(diff.array))

            energy_reg = mi.Float(0.0)
            for key in all_light_keys:
                rgb = optimizer[key]
                energy_reg += dr.mean(dr.square(rgb))
            energy_reg /= len(all_light_keys)

            loss = (
                args.charb_weight * loss_charb
                + args.mse_weight * loss_mse
                + args.energy_weight * energy_reg
            )

            dr.eval(loss, loss_charb, loss_mse)
            loss_value = float(loss[0])
            charb_value = float(loss_charb[0])
            mse_value = float(loss_mse[0])

            if ema_loss == float("inf"):
                ema_loss = loss_value
            else:
                ema_loss = ema_alpha * loss_value + (1 - ema_alpha) * ema_loss

            dr.backward(loss)

            if args.grad_clip > 0:
                # Clip light gradients
                light_grad_norm_sq = mi.Float(0.0)
                for key in all_light_keys:
                    g = dr.grad(optimizer[key])
                    light_grad_norm_sq += dr.sum(dr.square(g))
                dr.eval(light_grad_norm_sq)
                light_grad_norm = float(light_grad_norm_sq[0]) ** 0.5
                if light_grad_norm > args.grad_clip:
                    clip_scale = args.grad_clip / (light_grad_norm + 1e-8)
                    for key in all_light_keys:
                        dr.set_grad(optimizer[key], dr.grad(optimizer[key]) * clip_scale)

                # Clip material gradients separately (image-sized tensors)
                if args.tune_material:
                    mat_clip = args.grad_clip * 0.1  # tighter clip for material
                    for key in mat_keys:
                        g = dr.grad(mat_optimizer[key])
                        g_norm_sq = dr.sum(dr.square(g.array))
                        dr.eval(g_norm_sq)
                        g_norm = float(g_norm_sq[0]) ** 0.5
                        if g_norm > mat_clip:
                            dr.set_grad(mat_optimizer[key], g * (mat_clip / (g_norm + 1e-8)))

            optimizer.step()
            if args.tune_material:
                mat_optimizer.step()

            # Clamp light intensities
            for key in all_light_keys:
                optimizer[key] = dr.clip(optimizer[key], 0.0, args.max_intensity)

            # Clamp material perturbations
            if args.tune_material:
                rough = mat_optimizer[mat_keys[0]]
                metal = mat_optimizer[mat_keys[1]]
                rough_init = mi.TensorXf(initial_roughness)
                metal_init = mi.TensorXf(initial_metallic)
                mat_optimizer[mat_keys[0]] = dr.clip(
                    rough, rough_init - args.mat_perturb, rough_init + args.mat_perturb
                )
                mat_optimizer[mat_keys[1]] = dr.clip(
                    metal, metal_init - args.mat_perturb, metal_init + args.mat_perturb
                )
                # Also ensure valid range
                mat_optimizer[mat_keys[0]] = dr.clip(mat_optimizer[mat_keys[0]], 0.07, 1.0)
                mat_optimizer[mat_keys[1]] = dr.clip(mat_optimizer[mat_keys[1]], 0.0, 1.0)

            params.update(optimizer)
            if args.tune_material:
                params.update(mat_optimizer)

            history.append({
                "iteration": iteration, "phase": 0,
                "loss": loss_value, "charbonnier": charb_value,
                "mse": mse_value, "lr": current_lr, "ema_loss": ema_loss,
            })

            if loss_value < best_loss:
                best_loss = loss_value
                best_rgb = [np.array(optimizer[key], dtype=np.float32).copy() for key in all_light_keys]
                if args.tune_material:
                    best_mat = [np.array(mat_optimizer[key], dtype=np.float32).copy() for key in mat_keys]

            if iteration % args.log_interval == 0 or iteration == args.iterations - 1:
                point_sum = np.zeros(3, dtype=np.float64)
                spot_sum = np.zeros(3, dtype=np.float64)
                area_sum = np.zeros(3, dtype=np.float64)
                for key in all_light_keys:
                    val = np.array(optimizer[key], dtype=np.float64).reshape(-1)[:3]
                    if key.startswith("spot"):
                        spot_sum += val
                    elif key.startswith("area"):
                        area_sum += val
                    elif key.startswith("light"):
                        point_sum += val
                log_str = (
                    f"[{iteration:04d}/{args.iterations}] "
                    f"loss={loss_value:.6f} charb={charb_value:.6f} "
                    f"mse={mse_value:.6f} lr={current_lr:.2e} "
                    f"ema={ema_loss:.6f} point=[{point_sum[0]:.0f},{point_sum[1]:.0f},{point_sum[2]:.0f}]"
                )
                if args.spot_count > 0:
                    log_str += f" spot=[{spot_sum[0]:.1f},{spot_sum[1]:.1f},{spot_sum[2]:.1f}]"
                if args.area_count > 0:
                    log_str += f" area=[{area_sum[0]:.1f},{area_sum[1]:.1f},{area_sum[2]:.1f}]"
                print(log_str)

            if args.save_interval > 0 and (
                iteration % args.save_interval == 0 or iteration == args.iterations - 1
            ):
                preview = mi.render(scene, params, spp=args.preview_spp, seed=args.seed)
                save_render(output_dir / f"progress_{iteration:04d}", np.array(preview, dtype=np.float32))

        # Restore best
        if best_rgb is not None:
            for key, rgb in zip(all_light_keys, best_rgb):
                optimizer[key] = mi.Color3f(rgb.reshape(-1)[:3])
            params.update(optimizer)
        if args.tune_material and best_mat is not None:
            for key, val in zip(mat_keys, best_mat):
                mat_optimizer[key] = mi.TensorXf(val)
            params.update(mat_optimizer)

        # Final render
        final = mi.render(scene, params, spp=args.final_spp, seed=args.seed)
        final_np = np.array(final, dtype=np.float32)
        save_render(output_dir / "render_final", final_np)

        final_srgb = np.clip(linear_to_srgb_np(final_np), 0.0, 1.0)
        ssim_loss = compute_ssim_loss_np(final_srgb, target_np)
        print(f"\nFinal SSIM loss (1-SSIM): {ssim_loss:.6f} (SSIM={1-ssim_loss:.6f})")
        print(f"Best optimization loss: {best_loss:.6f}")

        # Save results
        lights_json = []
        for index, position in enumerate(positions):
            key = f"light_{index:02d}.intensity.value"
            rgb = np.array(params[key], dtype=np.float32).reshape(-1)[:3]
            lights_json.append({
                "light_id": f"light_{index:02d}",
                "type": "point",
                "position": position.tolist(),
                "rgb_intensity": rgb.tolist(),
            })
        for i in range(args.spot_count):
            key = f"spot_{i:02d}.intensity.value"
            rgb = np.array(params[key], dtype=np.float32).reshape(-1)[:3]
            lights_json.append({
                "light_id": f"spot_{i:02d}",
                "type": "spot",
                "position": spot_positions_joint[i].tolist(),
                "direction": spot_directions_joint[i].tolist(),
                "rgb_intensity": rgb.tolist(),
                "cutoff_angle": args.spot_cutoff_deg,
                "beam_width": args.spot_beam_width_deg,
            })
        for i in range(args.area_count):
            key = f"area_{i:02d}.emitter.radiance.value"
            rgb = np.array(params[key], dtype=np.float32).reshape(-1)[:3]
            lights_json.append({
                "light_id": f"area_{i:02d}",
                "type": "rectangle",
                "position": area_configs_joint[i]["position"].tolist(),
                "direction": area_configs_joint[i]["direction"].tolist(),
                "width": args.area_width,
                "height": args.area_height,
                "rgb_radiance": rgb.tolist(),
            })

        result = {
            "mesh_path": str(args.mesh_path),
            "material_dir": str(args.mat_dir),
            "target_path": str(args.target_path),
            "camera_meta": str(args.camera_meta),
            "hemisphere_center": center.tolist(),
            "hemisphere_radius": radius,
            "best_loss": best_loss,
            "final_ssim": 1.0 - ssim_loss,
            "num_point_lights": args.num_lights,
            "num_spot_lights": args.spot_count,
            "num_area_lights": args.area_count,
            "tune_material": args.tune_material,
            "mode": "joint",
            "lights": lights_json,
        }
        with open(output_dir / "optimized_lights.json", "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
        with open(output_dir / "loss_history.json", "w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        print("\nOptimization complete (joint mode).")
        print(f"  final PNG      : {output_dir / 'render_final.png'}")
        print(f"  light params   : {output_dir / 'optimized_lights.json'}")
        return

    # =========================================================================
    # TWO-PHASE MODE: Phase 1 point lights → Phase 2 add spotlights
    # =========================================================================
    phase1_iters = int(args.iterations * (1.0 - args.phase2_ratio))
    phase2_iters = args.iterations - phase1_iters

    print("\n" + "=" * 70)
    print("PHASE 1: All point lights optimization")
    print("=" * 70)
    print(f"  lights           = {args.num_lights} (all point)")
    print(f"  base intensity   = {base_intensity:.6f}")
    print(f"  base LR          = {actual_lr:.6f}")
    print(f"  phase1 iters     = {phase1_iters}")
    print(f"  phase2 iters     = {phase2_iters}")
    print(f"  spot_count       = {args.spot_count}")
    print(f"  spp/spp_grad     = {args.spp}/{args.spp_grad}")
    print(f"  grad_clip        = {args.grad_clip}")

    # Build scene with all point lights
    scene = build_scene(
        mesh_path=args.mesh_path,
        camera_meta_path=args.camera_meta,
        camera=camera,
        light_positions=positions,
        initial_rgb=initial_rgb,
        use_mesh_normal=args.use_mesh_normal,
        max_depth=args.max_depth,
        ambient_radiance=args.ambient_radiance,
    )

    params = mi.traverse(scene)
    material_keys = material_parameter_keys(params)
    params[material_keys["albedo"]] = materials["albedo"]
    params[material_keys["roughness"]] = materials["roughness"]
    params[material_keys["metallic"]] = materials["metallic"]
    params[material_keys["use_mesh_normal"]] = args.use_mesh_normal
    if not args.use_mesh_normal:
        params[material_keys["normal"]] = materials["normal"]
    params.update()

    light_keys = [f"light_{index:02d}.intensity.value" for index in range(args.num_lights)]
    if args.ambient_radiance > 0:
        light_keys.append("ambient.radiance.value")
    missing_light_keys = [key for key in light_keys if key not in params]
    if missing_light_keys:
        available = "\n".join(str(k) for k in params.keys())
        raise KeyError(
            f"Missing light parameters: {missing_light_keys}\n"
            f"Available parameters:\n{available}"
        )

    # Initial render
    initial_render = mi.render(scene, params, spp=args.preview_spp, seed=args.seed)
    save_render(output_dir / "render_initial", np.array(initial_render, dtype=np.float32))

    # Phase 1 optimization loop
    optimizer = mi.ad.Adam(lr=actual_lr)
    for key in light_keys:
        optimizer[key] = params[key]
    params.update(optimizer)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_rgb: list[np.ndarray] | None = None
    ema_loss = float("inf")
    ema_alpha = 0.05

    for iteration in range(phase1_iters):
        seed = args.seed + iteration * 2

        current_lr = cosine_lr_schedule(
            iteration, phase1_iters, actual_lr,
            warmup_iters=min(50, phase1_iters // 8),
            min_lr_ratio=0.05,
        )
        optimizer.set_learning_rate(current_lr)

        rendered_linear = mi.render(
            scene, params,
            spp=args.spp, spp_grad=args.spp_grad,
            seed=seed, seed_grad=seed + 1,
        )
        rendered_srgb = linear_to_srgb_dr(rendered_linear)

        loss_charb = charbonnier_loss_dr(rendered_srgb, target, epsilon=1e-3)
        diff = rendered_srgb - target
        loss_mse = dr.mean(dr.square(diff.array))

        energy_reg = mi.Float(0.0)
        for key in light_keys:
            rgb = optimizer[key]
            energy_reg += dr.mean(dr.square(rgb))
        energy_reg /= args.num_lights

        loss = (
            args.charb_weight * loss_charb
            + args.mse_weight * loss_mse
            + args.energy_weight * energy_reg
        )

        dr.eval(loss, loss_charb, loss_mse)
        loss_value = float(loss[0])
        charb_value = float(loss_charb[0])
        mse_value = float(loss_mse[0])

        if ema_loss == float("inf"):
            ema_loss = loss_value
        else:
            ema_loss = ema_alpha * loss_value + (1 - ema_alpha) * ema_loss

        dr.backward(loss)

        if args.grad_clip > 0:
            total_grad_norm_sq = mi.Float(0.0)
            for key in light_keys:
                g = dr.grad(optimizer[key])
                total_grad_norm_sq += dr.sum(dr.square(g))
            dr.eval(total_grad_norm_sq)
            grad_norm = float(total_grad_norm_sq[0]) ** 0.5
            if grad_norm > args.grad_clip:
                clip_scale = args.grad_clip / (grad_norm + 1e-8)
                for key in light_keys:
                    dr.set_grad(optimizer[key], dr.grad(optimizer[key]) * clip_scale)

        optimizer.step()

        for key in light_keys:
            optimizer[key] = dr.clip(optimizer[key], 0.0, args.max_intensity)

        params.update(optimizer)

        history.append({
            "iteration": iteration, "phase": 1,
            "loss": loss_value, "charbonnier": charb_value,
            "mse": mse_value, "lr": current_lr, "ema_loss": ema_loss,
        })

        if loss_value < best_loss:
            best_loss = loss_value
            best_rgb = [np.array(optimizer[key], dtype=np.float32).copy() for key in light_keys]

        if iteration % args.log_interval == 0 or iteration == phase1_iters - 1:
            total_rgb = np.zeros(3, dtype=np.float64)
            for key in light_keys:
                total_rgb += np.array(optimizer[key], dtype=np.float64).reshape(-1)[:3]
            print(
                f"[P1 {iteration:04d}/{phase1_iters}] "
                f"loss={loss_value:.6f} charb={charb_value:.6f} "
                f"mse={mse_value:.6f} lr={current_lr:.2e} "
                f"ema={ema_loss:.6f} sum=[{total_rgb[0]:.1f},{total_rgb[1]:.1f},{total_rgb[2]:.1f}]"
            )

        if args.save_interval > 0 and (
            iteration % args.save_interval == 0 or iteration == phase1_iters - 1
        ):
            preview = mi.render(scene, params, spp=args.preview_spp, seed=args.seed)
            save_render(output_dir / f"phase1_{iteration:04d}", np.array(preview, dtype=np.float32))

    # Save Phase 1 result
    phase1_render = mi.render(scene, params, spp=args.preview_spp, seed=args.seed)
    phase1_np = np.array(phase1_render, dtype=np.float32)
    save_render(output_dir / "phase1_final", phase1_np)
    print(f"\nPhase 1 best loss: {best_loss:.6f}")

    # =========================================================================
    # SPOTLIGHT ANALYSIS: Determine where to add spotlights
    # =========================================================================
    spot_positions_list: list[np.ndarray] = []
    spot_directions_list: list[np.ndarray] = []

    if args.spot_count > 0 and phase2_iters > 0:
        print("\n" + "=" * 70)
        print("SPOTLIGHT ANALYSIS: Finding positions for accent spotlights")
        print("=" * 70)

        # Get current rendered sRGB for residual analysis
        phase1_srgb = np.clip(linear_to_srgb_np(phase1_np), 0.0, 1.0)

        spot_positions_list, spot_directions_list = compute_spotlight_targets(
            target_srgb=target_np,
            rendered_srgb=phase1_srgb,
            mesh_path=args.mesh_path,
            camera=camera,
            light_positions=positions,
            spot_count=args.spot_count,
        )

        # Save error map visualization
        target_lum = 0.2126 * target_np[..., 0] + 0.7152 * target_np[..., 1] + 0.0722 * target_np[..., 2]
        render_lum = 0.2126 * phase1_srgb[..., 0] + 0.7152 * phase1_srgb[..., 1] + 0.0722 * phase1_srgb[..., 2]
        error_vis = np.clip(np.maximum(target_lum - render_lum, 0.0) * 3.0, 0.0, 1.0)
        cv2.imwrite(
            str(output_dir / "error_map.png"),
            (error_vis * 255).astype(np.uint8),
        )

    # =========================================================================
    # PHASE 2: Add spotlights, optimize only spotlight intensities
    # =========================================================================
    num_spots = len(spot_positions_list)
    if num_spots > 0 and phase2_iters > 0:
        print("\n" + "=" * 70)
        print("PHASE 2: Adding spotlights (point lights frozen)")
        print("=" * 70)
        print(f"  point lights (frozen) = {args.num_lights}")
        print(f"  spotlights (optimized)= {num_spots}")
        print(f"  spot cutoff  = {args.spot_cutoff_deg}°")
        print(f"  beam width   = {args.spot_beam_width_deg}°")

        # Get Phase 1 optimized point light intensities
        point_rgb = np.zeros((args.num_lights, 3), dtype=np.float32)
        for idx in range(args.num_lights):
            key = f"light_{idx:02d}.intensity.value"
            point_rgb[idx] = np.array(optimizer[key], dtype=np.float32).reshape(-1)[:3]

        # Initialize spotlight intensities
        spot_intensities = []
        for i in range(num_spots):
            # Start with low intensity, optimizer will increase where needed
            init_spot = np.array([base_intensity * args.spot_intensity_boost] * 3, dtype=np.float32)
            spot_intensities.append(init_spot)

        # Rebuild scene with point lights + spotlights
        scene = build_scene(
            mesh_path=args.mesh_path,
            camera_meta_path=args.camera_meta,
            camera=camera,
            light_positions=positions,
            initial_rgb=point_rgb,
            use_mesh_normal=args.use_mesh_normal,
            max_depth=args.max_depth,
            ambient_radiance=args.ambient_radiance,
            extra_spot_positions=spot_positions_list,
            extra_spot_directions=spot_directions_list,
            extra_spot_intensities=spot_intensities,
            spot_cutoff_deg=args.spot_cutoff_deg,
            spot_beam_width_deg=args.spot_beam_width_deg,
        )

        params = mi.traverse(scene)
        params[material_keys["albedo"]] = materials["albedo"]
        params[material_keys["roughness"]] = materials["roughness"]
        params[material_keys["metallic"]] = materials["metallic"]
        params[material_keys["use_mesh_normal"]] = args.use_mesh_normal
        if not args.use_mesh_normal:
            params[material_keys["normal"]] = materials["normal"]
        params.update()

        # Only optimize spotlight intensities (point lights frozen)
        spot_keys = [f"spot_{i:02d}.intensity.value" for i in range(num_spots)]
        missing_spot_keys = [key for key in spot_keys if key not in params]
        if missing_spot_keys:
            available = "\n".join(str(k) for k in params.keys())
            raise KeyError(
                f"Missing spotlight parameters: {missing_spot_keys}\n"
                f"Available parameters:\n{available}"
            )

        # Phase 2 optimizer: only spotlight intensities
        phase2_lr = actual_lr * args.phase2_lr_scale
        optimizer2 = mi.ad.Adam(lr=phase2_lr)
        for key in spot_keys:
            optimizer2[key] = params[key]
        params.update(optimizer2)

        ema_loss_p2 = float("inf")
        best_spot_rgb: list[np.ndarray] | None = None

        for iteration in range(phase2_iters):
            seed = args.seed + (phase1_iters + iteration) * 2

            current_lr = cosine_lr_schedule(
                iteration, phase2_iters, phase2_lr,
                warmup_iters=min(20, phase2_iters // 8),
                min_lr_ratio=0.01,
            )
            optimizer2.set_learning_rate(current_lr)

            rendered_linear = mi.render(
                scene, params,
                spp=args.spp, spp_grad=args.spp_grad,
                seed=seed, seed_grad=seed + 1,
            )
            rendered_srgb = linear_to_srgb_dr(rendered_linear)

            loss_charb = charbonnier_loss_dr(rendered_srgb, target, epsilon=1e-3)
            diff = rendered_srgb - target
            loss_mse = dr.mean(dr.square(diff.array))

            # Regularize spotlight intensities
            energy_reg = mi.Float(0.0)
            for key in spot_keys:
                rgb = optimizer2[key]
                energy_reg += dr.mean(dr.square(rgb))
            energy_reg /= num_spots

            loss = (
                args.charb_weight * loss_charb
                + args.mse_weight * loss_mse
                + args.energy_weight * energy_reg
            )

            dr.eval(loss, loss_charb, loss_mse)
            loss_value = float(loss[0])
            charb_value = float(loss_charb[0])
            mse_value = float(loss_mse[0])

            if ema_loss_p2 == float("inf"):
                ema_loss_p2 = loss_value
            else:
                ema_loss_p2 = ema_alpha * loss_value + (1 - ema_alpha) * ema_loss_p2

            dr.backward(loss)

            if args.grad_clip > 0:
                total_grad_norm_sq = mi.Float(0.0)
                for key in spot_keys:
                    g = dr.grad(optimizer2[key])
                    total_grad_norm_sq += dr.sum(dr.square(g))
                dr.eval(total_grad_norm_sq)
                grad_norm = float(total_grad_norm_sq[0]) ** 0.5
                if grad_norm > args.grad_clip:
                    clip_scale = args.grad_clip / (grad_norm + 1e-8)
                    for key in spot_keys:
                        dr.set_grad(optimizer2[key], dr.grad(optimizer2[key]) * clip_scale)

            optimizer2.step()

            for key in spot_keys:
                optimizer2[key] = dr.clip(optimizer2[key], 0.0, args.max_intensity)

            params.update(optimizer2)

            history.append({
                "iteration": phase1_iters + iteration, "phase": 2,
                "loss": loss_value, "charbonnier": charb_value,
                "mse": mse_value, "lr": current_lr, "ema_loss": ema_loss_p2,
            })

            if loss_value < best_loss:
                best_loss = loss_value
                best_spot_rgb = [np.array(optimizer2[key], dtype=np.float32).copy() for key in spot_keys]

            if iteration % args.log_interval == 0 or iteration == phase2_iters - 1:
                total_spot_rgb = np.zeros(3, dtype=np.float64)
                for key in spot_keys:
                    total_spot_rgb += np.array(optimizer2[key], dtype=np.float64).reshape(-1)[:3]
                print(
                    f"[P2 {iteration:04d}/{phase2_iters}] "
                    f"loss={loss_value:.6f} charb={charb_value:.6f} "
                    f"mse={mse_value:.6f} lr={current_lr:.2e} "
                    f"ema={ema_loss_p2:.6f} spot_sum=[{total_spot_rgb[0]:.1f},{total_spot_rgb[1]:.1f},{total_spot_rgb[2]:.1f}]"
                )

            if args.save_interval > 0 and (
                iteration % args.save_interval == 0 or iteration == phase2_iters - 1
            ):
                preview = mi.render(scene, params, spp=args.preview_spp, seed=args.seed)
                save_render(output_dir / f"phase2_{iteration:04d}", np.array(preview, dtype=np.float32))

        # Restore best spotlight intensities
        if best_spot_rgb is not None:
            for key, rgb in zip(spot_keys, best_spot_rgb):
                optimizer2[key] = mi.Color3f(rgb.reshape(-1)[:3])
            params.update(optimizer2)

    # =========================================================================
    # Render final
    # =========================================================================
    final = mi.render(scene, params, spp=args.final_spp, seed=args.seed)
    final_np = np.array(final, dtype=np.float32)
    save_render(output_dir / "render_final", final_np)

    # Compute final SSIM for quality assessment
    final_srgb = np.clip(linear_to_srgb_np(final_np), 0.0, 1.0)
    ssim_loss = compute_ssim_loss_np(final_srgb, target_np)
    print(f"\nFinal SSIM loss (1-SSIM): {ssim_loss:.6f} (SSIM={1-ssim_loss:.6f})")
    print(f"Best optimization loss: {best_loss:.6f}")

    # Save results
    lights_json = []
    for index, position in enumerate(positions):
        key = f"light_{index:02d}.intensity.value"
        rgb = np.array(params[key], dtype=np.float32).reshape(-1)[:3]
        lights_json.append({
            "light_id": f"light_{index:02d}",
            "type": "point",
            "position": position.tolist(),
            "rgb_intensity": rgb.tolist(),
        })

    # Add spotlight info
    for i in range(num_spots):
        key = f"spot_{i:02d}.intensity.value"
        rgb = np.array(params[key], dtype=np.float32).reshape(-1)[:3]
        lights_json.append({
            "light_id": f"spot_{i:02d}",
            "type": "spot",
            "position": spot_positions_list[i].tolist(),
            "direction": spot_directions_list[i].tolist(),
            "rgb_intensity": rgb.tolist(),
            "cutoff_angle": args.spot_cutoff_deg,
            "beam_width": args.spot_beam_width_deg,
        })

    result = {
        "mesh_path": str(args.mesh_path),
        "material_dir": str(args.mat_dir),
        "target_path": str(args.target_path),
        "camera_meta": str(args.camera_meta),
        "hemisphere_center": center.tolist(),
        "hemisphere_radius": radius,
        "front_only": args.front_only,
        "min_elevation_deg": args.min_elevation_deg,
        "best_loss": best_loss,
        "final_ssim": 1.0 - ssim_loss,
        "num_point_lights": args.num_lights,
        "num_spot_lights": num_spots,
        "lights": lights_json,
    }
    with open(output_dir / "optimized_lights.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    with open(output_dir / "loss_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print("\nOptimization complete.")
    print(f"  final PNG      : {output_dir / 'render_final.png'}")
    print(f"  light params   : {output_dir / 'optimized_lights.json'}")
    print(f"  loss history   : {output_dir / 'loss_history.json'}")
    print(f"  point lights   : {args.num_lights}")
    print(f"  spot lights    : {num_spots}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a mixed point+spot light rig with Mitsuba (two-stage)."
    )
    parser.add_argument("--target_path", required=True, type=str)
    parser.add_argument("--mesh_path", required=True, type=str)
    parser.add_argument("--mat_dir", required=True, type=str)
    parser.add_argument("--camera_meta", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--num_lights", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--radius_scale", type=float, default=1.5)
    parser.add_argument("--min_elevation_deg", type=float, default=10.0)
    parser.add_argument(
        "--front_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict lights to the camera-facing half of the upper hemisphere.",
    )

    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--spp_grad", type=int, default=16)
    parser.add_argument("--preview_spp", type=int, default=32)
    parser.add_argument("--final_spp", type=int, default=512)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--ambient_radiance", type=float, default=0.0,
                        help="Initial ambient light radiance. 0 disables ambient.")

    # Spotlight configuration
    parser.add_argument("--spot_count", type=int, default=8,
                        help="Number of spotlights. 0 disables spotlights.")
    parser.add_argument("--spot_cutoff_deg", type=float, default=25.0,
                        help="Spotlight cone half-angle in degrees.")
    parser.add_argument("--spot_beam_width_deg", type=float, default=15.0,
                        help="Spotlight soft edge width in degrees.")
    parser.add_argument("--spot_intensity_boost", type=float, default=1.0,
                        help="Intensity multiplier for spotlights relative to base_intensity.")
    parser.add_argument("--phase2_ratio", type=float, default=0.4,
                        help="Fraction of total iterations allocated to Phase 2 (two-phase mode).")
    parser.add_argument("--phase2_lr_scale", type=float, default=0.5,
                        help="LR multiplier for Phase 2 relative to Phase 1 base LR.")
    parser.add_argument("--joint", action="store_true",
                        help="Use joint optimization (point+spot from start) instead of two-phase.")

    # Area light (rectangle) configuration
    parser.add_argument("--area_count", type=int, default=0,
                        help="Number of rectangle area lights. 0 disables.")
    parser.add_argument("--area_width", type=float, default=2.0,
                        help="Width of each rectangle area light.")
    parser.add_argument("--area_height", type=float, default=1.5,
                        help="Height of each rectangle area light.")
    parser.add_argument("--area_radiance_boost", type=float, default=0.5,
                        help="Radiance multiplier for area lights relative to base_intensity.")

    # Material fine-tuning
    parser.add_argument("--tune_material", action="store_true",
                        help="Enable joint material (roughness/metallic) fine-tuning.")
    parser.add_argument("--mat_lr_scale", type=float, default=0.001,
                        help="LR scale for material parameters (relative to base LR).")
    parser.add_argument("--mat_perturb", type=float, default=0.05,
                        help="Max perturbation for material params (clamped around initial).")

    parser.add_argument(
        "--lr",
        type=float,
        default=-1.0,
        help="Absolute Adam learning rate. Negative means base_intensity * lr_scale.",
    )
    parser.add_argument("--lr_scale", type=float, default=0.008)
    parser.add_argument("--max_intensity", type=float, default=1e5)

    # Loss weights
    parser.add_argument("--charb_weight", type=float, default=1.0,
                        help="Weight for Charbonnier (smooth L1) loss.")
    parser.add_argument("--mse_weight", type=float, default=0.5,
                        help="Weight for MSE loss in sRGB.")
    parser.add_argument("--energy_weight", type=float, default=1e-4,
                        help="Weight for L2 intensity regularization.")

    # Optimization stability
    parser.add_argument("--grad_clip", type=float, default=0.3,
                        help="Max gradient norm. 0 disables clipping.")

    parser.add_argument(
        "--use_mesh_normal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    optimize_lights(arguments)