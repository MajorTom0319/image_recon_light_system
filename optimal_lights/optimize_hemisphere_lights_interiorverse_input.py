"""
#适配一下interiorverse合成数据集的数据输入

Optimize a 16-light hemispherical point-light rig for a Materialist scene.

The script keeps the PLY geometry and MatNet material maps fixed, randomly
places N point lights on an upper hemisphere around the scene, and optimizes
per-light RGB radiant intensity with Mitsuba 3 differentiable rendering.

Important limitation of this first version:
    Point-light `intensity` is differentiable in Mitsuba 3, while point-light
    `position` is only traversable. Therefore the randomly initialized light
    positions stay fixed during this optimization; only RGB intensity/color
    is optimized. The final JSON stores both positions and optimized RGB values.

Example:
python optimize_hemisphere_lights.py \
    --target_path /home/majortom/project/test2.png \
    --mesh_path /home/majortom/project/Materialist/output_imgs/test2_matnet_out/mesh_hdri.ply \
    --mat_dir /home/majortom/project/Materialist/output_imgs/test2_matnet_out \
    --camera_meta /home/majortom/project/Materialist/output_imgs/test2_matnet_out/camera_meta.json \
    --output_dir /home/majortom/project/Materialist/output_imgs/test2_matnet_out/light_optimization \
    --num_lights 16 \
    --iterations 500 \
    --spp 8 \
    --spp_grad 8 \
    --final_spp 256 \ 
    --no-front_only

python optimize_hemisphere_lights.py     --target_path /home/majortom/project/test2.png     --mesh_path /home/majortom/project/Materialist/output_imgs/test2_matnet_out/mesh_hdri.ply     --mat_dir /home/majortom/project/Materialist/output_imgs/test2_matnet_out     --camera_meta /home/majortom/project/Materialist/output_imgs/test2_matnet_out/camera_meta.json     --output_dir /home/majortom/project/Materialist/output_imgs/test2_matnet_out/light_optimization     --num_lights 64     --iterations 500     --color_weight 0.0     --radius_scale 1.2     --max_depth 2     --spp 16     --spp_grad 8     --final_spp 256 --no-front_only

Synthetic data mode (InteriorVerse format):
python optimize_hemisphere_lights_1.py \
    --data_dir /home/majortom/project/datasets/interiorverse1 \
    --output_dir /home/majortom/project/Materialist/output_imgs/synthetic_light_opt \
    --fov 60.0 \
    --num_lights 16 \
    --iterations 500 \
    --spp 16 \
    --spp_grad 8 \
    --final_spp 256
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Ensure the Materialist root (parent of optimal_lights/) is importable.
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
# Synthetic data support (InteriorVerse format)
# -----------------------------------------------------------------------------


def load_synthetic_data(data_dir: str | Path) -> dict[str, Any]:
    """Load InteriorVerse-style synthetic data from *data_dir*.

    Expected files (prefix ``000_``):
        000_albedo.exr   – RGB albedo
        000_material.exr – R = roughness, G = metallic
        000_normal.exr   – world-space normal
        000_im.exr       – linear HDR target image
        000_depth.exr    – depth (may contain inf for background)
        000_mask.exr     – optional foreground mask
    """
    data_dir = Path(data_dir)

    albedo = _read_bitmap(data_dir / "000_albedo.exr")[..., :3]

    material = _read_bitmap(data_dir / "000_material.exr")
    roughness = material[..., 0:1]
    metallic = material[..., 1:2]

    normal = _read_bitmap(data_dir / "000_normal.exr")[..., :3]

    # Target: linear HDR -> sRGB for loss comparison
    target_linear = _read_bitmap(data_dir / "000_im.exr")[..., :3]
    target_srgb = np.clip(linear_to_srgb_np(target_linear), 0.0, 1.0).astype(
        np.float32
    )

    # Depth (first channel, replace inf with 0)
    depth = _read_bitmap(data_dir / "000_depth.exr")
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)

    mask = None
    mask_path = data_dir / "000_mask.exr"
    if mask_path.exists():
        mask = _read_bitmap(mask_path)
        if mask.ndim == 3:
            mask = mask[..., 0]

    h, w = albedo.shape[:2]
    materials: dict[str, torch.Tensor] = {
        "albedo": torch.from_numpy(albedo.copy()).float().cuda().clamp(0.0, 1.0),
        "roughness": torch.from_numpy(roughness.copy())
        .float()
        .cuda()
        .clamp(0.07, 1.0),
        "metallic": torch.from_numpy(metallic.copy())
        .float()
        .cuda()
        .clamp(0.0, 1.0),
        "normal": torch.nn.functional.normalize(
            torch.from_numpy(normal.copy()).float().cuda(), p=2, dim=-1
        ),
    }

    print("Synthetic data loaded:")
    print(f"  directory : {data_dir}")
    print(f"  resolution: {w} x {h}")
    print(f"  roughness : [{roughness.min():.4f}, {roughness.max():.4f}]")
    print(f"  metallic  : [{metallic.min():.4f}, {metallic.max():.4f}]")
    print(f"  target HDR: [{target_linear.min():.4f}, {target_linear.max():.4f}]")

    return {
        "materials": materials,
        "target_srgb": target_srgb,
        "depth": depth,
        "mask": mask,
        "width": w,
        "height": h,
    }


def generate_default_camera_meta(
    output_dir: Path,
    width: int,
    height: int,
    hfov_deg: float = 60.0,
) -> Path:
    """Write a default ``camera_meta.json`` for synthetic-data mode."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hfov_rad = math.radians(hfov_deg)
    focal = 0.5 * width / math.tan(0.5 * hfov_rad)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    vfov_deg = math.degrees(2.0 * math.atan(math.tan(hfov_rad * 0.5) * height / width))

    meta: dict[str, Any] = {
        "schema": "synthetic_default",
        "camera_model": "pinhole",
        "film.size": [width, height],
        "K": [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        "x_fov": [hfov_deg],
        "y_fov": [vfov_deg],
        "near_clip": 0.01,
        "far_clip": 10000.0,
        "to_world": [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
    }
    meta_path = output_dir / "camera_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Default camera meta written to: {meta_path}")
    print(f"  hfov = {hfov_deg:.1f} deg, vfov = {vfov_deg:.1f} deg")
    return meta_path


def build_mesh_from_depth(
    depth: np.ndarray,
    output_dir: Path,
    focal: float,
    width: int,
    height: int,
) -> Path:
    """Reconstruct a PLY mesh from a depth map and save it to *output_dir*."""
    from myutils.mesh_recon import (
        depth_file_to_mesh,
        rotate_mesh_around_x,
        set_recon_camera,
    )

    depth_clean = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(
        np.float32
    )
    camera = set_recon_camera(width=width, height=height)
    camera.intrinsic_matrix = np.array(
        [[focal, 0, (width - 1) / 2.0], [0, focal, (height - 1) / 2.0], [0, 0, 1]],
        dtype=np.float64,
    )

    print("Building mesh from depth map ...")
    mesh, _ = depth_file_to_mesh(
        depth_clean, cameraMatrix=camera, minAngle=1.5, depthScale=1.0
    )
    mesh = rotate_mesh_around_x(mesh, 180)

    mesh_path = output_dir / "synthetic_mesh.ply"
    o3d.io.write_triangle_mesh(str(mesh_path), mesh)
    n_verts = len(mesh.vertices)
    n_tris = len(mesh.triangles)
    print(f"  mesh saved: {mesh_path}  ({n_verts} verts, {n_tris} tris)")
    return mesh_path


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
    Uniformly sample an upper hemisphere whose up axis is +Y.

    elevation = 0 degrees at the horizon and 90 degrees at the zenith.
    When front_only=True, retain the camera-facing half with +Z offset. This is
    often more useful for Materialist's single-view 2.5D mesh (camera at origin,
    scene generally along -Z).
    """
    rng = np.random.default_rng(seed)
    y_min = math.sin(math.radians(min_elevation_deg))
    y = rng.uniform(y_min, 1.0, size=count)
    radial = np.sqrt(np.maximum(1.0 - y * y, 0.0))

    if front_only:
        azimuth = rng.uniform(0.0, math.pi, size=count)  # sin(phi) >= 0
    else:
        azimuth = rng.uniform(0.0, 2.0 * math.pi, size=count)

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
) -> mi.Scene:
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
        # "ambient": {
        #     "type": "constant",
        #     "radiance": {
        #         "type": "rgb",
        #         "value": [0.5, 0.5, 0.5],
        #     },
        # },
    }

    for index, position in enumerate(light_positions):
        scene_dict[f"light_{index:02d}"] = {
            "type": "point",
            "position": position.tolist(),
            "intensity": {
                "type": "rgb",
                "value": initial_rgb[index].tolist(),
            },
        }

    return mi.load_dict(scene_dict)


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


def save_render(base_path: Path, image_linear: np.ndarray) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    # mi.util.write_bitmap(str(base_path.with_suffix(".exr")), image_linear)
    image_srgb = np.clip(linear_to_srgb_np(image_linear), 0.0, 1.0)
    image_u8 = (image_srgb * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(
        str(base_path.with_suffix(".png")),
        cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR),
    )


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

    # ---- Synthetic data mode ------------------------------------------------
    if args.data_dir:
        syn = load_synthetic_data(args.data_dir)
        materials = syn["materials"]

        camera_meta_path = generate_default_camera_meta(
            output_dir, syn["width"], syn["height"], hfov_deg=args.fov
        )
        args.camera_meta = str(camera_meta_path)

        if args.mesh_path is None:
            mesh_path = build_mesh_from_depth(
                syn["depth"],
                output_dir,
                focal=0.5 * syn["width"] / math.tan(0.5 * math.radians(args.fov)),
                width=syn["width"],
                height=syn["height"],
            )
            args.mesh_path = str(mesh_path)
    else:
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

    # If each point light is roughly `radius` away, sum(I_i / r^2) is on the
    # order of init_irradiance. This produces a scale-aware initialization.
    base_intensity = args.init_irradiance * radius * radius / args.num_lights
    rng = np.random.default_rng(args.seed + 1)
    color_jitter = rng.uniform(0.9, 1.1, size=(args.num_lights, 3)).astype(np.float32)
    initial_rgb = base_intensity * color_jitter

    scene = build_scene(
        mesh_path=args.mesh_path,
        camera_meta_path=args.camera_meta,
        camera=camera,
        light_positions=positions,
        initial_rgb=initial_rgb,
        use_mesh_normal=args.use_mesh_normal,
        max_depth=args.max_depth,
    )

    params = mi.traverse(scene)
    material_keys = material_parameter_keys(params)
    params[material_keys["albedo"]] = materials["albedo"]
    params[material_keys["roughness"]] = materials["roughness"]
    params[material_keys["metallic"]] = materials["metallic"]
    # params[material_keys["roughness"]] = torch.ones_like(materials["roughness"])
    # params[material_keys["metallic"]] = torch.zeros_like(materials["metallic"])
    params[material_keys["use_mesh_normal"]] = args.use_mesh_normal
    if not args.use_mesh_normal:
        params[material_keys["normal"]] = materials["normal"]
    params.update()

    light_keys = [f"light_{index:02d}.intensity.value" for index in range(args.num_lights)]
    missing_light_keys = [key for key in light_keys if key not in params]
    if missing_light_keys:
        available = "\n".join(str(k) for k in params.keys())
        raise KeyError(
            f"Missing point-light parameters: {missing_light_keys}\n"
            f"Available parameters:\n{available}"
        )

    if args.data_dir:
        target_np = syn["target_srgb"]
    else:
        target_np = load_target_srgb(
            args.target_path,
            width=camera["width"],
            height=camera["height"],
        )
    target = mi.TensorXf(target_np)
    cv2.imwrite(
        str(output_dir / "target.png"),
        cv2.cvtColor((target_np * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR),
    )

    actual_lr = args.lr if args.lr > 0 else max(base_intensity * args.lr_scale, 1e-5)
    optimizer = mi.ad.Adam(lr=actual_lr)
    for key in light_keys:
        optimizer[key] = params[key]
    params.update(optimizer)

    print("Optimization:")
    print(f"  lights       = {args.num_lights}")
    print(f"  base intensity per light = {base_intensity:.6f}")
    print(f"  Adam lr      = {actual_lr:.6f}")
    print(f"  iterations   = {args.iterations}")
    print(f"  spp/spp_grad = {args.spp}/{args.spp_grad}")

    initial = mi.render(scene, params, spp=args.preview_spp, seed=args.seed)
    save_render(output_dir / "render_initial", np.array(initial, dtype=np.float32))

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_rgb: list[np.ndarray] | None = None

    for iteration in range(args.iterations):
        seed = args.seed + iteration * 2
        rendered_linear = mi.render(
            scene,
            params,
            spp=args.spp,
            spp_grad=args.spp_grad,
            seed=seed,
            seed_grad=seed + 1,
        )
        rendered_srgb = linear_to_srgb_dr(rendered_linear)
        difference = rendered_srgb - target

        loss_mse = dr.mean(dr.square(difference.array))
        loss_l1 = dr.mean(dr.abs(difference.array))

        energy_reg = mi.Float(0.0)
        color_reg = mi.Float(0.0)
        for key in light_keys:
            rgb = optimizer[key]
            energy_reg += dr.mean(rgb)
            channel_mean = dr.mean(rgb)
            color_reg += dr.mean(dr.square(rgb - channel_mean))
        energy_reg /= args.num_lights
        color_reg /= args.num_lights

        loss = (
            args.mse_weight * loss_mse
            + args.l1_weight * loss_l1
            + args.energy_weight * energy_reg
            + args.color_weight * color_reg
        )

        # 先求值并转成 Python float，用于日志和 best checkpoint。
        dr.eval(
            loss,
            loss_mse,
            loss_l1,
        )

        loss_value = float(loss[0])
        mse_value = float(loss_mse[0])
        l1_value = float(loss_l1[0])

        # 然后执行反向传播和参数更新。
        dr.backward(loss)
        optimizer.step()

        for key in light_keys:
            optimizer[key] = dr.clip(
                optimizer[key],
                0.0,
                args.max_intensity,
            )

        params.update(optimizer)
        history.append(
            {
                "iteration": iteration,
                "loss": loss_value,
                "mse": mse_value,
                "l1": l1_value,
            }
        )

        if loss_value < best_loss:
            best_loss = loss_value
            best_rgb = [np.array(optimizer[key], dtype=np.float32).copy() for key in light_keys]

        if iteration % args.log_interval == 0 or iteration == args.iterations - 1:
            total_rgb = np.zeros(3, dtype=np.float64)
            for key in light_keys:
                total_rgb += np.array(optimizer[key], dtype=np.float64).reshape(-1)[:3]
            print(
                f"[{iteration:04d}/{args.iterations}] "
                f"loss={loss_value:.6f} mse={mse_value:.6f} "
                f"l1={l1_value:.6f} total_rgb={total_rgb.tolist()}"
            )

        if args.save_interval > 0 and (
            iteration % args.save_interval == 0 or iteration == args.iterations - 1
        ):
            preview = mi.render(
                scene,
                params,
                spp=args.preview_spp,
                seed=args.seed,
            )
            save_render(
                output_dir / f"progress_{iteration:04d}",
                np.array(preview, dtype=np.float32),
            )

    if best_rgb is not None:
        for key, rgb in zip(light_keys, best_rgb):
            optimizer[key] = mi.Color3f(rgb.reshape(-1)[:3])
        params.update(optimizer)

    final = mi.render(scene, params, spp=args.final_spp, seed=args.seed)
    final_np = np.array(final, dtype=np.float32)
    save_render(output_dir / "render_final", final_np)

    lights_json = []
    for index, (position, key) in enumerate(zip(positions, light_keys)):
        rgb = np.array(params[key], dtype=np.float32).reshape(-1)[:3]
        lights_json.append(
            {
                "light_id": f"light_{index:02d}",
                "position": position.tolist(),
                "rgb_intensity": rgb.tolist(),
            }
        )

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
        "lights": lights_json,
    }
    with open(output_dir / "optimized_lights.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    with open(output_dir / "loss_history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print("Optimization complete.")
    print(f"  final EXR/PNG : {output_dir / 'render_final'}")
    print(f"  light params  : {output_dir / 'optimized_lights.json'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a fixed-position 16-light hemisphere rig with Mitsuba."
    )
    parser.add_argument("--target_path", type=str, default=None)
    parser.add_argument("--mesh_path", type=str, default=None)
    parser.add_argument("--mat_dir", type=str, default=None)
    parser.add_argument("--camera_meta", type=str, default=None)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to InteriorVerse-style synthetic data directory containing "
            "000_albedo.exr, 000_material.exr (R=roughness, G=metallic), "
            "000_normal.exr, 000_im.exr, 000_depth.exr. "
            "When set, --target_path / --mesh_path / --mat_dir / --camera_meta "
            "are auto-generated and can be omitted."
        ),
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=60.0,
        help="Horizontal FOV in degrees for the default synthetic camera.",
    )

    parser.add_argument("--num_lights", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--radius_scale", type=float, default=1.5)
    parser.add_argument("--min_elevation_deg", type=float, default=10.0)
    parser.add_argument(
        "--front_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restrict lights to the camera-facing half of the upper hemisphere.",
    )

    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--spp_grad", type=int, default=8)
    parser.add_argument("--preview_spp", type=int, default=32)
    parser.add_argument("--final_spp", type=int, default=256)
    parser.add_argument("--max_depth", type=int, default=2)

    parser.add_argument(
        "--init_irradiance",
        type=float,
        default=1.0,
        help="Approximate initial total irradiance at the hemisphere center.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=-1.0,
        help="Absolute Adam learning rate. Negative means base_intensity * lr_scale.",
    )
    parser.add_argument("--lr_scale", type=float, default=0.02)
    parser.add_argument("--max_intensity", type=float, default=1e6)

    parser.add_argument("--mse_weight", type=float, default=0.25)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--energy_weight", type=float, default=1e-5)
    parser.add_argument("--color_weight", type=float, default=1e-4)

    parser.add_argument(
        "--use_mesh_normal",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=50)

    args = parser.parse_args()

    if not args.data_dir:
        missing = []
        if args.target_path is None:
            missing.append("--target_path")
        if args.mesh_path is None:
            missing.append("--mesh_path")
        if args.mat_dir is None:
            missing.append("--mat_dir")
        if args.camera_meta is None:
            missing.append("--camera_meta")
        if missing:
            parser.error(
                f"The following arguments are required when --data_dir is not "
                f"set: {', '.join(missing)}"
            )

    return args


if __name__ == "__main__":
    arguments = parse_args()
    optimize_lights(arguments)