#!/usr/bin/env python3
"""Optimize a 32x16 far-field HDRI, then Materialist RM and albedo maps.

This experiment deliberately disables Stage B: ILE lamp radiance and window
sun/sky/ground lobes stay fixed at their input values times ``--radiance-scale``.
Stage C first optimizes only the far-field envmap. Stage D then freezes all lighting, jointly optimizes
roughness/metallic, freezes their best checkpoint, and finally optimizes
albedo. Its loss and per-phase optimizer follow the real-image direct material
optimization used in ``inverse_img_w_mi_ori.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_light.io import load_ile_lights
from hybrid_light.mitsuba_builder import build_hybrid_scene_dict
from hybrid_light.visualization import render_projection_debug
from scripts.optimize_ile_farfield import (
    cosine_learning_rate,
    linear_to_srgb_dr,
    linear_to_srgb_np,
    load_target_linear,
    optimize_farfield,
    save_linear_image,
    set_material_parameters,
    write_json,
)
from scripts.render_ile_lights import (
    _find_material_maps,
    _load_camera,
    _load_material_arrays,
    _resolve_mesh,
)


MATERIAL_KEYS = {
    "a": "materialist_mesh.bsdf.a",
    "r": "materialist_mesh.bsdf.r",
    "m": "materialist_mesh.bsdf.m",
}
MATERIAL_NAMES = {"a": "albedo", "r": "roughness", "m": "metallic"}
MATERIAL_BOUNDS = {"a": (0.0, 1.0), "r": (0.07, 1.0), "m": (0.0, 1.0)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Keep all ILE lamp/window intensities fixed, optimize a 32x16 "
            "HDRI, then optimize Materialist roughness/metallic and albedo."
        ),
    )
    parser.add_argument("--materialist-dir", type=Path, required=True)
    parser.add_argument("--lights-json", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--camera-meta", type=Path, default=None)
    parser.add_argument("--material-dir", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--geometry-scale", type=float, default=None)
    parser.add_argument("--radiance-scale", type=float, default=1.0)
    parser.add_argument("--visible-offset", type=float, default=0.005)
    parser.add_argument(
        "--model_name",
        "--model-name",
        choices=("none", "pos_mlp"),
        default="pos_mlp",
        help="Optimization parameterization: direct pixels or Materialist PosMLP",
    )

    # Width x height = 32 x 16; the array layout is H x W x C.
    parser.add_argument("--farfield-width", type=int, default=32)
    parser.add_argument("--farfield-height", type=int, default=16)
    parser.add_argument("--farfield-iters", type=int, default=300)
    parser.add_argument("--farfield-lr", type=float, default=0.003)
    parser.add_argument("--farfield-init", type=float, default=0.01)
    parser.add_argument("--farfield-max", type=float, default=4.0)
    parser.add_argument("--farfield-tv-weight", type=float, default=1e-2)
    parser.add_argument("--farfield-energy-weight", type=float, default=1e-3)
    parser.add_argument(
        "--farfield-checkpoint-metric",
        choices=("mse", "loss"),
        default="mse",
        help="Metric used to select the exported Stage-C checkpoint",
    )

    parser.add_argument(
        "--material-order",
        nargs="+",
        choices=("a", "r", "m", "ar", "am", "rm", "arm"),
        default=argparse.SUPPRESS,
        help="Sequential material phases; default: rm a",
    )
    parser.add_argument(
        "--material-channels",
        choices=("a", "r", "m", "ar", "am", "rm", "arm"),
        default=None,
        help="Deprecated compatibility option for one simultaneous phase",
    )
    parser.add_argument(
        "--material-iters",
        type=int,
        default=500,
        help="Maximum iterations for each material phase",
    )
    parser.add_argument("--material-lr", type=float, default=3e-4)
    parser.add_argument("--material-lr-step", type=int, default=100)
    parser.add_argument("--material-lr-gamma", type=float, default=0.8)
    parser.add_argument("--material-patience", type=int, default=200)
    parser.add_argument(
        "--material-min-delta",
        type=float,
        default=None,
        help="Override relative MSE improvement; default: rm=0.001, a=0.005",
    )
    parser.add_argument(
        "--material-prior-weight",
        type=float,
        default=0.1,
        help="L1 weight to the input maps, equivalent to scale_delta",
    )
    parser.add_argument(
        "--posmlp-train-seeds",
        type=int,
        default=2,
        help="Independent Monte Carlo renders averaged per PosMLP training step",
    )
    parser.add_argument(
        "--posmlp-material-tv-weight",
        type=float,
        default=1e-2,
        help="TV weight on PosMLP material changes from their input maps",
    )
    parser.add_argument(
        "--posmlp-hidden-depth",
        type=int,
        default=4,
        help="Number of hidden layers in the far-field and material PosMLPs, for nerf it's 8",
    )
    parser.add_argument(
        "--posmlp-hidden-width",
        type=int,
        default=256,
        help="Hidden width of the far-field and material PosMLPs",
    )
    parser.add_argument(
        "--posmlp-multires",
        type=int,
        default=3,
        help="Number of Fourier frequency bands used by both PosMLPs",
    )
    parser.add_argument(
        "--posmlp-backbone",
        choices=("pos_mlp", "nerf_mlp"),
        default="pos_mlp",
        help="Coordinate-network backbone; nerf_mlp uses ReLU and a layer-4 skip",
    )
    exposure_group = parser.add_mutually_exclusive_group()
    exposure_group.add_argument(
        "--material-exposure-match",
        dest="material_exposure_match",
        action="store_true",
        help=(
            "Opt in to Materialist's detached mean-exposure ratio. This can "
            "improve scale-invariant matching but no longer optimizes physical brightness."
        ),
    )
    exposure_group.add_argument(
        "--no-material-exposure-match",
        dest="material_exposure_match",
        action="store_false",
        help="Explicitly keep physical-brightness matching (the default)",
    )
    parser.set_defaults(material_exposure_match=False)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--resample-each-iteration",
        dest="resample_each_iteration",
        action="store_true",
        help="Use a new Monte Carlo seed every iteration (the default)",
    )
    seed_group.add_argument(
        "--fixed-optimization-seed",
        dest="resample_each_iteration",
        action="store_false",
        help="Use common random numbers for debugging; can overfit one sample pattern",
    )
    parser.set_defaults(resample_each_iteration=True)

    parser.add_argument(
        "--use-mesh-normal",
        action="store_true",
        help="Use interpolated PLY normals instead of the loaded MoGe2 normal map",
    )
    parser.add_argument(
        "--integrator",
        choices=("prb", "path"),
        default="prb",
        help="Differentiable integrator used for optimization and validation",
    )
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-seeds", type=int, default=2)
    parser.add_argument(
        "--validation-min-delta",
        type=float,
        default=1e-3,
        help="Required relative validation-MSE improvement",
    )
    parser.add_argument(
        "--validation-patience",
        type=int,
        default=8,
        help="Validation checks without improvement before early stopping",
    )
    parser.add_argument("--emitter-mask-dilate", type=int, default=3)
    parser.add_argument("--mesh-edge-dilate", type=int, default=2)
    parser.add_argument("--depth-edge-rtol", type=float, default=0.05)

    parser.add_argument("--charbonnier-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--spp-grad", type=int, default=16)
    parser.add_argument(
        "--validation-spp",
        "--preview-spp",
        dest="validation_spp",
        type=int,
        default=64,
        help="SPP per fixed validation seed; --preview-spp is a compatibility alias",
    )
    parser.add_argument("--final-spp", type=int, default=256)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-denoise", action="store_true")
    args = parser.parse_args()

    material_order = getattr(args, "material_order", None)
    if material_order is not None and args.material_channels is not None:
        parser.error("Use --material-order or --material-channels, not both")
    if args.material_channels is not None:
        args.material_order = [args.material_channels]
    elif material_order is None:
        args.material_order = ["rm", "a"]

    positive = (
        "farfield_width",
        "farfield_height",
        "farfield_iters",
        "farfield_lr",
        "farfield_max",
        "material_iters",
        "material_lr",
        "material_lr_step",
        "material_patience",
        "posmlp_train_seeds",
        "posmlp_hidden_depth",
        "posmlp_hidden_width",
        "posmlp_multires",
        "spp",
        "spp_grad",
        "validation_spp",
        "validation_seeds",
        "validation_interval",
        "validation_patience",
        "final_spp",
        "max_depth",
        "log_interval",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "farfield_init",
        "farfield_tv_weight",
        "farfield_energy_weight",
        "material_prior_weight",
        "posmlp_material_tv_weight",
        "charbonnier_weight",
        "mse_weight",
        "visible_offset",
        "validation_min_delta",
        "emitter_mask_dilate",
        "mesh_edge_dilate",
        "depth_edge_rtol",
    )
    for name in nonnegative:
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if args.farfield_init > args.farfield_max:
        parser.error("--farfield-init cannot exceed --farfield-max")
    if args.radiance_scale <= 0:
        parser.error("--radiance-scale must be positive")
    if not 0 < args.material_lr_gamma <= 1:
        parser.error("--material-lr-gamma must be in (0, 1]")
    if args.material_min_delta is not None and args.material_min_delta < 0:
        parser.error("--material-min-delta must be non-negative")
    if args.posmlp_hidden_depth < 3:
        parser.error("--posmlp-hidden-depth must be at least 3")
    if args.posmlp_hidden_width <= 32:
        parser.error("--posmlp-hidden-width must be greater than 32")
    if args.posmlp_backbone == "nerf_mlp" and args.posmlp_hidden_depth < 6:
        parser.error("nerf_mlp requires --posmlp-hidden-depth of at least 6")

    # optimize_farfield() uses this only to keep random seeds disjoint from
    # Stage B. There is intentionally no Stage B in this experiment.
    args.stage_b_iters = 0
    return args


def material_learning_rate(iteration: int, args) -> float:
    """Closed-form equivalent of Materialist's StepLR schedule."""
    decay_count = iteration // args.material_lr_step
    return args.material_lr * args.material_lr_gamma**decay_count


def mean_abs_dr(dr, value):
    return dr.mean(dr.abs(value.array))


def _masked_mean_dr(dr, value, loss_mask):
    if loss_mask is None:
        return dr.mean(value)
    denominator = dr.maximum(dr.sum(loss_mask.array), 1e-8)
    return dr.sum(value * loss_mask.array) / denominator


def linear_to_display_srgb_np(value: np.ndarray) -> np.ndarray:
    """Convert LDR-referred linear RGB to the clipped display domain."""
    return linear_to_srgb_np(np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0))


def linear_to_display_srgb_dr(dr, value):
    return linear_to_srgb_dr(dr, dr.clip(value, 0.0, 1.0))


def linear_to_optimization_srgb_np(value: np.ndarray) -> np.ndarray:
    """Extended sRGB for optimization; preserve gradients above linear 1."""
    return linear_to_srgb_np(np.maximum(np.asarray(value, dtype=np.float32), 0.0))


def linear_to_optimization_srgb_dr(dr, value):
    return linear_to_srgb_dr(dr, dr.maximum(value, 0.0))


def _masked_mean_np(value: np.ndarray, mask: np.ndarray | None) -> float:
    value = np.asarray(value, dtype=np.float64)
    if mask is None:
        return float(np.mean(value))
    weights = np.asarray(mask, dtype=np.float64)
    if weights.ndim == value.ndim - 1:
        weights = weights[..., None]
    weights = np.broadcast_to(weights, value.shape)
    denominator = float(np.sum(weights))
    if denominator <= 0:
        raise ValueError("Loss mask contains no valid pixels")
    return float(np.sum(value * weights) / denominator)


def _load_binary_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None:
        raise FileNotFoundError(path)
    height, width = target_hw
    if value.shape != (height, width):
        value = cv2.resize(value, (width, height), interpolation=cv2.INTER_NEAREST)
    return value > 127


def _depth_discontinuity_mask(
    depth: np.ndarray,
    valid: np.ndarray,
    relative_threshold: float,
) -> np.ndarray:
    """Mark both sides of large horizontal/vertical depth jumps."""
    depth = np.asarray(depth, dtype=np.float32)
    edge = np.zeros(depth.shape, dtype=bool)
    horizontal_boundary = valid[:, 1:] != valid[:, :-1]
    edge[:, 1:] |= horizontal_boundary
    edge[:, :-1] |= horizontal_boundary
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    horizontal_scale = np.maximum(
        np.minimum(np.abs(depth[:, 1:]), np.abs(depth[:, :-1])), 1e-6
    )
    horizontal = horizontal_valid & (
        np.abs(depth[:, 1:] - depth[:, :-1]) / horizontal_scale
        > relative_threshold
    )
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal

    vertical_boundary = valid[1:, :] != valid[:-1, :]
    edge[1:, :] |= vertical_boundary
    edge[:-1, :] |= vertical_boundary
    vertical_valid = valid[1:, :] & valid[:-1, :]
    vertical_scale = np.maximum(
        np.minimum(np.abs(depth[1:, :]), np.abs(depth[:-1, :])), 1e-6
    )
    vertical = vertical_valid & (
        np.abs(depth[1:, :] - depth[:-1, :]) / vertical_scale
        > relative_threshold
    )
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return edge


def build_loss_masks(
    mi,
    *,
    materialist_dir: Path,
    light_set,
    target_hw: tuple[int, int],
    emitter_dilate: int,
    edge_dilate: int,
    depth_edge_rtol: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build separate far-field and material masks from existing assets."""
    height, width = target_hw
    emitter_mask = np.zeros((height, width), dtype=bool)
    emitter_paths: list[str] = []
    for light in light_set.lights:
        if not light.visible or light.mask_path is None or not light.mask_path.is_file():
            continue
        emitter_mask |= _load_binary_mask(light.mask_path, target_hw)
        emitter_paths.append(str(light.mask_path))
    if emitter_dilate > 0 and emitter_mask.any():
        kernel_size = 2 * emitter_dilate + 1
        emitter_mask = cv2.dilate(
            emitter_mask.astype(np.uint8),
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
        ).astype(bool)

    valid_mask = np.ones((height, width), dtype=bool)
    valid_path = None
    for candidate in ("mesh_valid_mask.png", "moge2_valid_mask.png"):
        path = materialist_dir / candidate
        if path.is_file():
            valid_mask = _load_binary_mask(path, target_hw)
            valid_path = str(path)
            break

    depth_edge = np.zeros((height, width), dtype=bool)
    depth_path = None
    for candidate in ("moge2_depth.exr", "mesh_depth.exr", "depthPred.exr"):
        path = materialist_dir / candidate
        if not path.is_file():
            continue
        depth = np.asarray(mi.Bitmap(str(path)), dtype=np.float32)
        depth = np.squeeze(depth)
        if depth.shape != (height, width):
            depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        depth_valid = np.isfinite(depth) & (depth > 0.0) & valid_mask
        valid_mask &= depth_valid
        depth_edge = _depth_discontinuity_mask(
            depth,
            depth_valid,
            depth_edge_rtol,
        )
        depth_path = str(path)
        break

    if edge_dilate > 0:
        kernel_size = 2 * edge_dilate + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        if depth_edge.any():
            depth_edge = cv2.dilate(depth_edge.astype(np.uint8), kernel).astype(bool)
        valid_mask = cv2.erode(valid_mask.astype(np.uint8), kernel).astype(bool)

    farfield_mask = ~(emitter_mask | depth_edge)
    material_mask = valid_mask & farfield_mask
    if not farfield_mask.any() or not material_mask.any():
        raise ValueError("Generated loss mask contains no valid pixels")
    metadata = {
        "emitter_masks": emitter_paths,
        "valid_mask": valid_path,
        "depth": depth_path,
        "emitter_excluded_fraction": float(emitter_mask.mean()),
        "depth_edge_excluded_fraction": float(depth_edge.mean()),
        "farfield_valid_fraction": float(farfield_mask.mean()),
        "material_valid_fraction": float(material_mask.mean()),
    }
    return (
        np.repeat(farfield_mask[..., None], 3, axis=-1).astype(np.float32),
        np.repeat(material_mask[..., None], 3, axis=-1).astype(np.float32),
        metadata,
    )


def _optimization_mse_np(
    target_linear: np.ndarray,
    rendered_linear: np.ndarray,
    loss_mask: np.ndarray | None,
) -> float:
    difference = (
        linear_to_optimization_srgb_np(rendered_linear)
        - linear_to_optimization_srgb_np(target_linear)
    )
    return _masked_mean_np(np.square(difference), loss_mask)


def render_validation_mean(
    mi,
    dr,
    scene,
    params,
    *,
    spp: int,
    seeds: list[int],
) -> np.ndarray:
    renders = []
    with dr.suspend_grad():
        for seed in seeds:
            rendered = mi.render(scene, params, spp=spp, seed=seed)
            dr.eval(rendered)
            renders.append(np.asarray(rendered, dtype=np.float32))
    return np.mean(np.stack(renders, axis=0), axis=0, dtype=np.float32)


def prepare_optix_denoiser_guides(
    albedo: np.ndarray,
    normals_world: np.ndarray,
    camera_meta: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare linear albedo and sensor-space normals for OptiX denoising."""
    albedo = np.asarray(albedo, dtype=np.float32)[..., :3]
    normals_world = np.asarray(normals_world, dtype=np.float32)[..., :3]
    if albedo.shape != normals_world.shape or albedo.ndim != 3:
        raise ValueError(
            "OptiX guide shapes must match as HxWx3: "
            f"albedo={albedo.shape}, normals={normals_world.shape}"
        )

    to_world = np.asarray(camera_meta["to_world"], dtype=np.float32)
    while to_world.ndim > 2:
        to_world = to_world[0]
    if to_world.shape != (4, 4):
        raise ValueError(f"Expected camera to_world 4x4, got {to_world.shape}")
    world_to_sensor = np.linalg.inv(to_world)[:3, :3]

    albedo = np.nan_to_num(albedo, nan=0.0, posinf=1.0, neginf=0.0)
    albedo = np.clip(albedo, 0.0, 1.0)
    normals_sensor = np.einsum(
        "ij,...j->...i", world_to_sensor, normals_world
    ).astype(np.float32)
    normals_sensor = np.nan_to_num(normals_sensor, nan=0.0, posinf=0.0, neginf=0.0)
    normal_length = np.linalg.norm(normals_sensor, axis=-1, keepdims=True)
    invalid = normal_length[..., 0] <= 1e-8
    normals_sensor /= np.maximum(normal_length, 1e-8)
    normals_sensor[invalid] = (0.0, 0.0, 1.0)
    return (
        np.ascontiguousarray(albedo, dtype=np.float32),
        np.ascontiguousarray(normals_sensor, dtype=np.float32),
    )


def _guided_optix_denoise(
    mi,
    rendered_raw: np.ndarray,
    albedo: np.ndarray,
    normals_sensor: np.ndarray,
) -> np.ndarray:
    rendered_raw = np.ascontiguousarray(rendered_raw, dtype=np.float32)
    albedo = np.ascontiguousarray(albedo, dtype=np.float32)
    normals_sensor = np.ascontiguousarray(normals_sensor, dtype=np.float32)
    if albedo.shape != rendered_raw.shape or normals_sensor.shape != rendered_raw.shape:
        raise ValueError(
            "OptiX radiance/albedo/normal inputs must have the same shape: "
            f"radiance={rendered_raw.shape}, albedo={albedo.shape}, "
            f"normals={normals_sensor.shape}"
        )
    denoiser = mi.OptixDenoiser(
        input_size=(int(rendered_raw.shape[1]), int(rendered_raw.shape[0])),
        albedo=True,
        normals=True,
        temporal=False,
    )
    return np.asarray(
        denoiser(
            mi.TensorXf(rendered_raw),
            mi.TensorXf(albedo),
            mi.TensorXf(normals_sensor),
        ),
        dtype=np.float32,
    )


def save_render_variants(
    mi,
    *,
    base_path: Path,
    rendered_raw: np.ndarray,
    denoise: bool,
    denoise_albedo: np.ndarray,
    denoise_normals: np.ndarray,
) -> tuple[np.ndarray, dict[str, str]]:
    raw_base = base_path.with_name(base_path.name + "_raw")
    save_linear_image(mi, raw_base, rendered_raw)
    displayed = rendered_raw
    outputs = {
        "raw_exr": str(raw_base.with_suffix(".exr")),
        "raw_png": str(raw_base.with_suffix(".png")),
    }
    if denoise:
        displayed = _guided_optix_denoise(
            mi,
            rendered_raw,
            denoise_albedo,
            denoise_normals,
        )
        outputs.update(
            {
                "denoised_exr": str(base_path.with_suffix(".exr")),
                "denoised_png": str(base_path.with_suffix(".png")),
            }
        )
    save_linear_image(mi, base_path, displayed)
    return displayed, outputs


def save_final_render_variants(
    mi,
    *,
    best_dir: Path,
    rendered_raw: np.ndarray,
    denoise: bool,
    denoise_albedo: np.ndarray,
    denoise_normals: np.ndarray,
) -> dict[str, str]:
    """Keep the authoritative final result raw; denoise only a named preview."""
    compatibility_base = best_dir / "rendered_img"
    raw_base = best_dir / "rendered_img_final_raw"
    save_linear_image(mi, compatibility_base, rendered_raw)
    save_linear_image(mi, raw_base, rendered_raw)
    outputs = {
        "rendered_img_exr": str(compatibility_base.with_suffix(".exr")),
        "rendered_img_png": str(compatibility_base.with_suffix(".png")),
        "final_raw_exr": str(raw_base.with_suffix(".exr")),
        "final_raw_png": str(raw_base.with_suffix(".png")),
    }
    if denoise:
        denoised = _guided_optix_denoise(
            mi,
            rendered_raw,
            denoise_albedo,
            denoise_normals,
        )
        denoised_base = best_dir / "rendered_img_final_denoised"
        save_linear_image(mi, denoised_base, denoised)
        outputs.update(
            {
                "final_denoised_exr": str(denoised_base.with_suffix(".exr")),
                "final_denoised_png": str(denoised_base.with_suffix(".png")),
            }
        )
    return outputs


def resolve_target_path(args, light_set, materialist_dir: Path) -> tuple[Path, str]:
    """Use a verified current GT, while avoiding legacy mislabeled EXRs.

    Older ``gt_image.exr`` files can contain sRGB PNG values mislabeled as
    linear EXR. Inference manifest v2 guarantees the corrected linear export
    at the same working resolution as the geometry and material maps.
    """
    if args.target is not None:
        path = args.target.expanduser().resolve()
        source = "explicit_cli"
    elif (materialist_dir / "inference_manifest.json").is_file():
        manifest = json.loads(
            (materialist_dir / "inference_manifest.json").read_text(encoding="utf-8")
        )
        gt_path = (materialist_dir / "gt_image.exr").resolve()
        if (
            manifest.get("schema_version", 0) >= 2
            and manifest.get("status") == "complete"
        ):
            path = gt_path
            source = "verified_materialist_gt"
        elif light_set.image_path is not None and light_set.image_path.is_file():
            path = light_set.image_path.resolve()
            source = "ile_original_image"
        else:
            path = gt_path
            source = "materialist_gt_fallback"
    elif light_set.image_path is not None and light_set.image_path.is_file():
        path = light_set.image_path.resolve()
        source = "ile_original_image"
    else:
        path = (materialist_dir / "gt_image.exr").resolve()
        source = "materialist_gt_fallback"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path, source


def image_metrics(
    target_linear: np.ndarray,
    rendered_linear: np.ndarray,
    loss_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Report display metrics separately from the HDR-aware optimization metric."""
    target_display = linear_to_display_srgb_np(target_linear)
    rendered_display = linear_to_display_srgb_np(rendered_linear)
    display_difference = rendered_display - target_display
    display_mse = _masked_mean_np(np.square(display_difference), loss_mask)
    optimization_difference = (
        linear_to_optimization_srgb_np(rendered_linear)
        - linear_to_optimization_srgb_np(target_linear)
    )
    optimization_mse = _masked_mean_np(
        np.square(optimization_difference), loss_mask
    )
    return {
        "optimization_mae": _masked_mean_np(
            np.abs(optimization_difference), loss_mask
        ),
        "optimization_mse": optimization_mse,
        "display_mae": _masked_mean_np(np.abs(display_difference), loss_mask),
        "display_mse": display_mse,
        "display_psnr": (
            float("inf")
            if display_mse == 0
            else float(-10.0 * np.log10(display_mse))
        ),
        "linear_mae": _masked_mean_np(
            np.abs(rendered_linear - target_linear), loss_mask
        ),
        "evaluated_fraction": (
            1.0 if loss_mask is None else float(np.asarray(loss_mask)[..., 0].mean())
        ),
        "target_linear_mean": float(np.mean(target_linear)),
        "render_linear_mean": float(np.mean(rendered_linear)),
        "target_display_mean": float(np.mean(target_display)),
        "render_display_mean": float(np.mean(rendered_display)),
        "render_linear_max": float(np.max(rendered_linear)),
    }


def select_validated_farfield(
    *,
    initial_storage: np.ndarray,
    initial_render: np.ndarray,
    initial_metrics: dict[str, float],
    candidate_storage: np.ndarray,
    candidate_render: np.ndarray,
    candidate_metrics: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, dict[str, float], str]:
    """Keep the candidate only when HDR-aware same-seed validation improves."""
    initial_mse = float(initial_metrics["optimization_mse"])
    candidate_mse = float(candidate_metrics["optimization_mse"])
    if not np.isfinite(initial_mse):
        raise ValueError("Initial Stage-C validation MSE must be finite")
    if np.isfinite(candidate_mse) and candidate_mse <= initial_mse:
        return candidate_storage, candidate_render, candidate_metrics, "optimized"
    return (
        initial_storage,
        initial_render,
        initial_metrics,
        "initial_validation_fallback",
    )


def save_comparison(
    path: Path,
    target_linear: np.ndarray,
    rendered_linear: np.ndarray,
) -> None:
    target = linear_to_display_srgb_np(target_linear)
    rendered = linear_to_display_srgb_np(rendered_linear)
    error = np.clip(np.abs(rendered - target) * 4.0, 0.0, 1.0)
    canvas = np.concatenate([target, rendered, error], axis=1)
    canvas_u8 = (canvas * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(canvas_u8, cv2.COLOR_RGB2BGR))


def save_envmap_preview(path: Path, envmap_linear: np.ndarray) -> float:
    """Save an auto-exposed PNG preview without changing HDRI radiance."""
    percentile_99 = float(np.percentile(envmap_linear, 99.0))
    exposure_scale = 0.8 / max(percentile_99, 1e-6)
    preview = np.clip(linear_to_srgb_np(envmap_linear * exposure_scale), 0.0, 1.0)
    preview_u8 = (preview * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(preview_u8, cv2.COLOR_RGB2BGR))
    return exposure_scale


def optimize_materials(
    mi,
    dr,
    *,
    scene,
    params,
    initial_materials: dict[str, np.ndarray],
    target_linear,
    target_srgb,
    args,
    loss_mask=None,
    validation_fn: Callable[[], float] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Sequential direct material optimization following optimize_order.

    Each entry in ``args.material_order`` gets a fresh Adam/StepLR state and
    early-stopping state. Only that phase's tensors are traversed by the
    optimizer; prior phases and the Stage-C HDRI remain frozen.
    """
    order = list(args.material_order)
    unique_channels = list(dict.fromkeys("".join(order)))
    for channel in unique_channels:
        key = MATERIAL_KEYS[channel]
        if key not in params:
            available = "\n".join(str(item) for item in params.keys())
            raise KeyError(f"Missing material parameter {key}\nAvailable:\n{available}")

    history: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    final_materials: dict[str, np.ndarray] = {}
    target_mean = dr.detach(dr.mean(target_linear.array))

    for phase_index, phase in enumerate(order):
        channels = tuple(phase)
        keys = [MATERIAL_KEYS[channel] for channel in channels]
        frozen_channels = tuple(channel for channel in "arm" if channel not in channels)
        frozen_before = {
            MATERIAL_KEYS[channel]: np.asarray(
                params[MATERIAL_KEYS[channel]], dtype=np.float32
            ).copy()
            for channel in frozen_channels
        }
        initial = {
            MATERIAL_KEYS[channel]: mi.TensorXf(initial_materials[channel])
            for channel in channels
        }
        optimizer = mi.ad.Adam(
            lr=args.material_lr,
            amsgrad=True,
            mask_updates=True,
        )
        for key in keys:
            optimizer[key] = params[key]
        params.update(optimizer)

        best_mse = float("inf")
        best_phase_materials: dict[str, np.ndarray] | None = None
        early_stop_reference: float | None = None
        stale_iterations = 0
        phase_min_delta = (
            args.material_min_delta
            if args.material_min_delta is not None
            else (0.005 if "a" in channels else 0.001)
        )
        phase_start = len(history)

        if validation_fn is not None:
            best_mse = validation_fn()
            early_stop_reference = best_mse
            best_phase_materials = {
                key: np.asarray(params[key], dtype=np.float32).copy()
                for key in keys
            }

        print(
            f"[Material phase {phase_index + 1}/{len(order)}] "
            f"optimize={phase} frozen={''.join(c for c in 'arm' if c not in channels) or 'none'}"
        )

        for iteration in range(args.material_iters):
            learning_rate = material_learning_rate(iteration, args)
            optimizer.set_learning_rate(float(learning_rate))
            iteration_seed = iteration if args.resample_each_iteration else 0
            seed_offset = phase_index * args.material_iters + iteration_seed
            seed = args.seed + (args.farfield_iters + seed_offset) * 2
            rendered_linear = mi.render(
                scene,
                params,
                spp=args.spp,
                spp_grad=args.spp_grad,
                seed=seed,
                seed_grad=seed + 1,
            )

            exposure_ratio = mi.Float(1.0)
            if args.material_exposure_match:
                rendered_mean = dr.maximum(
                    dr.detach(dr.mean(rendered_linear.array)), 1e-6
                )
                exposure_ratio = target_mean / rendered_mean
            rendered_matched = rendered_linear * exposure_ratio
            rendered_srgb = linear_to_optimization_srgb_dr(dr, rendered_matched)
            difference = rendered_srgb.array - target_srgb.array
            loss_mse = _masked_mean_dr(dr, dr.square(difference), loss_mask)
            loss_l1 = _masked_mean_dr(dr, dr.abs(difference), loss_mask)

            # Same adaptive render loss as inverse_img_w_mi_ori.py.
            loss_balance = dr.detach(loss_l1) / dr.maximum(
                dr.detach(loss_mse), 1e-8
            )
            loss_render = 3.0 * loss_balance * loss_mse + loss_l1

            prior_terms = []
            prior_values: dict[str, Any] = {}
            for channel, key in zip(channels, keys):
                value = mean_abs_dr(dr, optimizer[key] - initial[key])
                prior_terms.append(value)
                prior_values[channel] = value
            loss_prior = sum(prior_terms, mi.Float(0.0))
            loss = loss_render + args.material_prior_weight * loss_prior
            dr.eval(loss, loss_render, loss_mse, loss_l1, loss_prior, exposure_ratio)

            loss_value = float(loss[0])
            mse_value = float(loss_mse[0])
            history.append(
                {
                    "global_iteration": len(history),
                    "phase_index": phase_index,
                    "phase": phase,
                    "phase_iteration": iteration,
                    "loss": loss_value,
                    "render_loss": float(loss_render[0]),
                    "mse": mse_value,
                    "l1": float(loss_l1[0]),
                    "prior": float(loss_prior[0]),
                    "exposure_ratio": float(exposure_ratio[0]),
                    "learning_rate": float(learning_rate),
                    "channel_priors": {
                        channel: float(prior_values[channel][0])
                        for channel in channels
                    },
                }
            )

            if validation_fn is None and mse_value < best_mse:
                best_mse = mse_value
                best_phase_materials = {
                    key: np.asarray(optimizer[key], dtype=np.float32).copy()
                    for key in keys
                }

            if validation_fn is None:
                if (
                    early_stop_reference is None
                    or mse_value < early_stop_reference * (1.0 - phase_min_delta)
                ):
                    early_stop_reference = mse_value
                    stale_iterations = 0
                else:
                    stale_iterations += 1

            dr.backward(loss)
            optimizer.step()
            for channel, key in zip(channels, keys):
                lower, upper = MATERIAL_BOUNDS[channel]
                optimizer[key] = dr.clip(optimizer[key], lower, upper)
            params.update(optimizer)

            validation_mse = None
            should_validate = (
                validation_fn is not None
                and (
                    (iteration + 1) % args.validation_interval == 0
                    or iteration == args.material_iters - 1
                )
            )
            if should_validate:
                validation_mse = validation_fn()
                history[-1]["validation_mse"] = validation_mse
                if validation_mse < best_mse:
                    best_mse = validation_mse
                    best_phase_materials = {
                        key: np.asarray(optimizer[key], dtype=np.float32).copy()
                        for key in keys
                    }
                if validation_mse < early_stop_reference * (
                    1.0 - args.validation_min_delta
                ):
                    early_stop_reference = validation_mse
                    stale_iterations = 0
                else:
                    stale_iterations += 1

            if iteration % args.log_interval == 0 or iteration == args.material_iters - 1:
                print(
                    f"[Material {phase} {iteration:04d}/{args.material_iters}] "
                    f"loss={loss_value:.6f} render={float(loss_render[0]):.6f} "
                    f"mse={mse_value:.6f} l1={float(loss_l1[0]):.6f} "
                    f"prior={float(loss_prior[0]):.6f} "
                    f"exposure={float(exposure_ratio[0]):.4f} "
                    f"val={validation_mse if validation_mse is not None else '-'}"
                )
            stopping_patience = (
                args.validation_patience
                if validation_fn is not None
                else args.material_patience
            )
            if stale_iterations >= stopping_patience:
                reason = (
                    "validation did not improve"
                    if validation_fn is not None
                    else f"no relative MSE improvement >= {phase_min_delta:.4g}"
                )
                print(
                    f"[Material {phase}] early stopping at iteration {iteration}: "
                    f"{reason}"
                )
                break

        if best_phase_materials is None:
            raise RuntimeError(f"Material phase {phase!r} produced no checkpoint")
        # Freeze this phase at its best checkpoint before starting the next one.
        for key, value in best_phase_materials.items():
            params[key] = mi.TensorXf(value)
            final_materials[key] = value
        params.update()
        frozen_max_abs_diff = max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(params[key], dtype=np.float32)
                            - value
                        )
                    )
                )
                for key, value in frozen_before.items()
            ),
            default=0.0,
        )
        if frozen_max_abs_diff > 1e-7:
            raise RuntimeError(
                f"Frozen material channels changed during phase {phase!r}: "
                f"max_abs_diff={frozen_max_abs_diff:.6g}"
            )
        phase_summaries.append(
            {
                "phase_index": phase_index,
                "phase": phase,
                "channels": list(channels),
                "iterations": len(history) - phase_start,
                "best_optimization_mse": best_mse,
                "min_delta": phase_min_delta,
                "frozen_channels": list(frozen_channels),
                "frozen_max_abs_diff": frozen_max_abs_diff,
            }
        )

    return final_materials, history, phase_summaries


def _torch_linear_to_srgb(torch, value):
    value = torch.clamp_min(value, 0.0)
    return torch.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * torch.pow(torch.clamp_min(value, 0.0031308), 1.0 / 2.4) - 0.055,
    )


def _torch_masked_mean(torch, value, loss_mask):
    if loss_mask is None:
        return torch.mean(value)
    return torch.sum(value * loss_mask) / torch.clamp_min(torch.sum(loss_mask), 1e-8)


def _torch_spatial_total_variation(torch, value):
    terms = []
    if value.shape[0] > 1:
        terms.append(torch.mean(torch.abs(value[1:] - value[:-1])))
    if value.shape[1] > 1:
        terms.append(torch.mean(torch.abs(value[:, 1:] - value[:, :-1])))
    return sum(terms, torch.zeros((), dtype=value.dtype, device=value.device))


def optimize_farfield_pos_mlp(
    mi,
    dr,
    *,
    scene,
    params,
    target_srgb,
    args,
    loss_mask=None,
    validation_fn: Callable[[], float] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """PosMLP parameterization of the existing Stage-C objective."""
    import torch
    from mymodels.mlps import NeRFMLP, PosMLP

    if not torch.cuda.is_available():
        raise RuntimeError("--model_name pos_mlp requires CUDA")
    env_key = "far_field_env.data"
    storage_shape = tuple(int(value) for value in params[env_key].shape)
    expected_shapes = {
        (args.farfield_height, args.farfield_width, 3),
        (args.farfield_height, args.farfield_width + 1, 3),
    }
    if storage_shape not in expected_shapes:
        raise ValueError(f"Unexpected far-field tensor shape: {storage_shape}")

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    start_envmap = torch.full(
        (args.farfield_height * args.farfield_width, 3),
        args.farfield_init,
        dtype=torch.float32,
        device=device,
    )
    network_class = NeRFMLP if args.posmlp_backbone == "nerf_mlp" else PosMLP
    skip_connection = [4] if args.posmlp_backbone == "nerf_mlp" else [1, 3]
    envmap_net = network_class(
        in_dims=6,
        out_dims=3,
        dims=[args.posmlp_hidden_width] * args.posmlp_hidden_depth,
        skip_connection=skip_connection,
        weight_norm=False,
        multires_view=args.posmlp_multires,
        output_type="envmap",
        color_ch=3,
        img_h=args.farfield_height,
        img_w=args.farfield_width,
        coordinate_type="spherical",
    ).to(device)
    # PosMLP's zero-initialized head otherwise starts at softplus(0)=0.693.
    # Match the direct branch's --farfield-init without changing the network.
    output_layer = envmap_net.output_layer
    initial_bias = float(np.log(np.expm1(max(args.farfield_init, 1e-6))))
    with torch.no_grad():
        output_layer.bias.fill_(initial_bias)
    optimizer = torch.optim.Adam(
        envmap_net.parameters(),
        lr=args.farfield_lr,
        weight_decay=0.0,
    )
    target = torch.as_tensor(
        np.asarray(target_srgb, dtype=np.float32), device=device
    )
    loss_mask_torch = (
        None
        if loss_mask is None
        else torch.as_tensor(
            np.asarray(loss_mask, dtype=np.float32), device=device
        )
    )

    @dr.wrap(source="torch", target="drjit")
    def render_envmap(envmap, seed):
        params[env_key] = envmap
        params.update()
        return mi.render(
            scene,
            params,
            spp=args.spp,
            spp_grad=args.spp_grad,
            seed=seed,
            seed_grad=seed + 1,
        )

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_env: np.ndarray | None = None
    validation_reference: float | None = None
    stale_validations = 0
    if validation_fn is not None:
        best_loss = validation_fn()
        validation_reference = best_loss
        best_env = np.asarray(params[env_key], dtype=np.float32).copy()
    for iteration in range(args.farfield_iters):
        learning_rate = cosine_learning_rate(
            iteration, args.farfield_iters, args.farfield_lr
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        logical_env = envmap_net(start_envmap).reshape(
            args.farfield_height, args.farfield_width, 3
        )
        logical_env = torch.clamp(logical_env, 0.0, args.farfield_max)
        storage_env = (
            torch.cat([logical_env, logical_env[:, :1]], dim=1)
            if storage_shape[1] == args.farfield_width + 1
            else logical_env
        )
        iteration_seed = iteration if args.resample_each_iteration else 0
        seed = args.seed + (
            args.stage_b_iters + iteration_seed
        ) * 2 * args.posmlp_train_seeds
        rendered_linear = torch.stack(
            [
                render_envmap(storage_env, seed + sample_index * 2)
                for sample_index in range(args.posmlp_train_seeds)
            ],
            dim=0,
        ).mean(dim=0)
        rendered_srgb = _torch_linear_to_srgb(
            torch, rendered_linear
        )
        difference = rendered_srgb - target
        loss_charbonnier = _torch_masked_mean(
            torch,
            torch.sqrt(difference.square() + args.charbonnier_epsilon**2)
            - args.charbonnier_epsilon,
            loss_mask_torch,
        )
        loss_mse = _torch_masked_mean(
            torch, difference.square(), loss_mask_torch
        )
        right = torch.roll(logical_env, shifts=-1, dims=1)
        down = torch.cat([logical_env[1:], logical_env[-1:]], dim=0)
        loss_tv = torch.mean(torch.abs(logical_env - right)) + torch.mean(
            torch.abs(logical_env - down)
        )
        loss_energy = torch.mean(logical_env.square())
        loss = (
            args.charbonnier_weight * loss_charbonnier
            + args.mse_weight * loss_mse
            + args.farfield_tv_weight * loss_tv
            + args.farfield_energy_weight * loss_energy
        )
        loss_value = float(loss.detach())
        mse_value = float(loss_mse.detach())
        history.append(
            {
                "iteration": iteration,
                "loss": loss_value,
                "charbonnier": float(loss_charbonnier.detach()),
                "mse": mse_value,
                "tv": float(loss_tv.detach()),
                "energy": float(loss_energy.detach()),
                "learning_rate": learning_rate,
            }
        )
        checkpoint_value = (
            mse_value
            if args.farfield_checkpoint_metric == "mse"
            else loss_value
        )
        if validation_fn is None and checkpoint_value < best_loss:
            best_loss = checkpoint_value
            best_env = storage_env.detach().cpu().numpy().copy()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        validation_mse = None
        should_validate = (
            validation_fn is not None
            and (
                (iteration + 1) % args.validation_interval == 0
                or iteration == args.farfield_iters - 1
            )
        )
        if should_validate:
            with torch.no_grad():
                validation_logical = envmap_net(start_envmap).reshape(
                    args.farfield_height, args.farfield_width, 3
                )
                validation_logical = torch.clamp(
                    validation_logical, 0.0, args.farfield_max
                )
                validation_storage = (
                    torch.cat(
                        [validation_logical, validation_logical[:, :1]], dim=1
                    )
                    if storage_shape[1] == args.farfield_width + 1
                    else validation_logical
                )
            validation_storage_np = (
                validation_storage.detach().cpu().numpy().copy()
            )
            params[env_key] = mi.TensorXf(validation_storage_np)
            params.update()
            validation_mse = validation_fn()
            history[-1]["validation_mse"] = validation_mse
            if validation_mse < best_loss:
                best_loss = validation_mse
                best_env = validation_storage_np
            if validation_mse < validation_reference * (
                1.0 - args.validation_min_delta
            ):
                validation_reference = validation_mse
                stale_validations = 0
            else:
                stale_validations += 1
        if iteration % args.log_interval == 0 or iteration == args.farfield_iters - 1:
            print(
                f"[Far-field PosMLP {iteration:04d}/{args.farfield_iters}] "
                f"loss={loss_value:.6f} charb={float(loss_charbonnier.detach()):.6f} "
                f"mse={mse_value:.6f} tv={float(loss_tv.detach()):.6f} "
                f"energy={float(loss_energy.detach()):.6f} "
                f"val={validation_mse if validation_mse is not None else '-'}"
            )
        if (
            validation_fn is not None
            and stale_validations >= args.validation_patience
        ):
            print(
                f"[Far-field PosMLP] early stopping at {iteration}: "
                "validation did not improve"
            )
            break

    if best_env is None:
        raise RuntimeError("PosMLP far-field optimization produced no checkpoint")
    params[env_key] = mi.TensorXf(best_env)
    params.update()
    return best_env, history, best_loss


def optimize_materials_pos_mlp(
    mi,
    dr,
    *,
    scene,
    params,
    initial_materials: dict[str, np.ndarray],
    target_linear,
    target_srgb,
    args,
    loss_mask=None,
    validation_fn: Callable[[], float] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """PosMLP parameterization of the existing sequential material phases."""
    import torch
    from mymodels.mlps import NeRFMLP, PosMLP

    if not torch.cuda.is_available():
        raise RuntimeError("--model_name pos_mlp requires CUDA")
    device = torch.device("cuda")
    target = torch.as_tensor(
        np.asarray(target_srgb, dtype=np.float32), device=device
    )
    loss_mask_torch = (
        None
        if loss_mask is None
        else torch.as_tensor(
            np.asarray(loss_mask, dtype=np.float32), device=device
        )
    )
    target_mean = float(np.asarray(target_linear, dtype=np.float32).mean())

    @dr.wrap(source="torch", target="drjit")
    def render_materials(albedo, roughness, metallic, seed):
        params[MATERIAL_KEYS["a"]] = albedo
        params[MATERIAL_KEYS["r"]] = roughness
        params[MATERIAL_KEYS["m"]] = metallic
        params.update()
        return mi.render(
            scene,
            params,
            spp=args.spp,
            spp_grad=args.spp_grad,
            seed=seed,
            seed_grad=seed + 1,
        )

    order = list(args.material_order)
    history: list[dict[str, Any]] = []
    phase_summaries: list[dict[str, Any]] = []
    final_materials: dict[str, np.ndarray] = {}
    for phase_index, phase in enumerate(order):
        channels = tuple(phase)
        current = {
            channel: np.asarray(params[MATERIAL_KEYS[channel]], dtype=np.float32).copy()
            for channel in "arm"
        }
        frozen_channels = tuple(channel for channel in "arm" if channel not in channels)
        frozen_before = {channel: current[channel].copy() for channel in frozen_channels}
        base_arm = np.concatenate(
            [
                current["a"],
                (current["r"] - 0.07) / 0.93,
                current["m"],
            ],
            axis=-1,
        )
        base_arm_torch = torch.as_tensor(
            base_arm.reshape(-1, 5), dtype=torch.float32, device=device
        )
        torch.manual_seed(args.seed + phase_index + 1)
        network_class = NeRFMLP if args.posmlp_backbone == "nerf_mlp" else PosMLP
        skip_connection = [4] if args.posmlp_backbone == "nerf_mlp" else [1, 3]
        brdf_net = network_class(
            in_dims=7,
            out_dims=5,
            dims=[args.posmlp_hidden_width] * args.posmlp_hidden_depth,
            skip_connection=skip_connection,
            weight_norm=False,
            multires_view=args.posmlp_multires,
            output_type="arm",
            color_ch=5,
            img_h=current["a"].shape[0],
            img_w=current["a"].shape[1],
            coordinate_type="uv",
            normalize_uv=True,
            use_ste_clamp=False,
        ).to(device)
        optimizer = torch.optim.AdamW(
            brdf_net.parameters(),
            lr=args.material_lr,
            weight_decay=0.0,
        )
        initial = {
            channel: torch.as_tensor(
                initial_materials[channel], dtype=torch.float32, device=device
            )
            for channel in channels
        }
        current_torch = {
            channel: torch.as_tensor(value, dtype=torch.float32, device=device)
            for channel, value in current.items()
        }
        best_mse = float("inf")
        best_phase_materials: dict[str, np.ndarray] | None = None
        early_stop_reference: float | None = None
        stale_iterations = 0
        phase_min_delta = (
            args.material_min_delta
            if args.material_min_delta is not None
            else (0.005 if "a" in channels else 0.001)
        )
        phase_start = len(history)
        if validation_fn is not None:
            best_mse = validation_fn()
            early_stop_reference = best_mse
            best_phase_materials = {
                MATERIAL_KEYS[channel]: current[channel].copy()
                for channel in channels
            }
        print(
            f"[Material PosMLP phase {phase_index + 1}/{len(order)}] "
            f"optimize={phase} frozen={''.join(frozen_channels) or 'none'}"
        )

        for iteration in range(args.material_iters):
            learning_rate = material_learning_rate(iteration, args)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            arm = brdf_net(base_arm_torch)
            predicted = {
                "a": arm[:, :3].reshape(current["a"].shape),
                "r": (arm[:, 3:4] * 0.93 + 0.07).reshape(current["r"].shape),
                "m": arm[:, 4:5].reshape(current["m"].shape),
            }
            materials = {
                channel: predicted[channel] if channel in channels else current_torch[channel]
                for channel in "arm"
            }
            iteration_seed = iteration if args.resample_each_iteration else 0
            seed_offset = phase_index * args.material_iters + iteration_seed
            seed = args.seed + (
                args.farfield_iters + seed_offset
            ) * 2 * args.posmlp_train_seeds
            rendered_linear = torch.stack(
                [
                    render_materials(
                        materials["a"],
                        materials["r"],
                        materials["m"],
                        seed + sample_index * 2,
                    )
                    for sample_index in range(args.posmlp_train_seeds)
                ],
                dim=0,
            ).mean(dim=0)
            exposure_ratio = 1.0
            if args.material_exposure_match:
                exposure_ratio = target_mean / max(
                    float(rendered_linear.detach().mean()), 1e-6
                )
            rendered_srgb = _torch_linear_to_srgb(
                torch, rendered_linear * exposure_ratio
            )
            difference = rendered_srgb - target
            loss_mse = _torch_masked_mean(
                torch, difference.square(), loss_mask_torch
            )
            loss_l1 = _torch_masked_mean(
                torch, torch.abs(difference), loss_mask_torch
            )
            # loss_balance = loss_l1.detach() / torch.clamp_min(
            #     loss_mse.detach(), 1e-8
            # )
            loss_render = loss_mse + loss_l1
            prior_values = {
                channel: torch.mean(torch.abs(materials[channel] - initial[channel]))
                for channel in channels
            }
            loss_prior = sum(prior_values.values(), torch.zeros((), device=device))
            tv_values = {
                channel: _torch_spatial_total_variation(
                    torch, materials[channel] - initial[channel]
                )
                for channel in channels
            }
            loss_tv = sum(tv_values.values(), torch.zeros((), device=device))
            loss = (
                loss_render
                + args.material_prior_weight * loss_prior
                + args.posmlp_material_tv_weight * loss_tv
            )
            loss_value = float(loss.detach())
            mse_value = float(loss_mse.detach())
            history.append(
                {
                    "global_iteration": len(history),
                    "phase_index": phase_index,
                    "phase": phase,
                    "phase_iteration": iteration,
                    "loss": loss_value,
                    "render_loss": float(loss_render.detach()),
                    "mse": mse_value,
                    "l1": float(loss_l1.detach()),
                    "prior": float(loss_prior.detach()),
                    "material_delta_tv": float(loss_tv.detach()),
                    "exposure_ratio": float(exposure_ratio),
                    "learning_rate": float(learning_rate),
                    "channel_priors": {
                        channel: float(value.detach())
                        for channel, value in prior_values.items()
                    },
                    "channel_delta_tv": {
                        channel: float(value.detach())
                        for channel, value in tv_values.items()
                    },
                }
            )
            if validation_fn is None and mse_value < best_mse:
                best_mse = mse_value
                best_phase_materials = {
                    MATERIAL_KEYS[channel]: materials[channel].detach().cpu().numpy().copy()
                    for channel in channels
                }
            if validation_fn is None:
                if (
                    early_stop_reference is None
                    or mse_value < early_stop_reference * (1.0 - phase_min_delta)
                ):
                    early_stop_reference = mse_value
                    stale_iterations = 0
                else:
                    stale_iterations += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            validation_mse = None
            should_validate = (
                validation_fn is not None
                and (
                    (iteration + 1) % args.validation_interval == 0
                    or iteration == args.material_iters - 1
                )
            )
            if should_validate:
                with torch.no_grad():
                    validation_arm = brdf_net(base_arm_torch)
                    validation_predicted = {
                        "a": validation_arm[:, :3].reshape(current["a"].shape),
                        "r": (
                            validation_arm[:, 3:4] * 0.93 + 0.07
                        ).reshape(current["r"].shape),
                        "m": validation_arm[:, 4:5].reshape(current["m"].shape),
                    }
                    validation_materials = {
                        channel: (
                            validation_predicted[channel]
                            if channel in channels
                            else current_torch[channel]
                        )
                        for channel in "arm"
                    }
                validation_numpy = {
                    channel: value.detach().cpu().numpy().copy()
                    for channel, value in validation_materials.items()
                }
                for channel in "arm":
                    params[MATERIAL_KEYS[channel]] = mi.TensorXf(
                        validation_numpy[channel]
                    )
                params.update()
                validation_mse = validation_fn()
                history[-1]["validation_mse"] = validation_mse
                if validation_mse < best_mse:
                    best_mse = validation_mse
                    best_phase_materials = {
                        MATERIAL_KEYS[channel]: validation_numpy[channel]
                        for channel in channels
                    }
                if validation_mse < early_stop_reference * (
                    1.0 - args.validation_min_delta
                ):
                    early_stop_reference = validation_mse
                    stale_iterations = 0
                else:
                    stale_iterations += 1
            if iteration % args.log_interval == 0 or iteration == args.material_iters - 1:
                print(
                    f"[Material PosMLP {phase} {iteration:04d}/{args.material_iters}] "
                    f"loss={loss_value:.6f} render={float(loss_render.detach()):.6f} "
                    f"mse={mse_value:.6f} l1={float(loss_l1.detach()):.6f} "
                    f"prior={float(loss_prior.detach()):.6f} "
                    f"tv={float(loss_tv.detach()):.6f} "
                    f"exposure={float(exposure_ratio):.4f} "
                    f"val={validation_mse if validation_mse is not None else '-'}"
                )
            stopping_patience = (
                args.validation_patience
                if validation_fn is not None
                else args.material_patience
            )
            if stale_iterations >= stopping_patience:
                print(f"[Material PosMLP {phase}] early stopping at {iteration}")
                break

        if best_phase_materials is None:
            raise RuntimeError(f"PosMLP material phase {phase!r} produced no checkpoint")
        for key, value in best_phase_materials.items():
            params[key] = mi.TensorXf(value)
            final_materials[key] = value
        params.update()
        frozen_max_abs_diff = max(
            (
                float(
                    np.max(
                        np.abs(
                            np.asarray(params[MATERIAL_KEYS[channel]], dtype=np.float32)
                            - frozen_before[channel]
                        )
                    )
                )
                for channel in frozen_channels
            ),
            default=0.0,
        )
        if frozen_max_abs_diff > 1e-7:
            raise RuntimeError(
                f"Frozen material channels changed during phase {phase!r}: "
                f"max_abs_diff={frozen_max_abs_diff:.6g}"
            )
        phase_summaries.append(
            {
                "phase_index": phase_index,
                "phase": phase,
                "channels": list(channels),
                "iterations": len(history) - phase_start,
                "best_optimization_mse": best_mse,
                "min_delta": phase_min_delta,
                "frozen_channels": list(frozen_channels),
                "frozen_max_abs_diff": frozen_max_abs_diff,
            }
        )
    return final_materials, history, phase_summaries


def save_material_outputs(
    mi,
    output_dir: Path,
    optimized: dict[str, np.ndarray],
    initial_materials: dict[str, np.ndarray],
) -> dict[str, str]:
    best_dir = output_dir / "best_results"
    best_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for channel in "arm":
        key = MATERIAL_KEYS[channel]
        value = optimized.get(key, initial_materials[channel])
        name = MATERIAL_NAMES[channel]
        path = best_dir / f"{name}.exr"
        mi.util.write_bitmap(str(path), np.asarray(value, dtype=np.float32))
        outputs[name] = str(path)

        preview = np.clip(value, 0.0, 1.0)
        if preview.shape[-1] == 1:
            preview = np.repeat(preview, 3, axis=-1)
        if channel == "a":
            preview = np.clip(linear_to_srgb_np(preview), 0.0, 1.0)
        preview_u8 = (preview * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(
            str(best_dir / f"{name}.png"),
            cv2.cvtColor(preview_u8, cv2.COLOR_RGB2BGR),
        )

    normal_path = best_dir / "normal.exr"
    mi.util.write_bitmap(str(normal_path), initial_materials["n"])
    outputs["normal"] = str(normal_path)
    return outputs


def main() -> None:
    args = parse_args()
    materialist_dir = args.materialist_dir.expanduser().resolve()
    if not materialist_dir.is_dir():
        raise NotADirectoryError(materialist_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else materialist_dir / "hybrid_ile_win_farfield_material_opt_nerf_mlp_d8m3"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_path = (
        args.camera_meta.expanduser().resolve()
        if args.camera_meta is not None
        else materialist_dir / "camera_meta.json"
    )
    if not camera_path.is_file():
        raise FileNotFoundError(camera_path)
    camera_meta = _load_camera(camera_path)
    width, height = [int(value) for value in camera_meta["film.size"]]
    mesh_path = _resolve_mesh(materialist_dir, args.mesh)
    material_paths = _find_material_maps(materialist_dir, args.material_dir)
    light_set = load_ile_lights(
        args.lights_json,
        include_windows=True,
        geometry_scale=args.geometry_scale,
    )

    if light_set.image_path is not None and light_set.image_path.is_file():
        render_projection_debug(
            light_set.image_path,
            light_set.lights,
            np.asarray(camera_meta["K"], dtype=np.float32),
            (width, height),
            output_dir / "lights_projected.png",
            visible_offset=args.visible_offset,
        )

    import drjit as dr
    import mitsuba as mi
    from myutils.mi_plugin import MatDiffBSDF

    mi.register_bsdf("MatDiffBSDF", lambda props: MatDiffBSDF(props))
    material_arrays = _load_material_arrays(mi, material_paths, (height, width))
    target_path, target_source = resolve_target_path(args, light_set, materialist_dir)
    target_linear_np = load_target_linear(mi, target_path, (height, width))
    target_linear = mi.TensorXf(target_linear_np)
    target_srgb_np = linear_to_optimization_srgb_np(target_linear_np)
    target_srgb = mi.TensorXf(target_srgb_np)
    save_linear_image(mi, output_dir / "target", target_linear_np)
    (
        diagnostic_farfield_mask_np,
        diagnostic_material_mask_np,
        diagnostic_mask_metadata,
    ) = (
        build_loss_masks(
            mi,
            materialist_dir=materialist_dir,
            light_set=light_set,
            target_hw=(height, width),
            emitter_dilate=args.emitter_mask_dilate,
            edge_dilate=args.mesh_edge_dilate,
            depth_edge_rtol=args.depth_edge_rtol,
        )
    )
    # Keep the former exclusion masks only as diagnostics. No spatial loss mask
    # is passed to either optimizer, validation gate, or final metric: every
    # target pixel now participates in both Stage C and Stage D.
    farfield_loss_mask_np = None
    material_loss_mask_np = None
    farfield_loss_mask = None
    material_loss_mask = None
    full_image_mask = np.ones((height, width), dtype=np.uint8) * 255
    for name, value in (
        ("farfield_exclusion_diagnostic_mask.png", diagnostic_farfield_mask_np[..., 0]),
        ("material_exclusion_diagnostic_mask.png", diagnostic_material_mask_np[..., 0]),
    ):
        cv2.imwrite(str(output_dir / name), (value * 255.0 + 0.5).astype(np.uint8))
    cv2.imwrite(str(output_dir / "farfield_loss_mask.png"), full_image_mask)
    cv2.imwrite(str(output_dir / "material_loss_mask.png"), full_image_mask)
    loss_mask_metadata = {
        "mode": "full_image",
        "spatial_loss_mask_enabled": False,
        "farfield_valid_fraction": 1.0,
        "material_valid_fraction": 1.0,
        "farfield_mask": str(output_dir / "farfield_loss_mask.png"),
        "material_mask": str(output_dir / "material_loss_mask.png"),
        "diagnostic_previous_exclusion": {
            **diagnostic_mask_metadata,
            "farfield_mask": str(
                output_dir / "farfield_exclusion_diagnostic_mask.png"
            ),
            "material_mask": str(
                output_dir / "material_exclusion_diagnostic_mask.png"
            ),
        },
    }
    initial_denoise_albedo, denoise_normals_sensor = prepare_optix_denoiser_guides(
        material_arrays["a"], material_arrays["n"], camera_meta
    )
    # Stage C: fixed ILE lamps and fixed input materials; optimize only HDRI.
    farfield_initial = np.full(
        (args.farfield_height, args.farfield_width, 3),
        args.farfield_init,
        dtype=np.float32,
    )
    farfield_initial_path = output_dir / "farfield_initial_32x16.exr"
    mi.util.write_bitmap(str(farfield_initial_path), farfield_initial)
    scene_dict = build_hybrid_scene_dict(
        mi,
        mesh_path=mesh_path,
        camera_meta_path=camera_path,
        camera_meta=camera_meta,
        lights=light_set.lights,
        mode="combined",
        envmap_path=farfield_initial_path,
        radiance_scale=args.radiance_scale,
        visible_offset=args.visible_offset,
        use_mesh_normal=args.use_mesh_normal,
        max_depth=args.max_depth,
    )
    scene_dict["integrator"]["type"] = args.integrator
    scene = mi.load_dict(scene_dict)
    params = mi.traverse(scene)
    set_material_parameters(mi, params, material_arrays)
    env_key = "far_field_env.data"
    if env_key not in params:
        raise KeyError(f"Combined scene is missing {env_key}")
    farfield_initial_storage = np.asarray(params[env_key], dtype=np.float32).copy()
    farfield_validation_seed = args.seed + 100000
    farfield_validation_seeds = [
        farfield_validation_seed + index * 2
        for index in range(args.validation_seeds)
    ]
    farfield_initial_render = render_validation_mean(
        mi,
        dr,
        scene,
        params,
        spp=args.validation_spp,
        seeds=farfield_validation_seeds,
    )
    save_render_variants(
        mi,
        base_path=output_dir / "farfield_initial_combined",
        rendered_raw=farfield_initial_render,
        denoise=not args.no_denoise,
        denoise_albedo=initial_denoise_albedo,
        denoise_normals=denoise_normals_sensor,
    )
    for output_name, initial_lights in (
        (
            "farfield_initial_lambs_combined",
            [light for light in light_set.lights if not light.is_window],
        ),
        (
            "farfield_initial_windows_combined",
            [light for light in light_set.lights if light.is_window],
        ),
    ):
        initial_scene_dict = build_hybrid_scene_dict(
            mi,
            mesh_path=mesh_path,
            camera_meta_path=camera_path,
            camera_meta=camera_meta,
            lights=initial_lights,
            mode="combined",
            envmap_path=farfield_initial_path,
            radiance_scale=args.radiance_scale,
            visible_offset=args.visible_offset,
            use_mesh_normal=args.use_mesh_normal,
            max_depth=args.max_depth,
        )
        initial_scene_dict["integrator"]["type"] = args.integrator
        initial_scene = mi.load_dict(initial_scene_dict)
        initial_params = mi.traverse(initial_scene)
        set_material_parameters(mi, initial_params, material_arrays)
        initial_render = render_validation_mean(
            mi,
            dr,
            initial_scene,
            initial_params,
            spp=args.validation_spp,
            seeds=farfield_validation_seeds,
        )
        save_render_variants(
            mi,
            base_path=output_dir / output_name,
            rendered_raw=initial_render,
            denoise=not args.no_denoise,
            denoise_albedo=initial_denoise_albedo,
            denoise_normals=denoise_normals_sensor,
        )
        del initial_params, initial_scene

    def validate_farfield_checkpoint() -> float:
        rendered = render_validation_mean(
            mi,
            dr,
            scene,
            params,
            spp=args.validation_spp,
            seeds=farfield_validation_seeds,
        )
        return _optimization_mse_np(
            target_linear_np,
            rendered,
            farfield_loss_mask_np,
        )

    farfield_optimizer = (
        optimize_farfield_pos_mlp
        if args.model_name == "pos_mlp"
        else optimize_farfield
    )
    best_farfield, farfield_history, farfield_best_loss = farfield_optimizer(
        mi,
        dr,
        scene=scene,
        params=params,
        target_srgb=target_srgb,
        args=args,
        loss_mask=farfield_loss_mask,
        validation_fn=validate_farfield_checkpoint,
    )
    farfield_render = render_validation_mean(
        mi,
        dr,
        scene,
        params,
        spp=args.validation_spp,
        seeds=farfield_validation_seeds,
    )
    save_render_variants(
        mi,
        base_path=output_dir / "farfield_candidate_combined",
        rendered_raw=farfield_render,
        denoise=not args.no_denoise,
        denoise_albedo=initial_denoise_albedo,
        denoise_normals=denoise_normals_sensor,
    )
    farfield_initial_metrics = image_metrics(
        target_linear_np,
        farfield_initial_render,
        farfield_loss_mask_np
    )
    farfield_candidate_metrics = image_metrics(
        target_linear_np,
        farfield_render,
        farfield_loss_mask_np
    )
    (
        selected_farfield,
        selected_farfield_render,
        farfield_metrics,
        farfield_selection,
    ) = select_validated_farfield(
        initial_storage=farfield_initial_storage,
        initial_render=farfield_initial_render,
        initial_metrics=farfield_initial_metrics,
        candidate_storage=best_farfield,
        candidate_render=farfield_render,
        candidate_metrics=farfield_candidate_metrics,
    )
    selected_validation_base = output_dir / (
        "farfield_candidate_combined"
        if farfield_selection == "optimized"
        else "farfield_initial_combined"
    )
    # Make the validation choice authoritative for the export and Stage D.
    params[env_key] = mi.TensorXf(selected_farfield)
    params.update()
    farfield_export = selected_farfield[:, : args.farfield_width, :]
    farfield_exr = output_dir / "farfield_optimized_32x16.exr"
    farfield_hdr = output_dir / "farfield_optimized_32x16.hdr"
    farfield_preview = output_dir / "farfield_optimized_32x16_preview.png"
    mi.util.write_bitmap(str(farfield_exr), farfield_export)
    mi.util.write_bitmap(str(farfield_hdr), farfield_export)
    farfield_preview_exposure = save_envmap_preview(
        farfield_preview, farfield_export
    )
    save_linear_image(
        mi,
        output_dir / "farfield_optimized_combined",
        selected_farfield_render,
    )
    for suffix in (".exr", ".png"):
        selected_raw_path = selected_validation_base.with_name(
            selected_validation_base.name + "_raw"
        ).with_suffix(suffix)
        optimized_raw_path = (
            output_dir / "farfield_optimized_combined_raw"
        ).with_suffix(suffix)
        shutil.copyfile(selected_raw_path, optimized_raw_path)
    write_json(output_dir / "farfield_history.json", farfield_history)
    write_json(
        output_dir / "farfield_metrics.json",
        {
            "validation_spp": args.validation_spp,
            "validation_seed": farfield_validation_seed,
            "validation_seeds": farfield_validation_seeds,
            "initial": farfield_initial_metrics,
            "candidate": farfield_candidate_metrics,
            "selected": farfield_metrics,
            "selection": farfield_selection,
            # Retained for readers of the previous metrics schema.
            "optimized": farfield_metrics,
            "candidate_optimization_mse_delta": (
                farfield_candidate_metrics["optimization_mse"]
                - farfield_initial_metrics["optimization_mse"]
            ),
            "optimization_mse_delta": (
                farfield_metrics["optimization_mse"]
                - farfield_initial_metrics["optimization_mse"]
            ),
            "candidate_display_mse_delta": (
                farfield_candidate_metrics["display_mse"]
                - farfield_initial_metrics["display_mse"]
            ),
            "display_mse_delta": (
                farfield_metrics["display_mse"]
                - farfield_initial_metrics["display_mse"]
            ),
        },
    )
    save_comparison(
        output_dir / "farfield_target_render_error.png",
        target_linear_np,
        selected_farfield_render,
    )

    # Rebuild Stage D from the exported best EXR. This makes the material stage
    # consume exactly the same file users can open in Blender, rather than an
    # implicit or potentially replaced in-memory envmap tensor.
    material_scene_dict = build_hybrid_scene_dict(
        mi,
        mesh_path=mesh_path,
        camera_meta_path=camera_path,
        camera_meta=camera_meta,
        lights=light_set.lights,
        mode="combined",
        envmap_path=farfield_exr,
        radiance_scale=args.radiance_scale,
        visible_offset=args.visible_offset,
        use_mesh_normal=args.use_mesh_normal,
        max_depth=args.max_depth,
    )
    material_scene_dict["integrator"]["type"] = args.integrator
    scene = mi.load_dict(material_scene_dict)
    params = mi.traverse(scene)
    set_material_parameters(mi, params, material_arrays)
    if env_key not in params:
        raise KeyError(f"Material scene is missing {env_key}")
    farfield_before_material = np.asarray(params[env_key], dtype=np.float32).copy()
    reloaded_logical = farfield_before_material[:, : args.farfield_width, :]
    farfield_reload_max_abs_diff = float(
        np.max(np.abs(reloaded_logical - farfield_export))
    )
    if farfield_reload_max_abs_diff > 1e-6:
        raise RuntimeError(
            "Reloaded material-stage HDRI differs from the optimized export: "
            f"max_abs_diff={farfield_reload_max_abs_diff:.6g}"
        )

    # Stage D: optimizer contains ARM tensors only. Lights and HDRI stay fixed.
    material_validation_seed = args.seed + 300000
    material_validation_seeds = [
        material_validation_seed + index * 2
        for index in range(args.validation_seeds)
    ]
    material_initial_render = render_validation_mean(
        mi,
        dr,
        scene,
        params,
        spp=args.validation_spp,
        seeds=material_validation_seeds,
    )
    save_render_variants(
        mi,
        base_path=output_dir / "material_initial_combined",
        rendered_raw=material_initial_render,
        denoise=not args.no_denoise,
        denoise_albedo=initial_denoise_albedo,
        denoise_normals=denoise_normals_sensor,
    )

    def validate_material_checkpoint() -> float:
        rendered = render_validation_mean(
            mi,
            dr,
            scene,
            params,
            spp=args.validation_spp,
            seeds=material_validation_seeds,
        )
        return _optimization_mse_np(
            target_linear_np,
            rendered,
            material_loss_mask_np,
        )

    material_optimizer = (
        optimize_materials_pos_mlp
        if args.model_name == "pos_mlp"
        else optimize_materials
    )
    best_materials, material_history, material_phase_summaries = material_optimizer(
        mi,
        dr,
        scene=scene,
        params=params,
        initial_materials=material_arrays,
        target_linear=target_linear,
        target_srgb=target_srgb,
        args=args,
        loss_mask=material_loss_mask,
        validation_fn=validate_material_checkpoint,
    )
    material_candidate_render = render_validation_mean(
        mi,
        dr,
        scene,
        params,
        spp=args.validation_spp,
        seeds=material_validation_seeds,
    )
    save_render_variants(
        mi,
        base_path=output_dir / "material_optimized_combined",
        rendered_raw=material_candidate_render,
        denoise=not args.no_denoise,
        denoise_albedo=np.asarray(params[MATERIAL_KEYS["a"]], dtype=np.float32),
        denoise_normals=denoise_normals_sensor,
    )
    farfield_after_material = np.asarray(params[env_key], dtype=np.float32)
    farfield_frozen_max_abs_diff = float(
        np.max(np.abs(farfield_after_material - farfield_before_material))
    )
    if farfield_frozen_max_abs_diff > 1e-7:
        raise RuntimeError(
            "Far-field HDRI changed during material optimization: "
            f"max_abs_diff={farfield_frozen_max_abs_diff:.6g}"
        )
    write_json(output_dir / "material_history.json", material_history)
    write_json(output_dir / "material_phase_summaries.json", material_phase_summaries)
    material_initial_metrics = image_metrics(
        target_linear_np,
        material_initial_render,
        material_loss_mask_np
    )
    material_candidate_metrics = image_metrics(
        target_linear_np,
        material_candidate_render,
        material_loss_mask_np,
    )
    material_selection = "optimized"
    if (
        material_candidate_metrics["optimization_mse"]
        > material_initial_metrics["optimization_mse"]
    ):
        material_selection = "initial_validation_fallback"
        optimized_channels = list(dict.fromkeys("".join(args.material_order)))
        best_materials = {
            MATERIAL_KEYS[channel]: material_arrays[channel]
            for channel in optimized_channels
        }
        for key, value in best_materials.items():
            params[key] = mi.TensorXf(value)
        params.update()
        selected_validation_render = material_initial_render
    else:
        selected_validation_render = material_candidate_render
    selected_validation_metrics = image_metrics(
        target_linear_np,
        selected_validation_render,
        material_loss_mask_np,
    )
    save_linear_image(
        mi, output_dir / "material_selected_combined", selected_validation_render
    )
    material_outputs = save_material_outputs(
        mi, output_dir, best_materials, material_arrays
    )
    final_seed = args.seed + 500000
    final_render = render_validation_mean(
        mi,
        dr,
        scene,
        params,
        spp=args.final_spp,
        seeds=[final_seed],
    )
    final_render_outputs = save_final_render_variants(
        mi,
        best_dir=output_dir / "best_results",
        rendered_raw=final_render,
        denoise=not args.no_denoise,
        denoise_albedo=np.asarray(params[MATERIAL_KEYS["a"]], dtype=np.float32),
        denoise_normals=denoise_normals_sensor,
    )
    final_metrics = image_metrics(
        target_linear_np,
        final_render,
        material_loss_mask_np,
    )
    final_metrics_report = {
        "validation_spp": args.validation_spp,
        "validation_seeds": material_validation_seeds,
        "final_spp": args.final_spp,
        "final_seed": final_seed,
        "initial": material_initial_metrics,
        "candidate": material_candidate_metrics,
        "selected_validation": selected_validation_metrics,
        "final_raw": final_metrics,
        "selection": material_selection,
        "optimization_mse_delta": (
            selected_validation_metrics["optimization_mse"]
            - material_initial_metrics["optimization_mse"]
        ),
        "display_mse_delta": (
            selected_validation_metrics["display_mse"]
            - material_initial_metrics["display_mse"]
        ),
    }
    write_json(output_dir / "final_metrics.json", final_metrics_report)
    save_comparison(
        output_dir / "final_target_render_error.png",
        target_linear_np,
        final_render,
    )

    manifest = {
        "schema_version": 7,
        "status": "complete",
        "stage_b_enabled": False,
        "model_name": args.model_name,
        "materialist_dir": str(materialist_dir),
        "mesh": str(mesh_path),
        "camera_meta": str(camera_path),
        "target": str(target_path),
        "target_source": target_source,
        "target_color_space": (
            "linear" if target_path.suffix.lower() in {".exr", ".hdr"} else "srgb"
        ),
        "lights_json": str(light_set.source_path),
        "fixed_local_lights": [light.name for light in light_set.lights],
        "fixed_window_count": light_set.metadata["windows_included"],
        "fixed_local_radiance_scale": args.radiance_scale,
        # Retained for compatibility with manifests from lamp-only runs.
        "fixed_lamp_radiance_scale": args.radiance_scale,
        "geometry_scale": light_set.metadata["geometry_scale"],
        "normal_source": "mesh" if args.use_mesh_normal else "moge2_normal_map",
        "use_mesh_normal": args.use_mesh_normal,
        "window_bsdf": "null",
        "input_materials": {key: str(value) for key, value in material_paths.items()},
        "optimization": {
            "integrator": args.integrator,
            "max_depth": args.max_depth,
            "spp": args.spp,
            "spp_grad": args.spp_grad,
            "resample_each_iteration": args.resample_each_iteration,
            "render_loss": "extended_standard_srgb",
            "checkpoint_metric": "optimization_mse",
            "display_metrics_clipped": True,
            "validation_spp_per_seed": args.validation_spp,
            "validation_seed_count": args.validation_seeds,
            "validation_interval": args.validation_interval,
            "validation_min_delta": args.validation_min_delta,
            "validation_patience": args.validation_patience,
            "final_spp": args.final_spp,
            "final_seed": final_seed,
            "spatial_loss_scope": "full_image",
            "direct_optimizer": {
                "type": "adam",
                "amsgrad": True,
                "mask_updates": True,
            },
            "pos_mlp_train_seeds": args.posmlp_train_seeds,
            "pos_mlp_backbone": args.posmlp_backbone,
            "pos_mlp_hidden_depth": args.posmlp_hidden_depth,
            "pos_mlp_hidden_width": args.posmlp_hidden_width,
            "pos_mlp_multires": args.posmlp_multires,
            "pos_mlp_material_uv_normalized": True,
            "pos_mlp_material_uv_frequency_scale": "pi",
            "pos_mlp_material_ste_clamp": False,
            "pos_mlp_material_bound": "sigmoid_logit_residual",
            "pos_mlp_material_delta_tv_weight": args.posmlp_material_tv_weight,
            "pos_mlp_weight_decay": 0.0,
        },
        "loss_masks": loss_mask_metadata,
        "denoiser": {
            "type": "optix",
            "enabled": not args.no_denoise,
            "albedo_guide": True,
            "normal_guide": True,
            "normal_source": "moge2_normal_map",
            "normal_space": "sensor",
        },
        "farfield": {
            "width": args.farfield_width,
            "height": args.farfield_height,
            "iterations": args.farfield_iters,
            "best_loss": farfield_best_loss,
            "best_checkpoint_value": farfield_best_loss,
            "checkpoint_metric": "validation_optimization_mse",
            "selection": farfield_selection,
            "validation_spp": args.validation_spp,
            "validation_seed": farfield_validation_seed,
            "validation_seeds": farfield_validation_seeds,
            "initial_display_mse": farfield_initial_metrics["display_mse"],
            "candidate_display_mse": farfield_candidate_metrics["display_mse"],
            "selected_display_mse": farfield_metrics["display_mse"],
            "initial_optimization_mse": farfield_initial_metrics[
                "optimization_mse"
            ],
            "candidate_optimization_mse": farfield_candidate_metrics[
                "optimization_mse"
            ],
            "selected_optimization_mse": farfield_metrics[
                "optimization_mse"
            ],
            "optimized_exr": str(farfield_exr),
            "optimized_hdr": str(farfield_hdr),
            "preview_png": str(farfield_preview),
            "preview_exposure_scale": farfield_preview_exposure,
            "used_for_material_optimization": True,
            "reload_max_abs_diff": farfield_reload_max_abs_diff,
            "frozen_during_material_max_abs_diff": farfield_frozen_max_abs_diff,
        },
        "materials": {
            "order": args.material_order,
            "iterations_per_phase": args.material_iters,
            "completed_iterations": len(material_history),
            "phase_summaries": material_phase_summaries,
            "final_phase_best_optimization_mse": material_phase_summaries[-1][
                "best_optimization_mse"
            ],
            "prior_weight": args.material_prior_weight,
            "exposure_match": args.material_exposure_match,
            "learning_rate": args.material_lr,
            "learning_rate_step": args.material_lr_step,
            "learning_rate_gamma": args.material_lr_gamma,
            "patience": args.material_patience,
            "min_delta": args.material_min_delta,
            "outputs": material_outputs,
            "validation_spp": args.validation_spp,
            "validation_seed": material_validation_seed,
            "validation_seeds": material_validation_seeds,
            "final_spp": args.final_spp,
            "final_seed": final_seed,
            "renders": final_render_outputs,
            "rendered_img": final_render_outputs["rendered_img_exr"],
            "selection": material_selection,
            "metrics": final_metrics,
            "validation_metrics": selected_validation_metrics,
        },
        "farfield_metrics": farfield_metrics,
        "farfield_candidate_metrics": farfield_candidate_metrics,
        "farfield_initial_metrics": farfield_initial_metrics,
    }
    write_json(output_dir / "optimization_manifest.json", manifest)
    if farfield_selection != "optimized":
        print(
            "Far-field candidate failed the validation gate; selected the initial "
            "HDRI for export and material optimization."
        )
    if material_selection != "optimized":
        print(
            "Material candidate failed the validation gate; selected the input ARM maps "
            "for final export."
        )
    print(f"Far-field + material optimization complete: {output_dir}")


if __name__ == "__main__":
    main()
