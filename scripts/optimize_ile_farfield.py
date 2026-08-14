#!/usr/bin/env python3
"""Optimize ILE lamp scales, then a 32x16 far-field HDR environment map.

Stage B freezes Materialist geometry/materials and ILE lamp geometry/color,
optimizing one non-negative scalar multiplier per lamp. The far-field stage
is called Stage C in this prototype: it freezes those optimized local lamps
and optimizes a low-resolution HDR environment map with energy and
total-variation regularization.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_light.io import load_ile_lights
from hybrid_light.mitsuba_builder import build_hybrid_scene_dict
from hybrid_light.visualization import render_projection_debug
from scripts.render_ile_lights import (
    _find_material_maps,
    _load_camera,
    _load_material_arrays,
    _resolve_mesh,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Stage B: optimize one scale per ILE lamp; then optimize a 32x16 "
            "far-field HDR environment map."
        ),
    )
    parser.add_argument("--materialist-dir", type=Path, required=True)
    parser.add_argument("--lights-json", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--camera-meta", type=Path, default=None)
    parser.add_argument("--material-dir", type=Path, default=None)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--use-pred-normal", action="store_true")
    parser.add_argument("--geometry-scale", type=float, default=None)
    parser.add_argument("--radiance-scale", type=float, default=1.0)
    parser.add_argument("--visible-offset", type=float, default=0.005)

    parser.add_argument("--stage-b-iters", type=int, default=200)
    parser.add_argument("--stage-b-lr", type=float, default=0.1)
    parser.add_argument("--light-scale-min", type=float, default=0.01)
    parser.add_argument("--light-scale-max", type=float, default=20.0)
    parser.add_argument("--light-prior-weight", type=float, default=1e-3)

    # Width x height is 32 x 16, stored as an HxWx3 tensor (16, 32, 3).
    parser.add_argument("--farfield-width", type=int, default=32)
    parser.add_argument("--farfield-height", type=int, default=16)
    parser.add_argument("--farfield-iters", type=int, default=300)
    parser.add_argument("--farfield-lr", type=float, default=0.03)
    parser.add_argument("--farfield-init", type=float, default=0.02)
    parser.add_argument("--farfield-max", type=float, default=4.0)
    parser.add_argument("--farfield-tv-weight", type=float, default=1e-2)
    parser.add_argument("--farfield-energy-weight", type=float, default=1e-3)
    parser.add_argument("--stage-b-only", action="store_true")

    parser.add_argument("--charbonnier-weight", type=float, default=1.0)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--charbonnier-epsilon", type=float, default=1e-3)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--spp-grad", type=int, default=16)
    parser.add_argument("--preview-spp", type=int, default=128)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-denoise", action="store_true")
    args = parser.parse_args()

    positive_names = (
        "stage_b_iters",
        "stage_b_lr",
        "light_scale_min",
        "light_scale_max",
        "farfield_width",
        "farfield_height",
        "farfield_iters",
        "farfield_lr",
        "farfield_max",
        "spp",
        "spp_grad",
        "preview_spp",
        "max_depth",
        "log_interval",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.light_scale_min >= args.light_scale_max:
        parser.error("--light-scale-min must be smaller than --light-scale-max")
    if args.radiance_scale <= 0 or args.visible_offset < 0:
        parser.error("radiance scale must be positive and visible offset non-negative")
    if args.farfield_init < 0 or args.farfield_init > args.farfield_max:
        parser.error("far-field init must be between zero and --farfield-max")
    return args


def linear_to_srgb_np(value: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(value, dtype=np.float32), 0.0)
    return np.where(
        value <= 0.0031308,
        12.92 * value,
        1.055 * np.power(value, 1.0 / 2.4) - 0.055,
    )


def srgb_to_linear_np(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return np.where(
        value <= 0.04045,
        value / 12.92,
        np.power((value + 0.055) / 1.055, 2.4),
    )


def linear_to_srgb_dr(dr, value):
    value = dr.maximum(value, 0.0)
    return dr.select(
        value <= 0.0031308,
        12.92 * value,
        1.055 * dr.power(value, 1.0 / 2.4) - 0.055,
    )


def charbonnier_loss_dr(dr, prediction, target, epsilon: float):
    difference_sq = dr.square(prediction.array - target.array)
    return dr.mean(dr.sqrt(difference_sq + epsilon * epsilon)) - epsilon


def cosine_learning_rate(iteration: int, total: int, base_lr: float) -> float:
    if total <= 1:
        return base_lr
    warmup = min(10, max(total // 20, 1))
    if iteration < warmup:
        return base_lr * (iteration + 1) / warmup
    progress = (iteration - warmup) / max(total - warmup - 1, 1)
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def load_target_linear(mi, path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    value = np.asarray(mi.Bitmap(str(path)), dtype=np.float32)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=-1)
    value = value[..., :3]

    # EXR/HDR are linear. Ordinary image files are display-referred sRGB.
    if path.suffix.lower() not in {".exr", ".hdr"}:
        if value.max(initial=0.0) > 1.5:
            value = value / 255.0
        value = srgb_to_linear_np(value)

    height, width = target_hw
    if value.shape[:2] != (height, width):
        interpolation = cv2.INTER_AREA if value.shape[0] >= height else cv2.INTER_CUBIC
        value = cv2.resize(value, (width, height), interpolation=interpolation)
    if not np.isfinite(value).all():
        raise ValueError(f"Target contains non-finite values: {path}")
    return np.ascontiguousarray(np.maximum(value, 0.0), dtype=np.float32)


def save_linear_image(mi, base_path: Path, image_linear: Any) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    image_np = np.asarray(image_linear, dtype=np.float32)
    mi.util.write_bitmap(str(base_path.with_suffix(".exr")), image_np)
    image_srgb = np.clip(linear_to_srgb_np(image_np), 0.0, 1.0)
    image_u8 = (image_srgb * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(
        str(base_path.with_suffix(".png")),
        cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR),
    )


def render_and_save(
    mi,
    scene,
    params,
    base_path: Path,
    *,
    spp: int,
    seed: int,
    denoise: bool,
) -> np.ndarray:
    rendered_raw = mi.render(scene, params, spp=spp, seed=seed)
    raw_np = np.asarray(rendered_raw, dtype=np.float32)
    save_linear_image(mi, base_path.with_name(base_path.name + "_raw"), raw_np)

    rendered = rendered_raw
    if denoise:
        denoiser = mi.OptixDenoiser(
            input_size=(int(rendered_raw.shape[1]), int(rendered_raw.shape[0])),
            albedo=False,
            normals=False,
            temporal=False,
        )
        rendered = denoiser(rendered_raw)
    rendered_np = np.asarray(rendered, dtype=np.float32)
    save_linear_image(mi, base_path, rendered_np)
    return rendered_np


def set_material_parameters(mi, params, material_arrays: dict[str, np.ndarray]) -> None:
    for short_name, value in material_arrays.items():
        key = f"materialist_mesh.bsdf.{short_name}"
        if key not in params:
            available = "\n".join(str(item) for item in params.keys())
            raise KeyError(f"Missing material parameter {key}\nAvailable:\n{available}")
        params[key] = mi.TensorXf(value)
    params.update()


def lamp_parameter_data(params, lights) -> tuple[list[str], dict[str, np.ndarray]]:
    keys = []
    bases = {}
    for light in lights:
        key = f"ile_{light.name}.emitter.radiance.value"
        if key not in params:
            available = "\n".join(str(item) for item in params.keys())
            raise KeyError(f"Missing lamp parameter {key}\nAvailable:\n{available}")
        base = np.asarray(light.rgb, dtype=np.float32).reshape(3)
        if not np.isfinite(base).all() or np.dot(base, base) <= 1e-12:
            raise ValueError(f"Lamp {light.name} needs a finite non-zero base RGB")
        keys.append(key)
        bases[key] = base
    return keys, bases


def projected_scale(rgb: np.ndarray, base: np.ndarray) -> float:
    return float(np.dot(rgb.reshape(3), base) / max(float(np.dot(base, base)), 1e-12))


def optimize_lamp_scales(
    mi,
    dr,
    *,
    scene,
    params,
    lights,
    target_srgb,
    args,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], float]:
    """Stage B: optimize RGB parameters but project them to base_rgb * scalar."""
    keys, bases = lamp_parameter_data(params, lights)
    optimizer = mi.ad.Adam(lr=args.stage_b_lr)
    for key in keys:
        optimizer[key] = params[key]
    params.update(optimizer)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_rgb: dict[str, np.ndarray] | None = None

    for iteration in range(args.stage_b_iters):
        learning_rate = cosine_learning_rate(iteration, args.stage_b_iters, args.stage_b_lr)
        optimizer.set_learning_rate(learning_rate)
        seed = args.seed + iteration * 2
        rendered_linear = mi.render(
            scene,
            params,
            spp=args.spp,
            spp_grad=args.spp_grad,
            seed=seed,
            seed_grad=seed + 1,
        )
        rendered_srgb = linear_to_srgb_dr(dr, rendered_linear)
        difference = rendered_srgb.array - target_srgb.array
        loss_charbonnier = charbonnier_loss_dr(
            dr,
            rendered_srgb,
            target_srgb,
            args.charbonnier_epsilon,
        )
        loss_mse = dr.mean(dr.square(difference))

        scale_prior = mi.Float(0.0)
        current_scales = {}
        for key in keys:
            base = bases[key]
            base_mi = mi.Color3f(base)
            scale = dr.sum(optimizer[key] * base_mi) / float(np.dot(base, base))
            scale = dr.maximum(scale, 1e-6)
            scale_prior += dr.square(dr.log(scale))
            current_scales[key] = scale
        scale_prior /= len(keys)

        loss = (
            args.charbonnier_weight * loss_charbonnier
            + args.mse_weight * loss_mse
            + args.light_prior_weight * scale_prior
        )
        dr.eval(loss, loss_charbonnier, loss_mse, scale_prior)
        loss_value = float(loss[0])
        scale_values = {key: float(current_scales[key][0]) for key in keys}
        history.append(
            {
                "iteration": iteration,
                "loss": loss_value,
                "charbonnier": float(loss_charbonnier[0]),
                "mse": float(loss_mse[0]),
                "scale_prior": float(scale_prior[0]),
                "learning_rate": learning_rate,
                "scales": scale_values,
            }
        )
        if loss_value < best_loss:
            best_loss = loss_value
            best_rgb = {
                key: np.asarray(optimizer[key], dtype=np.float32).reshape(3).copy()
                for key in keys
            }

        dr.backward(loss)
        optimizer.step()

        # The optimizer operates on RGB because Mitsuba exposes an RGB value.
        # Project it back to one scalar along the fixed ILE chromaticity.
        for key in keys:
            base = bases[key]
            base_mi = mi.Color3f(base)
            scale = dr.sum(optimizer[key] * base_mi) / float(np.dot(base, base))
            scale = dr.clip(scale, args.light_scale_min, args.light_scale_max)
            optimizer[key] = base_mi * scale
        params.update(optimizer)

        if iteration % args.log_interval == 0 or iteration == args.stage_b_iters - 1:
            compact = ", ".join(
                f"{key.split('.')[0]}={value:.4f}" for key, value in scale_values.items()
            )
            print(
                f"[Stage B {iteration:04d}/{args.stage_b_iters}] "
                f"loss={loss_value:.6f} charb={float(loss_charbonnier[0]):.6f} "
                f"mse={float(loss_mse[0]):.6f} scales=[{compact}]"
            )

    if best_rgb is None:
        raise RuntimeError("Stage B did not produce a valid checkpoint")
    for key, value in best_rgb.items():
        params[key] = mi.Color3f(value)
    params.update()
    return best_rgb, history, best_loss


def farfield_regularizers(mi, dr, tensor, height: int, width: int):
    """Return spherical horizontal-wrap TV and mean-square energy.

    Mitsuba appends one periodic seam column when loading an envmap, so a
    16x32 file is traversed as a (16, 33, 3) tensor. Regularization operates on
    the 32 logical columns and explicitly wraps column 31 back to column 0.
    """
    storage_height, storage_width, channels = [int(value) for value in tensor.shape]
    if storage_height != height or channels != 3 or storage_width not in {width, width + 1}:
        raise ValueError(
            f"Unexpected far-field storage shape {tensor.shape} for logical "
            f"shape {(height, width, 3)}"
        )
    pixel_count = height * width
    indices_np = np.arange(pixel_count, dtype=np.uint32)
    rows = indices_np // width
    cols = indices_np % width
    indices_np = rows * storage_width + cols
    right_np = rows * storage_width + (cols + 1) % width
    down_np = np.minimum(rows + 1, height - 1) * storage_width + cols

    indices = mi.UInt32(indices_np)
    right_indices = mi.UInt32(right_np.astype(np.uint32))
    down_indices = mi.UInt32(down_np.astype(np.uint32))
    colors = dr.gather(mi.Color3f, tensor.array, indices)
    colors_right = dr.gather(mi.Color3f, tensor.array, right_indices)
    colors_down = dr.gather(mi.Color3f, tensor.array, down_indices)
    tv_horizontal = dr.mean(dr.mean(dr.abs(colors - colors_right)))
    tv_vertical = dr.mean(dr.mean(dr.abs(colors - colors_down)))
    tv = tv_horizontal + tv_vertical
    energy = dr.mean(dr.mean(dr.square(colors)))
    return tv, energy


def clamp_and_tie_farfield_seam(mi, dr, tensor, height: int, width: int, maximum: float):
    """Clamp HDR radiance and keep Mitsuba's extra seam column periodic."""
    tensor = dr.clip(tensor, 0.0, maximum)
    storage_height, storage_width, channels = [int(value) for value in tensor.shape]
    if storage_height != height or channels != 3 or storage_width not in {width, width + 1}:
        raise ValueError(f"Unexpected far-field tensor shape after update: {tensor.shape}")
    if storage_width == width + 1:
        rows = np.arange(height, dtype=np.uint32)
        first_indices = mi.UInt32(rows * storage_width)
        seam_indices = mi.UInt32(rows * storage_width + width)
        first_colors = dr.gather(mi.Color3f, tensor.array, first_indices)
        flat = tensor.array
        dr.scatter(flat, first_colors, seam_indices)
        tensor = mi.TensorXf(flat, shape=tensor.shape)
    return tensor


def optimize_farfield(
    mi,
    dr,
    *,
    scene,
    params,
    target_srgb,
    args,
    loss_mask=None,
    validation_fn=None,
) -> tuple[np.ndarray, list[dict[str, Any]], float]:
    """Optimize only the low-resolution far-field envmap; lamps stay frozen."""
    env_key = "far_field_env.data"
    if env_key not in params:
        available = "\n".join(str(item) for item in params.keys())
        raise KeyError(f"Missing far-field parameter {env_key}\nAvailable:\n{available}")
    expected_shapes = {
        (args.farfield_height, args.farfield_width, 3),
        (args.farfield_height, args.farfield_width + 1, 3),
    }
    if tuple(params[env_key].shape) not in expected_shapes:
        raise ValueError(
            f"Far-field tensor shape {params[env_key].shape} is not one of "
            f"{sorted(expected_shapes)}"
        )

    optimizer = mi.ad.Adam(
        lr=args.farfield_lr,
        amsgrad=True,
        mask_updates=True,
    )
    optimizer[env_key] = params[env_key]
    params.update(optimizer)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_env: np.ndarray | None = None
    validation_reference: float | None = None
    stale_validations = 0
    if validation_fn is not None:
        best_loss = validation_fn()
        validation_reference = best_loss
        best_env = np.asarray(params[env_key], dtype=np.float32).copy()

    def masked_mean(value):
        if loss_mask is None:
            return dr.mean(value)
        return dr.sum(value * loss_mask.array) / dr.maximum(
            dr.sum(loss_mask.array), 1e-8
        )

    for iteration in range(args.farfield_iters):
        learning_rate = cosine_learning_rate(iteration, args.farfield_iters, args.farfield_lr)
        optimizer.set_learning_rate(learning_rate)
        iteration_seed = iteration if getattr(args, "resample_each_iteration", True) else 0
        seed = args.seed + (args.stage_b_iters + iteration_seed) * 2
        rendered_linear = mi.render(
            scene,
            params,
            spp=args.spp,
            spp_grad=args.spp_grad,
            seed=seed,
            seed_grad=seed + 1,
        )
        if getattr(args, "clip_display", False):
            rendered_linear = dr.clip(rendered_linear, 0.0, 1.0)
        rendered_srgb = linear_to_srgb_dr(dr, rendered_linear)
        difference = rendered_srgb.array - target_srgb.array
        loss_charbonnier = masked_mean(
            dr.sqrt(dr.square(difference) + args.charbonnier_epsilon**2)
            - args.charbonnier_epsilon
        )
        loss_mse = masked_mean(dr.square(difference))
        loss_tv, loss_energy = farfield_regularizers(
            mi,
            dr,
            optimizer[env_key],
            args.farfield_height,
            args.farfield_width,
        )
        loss = (
            args.charbonnier_weight * loss_charbonnier
            + args.mse_weight * loss_mse
            + args.farfield_tv_weight * loss_tv
            + args.farfield_energy_weight * loss_energy
        )
        dr.eval(loss, loss_charbonnier, loss_mse, loss_tv, loss_energy)
        loss_value = float(loss[0])
        history.append(
            {
                "iteration": iteration,
                "loss": loss_value,
                "charbonnier": float(loss_charbonnier[0]),
                "mse": float(loss_mse[0]),
                "tv": float(loss_tv[0]),
                "energy": float(loss_energy[0]),
                "learning_rate": learning_rate,
            }
        )
        checkpoint_metric_name = getattr(args, "farfield_checkpoint_metric", "loss")
        checkpoint_value = (
            float(loss_mse[0]) if checkpoint_metric_name == "mse" else loss_value
        )
        if validation_fn is None and checkpoint_value < best_loss:
            best_loss = checkpoint_value
            best_env = np.asarray(optimizer[env_key], dtype=np.float32).copy()

        dr.backward(loss)
        optimizer.step()
        optimizer[env_key] = clamp_and_tie_farfield_seam(
            mi,
            dr,
            optimizer[env_key],
            args.farfield_height,
            args.farfield_width,
            args.farfield_max,
        )
        params.update(optimizer)

        validation_mse = None
        should_validate = (
            validation_fn is not None
            and (
                (iteration + 1) % args.validation_interval == 0
                or iteration == args.farfield_iters - 1
            )
        )
        if should_validate:
            validation_mse = validation_fn()
            history[-1]["validation_mse"] = validation_mse
            if validation_mse < best_loss:
                best_loss = validation_mse
                best_env = np.asarray(
                    optimizer[env_key], dtype=np.float32
                ).copy()
            if validation_mse < validation_reference * (
                1.0 - args.validation_min_delta
            ):
                validation_reference = validation_mse
                stale_validations = 0
            else:
                stale_validations += 1

        if iteration % args.log_interval == 0 or iteration == args.farfield_iters - 1:
            print(
                f"[Far-field {iteration:04d}/{args.farfield_iters}] "
                f"loss={loss_value:.6f} charb={float(loss_charbonnier[0]):.6f} "
                f"mse={float(loss_mse[0]):.6f} tv={float(loss_tv[0]):.6f} "
                f"energy={float(loss_energy[0]):.6f} "
                f"val={validation_mse if validation_mse is not None else '-'}"
            )
        if (
            validation_fn is not None
            and stale_validations >= args.validation_patience
        ):
            print(
                f"[Far-field] early stopping at {iteration}: "
                "validation did not improve"
            )
            break

    if best_env is None:
        raise RuntimeError("Far-field optimization did not produce a valid checkpoint")
    params[env_key] = mi.TensorXf(best_env)
    params.update()
    return best_env, history, best_loss


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    materialist_dir = args.materialist_dir.expanduser().resolve()
    if not materialist_dir.is_dir():
        raise NotADirectoryError(materialist_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else materialist_dir / "hybrid_ile_farfield_opt"
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
    if args.target is not None:
        target_path = args.target.expanduser().resolve()
        target_source = "explicit_cli"
    elif light_set.image_path is not None and light_set.image_path.is_file():
        target_path = light_set.image_path.resolve()
        target_source = "ile_original_image"
    else:
        target_path = (materialist_dir / "gt_image.exr").resolve()
        target_source = "materialist_gt_fallback"
    if not target_path.is_file():
        raise FileNotFoundError(target_path)
    target_linear = load_target_linear(mi, target_path, (height, width))
    target_srgb_np = linear_to_srgb_np(target_linear)
    target_srgb = mi.TensorXf(target_srgb_np)
    save_linear_image(mi, output_dir / "target", target_linear)

    # ------------------------------------------------------------------
    # Stage B: local ILE lamps, one fixed-chromaticity scale per lamp.
    # ------------------------------------------------------------------
    local_scene_dict = build_hybrid_scene_dict(
        mi,
        mesh_path=mesh_path,
        camera_meta_path=camera_path,
        camera_meta=camera_meta,
        lights=light_set.lights,
        mode="local",
        radiance_scale=args.radiance_scale,
        visible_offset=args.visible_offset,
        use_mesh_normal=not args.use_pred_normal,
        max_depth=args.max_depth,
    )
    local_scene = mi.load_dict(local_scene_dict)
    local_params = mi.traverse(local_scene)
    set_material_parameters(mi, local_params, material_arrays)
    render_and_save(
        mi,
        local_scene,
        local_params,
        output_dir / "stage_b_initial",
        spp=args.preview_spp,
        seed=args.seed,
        denoise=not args.no_denoise,
    )

    best_lamp_rgb, stage_b_history, stage_b_best_loss = optimize_lamp_scales(
        mi,
        dr,
        scene=local_scene,
        params=local_params,
        lights=light_set.lights,
        target_srgb=target_srgb,
        args=args,
    )
    render_and_save(
        mi,
        local_scene,
        local_params,
        output_dir / "stage_b_optimized_local",
        spp=args.preview_spp,
        seed=args.seed + 100000,
        denoise=not args.no_denoise,
    )

    _, lamp_bases = lamp_parameter_data(local_params, light_set.lights)
    lamp_summary = []
    for light in light_set.lights:
        key = f"ile_{light.name}.emitter.radiance.value"
        optimized_rgb = best_lamp_rgb[key]
        lamp_summary.append(
            {
                "name": light.name,
                "parameter": key,
                "base_rgb": lamp_bases[key].tolist(),
                "initial_scale": args.radiance_scale,
                "optimized_scale": projected_scale(optimized_rgb, lamp_bases[key]),
                "optimized_rgb": optimized_rgb.tolist(),
            }
        )
    write_json(output_dir / "stage_b_history.json", stage_b_history)
    write_json(
        output_dir / "stage_b_lamps.json",
        {"best_loss": stage_b_best_loss, "lamps": lamp_summary},
    )

    result_manifest: dict[str, Any] = {
        "status": "stage_b_complete" if args.stage_b_only else "farfield_optimizing",
        "materialist_dir": str(materialist_dir),
        "mesh": str(mesh_path),
        "camera_meta": str(camera_path),
        "target": str(target_path),
        "target_source": target_source,
        "target_color_space": (
            "linear" if target_path.suffix.lower() in {".exr", ".hdr"} else "srgb"
        ),
        "lights_json": str(light_set.source_path),
        "materials": {key: str(value) for key, value in material_paths.items()},
        "geometry_scale": light_set.metadata["geometry_scale"],
        "stage_b": {
            "iterations": args.stage_b_iters,
            "best_loss": stage_b_best_loss,
            "lamps": lamp_summary,
        },
    }
    if args.stage_b_only:
        write_json(output_dir / "optimization_manifest.json", result_manifest)
        print(f"Stage B complete: {output_dir}")
        return

    # ------------------------------------------------------------------
    # Stage C: freeze optimized lamps, optimize a 32x16 far-field HDR map.
    # ------------------------------------------------------------------
    farfield_initial = np.full(
        (args.farfield_height, args.farfield_width, 3),
        args.farfield_init,
        dtype=np.float32,
    )
    farfield_initial_path = output_dir / "farfield_initial_32x16.exr"
    mi.util.write_bitmap(str(farfield_initial_path), farfield_initial)

    combined_scene_dict = build_hybrid_scene_dict(
        mi,
        mesh_path=mesh_path,
        camera_meta_path=camera_path,
        camera_meta=camera_meta,
        lights=light_set.lights,
        mode="combined",
        envmap_path=farfield_initial_path,
        radiance_scale=args.radiance_scale,
        visible_offset=args.visible_offset,
        use_mesh_normal=not args.use_pred_normal,
        max_depth=args.max_depth,
    )
    combined_scene = mi.load_dict(combined_scene_dict)
    combined_params = mi.traverse(combined_scene)
    set_material_parameters(mi, combined_params, material_arrays)
    for key, value in best_lamp_rgb.items():
        if key not in combined_params:
            raise KeyError(f"Combined scene is missing optimized lamp parameter {key}")
        combined_params[key] = mi.Color3f(value)
    combined_params.update()

    render_and_save(
        mi,
        combined_scene,
        combined_params,
        output_dir / "farfield_initial_combined",
        spp=args.preview_spp,
        seed=args.seed + 200000,
        denoise=not args.no_denoise,
    )
    best_farfield, farfield_history, farfield_best_loss = optimize_farfield(
        mi,
        dr,
        scene=combined_scene,
        params=combined_params,
        target_srgb=target_srgb,
        args=args,
    )
    render_and_save(
        mi,
        combined_scene,
        combined_params,
        output_dir / "farfield_optimized_combined",
        spp=args.preview_spp,
        seed=args.seed + 300000,
        denoise=not args.no_denoise,
    )

    # Drop Mitsuba's appended periodic seam column before exporting the
    # requested 32x16 HDRI. It will be re-created automatically when reloaded.
    farfield_export = best_farfield[:, : args.farfield_width, :]
    farfield_exr = output_dir / "farfield_optimized_32x16.exr"
    farfield_hdr = output_dir / "farfield_optimized_32x16.hdr"
    mi.util.write_bitmap(str(farfield_exr), farfield_export)
    mi.util.write_bitmap(str(farfield_hdr), farfield_export)
    write_json(output_dir / "farfield_history.json", farfield_history)

    result_manifest["status"] = "complete"
    result_manifest["farfield"] = {
        "width": args.farfield_width,
        "height": args.farfield_height,
        "iterations": args.farfield_iters,
        "best_loss": farfield_best_loss,
        "initial_value": args.farfield_init,
        "tv_weight": args.farfield_tv_weight,
        "energy_weight": args.farfield_energy_weight,
        "optimized_exr": str(farfield_exr),
        "optimized_hdr": str(farfield_hdr),
    }
    write_json(output_dir / "optimization_manifest.json", result_manifest)
    print(f"Stage B + far-field optimization complete: {output_dir}")


if __name__ == "__main__":
    main()
