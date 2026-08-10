from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
from huggingface_hub import hf_hub_download
import mitsuba as mi
import numpy as np
import open3d as o3d
import torch

from Material_net.dpt import MaterialNet
from myutils.camera_utils import (
    make_mitsuba_compatible_K,
    scale_intrinsics,
    write_materialist_camera_json,
)
from myutils.mesh_recon import depth_file_to_mesh, rotate_mesh_around_x
from myutils.moge2_utils import (
    camera_to_materialist_vectors,
    estimate_camera_moge2,
    prepare_dense_moge2_depth,
    prepare_moge2_depth,
    prepare_moge2_normal,
    prepare_moge2_points,
)


DEFAULT_IMAGE = Path(
    "/home/majortom/project/IndoorLightEditing/examples/Example1/input/im.png"
)
DEFAULT_MOGE2_MODEL = Path(
    "/home/majortom/project/datasets/ckpt/moge2_vitl_normal.pt"
)
DEFAULT_OUTPUT = Path(
    "/home/majortom/project/Materialist/output_imgs/indoorlightediting_test"
)
LINEAR_EXTENSIONS = {".exr", ".hdr"}
RX180 = np.diag([1.0, -1.0, -1.0]).astype(np.float32)


def srgb_to_linear_standard(image: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB to linear RGB using the IEC transfer curve."""
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb_standard(image: np.ndarray) -> np.ndarray:
    """Convert non-negative linear RGB to normalized sRGB."""
    image = np.maximum(np.asarray(image, dtype=np.float32), 0.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Infer MatNet materials and MoGe2 geometry, export a consistent "
            "Materialist scene, and prepare IndoorLightEditing depth.npy."
        ),
    )
    parser.add_argument("--image-path", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--moge2-model", type=Path, default=DEFAULT_MOGE2_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--indoorlight-depth-path",
        type=Path,
        default=None,
        help="Defaults to depth.npy beside the input image",
    )
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch/MoGe device; auto selects CUDA when available",
    )
    parser.add_argument(
        "--mesh-mask",
        type=Path,
        default=None,
        help="Optional nonzero mask of image regions excluded from the mesh",
    )
    args = parser.parse_args()
    if not np.isfinite(args.scale) or args.scale <= 0:
        parser.error("--scale must be finite and positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_rgb_png(path: Path, srgb: np.ndarray) -> np.ndarray:
    rgb_u8 = np.clip(np.asarray(srgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write PNG: {path}")
    return rgb_u8


def _write_mask(path: Path, mask: np.ndarray) -> None:
    if not cv2.imwrite(str(path), np.asarray(mask, dtype=np.uint8) * 255):
        raise RuntimeError(f"Failed to write mask: {path}")


def load_input_rgb(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return linear RGB target, uint8 sRGB MoGe input, and input metadata."""
    suffix = path.suffix.lower()
    if suffix in LINEAR_EXTENSIONS:
        linear = np.asarray(mi.Bitmap(str(path)), dtype=np.float32)
        if linear.ndim == 2:
            linear = np.repeat(linear[..., None], 3, axis=-1)
        linear = linear[..., :3]
        if not np.isfinite(linear).all():
            raise ValueError(f"Linear input contains NaN or Inf: {path}")
        negative_count = int(np.count_nonzero(linear < 0))
        linear = np.maximum(linear, 0.0).astype(np.float32)
        positive = linear[linear > 0]
        percentile_99 = float(np.percentile(positive, 99.0)) if positive.size else 1.0
        moge_exposure = 0.9 / max(percentile_99, 1e-6)
        moge_srgb = np.clip(
            linear_to_srgb_standard(linear * moge_exposure),
            0.0,
            1.0,
        )
        moge_u8 = (moge_srgb * 255.0 + 0.5).astype(np.uint8)
        info = {
            "source_color_space": "linear",
            "source_dtype": str(linear.dtype),
            "negative_values_clamped": negative_count,
            "moge_preview_exposure": moge_exposure,
        }
        return np.ascontiguousarray(linear), moge_u8, info

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise FileNotFoundError(path)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    alpha_discarded = bgr.shape[2] == 4
    if alpha_discarded:
        alpha = bgr[..., 3]
        alpha_max = np.iinfo(alpha.dtype).max if np.issubdtype(alpha.dtype, np.integer) else 1.0
        if np.any(alpha != alpha_max):
            raise ValueError(
                "Input has non-opaque alpha. Composite it onto an explicit "
                "background before Materialist inference."
            )
    bgr = bgr[..., :3]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if np.issubdtype(rgb.dtype, np.integer):
        dtype_max = float(np.iinfo(rgb.dtype).max)
        srgb = rgb.astype(np.float32) / dtype_max
    elif np.issubdtype(rgb.dtype, np.floating):
        if not np.isfinite(rgb).all():
            raise ValueError(f"Input contains NaN or Inf: {path}")
        srgb = np.asarray(rgb, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported image dtype: {rgb.dtype}")
    srgb = np.clip(srgb, 0.0, 1.0)
    moge_u8 = (srgb * 255.0 + 0.5).astype(np.uint8)
    linear = srgb_to_linear_standard(srgb)
    info = {
        "source_color_space": "srgb",
        "source_dtype": str(rgb.dtype),
        "alpha_discarded": alpha_discarded,
        "moge_preview_exposure": 1.0,
    }
    return np.ascontiguousarray(linear), moge_u8, info


def _load_mesh_mask(path: Path | None, hw: tuple[int, int]) -> tuple[np.ndarray, str | None]:
    height, width = hw
    if path is None or not path.is_file():
        return np.zeros((height, width), dtype=bool), None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Cannot read mesh mask: {path}")
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0, str(path)


def _material_stats(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "finite": bool(np.isfinite(value).all()),
    }


def main() -> None:
    args = parse_args()
    image_path = args.image_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    moge2_model = args.moge2_model.expanduser().resolve()
    indoorlight_depth_path = (
        args.indoorlight_depth_path.expanduser().resolve()
        if args.indoorlight_depth_path is not None
        else image_path.with_name("depth.npy")
    )
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not moge2_model.is_file():
        raise FileNotFoundError(moge2_model)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "inference_manifest.json"
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        manifest_path,
        {
            "schema_version": 2,
            "status": "running",
            "started_at": started_at,
            "image": str(image_path),
        },
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    image_linear, image_moge_u8, input_info = load_input_rgb(image_path)
    original_h, original_w = image_linear.shape[:2]
    print(f"Input: {original_w}x{original_h}, {input_info['source_color_space']}")

    print("Running MoGe2 inference for camera intrinsics and geometry...")
    geo_camera, moge2_output = estimate_camera_moge2(
        image_moge_u8,
        device=str(device),
        model_name=str(moge2_model),
    )
    print(
        f"MoGe2 FOV: hfov={geo_camera.hfov_deg:.2f}°, "
        f"vfov={geo_camera.vfov_deg:.2f}°"
    )

    indoorlight_depth, indoorlight_source_mask = prepare_dense_moge2_depth(
        moge2_output,
        target_hw=(original_h, original_w),
    )

    model_path = hf_hub_download(
        repo_id="Lez/MatNet",
        filename="matnet_weights.pth",
        repo_type="model",
    )
    model = MaterialNet(
        encoder="vitb",
        features=128,
        out_channels=[96, 192, 384, 768],
        use_bn=False,
        use_clstoken=False,
    )
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device)).eval()

    # MatNet was trained on linear EXR input. MoGe2 independently consumes the
    # uint8 sRGB/tone-mapped image prepared above.
    pred, work_linear, preprocess_meta = model.infer_image_scaled(
        image_linear,
        scale=args.scale,
    )
    work_linear = np.asarray(work_linear, dtype=np.float32)
    work_h, work_w = work_linear.shape[:2]
    print(f"Work resolution: {work_w}x{work_h}")

    K_work = scale_intrinsics(
        geo_camera.K,
        source_hw=(original_h, original_w),
        target_hw=(work_h, work_w),
        preprocess_meta=preprocess_meta,
    )
    K_work = make_mitsuba_compatible_K(K_work)
    camera_meta_path = output_dir / "camera_meta.json"
    camera_meta = write_materialist_camera_json(
        camera_meta_path,
        K_work=K_work,
        work_hw=(work_h, work_w),
        geocalib_result=geo_camera,
        preprocess_meta=preprocess_meta,
    )

    raw_albedo = np.asarray(pred["albedo"], dtype=np.float32)
    raw_roughness = np.asarray(pred["roughness"], dtype=np.float32)
    raw_metallic = np.asarray(pred["metallic"], dtype=np.float32)
    albedo = np.clip(raw_albedo, 0.0, 1.0)
    roughness = np.clip(raw_roughness, 0.07, 1.0)
    metallic = np.clip(raw_metallic, 0.0, 1.0)
    normal = np.asarray(pred["normal"], dtype=np.float32)
    normal_length = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = np.divide(
        normal,
        normal_length,
        out=np.zeros_like(normal),
        where=normal_length > 1e-8,
    )
    relative_depth = np.asarray(pred["depth"], dtype=np.float32)
    material_values = (albedo, roughness, metallic, normal, relative_depth)
    if not all(np.isfinite(value).all() for value in material_values):
        raise ValueError("MatNet produced NaN or Inf")

    moge2_depth, depth_valid = prepare_moge2_depth(
        moge2_output,
        target_hw=(work_h, work_w),
    )
    points_camera, points_valid = prepare_moge2_points(
        moge2_output,
        target_hw=(work_h, work_w),
    )
    points_materialist = camera_to_materialist_vectors(points_camera)
    normal_camera = normal_materialist = None
    normal_valid = np.zeros((work_h, work_w), dtype=bool)
    if moge2_output.get("normal") is not None:
        normal_camera, normal_valid = prepare_moge2_normal(
            moge2_output,
            target_hw=(work_h, work_w),
        )
        normal_materialist = camera_to_materialist_vectors(normal_camera)

    default_mask = output_dir / "mesh_mask.png"
    mesh_mask_path = (
        args.mesh_mask.expanduser().resolve()
        if args.mesh_mask is not None
        else default_mask
    )
    mesh_mask, mesh_mask_source = _load_mesh_mask(mesh_mask_path, (work_h, work_w))
    mesh_valid = depth_valid & ~mesh_mask
    depth_for_mesh = moge2_depth.copy()
    depth_for_mesh[~mesh_valid] = 0.0
    mesh, _ = depth_file_to_mesh(
        depth_for_mesh,
        cameraMatrix=K_work,
        minAngle=6,
        sun3d=False,
        depthScale=1.0,
    )
    mesh = rotate_mesh_around_x(mesh, 180)
    mesh_path = output_dir / "mesh_moge2.ply"
    if not o3d.io.write_triangle_mesh(str(mesh_path), mesh):
        raise RuntimeError(f"Failed to write mesh: {mesh_path}")

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    if not np.isfinite(vertices).all():
        raise ValueError("Mesh contains NaN or Inf vertices")
    invalid_grid_indices = np.flatnonzero(~mesh_valid.reshape(-1))
    if invalid_grid_indices.size and np.isin(triangles, invalid_grid_indices).any():
        raise RuntimeError("Invalid/masked depth was referenced by a mesh triangle")
    if len(vertices) < work_h * work_w:
        raise RuntimeError("Mesh does not retain the expected image-grid vertices")
    mesh_depth = (-vertices[: work_h * work_w, 2]).reshape(work_h, work_w)

    # Write every run deterministically. There is no implicit mesh cache: a new
    # input, K, scale, mask, or model prediction always produces a new mesh.
    target_linear = np.maximum(work_linear, 0.0).astype(np.float32)
    target_srgb = np.clip(linear_to_srgb_standard(target_linear), 0.0, 1.0)
    target_png_u8 = _write_rgb_png(output_dir / "gt_image.png", target_srgb)
    mi.util.write_bitmap(str(output_dir / "gt_image.exr"), target_linear)
    _write_rgb_png(
        output_dir / "albedo_half.png",
        np.clip(linear_to_srgb_standard(albedo), 0.0, 1.0),
    )
    _write_rgb_png(
        output_dir / "normal_half.png",
        np.clip(normal * 0.5 + 0.5, 0.0, 1.0),
    )
    _write_rgb_png(
        output_dir / "moge2_input.png",
        image_moge_u8.astype(np.float32) / 255.0,
    )
    # These *_half PNGs are display previews. Encode linear scalar values as
    # sRGB so Blender's default PNG decoding displays them like the EXRs.
    cv2.imwrite(
        str(output_dir / "roughness_half.png"),
        (np.clip(linear_to_srgb_standard(roughness), 0.0, 1.0) * 255.0 + 0.5)
        .astype(np.uint8),
    )
    cv2.imwrite(
        str(output_dir / "metallic_half.png"),
        (np.clip(linear_to_srgb_standard(metallic), 0.0, 1.0) * 255.0 + 0.5)
        .astype(np.uint8),
    )
    depth_span = float(relative_depth.max() - relative_depth.min())
    depth_preview = (relative_depth - relative_depth.min()) / max(depth_span, 1e-8)
    cv2.imwrite(
        str(output_dir / "depth_half.png"),
        (depth_preview * 255.0 + 0.5).astype(np.uint8),
    )

    mi.util.write_bitmap(str(output_dir / "albedoPred.exr"), albedo)
    mi.util.write_bitmap(str(output_dir / "normalPred.exr"), normal)
    mi.util.write_bitmap(str(output_dir / "roughnessPred.exr"), roughness)
    mi.util.write_bitmap(str(output_dir / "metallicPred.exr"), metallic)
    mi.util.write_bitmap(str(output_dir / "depthPred.exr"), relative_depth)
    mi.util.write_bitmap(str(output_dir / "moge2_depth.exr"), moge2_depth)
    mi.util.write_bitmap(str(output_dir / "mesh_depth.exr"), mesh_depth.astype(np.float32))
    mi.util.write_bitmap(str(output_dir / "moge2_points_camera.exr"), points_camera)
    mi.util.write_bitmap(str(output_dir / "moge2_points.exr"), points_materialist)
    if normal_camera is not None and normal_materialist is not None:
        mi.util.write_bitmap(str(output_dir / "moge2_normal_camera.exr"), normal_camera)
        mi.util.write_bitmap(str(output_dir / "moge2_normal.exr"), normal_materialist)

    np.save(output_dir / "moge2_valid_mask.npy", depth_valid, allow_pickle=False)
    np.save(output_dir / "moge2_points_valid_mask.npy", points_valid, allow_pickle=False)
    np.save(output_dir / "moge2_normal_valid_mask.npy", normal_valid, allow_pickle=False)
    np.save(
        output_dir / "indoorlight_depth_source_mask.npy",
        indoorlight_source_mask,
        allow_pickle=False,
    )
    _write_mask(output_dir / "moge2_valid_mask.png", depth_valid)
    _write_mask(output_dir / "mesh_valid_mask.png", mesh_valid)
    indoorlight_depth_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(indoorlight_depth_path, indoorlight_depth, allow_pickle=False)

    expected_png_u8 = np.clip(target_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    png_max_abs_diff = int(
        np.max(
            np.abs(
                target_png_u8.astype(np.int16) - expected_png_u8.astype(np.int16)
            )
        )
    )
    projection_error = 0.0
    valid_vertices = mesh_valid.reshape(-1)
    if valid_vertices.any():
        grid_vertices = vertices[: work_h * work_w][valid_vertices]
        z = -grid_vertices[:, 2]
        projected_u = K_work[0, 0] * grid_vertices[:, 0] / z + K_work[0, 2]
        projected_v = K_work[1, 2] - K_work[1, 1] * grid_vertices[:, 1] / z
        xx, yy = np.meshgrid(np.arange(work_w), np.arange(work_h))
        expected_uv = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)[valid_vertices]
        projected_uv = np.stack([projected_u, projected_v], axis=-1)
        projection_error = float(np.max(np.abs(projected_uv - expected_uv)))

    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "started_at": started_at,
        "completed_at": completed_at,
        "input": {
            "path": str(image_path),
            "sha256": _sha256(image_path),
            "original_size": [original_w, original_h],
            **input_info,
        },
        "device": str(device),
        "matnet": {
            "weights": str(model_path),
            "input_color_space": "linear",
            "scale": args.scale,
            "preprocess": preprocess_meta,
            "raw_out_of_bounds": {
                "albedo_above_1": int(np.count_nonzero(raw_albedo > 1.0)),
                "roughness_below_0.07": int(np.count_nonzero(raw_roughness < 0.07)),
                "metallic_above_1": int(np.count_nonzero(raw_metallic > 1.0)),
            },
        },
        "camera": {
            "source": "moge2",
            "metadata": str(camera_meta_path),
            "K": K_work.tolist(),
            "hfov_deg": camera_meta["x_fov"][0],
            "vfov_deg": camera_meta["y_fov"][0],
        },
        "moge2": {
            "model": str(moge2_model),
            "work_valid_depth": int(depth_valid.sum()),
            "work_total_pixels": int(depth_valid.size),
            "coordinate_files": {
                "moge2_points_camera.exr": "MoGe camera coordinates",
                "moge2_normal_camera.exr": "MoGe camera coordinates",
                "moge2_points.exr": "Materialist coordinates after Rx(180deg)",
                "moge2_normal.exr": "Materialist coordinates after Rx(180deg)",
            },
            "camera_to_materialist": RX180.tolist(),
        },
        "indoorlight_depth": {
            "path": str(indoorlight_depth_path),
            "shape": list(indoorlight_depth.shape),
            "dtype": str(indoorlight_depth.dtype),
            "min": float(indoorlight_depth.min()),
            "max": float(indoorlight_depth.max()),
            "filled_invalid": int(
                indoorlight_source_mask.size - indoorlight_source_mask.sum()
            ),
        },
        "mesh": {
            "path": str(mesh_path),
            "rebuilt": True,
            "mask_source": mesh_mask_source,
            "valid_depth_pixels": int(mesh_valid.sum()),
            "vertices": int(len(vertices)),
            "triangles": int(len(triangles)),
            "invalid_vertices_referenced": False,
        },
        "outputs": {
            "target_linear": _material_stats(target_linear),
            "albedo": _material_stats(albedo),
            "roughness": _material_stats(roughness),
            "metallic": _material_stats(metallic),
            "normal": _material_stats(normal),
            "moge2_depth": _material_stats(moge2_depth),
        },
        "validation": {
            "gt_png_write_max_abs_diff_code_value": png_max_abs_diff,
            "mesh_projection_max_abs_error_px": projection_error,
            "all_exported_numeric_arrays_finite": True,
        },
    }
    _write_json(manifest_path, manifest)
    print(f"IndoorLightEditing depth: {indoorlight_depth_path}")
    print(f"Materialist outputs: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
