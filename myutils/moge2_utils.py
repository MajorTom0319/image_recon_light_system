"""Lightweight MoGe-2 inference and depth-preparation helpers."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from myutils.camera_utils import GeoCalibResult, _fovs_from_K


_MOGE2_MODEL_CACHE: dict[str, torch.nn.Module] = {}


def load_moge2_model(
    device: str = "cuda",
    model_name: str = "Ruicheng/moge-2-vitl-normal",
):
    """Load and cache a MoGe-2 model on the requested device."""
    cache_key = f"{device}_{model_name}"
    if cache_key in _MOGE2_MODEL_CACHE:
        return _MOGE2_MODEL_CACHE[cache_key]

    try:
        from moge.model.v2 import MoGeModel
    except ImportError as exc:
        raise ImportError(
            "MoGe is not installed. Install it via: "
            "pip install git+https://github.com/microsoft/MoGe.git"
        ) from exc

    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    _MOGE2_MODEL_CACHE[cache_key] = model
    return model


def infer_moge2(
    image_rgb: np.ndarray,
    device: str = "cuda",
    model_name: str = "Ruicheng/moge-2-vitl-normal",
) -> dict[str, np.ndarray | None]:
    """Run MoGe-2 while preserving a separate finite-pixel validity mask."""
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError(f"image_rgb must be np.ndarray, got {type(image_rgb)}")
    if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image_rgb.shape}")

    image_rgb = image_rgb[..., :3]
    height, width = image_rgb.shape[:2]
    model = load_moge2_model(device, model_name)
    input_tensor = torch.as_tensor(
        image_rgb.astype(np.float32) / 255.0,
        dtype=torch.float32,
        device=device,
    ).permute(2, 0, 1)

    # MoGe's default apply_mask=True replaces invalid depth/points with inf.
    # Keep predictions finite and propagate the validity mask explicitly so
    # later interpolation and legacy mesh code never receive masked infinities.
    with torch.inference_mode():
        output = model.infer(input_tensor, apply_mask=False)

    points = output["points"].cpu().numpy()
    depth = output["depth"].cpu().numpy()
    mask = output["mask"].cpu().numpy().astype(bool)
    mask &= np.isfinite(depth) & (depth > 0)
    mask &= np.isfinite(points).all(axis=-1)

    # Keep every exported geometry map finite, not only the map eventually
    # passed to the mesh builder. The separate mask remains authoritative.
    points = np.where(mask[..., None], points, 0.0).astype(np.float32)
    depth = np.where(mask, depth, 0.0).astype(np.float32)

    normal = None
    if output.get("normal") is not None:
        normal = output["normal"].cpu().numpy()
        normal_valid = mask & np.isfinite(normal).all(axis=-1)
        normal = np.where(normal_valid[..., None], normal, 0.0).astype(np.float32)

    result: dict[str, np.ndarray | None] = {
        "points": points,
        "depth": depth,
        "mask": mask,
        "normal": normal,
    }

    # MoGe returns intrinsics normalized by image width/height.
    intrinsics = output["intrinsics"].cpu().numpy()
    result["intrinsics"] = np.array(
        [
            [intrinsics[0, 0] * width, intrinsics[0, 1] * width, intrinsics[0, 2] * width],
            [intrinsics[1, 0] * height, intrinsics[1, 1] * height, intrinsics[1, 2] * height],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return result


def prepare_moge2_depth(
    moge2_output: dict,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite metric depth and a validity mask at ``target_hw``.

    Invalid depth is set to zero, the sentinel expected by
    ``depth_file_to_mesh``. Resizing uses normalized convolution so invalid
    source pixels cannot bleed into valid geometry.
    """
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target_hw must be positive, got {target_hw}")

    depth = np.asarray(moge2_output["depth"], dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected MoGe depth with shape HxW, got {depth.shape}")

    raw_mask = moge2_output.get("mask")
    if raw_mask is None:
        valid = np.ones(depth.shape, dtype=bool)
    else:
        valid = np.asarray(raw_mask, dtype=bool)
        if valid.shape != depth.shape:
            raise ValueError(
                f"MoGe mask shape {valid.shape} does not match depth shape {depth.shape}"
            )

    valid &= np.isfinite(depth) & (depth > 0)

    if depth.shape != (target_h, target_w):
        valid_f32 = valid.astype(np.float32)
        weighted_depth = cv2.resize(
            np.where(valid, depth, 0.0),
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        valid_weight = cv2.resize(
            valid_f32,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_mask = cv2.resize(
            valid.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        depth = np.divide(
            weighted_depth,
            valid_weight,
            out=np.zeros_like(weighted_depth, dtype=np.float32),
            where=valid_weight > 1e-6,
        )
        valid = resized_mask & (valid_weight > 1e-6)

    valid &= np.isfinite(depth) & (depth > 0)
    clean_depth = np.zeros((target_h, target_w), dtype=np.float32)
    clean_depth[valid] = depth[valid]
    return clean_depth, valid


def prepare_dense_moge2_depth(
    moge2_output: dict,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Return dense metric depth suitable for IndoorLightEditing.

    Unlike mesh reconstruction, IndoorLightEditing expects a finite, dense
    ``float32`` HxW depth map. Invalid MoGe pixels are therefore filled from
    their nearest valid neighbour. The returned mask still describes which
    pixels were valid before filling.
    """
    depth, valid = prepare_moge2_depth(moge2_output, target_hw)
    if not valid.any():
        raise ValueError("MoGe produced no valid depth pixels")

    if not valid.all():
        from scipy.ndimage import distance_transform_edt

        nearest_valid_indices = distance_transform_edt(
            ~valid,
            return_distances=False,
            return_indices=True,
        )
        depth = depth[tuple(nearest_valid_indices)]

    depth = np.ascontiguousarray(depth, dtype=np.float32)
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("Dense MoGe depth must contain only finite positive values")
    return depth, valid


def estimate_camera_moge2(
    image_rgb: np.ndarray,
    device: str = "cuda",
    model_name: str = "Ruicheng/moge-2-vitl-normal",
) -> tuple[GeoCalibResult, dict[str, np.ndarray | None]]:
    """Estimate pinhole intrinsics and geometry using MoGe-2."""
    moge2_output = infer_moge2(image_rgb, device, model_name)
    K = np.asarray(moge2_output["intrinsics"], dtype=np.float32)
    height, width = image_rgb.shape[:2]
    hfov_deg, vfov_deg = _fovs_from_K(K, width=width, height=height)

    return GeoCalibResult(
        width=width,
        height=height,
        K=K,
        hfov_deg=hfov_deg,
        vfov_deg=vfov_deg,
        extra={"pose_used": False, "camera_model": "pinhole", "source": "moge2"},
    ), moge2_output
