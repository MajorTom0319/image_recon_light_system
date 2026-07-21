"""
render_matnet_pre_with_hdri.py

Reads MatNet predictions (albedo / roughness / metallic / normal / depth)
together with the original image, estimates camera intrinsics via GeoCalib,
reconstructs a mesh from the predicted depth, and renders the scene under a
user-supplied HDRI using Mitsuba.

Usage:
python render_matnet_pre_with_hdri.py \
    --img_path  /home/majortom/project/test2.png \
    --matnet_dir /home/majortom/project/Materialist/output_imgs/test2_matnet_out \
    --env_path  /home/majortom/project/Materialist/output_imgs/test2_syn_para/best_results/envmap.hdr
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import open3d as o3d

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import drjit as dr
dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)

from myutils.mi_plugin import MatDiffBSDF
mi.register_bsdf("MatDiffBSDF", lambda props: MatDiffBSDF(props))

from myutils.camera_utils import (
    estimate_camera_geocalib,
    make_mitsuba_compatible_K,
    scale_intrinsics,
    write_materialist_camera_json,
)
from myutils.mesh_recon import depth_file_to_mesh, rotate_mesh_around_x
from myutils.misc import linear_to_srgb

import global_config

# ---------------------------------------------------------------------------
# Material loading
# ---------------------------------------------------------------------------

def load_matnet_predictions(matnet_dir: str) -> dict:
    """Load albedo / roughness / metallic / normal / depth saved by test_matnet_infer.py."""
    albedo_path   = os.path.join(matnet_dir, "albedoPred.exr")
    roughness_path = os.path.join(matnet_dir, "roughnessPred.png")
    metallic_path  = os.path.join(matnet_dir, "metallicPred.png")
    normal_path    = os.path.join(matnet_dir, "normalPred.exr")
    depth_path     = os.path.join(matnet_dir, "depthPred.exr")

    for p in [albedo_path, roughness_path, metallic_path, normal_path, depth_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing MatNet prediction file: {p}")

    albedo    = np.array(mi.Bitmap(albedo_path),   dtype=np.float32)
    roughness = np.array(mi.Bitmap(roughness_path), dtype=np.float32)
    metallic  = np.array(mi.Bitmap(metallic_path),  dtype=np.float32)
    normal    = np.array(mi.Bitmap(normal_path),    dtype=np.float32)
    depth     = np.array(mi.Bitmap(depth_path),     dtype=np.float32)

    # Ensure roughness / metallic are single-channel
    if roughness.ndim == 3:
        roughness = roughness[..., 0]
    if metallic.ndim == 3:
        metallic = metallic[..., 0]

    mat = {
        "albedo":    torch.from_numpy(albedo).cuda().clamp(0, 1),
        "roughness": torch.from_numpy(roughness).unsqueeze(-1).cuda().clamp(0.07, 1),
        "metallic":  torch.from_numpy(metallic).unsqueeze(-1).cuda().clamp(0, 1),
        "normal":    torch.from_numpy(normal).cuda(),
        "depth":     torch.from_numpy(depth).unsqueeze(-1).cuda(),
    }
    return mat


# ---------------------------------------------------------------------------
# Mesh reconstruction
# ---------------------------------------------------------------------------

def build_mesh(mat: dict, K_work: np.ndarray, mesh_path: str, rebuild: bool = False) -> str:
    """Build (or reuse) a PLY mesh from the predicted depth map."""
    if not rebuild and os.path.exists(mesh_path):
        print(f"Using existing mesh: {mesh_path}")
        return mesh_path

    depth = mat["depth"].cpu().numpy()[..., 0]
    depth_for_mesh = 2.0 * depth.max() - depth   # flip so near = small

    mesh, _ = depth_file_to_mesh(
        depth_for_mesh,
        cameraMatrix=K_work,
        minAngle=6,
        sun3d=False,
        depthScale=1.0,
    )
    mesh = rotate_mesh_around_x(mesh, 180)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    print(f"Mesh saved to: {mesh_path}")
    return mesh_path


# ---------------------------------------------------------------------------
# Mitsuba scene setup & render
# ---------------------------------------------------------------------------

def build_scene(mesh_path: str, env_path: str, cam_meta_path: str, use_mesh_normal: bool):
    """Construct a Mitsuba scene with MatDiffBSDF and an HDRI emitter."""
    with open(cam_meta_path, "r", encoding="utf-8") as f:
        camera_meta = json.load(f)

    width, height = [int(v) for v in camera_meta["film.size"]]
    vfov_deg      = float(camera_meta["y_fov"][0])

    camera = mi.load_dict({
        "type":      "perspective",
        "fov":       vfov_deg,
        "fov_axis":  "y",
        "near_clip": float(camera_meta.get("near_clip", 0.01)),
        "far_clip":  float(camera_meta.get("far_clip", 10000.0)),
        "to_world":  mi.ScalarTransform4f.look_at(
                        origin=[0, 0, 0],
                        target=[0, 0, -1],
                        up=[0, 1, 0]),
        "film": {
            "type":         "hdrfilm",
            "width":        width,
            "height":       height,
            "pixel_format": "rgb",
        },
    })

    scene = mi.load_dict({
        "type": "scene",
        "shape": {
            "type":     "ply",
            "filename": mesh_path,
            "bsdf": {
                "type":             "MatDiffBSDF",
                "cam_meta":         str(cam_meta_path),
                "use_mesh_normal":  use_mesh_normal,
            },
        },
        "integrator": {"type": "path", "max_depth": 4},
        "sensor":     camera,
        "emitter":    {"type": "envmap", "filename": env_path},
    })
    return scene


def render_scene(scene, mat: dict, use_mesh_normal: bool, spp: int = 64, n_iter: int = 4):
    """Render the scene with denoising, accumulating multiple seeds."""
    mi_params = mi.traverse(scene)
    mi_params["shape.bsdf.a"] = mat["albedo"]
    mi_params["shape.bsdf.r"] = mat["roughness"]
    mi_params["shape.bsdf.m"] = mat["metallic"]
    if not use_mesh_normal:
        mi_params["shape.bsdf.n"] = mat["normal"]
    mi_params.update()

    # First render to get actual output size (for lazy denoiser init)
    img0 = mi.render(scene, spp=spp, seed=0)
    h, w = int(img0.shape[0]), int(img0.shape[1])
    denoiser  = mi.OptixDenoiser(input_size=(w, h), albedo=False, normals=False, temporal=False)
    empty_img = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(n_iter):
        img = mi.render(scene, spp=spp, seed=i)
        img = denoiser(img)
        empty_img += img.numpy()

    return empty_img / n_iter


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Render MatNet predictions under a given HDRI using Mitsuba."
    )
    parser.add_argument("--img_path",   required=True, type=str,
                        help="Path to the original input image (PNG/JPG).")
    parser.add_argument("--matnet_dir", required=True, type=str,
                        help="Directory containing MatNet prediction files "
                             "(albedoPred.exr, roughnessPred.png, ...).")
    parser.add_argument("--env_path",   required=True, type=str,
                        help="Path to the HDRI environment map (.hdr / .exr).")
    parser.add_argument("--spp",        type=int,   default=64,
                        help="Samples per pixel per render pass.")
    parser.add_argument("--n_iter",     type=int,   default=4,
                        help="Number of denoised render passes to accumulate.")
    parser.add_argument("--rebuild_mesh", action="store_true",
                        help="Force mesh rebuild even if PLY already exists.")
    parser.add_argument("--use_mesh_normal", action="store_true", default=True,
                        help="Use mesh geometric normal (default). "
                             "Pass --no_mesh_normal to use MatNet predicted normal.")
    parser.add_argument("--no_mesh_normal", dest="use_mesh_normal", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()

    img_path   = args.img_path
    matnet_dir = args.matnet_dir
    env_path   = args.env_path

    if not os.path.exists(env_path):
        raise FileNotFoundError(f"HDRI not found: {env_path}")

    # ── 1. Read original image ─────────────────────────────────────────────
    image_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image_bgr is None:
        raise FileNotFoundError(img_path)
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    if image_bgr.shape[2] == 4:
        image_bgr = image_bgr[:, :, :3]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    print(f"Original image: {original_w} x {original_h}")

    # ── 2. Load MatNet predictions ──────────────────────────────────────────
    mat = load_matnet_predictions(matnet_dir)
    work_h, work_w = int(mat["albedo"].shape[0]), int(mat["albedo"].shape[1])
    print(f"MatNet working resolution: {work_w} x {work_h}")

    # ── 3. GeoCalib → K + FOV ───────────────────────────────────────────────
    print("Estimating camera intrinsics with GeoCalib ...")
    geo_camera = estimate_camera_geocalib(img_path, device="cuda")

    K_work = scale_intrinsics(
        geo_camera.K,
        source_hw=(original_h, original_w),
        target_hw=(work_h, work_w),
    )
    K_work = make_mitsuba_compatible_K(K_work)

    hfov_deg = 2.0 * np.degrees(np.arctan2(work_w, 2.0 * K_work[0, 0]))
    vfov_deg = 2.0 * np.degrees(np.arctan2(work_h, 2.0 * K_work[1, 1]))
    print(f"  hfov / vfov = {hfov_deg:.3f} / {vfov_deg:.3f} deg")
    print(f"  fx / fy     = {K_work[0,0]:.3f} / {K_work[1,1]:.3f} px")

    # ── 4. Write camera_meta.json ───────────────────────────────────────────
    cam_meta_path = os.path.join(matnet_dir, "camera_meta.json")
    write_materialist_camera_json(
        cam_meta_path,
        K_work=K_work,
        work_hw=(work_h, work_w),
        geocalib_result=geo_camera,
    )
    print(f"Camera meta written to: {cam_meta_path}")

    # ── 5. Reconstruct mesh from depth ──────────────────────────────────────
    mesh_path = os.path.join(matnet_dir, "mesh_hdri.ply")
    build_mesh(mat, K_work, mesh_path, rebuild=args.rebuild_mesh)

    # ── 6. Build Mitsuba scene & render ─────────────────────────────────────
    print(f"Building Mitsuba scene (HDRI: {env_path}) ...")
    scene = build_scene(
        mesh_path=mesh_path,
        env_path=env_path,
        cam_meta_path=cam_meta_path,
        use_mesh_normal=args.use_mesh_normal,
    )

    print(f"Rendering (spp={args.spp}, n_iter={args.n_iter}) ...")
    img_linear = render_scene(
        scene, mat,
        use_mesh_normal=args.use_mesh_normal,
        spp=args.spp,
        n_iter=args.n_iter,
    )

    # ── 7. Save results ─────────────────────────────────────────────────────
    env_id   = os.path.splitext(os.path.basename(env_path))[0]
    out_exr  = os.path.join(matnet_dir, f"render_{env_id}.exr")
    out_png  = os.path.join(matnet_dir, f"render_{env_id}.png")

    mi.util.write_bitmap(out_exr, img_linear)

    img_srgb  = linear_to_srgb(torch.from_numpy(img_linear)).clamp(0, 1)
    img_srgb_np = (img_srgb.numpy() * 255).astype(np.uint8)
    img_bgr   = cv2.cvtColor(img_srgb_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_png, img_bgr)

    print(f"EXR saved to: {out_exr}")
    print(f"PNG saved to: {out_png}")


if __name__ == "__main__":
    try:
        main()
    finally:
        dr.sync_thread()
        torch.cuda.empty_cache()
