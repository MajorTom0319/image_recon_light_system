"""
render_matnet_pre_with_hdri.py

Reads MatNet predictions (albedo / roughness / metallic / normal / depth)
together with the original image, estimates camera intrinsics via GeoCalib,
reconstructs a mesh from the predicted depth, and renders the scene under a
user-supplied HDRI using Mitsuba.

Usage:
# 单帧渲染
python render_matnet_pre_with_hdri.py \
    --img_path  /home/majortom/project/test2.png \
    --matnet_dir /home/majortom/project/Materialist/output_imgs/test2_matnet_out \
    --env_path  /home/majortom/project/Materialist/output_imgs/test2_syn_para/best_results/envmap.hdr \
    --mode single

# 滚动渲染
python render_matnet_pre_with_hdri.py \
    --img_path /home/majortom/project/vin.jpg \
    --matnet_dir /home/majortom/project/Materialist/output_imgs/vin_test_out1 \
    --env_path /home/majortom/project/datasets/cowboy_town_saloon_4k.exr \
    --mode rolling --frames 36 --rotation_step 10
"""

import argparse
import json
import os
import sys
import time

import cv2
import imageio
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
from torchvision.utils import save_image

import global_config

# ---------------------------------------------------------------------------
# Material loading
# ---------------------------------------------------------------------------

def load_matnet_predictions(matnet_dir: str) -> dict:
    """Load albedo / roughness / metallic / normal / depth saved by test_matnet_infer.py."""
    albedo_path   = os.path.join(matnet_dir, "albedoPred.exr")
    roughness_path = os.path.join(matnet_dir, "roughnessPred.exr")
    metallic_path  = os.path.join(matnet_dir, "metallicPred.exr")
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
# Rolling-envmap helpers
# ---------------------------------------------------------------------------

def rotate_envmap(envmap: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Circular-shift the envmap along the horizontal axis."""
    height, width = envmap.shape[:2]
    shift_pixels = int((angle_degrees / 360.0) * width)
    rotated = np.zeros_like(envmap)
    for c in range(envmap.shape[2]):
        rotated[:, :, c] = np.roll(envmap[:, :, c], shift_pixels, axis=1)
    return rotated


def render_rolling(
    mesh_path: str,
    env_path: str,
    cam_meta_path: str,
    mat: dict,
    use_mesh_normal: bool,
    matnet_dir: str,
    frames: int,
    rotation_step: float,
    spp: int,
    n_iter: int,
):
    """Render a rolling-envmap animation: rotate the HDRI step-by-step."""
    animation_dir = os.path.join(matnet_dir, "rolling_envmap_animation")
    os.makedirs(animation_dir, exist_ok=True)

    original_envmap = np.array(mi.Bitmap(env_path), dtype=np.float32)
    env_id = os.path.splitext(os.path.basename(env_path))[0]

    print(f"Generating {frames} frames, {rotation_step}° per frame")
    frame_paths = []

    for frame_idx in range(frames):
        angle = frame_idx * rotation_step
        rotated = rotate_envmap(original_envmap, angle)

        temp_env = os.path.join(animation_dir, f"temp_envmap_{frame_idx}.hdr")
        mi.Bitmap(rotated).write(temp_env)
        while not os.path.exists(temp_env):
            time.sleep(0.5)

        print(f"  Frame {frame_idx + 1}/{frames}  (angle={angle:.1f}°)")

        scene = build_scene(mesh_path, temp_env, cam_meta_path, use_mesh_normal)
        img_linear = render_scene(scene, mat, use_mesh_normal, spp=spp, n_iter=n_iter)

        img_srgb = linear_to_srgb(torch.from_numpy(img_linear)).clamp(0, 1)
        frame_path = os.path.join(animation_dir, f"frame_{frame_idx:04d}.png")
        save_image(img_srgb.permute(2, 0, 1), frame_path)
        frame_paths.append(frame_path)

        del scene
        torch.cuda.empty_cache()
        os.remove(temp_env)

    # ── Assemble video ──────────────────────────────────────────────────────
    mp4_path = os.path.join(matnet_dir, f"rolling_envmap_{env_id}.mp4")
    img_list = [imageio.imread(p) for p in frame_paths]
    imageio.mimwrite(mp4_path, img_list, format="mp4", fps=10, quality=8)
    print(f"MP4 saved to: {mp4_path}")

    gif_path = os.path.join(matnet_dir, f"rolling_envmap_{env_id}.gif")
    with imageio.get_writer(gif_path, mode="I", duration=0.1) as writer:
        for p in frame_paths:
            writer.append_data(imageio.imread(p))
    print(f"GIF saved to: {gif_path}")

    return animation_dir


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
    parser.add_argument("--mode", choices=["single", "rolling"], default="single",
                        help="'single' = one render; 'rolling' = rotating HDRI animation.")
    parser.add_argument("--frames", type=int, default=36,
                        help="Number of frames for rolling animation.")
    parser.add_argument("--rotation_step", type=float, default=10.0,
                        help="Rotation angle per frame in degrees.")
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
    # mat["roughness"] = 0.7
    # mat["metallic"] = 0
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

    # ── 6. Render (single or rolling) ──────────────────────────────────────
    if args.mode == "rolling":
        render_rolling(
            mesh_path=mesh_path,
            env_path=env_path,
            cam_meta_path=cam_meta_path,
            mat=mat,
            use_mesh_normal=args.use_mesh_normal,
            matnet_dir=matnet_dir,
            frames=args.frames,
            rotation_step=args.rotation_step,
            spp=args.spp,
            n_iter=args.n_iter,
        )
    else:
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

        # ── 7. Save results ─────────────────────────────────────────────────
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
