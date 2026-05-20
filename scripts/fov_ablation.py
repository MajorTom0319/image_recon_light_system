#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import mitsuba as mi
import numpy as np
import open3d as o3d
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import global_config
from myutils.mesh_recon import depth_file_to_mesh, rotate_mesh_around_x, set_recon_camera
from myutils.mi_plugin import MatDiffBSDF, load_estimated_brdf
from myutils.misc import linear_to_srgb


def load_depth_with_mask(depth_path, mask_path):
    depth_np = np.array(mi.Bitmap(depth_path))
    depth_np = 2 * depth_np.max() - depth_np

    if os.path.exists(mask_path):
        mask_img = imageio.imread(mask_path)
        inv = False
        if mask_img.ndim > 2:
            inv = True
            mask_img = mask_img[..., 0]
        mask_img = mask_img.astype(np.float32)
        if mask_img.max() > 1.0:
            mask_img = mask_img / 255.0
        mesh_mask = mask_img > 0.5
        if inv:
            mesh_mask = ~mesh_mask
        depth_np[~mesh_mask] = 0
    return depth_np


def reconstruct_mesh(depth_path, mask_path, fov_deg, min_angle=6.0, depth_scale=1.0, save_ply_path=None):
    if save_ply_path and os.path.exists(save_ply_path):
        print(f"[Info] Mesh already exists: {save_ply_path}")
        return save_ply_path

    depth_np = load_depth_with_mask(depth_path, mask_path)
    cam_matrix = set_recon_camera(fov_deg=fov_deg, width=depth_np.shape[1], height=depth_np.shape[0])
    mesh, _ = depth_file_to_mesh(
        depth_np,
        cameraMatrix=cam_matrix,
        minAngle=min_angle,
        sun3d=False,
        depthScale=depth_scale,
    )
    mesh = rotate_mesh_around_x(mesh, 180)
    if save_ply_path:
        os.makedirs(os.path.dirname(save_ply_path), exist_ok=True)
        o3d.io.write_triangle_mesh(save_ply_path, mesh)
    return save_ply_path


def build_scene(mesh_path, mat_dir, env_path, fov_deg):
    mi.register_bsdf("MatDiffBSDF", lambda props: MatDiffBSDF(props))
    cam_cfg_path = os.path.join(global_config.BASE_DIR, "myutils", "default_cam.json")
    camera = mi.load_dict({
        "type": "perspective",
        "fov": float(fov_deg),
        "to_world": mi.ScalarTransform4f.look_at(origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": 512, "height": 512},
    })
    scene = mi.load_dict({
        "type": "scene",
        "shape": {
            "type": "ply",
            "filename": mesh_path,
            "bsdf": {
                "type": "MatDiffBSDF",
                "cam_meta": cam_cfg_path,
                "use_mesh_normal": True,
                "fov": float(fov_deg),
            },
        },
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": camera,
        "emitter": {"type": "envmap", "filename": env_path},
    })
    mat = load_estimated_brdf(mat_dir)
    params = mi.traverse(scene)
    params["shape.bsdf.a"] = mat["albedo"]
    params["shape.bsdf.r"] = mat["roughness"]
    params["shape.bsdf.m"] = mat["metallic"]
    params.update()
    return scene


def render_single(scene, spp=32, use_denoiser=False):
    img = mi.render(scene, spp=spp, seed=0)
    if use_denoiser:
        denoiser = mi.OptixDenoiser(input_size=img.shape[:2], albedo=False, normals=False, temporal=False)
        img = denoiser(img)
    return img.numpy()


def roll_envmap(envmap, angle_degrees):
    width = envmap.shape[1]
    shift_pixels = int((angle_degrees / 360.0) * width) % width
    return np.roll(envmap, shift_pixels, axis=1)


def render_rolling_envmap_for_mesh(mesh_path, mat_dir, env_path, fov_deg, out_dir,
                                   frames=36, rotation_step=10.0, spp=32):
    env_basename = os.path.splitext(os.path.basename(env_path))[0]
    out_dir = os.path.join(out_dir, env_basename)
    os.makedirs(out_dir, exist_ok=True)

    frame_dir = os.path.join(out_dir, "frames")
    os.makedirs(frame_dir, exist_ok=True)

    scene = build_scene(mesh_path, mat_dir, env_path, fov_deg)
    denoiser = mi.OptixDenoiser(input_size=(512, 512), albedo=False, normals=False, temporal=False)

    if float(rotation_step) == 0.0:
        img = mi.render(scene, spp=spp, seed=0)
        img = denoiser(img)
        img_np = img.numpy()
        mi.util.write_bitmap(os.path.join(frame_dir, "frame_0000.exr"), img_np)
        png = linear_to_srgb(img_np)
        imageio.imwrite(
            os.path.join(frame_dir, "frame_0000.png"),
            np.clip(png * 255.0, 0, 255).astype(np.uint8),
        )
        print(f"[Info] Saved single frame at {frame_dir}")
        return frame_dir

    base_env = np.array(mi.Bitmap(env_path))
    params = mi.traverse(scene)

    png_paths = []
    for idx in range(frames):
        angle = idx * rotation_step
        rotated_env = roll_envmap(base_env, angle)
        params["emitter.data"] = rotated_env
        params.update()

        img = mi.render(scene, spp=spp, seed=idx)
        img = denoiser(img)
        img_np = img.numpy()

        mi.util.write_bitmap(os.path.join(frame_dir, f"frame_{idx:04d}.exr"), img_np)
        png = linear_to_srgb(img_np)
        png_path = os.path.join(frame_dir, f"frame_{idx:04d}.png")
        imageio.imwrite(png_path, np.clip(png * 255.0, 0, 255).astype(np.uint8))
        png_paths.append(png_path)

    video_path = os.path.join(out_dir, "rolling.mp4")
    imageio.mimwrite(video_path, [imageio.imread(p) for p in png_paths], format="mp4", fps=10, quality=8)
    return frame_dir


def ablate_jinjya(args):
    input_root = args.input_root
    depth_path = os.path.join(input_root, "depthPred.exr")
    mask_path = os.path.join(input_root, "mesh_mask.png")
    mat_dir = os.path.join(input_root, "best_results")
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    pct = args.pct
    fov_deg = args.baseline_fov * pct
    pct_tag = int(round(pct * 100))
    ply_path = os.path.join(out_dir, f"{pct_tag}_{fov_deg:.2f}.ply")
    reconstruct_mesh(depth_path, mask_path, fov_deg, min_angle=args.min_angle, save_ply_path=ply_path)
    scene = build_scene(ply_path, mat_dir, args.env_path, fov_deg)
    img = render_single(scene, spp=args.spp, use_denoiser=False)
    save_path = os.path.join(out_dir, f"{pct_tag}_{fov_deg:.2f}.exr")
    mi.util.write_bitmap(save_path, img)
    rolling_dir = os.path.join(out_dir, f"{pct_tag}_{fov_deg:.2f}_rolling")
    render_rolling_envmap_for_mesh(
        ply_path,
        mat_dir,
        args.env_path,
        fov_deg,
        rolling_dir,
        frames=args.frames,
        rotation_step=args.rotation_step,
        spp=args.spp,
    )


def _load_depth_exr(depth_path):
    depth = np.array(mi.Bitmap(depth_path))
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def reconstruct_mesh_from_depth_exr(depth_path, fov_deg, min_angle=6.0, depth_scale=1.0, save_ply_path=None):
    depth_np = _load_depth_exr(depth_path)
    cam_matrix = set_recon_camera(fov_deg=fov_deg, width=depth_np.shape[1], height=depth_np.shape[0])
    mesh, _ = depth_file_to_mesh(
        depth_np,
        cameraMatrix=cam_matrix,
        minAngle=min_angle,
        sun3d=False,
        depthScale=depth_scale,
    )
    mesh = rotate_mesh_around_x(mesh, 180)

    if save_ply_path is not None:
        os.makedirs(os.path.dirname(save_ply_path), exist_ok=True)
        o3d.io.write_triangle_mesh(save_ply_path, mesh)
    return mesh


def build_recon_scene(mesh_path, env_path, fov_deg, res=512):
    camera = mi.load_dict({
        "type": "perspective",
        "fov": float(fov_deg),
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]
        ),
        "film": {"type": "hdrfilm", "width": int(res), "height": int(res)},
    })

    return mi.load_dict({
        "type": "scene",
        "shape": {
            "type": "ply",
            "filename": mesh_path,
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.5, 0.5, 0.5]},
            },
        },
        "integrator": {"type": "path", "max_depth": 4},
        "sensor": camera,
        "emitter": {"type": "envmap", "filename": env_path},
    })


def parse_number_list(value, cast=float):
    return [cast(item) for item in value.split(",") if item.strip()]


def recon_mesh_fov_ablation(results_root, scene_ids, fovs, depth_filename="depth.exr",
                            depth_source_fov=35, min_angle=3.0, depth_scale=1.0):
    for scene_id in scene_ids:
        scene_dir = os.path.join(results_root, f"scene{scene_id}")
        depth_dir = os.path.join(scene_dir, f"fov_{int(depth_source_fov)}")
        depth_path = os.path.join(depth_dir, depth_filename)
        if not os.path.exists(depth_path):
            print(f"[Skip] Missing depth: {depth_path}")
            continue

        recon_dir = os.path.join(scene_dir, "recon")
        os.makedirs(recon_dir, exist_ok=True)

        for fov in fovs:
            fov_tag = int(round(float(fov)))
            ply_path = os.path.join(recon_dir, f"fov_{fov_tag}.ply")
            reconstruct_mesh_from_depth_exr(
                depth_path=depth_path,
                fov_deg=float(fov),
                min_angle=float(min_angle),
                depth_scale=float(depth_scale),
                save_ply_path=ply_path,
            )
            print(f"[OK] Saved mesh: {ply_path}")


def render_recon_fov_ablation(results_root, env_root, env_ids, scene_ids, fovs, spp=256, write_png=True):
    for scene_id in scene_ids:
        scene_dir = os.path.join(results_root, f"scene{scene_id}")
        recon_dir = os.path.join(scene_dir, "recon")

        for fov in fovs:
            fov_tag = int(round(float(fov)))
            mesh_path = os.path.join(recon_dir, f"fov_{fov_tag}.ply")
            if not os.path.exists(mesh_path):
                print(f"[Skip] Missing mesh: {mesh_path}")
                continue

            out_dir = os.path.join(recon_dir, f"fov_{fov_tag}")
            os.makedirs(out_dir, exist_ok=True)

            for env_id in tqdm(list(env_ids), desc=f"scene{scene_id} fov{fov_tag} env"):
                env_path = os.path.join(env_root, f"{env_id}.hdr")
                if not os.path.exists(env_path):
                    print(f"[Skip] Missing envmap: {env_path}")
                    continue

                scene = build_recon_scene(mesh_path=mesh_path, env_path=env_path, fov_deg=float(fov), res=512)
                img = mi.render(scene, spp=int(spp), seed=int(env_id)).numpy()

                exr_path = os.path.join(out_dir, f"env_{env_id:02d}.exr")
                mi.util.write_bitmap(exr_path, img)

                if write_png:
                    png = linear_to_srgb(img)
                    png_path = os.path.join(out_dir, f"env_{env_id:02d}.png")
                    imageio.imwrite(png_path, np.clip(png * 255.0, 0, 255).astype(np.uint8))


def parse_args():
    example_root = os.path.join(global_config.BASE_DIR, "examples", "fov_ablation")
    parser = argparse.ArgumentParser(description="FOV ablation utilities.")
    parser.add_argument("--mode", choices=["jinjya", "recon", "render", "batch"], default="jinjya")
    parser.add_argument("--pct", type=float, default=1.0, help="FOV percentage for jinjya mode.")
    parser.add_argument("--baseline_fov", type=float, default=35.0)
    parser.add_argument("--input_root", default=os.path.join(global_config.OUT_DIR, "jinjya"))
    parser.add_argument("--output_dir", default=os.path.join(example_root, "jinjya"))
    parser.add_argument("--env_path", default=os.path.join(global_config.ENVMAP_DIR, "41.hdr"))
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--rotation_step", type=float, default=10.0)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--min_angle", type=float, default=6.0)
    parser.add_argument("--results_root", default=example_root)
    parser.add_argument("--scene_ids", default="0")
    parser.add_argument("--fovs", default="28,35,42")
    parser.add_argument("--depth_source_fov", type=int, default=35)
    parser.add_argument("--env_root", default=global_config.ENVMAP_DIR)
    parser.add_argument("--env_ids", default="41")
    parser.add_argument("--no_png", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "jinjya":
        ablate_jinjya(args)
        return

    scene_ids = parse_number_list(args.scene_ids, int)
    fovs = parse_number_list(args.fovs, float)
    env_ids = parse_number_list(args.env_ids, int)
    if args.mode in ("recon", "batch"):
        recon_mesh_fov_ablation(
            results_root=args.results_root,
            scene_ids=scene_ids,
            fovs=fovs,
            depth_source_fov=args.depth_source_fov,
            min_angle=args.min_angle,
        )
    if args.mode in ("render", "batch"):
        render_recon_fov_ablation(
            results_root=args.results_root,
            env_root=args.env_root,
            env_ids=env_ids,
            scene_ids=scene_ids,
            fovs=fovs,
            spp=args.spp,
            write_png=not args.no_png,
        )


if __name__ == "__main__":
    main()
