#!/usr/bin/env python
import argparse
import os
import sys
from pathlib import Path

import mitsuba as mi
import numpy as np
import open3d as o3d

mi.set_variant("cuda_ad_rgb")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import global_config
from myutils.mesh_recon import depth_file_to_mesh, rotate_mesh_around_x, rotate_pc_around_x


DEFAULT_RADII = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3]
DEFAULT_MIN_ANGLES = [i * 0.1 for i in range(1, 11)] + list(range(2, 11, 2))


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def angle_tag(min_angle):
    return f"{min_angle:.1f}" if min_angle < 1 else f"{int(min_angle)}"


def reconstruct_depth(depth_path, output_dir, min_angles):
    depth_path = Path(depth_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading depth from {depth_path}")
    depth_np = np.array(mi.Bitmap(str(depth_path)))

    print(f"Depth shape: {depth_np.shape}")
    print(f"Depth range: [{depth_np.min()}, {depth_np.max()}]")
    print(f"\nTotal number of minAngle values to test: {len(min_angles)}")
    print(f"minAngle values: {min_angles}\n")

    for idx, min_angle in enumerate(min_angles, 1):
        print(f"\n{'=' * 60}")
        print(f"Processing [{idx}/{len(min_angles)}] minAngle = {min_angle}")
        print(f"{'=' * 60}")

        mesh, b_points = depth_file_to_mesh(
            depth_np,
            cameraMatrix=None,
            minAngle=float(min_angle),
            sun3d=False,
            depthScale=1.0,
        )
        mesh = rotate_mesh_around_x(mesh, 180)

        tag = angle_tag(float(min_angle))
        mesh_path = output_dir / f"sphere_minAngle_{tag}.ply"
        o3d.io.write_triangle_mesh(str(mesh_path), mesh)
        print(f"Saved mesh to {mesh_path}")

        if len(b_points.points) > 0:
            b_points = rotate_pc_around_x(b_points, 180)
            b_points_path = output_dir / f"sphere_minAngle_{tag}_boundary.ply"
            o3d.io.write_point_cloud(str(b_points_path), b_points)
            print(f"Saved boundary points to {b_points_path}")

    print("\n" + "=" * 60)
    print("Ablation study completed!")
    print(f"Total meshes generated: {len(min_angles)}")
    print("=" * 60)


def load_estimate_scene(mesh_path):
    camera_cfg = {
        "type": "perspective",
        "fov": 35,
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[0, 0, 0], target=[0, 0, -1], up=[0, 1, 0]
        ),
        "film": {"type": "hdrfilm", "width": 512, "height": 512},
    }
    camera = mi.load_dict(camera_cfg)

    return mi.load_dict({
        "type": "scene",
        "shape": {
            "type": "ply",
            "filename": str(mesh_path),
            "bsdf": {
                "type": "diffuse",
                "reflectance": {"type": "rgb", "value": [0.8, 0.8, 0.8]},
            },
        },
        "integrator": {"type": "path"},
        "sensor": camera,
        "emitter": {
            "type": "constant",
            "radiance": {"type": "rgb", "value": [1.0, 1.0, 1.0]},
        },
    })


def run_bd_ablation_recon(results_root, radii, min_angles):
    for radius in radii:
        depth_path = os.path.join(results_root, "cylinder_render", f"depth_{radius:.3f}.exr")
        output_dir = os.path.join(results_root, f"cylinder_recon_{radius:.3f}")
        reconstruct_depth(depth_path, output_dir, min_angles)


def render_recon_mesh(results_root, radii, min_angles, spp=256):
    for radius in radii:
        for min_angle in min_angles:
            tag = angle_tag(float(min_angle))
            mesh_path = os.path.join(results_root, f"cylinder_recon_{radius:.3f}", f"sphere_minAngle_{tag}.ply")
            output_path = os.path.join(results_root, f"cylinder_recon_{radius:.3f}", f"render_sphere_minAngle_{tag}.png")
            if not os.path.exists(mesh_path):
                print(f"Skip missing mesh: {mesh_path}")
                continue
            scene = load_estimate_scene(mesh_path)
            img = mi.render(scene, sensor=scene.sensors()[0], spp=spp)
            mi.util.write_bitmap(output_path, img)
            print(f"Saved rendered image to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Boundary duplication ablation reconstruction/rendering.")
    parser.add_argument(
        "--results_root",
        default=os.path.join(global_config.BASE_DIR, "examples", "bd_ablation"),
        help="Root containing cylinder_render/depth_*.exr and output cylinder_recon_* folders.",
    )
    parser.add_argument("--radii", default="0.030", help="Comma-separated cylinder radii.")
    parser.add_argument("--min_angles", default="0.1,0.5,1,4,8", help="Comma-separated minAngle values.")
    parser.add_argument("--mode", choices=["reconstruct", "render", "both"], default="both")
    parser.add_argument("--spp", type=int, default=256, help="Samples per pixel for preview renders.")
    return parser.parse_args()


def main():
    args = parse_args()
    radii = parse_float_list(args.radii)
    min_angles = parse_float_list(args.min_angles)
    if args.mode in ("reconstruct", "both"):
        run_bd_ablation_recon(args.results_root, radii, min_angles)
    if args.mode in ("render", "both"):
        render_recon_mesh(args.results_root, radii, min_angles, spp=args.spp)


if __name__ == "__main__":
    main()
