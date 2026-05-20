#!/usr/bin/env python
"""Approximate transparency editing.
!IMPORTANT!
This renderer does not trace light through the edited object. Refraction is an
approximation and can be inaccurate under very strong/weak illumination or for
geometrically complex transparent objects.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np
import torch
from tqdm import tqdm

mi.set_variant("cuda_ad_rgb")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import global_config
from myutils.mi_plugin import load_estimated_brdf
from render_final import load_estimated_mesh_w_env

dr.set_flag(dr.JitFlag.VCallRecord, False)
dr.set_flag(dr.JitFlag.LoopRecord, False)


def prepare_material_dir(source_mat_dir, work_mat_dir, env_path=None, bg_path=None, mask_path=None):
    os.makedirs(work_mat_dir, exist_ok=True)
    for filename in ("albedo.exr", "roughness.exr", "metallic.exr", "normal.exr"):
        shutil.copyfile(os.path.join(source_mat_dir, filename), os.path.join(work_mat_dir, filename))
    if env_path is not None:
        shutil.copyfile(env_path, os.path.join(work_mat_dir, "envmap.hdr"))
    if bg_path is not None:
        ext = os.path.splitext(bg_path)[1].lower()
        shutil.copyfile(bg_path, os.path.join(work_mat_dir, "bg.png" if ext == ".png" else "bg.exr"))
    if mask_path is not None:
        shutil.copyfile(mask_path, os.path.join(work_mat_dir, "mask.png"))
    return work_mat_dir


def render_transparency_edit(mesh_path, mat_dir, env_path, output_path, ior=1.1,
                             n_iter=10, keep_albedo_color=False, specTrans=0.95):
    scene = load_estimated_mesh_w_env(
        mesh_path,
        env_path,
        use_mesh_normal=True,
        bsdf={"name": "TransBSDF", "ior": ior, "keep_albedo_color": keep_albedo_color},
    )
    mat = load_estimated_brdf(mat_dir)
    mask = mat["mask"]
    albedo = mat["albedo"]
    roughness = mat["roughness"]
    metallic = mat["metallic"]

    if not keep_albedo_color:
        albedo[mask] = 0.9
    roughness[mask] = roughness[mask] * 0 + 0.5
    metallic[mask] = metallic[mask] * 0

    params = mi.traverse(scene)
    params["shape.bsdf.a"] = albedo
    params["shape.bsdf.r"] = roughness
    params["shape.bsdf.m"] = metallic
    params["emitter.data"] = mat["envmap"]
    params["shape.bsdf.bg"] = mat["bg"]
    params["shape.bsdf.mask"] = mi.TensorXf(mat["mask"].float()) >= 1
    params["shape.bsdf.specTrans"] = specTrans
    params["shape.bsdf.ior"] = ior
    params.update()

    image_accum = mi.TensorXf(np.zeros((512, 512, 3), dtype=np.float32))
    for seed in tqdm(range(n_iter), desc="Rendering"):
        image_accum += mi.render(scene, spp=64, seed=seed)
    image = image_accum / n_iter

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mi.util.write_bitmap(output_path, image)
    print(f"Saved {output_path}")


def parse_args():
    example_root = os.path.join(global_config.BASE_DIR, "examples", "edit_transparency", "jug")
    output_root = os.path.join(global_config.BASE_DIR, "examples", "edit_transparency", "jug", "rendered")
    parser = argparse.ArgumentParser(description="Approximate transparency-edit ablation renderer.")
    parser.add_argument("--mesh_path", default=os.path.join(example_root, "jug.ply"))
    parser.add_argument("--material_dir", default=os.path.join(example_root, "best_results"))
    parser.add_argument("--work_mat_dir", default=os.path.join(output_root, "best_results"))
    parser.add_argument("--env_path", default=os.path.join(global_config.ENVMAP_DIR, "57.hdr"))
    parser.add_argument("--bg_path", default=os.path.join(example_root, "bg.png"))
    parser.add_argument("--mask_path", default=os.path.join(example_root, "mask.png"))
    parser.add_argument("--output_path", default=os.path.join(output_root, "edited_1.1.exr"))
    parser.add_argument("--ior", type=float, default=1.1)
    parser.add_argument("--specTrans", type=float, default=0.95)
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--keep_albedo_color", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    mat_dir = prepare_material_dir(args.material_dir, args.work_mat_dir, args.env_path, args.bg_path, args.mask_path)
    render_transparency_edit(
        args.mesh_path,
        mat_dir,
        args.env_path,
        args.output_path,
        ior=args.ior,
        n_iter=args.n_iter,
        keep_albedo_color=args.keep_albedo_color,
        specTrans=args.specTrans,
    )


if __name__ == "__main__":
    main()
