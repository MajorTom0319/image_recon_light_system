#!/usr/bin/env python3
"""Render IndoorLightEditing lamps on Materialist geometry and materials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_light.io import load_ile_lights
from hybrid_light.mitsuba_builder import build_hybrid_scene_dict
from hybrid_light.visualization import render_projection_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Render ILE lamp meshes as Mitsuba area emitters on a Materialist scene.",
    )
    parser.add_argument("--materialist-dir", type=Path, required=True)
    parser.add_argument("--lights-json", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--camera-meta", type=Path, default=None)
    parser.add_argument("--material-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["local", "env", "combined"], default="local")
    parser.add_argument("--envmap", type=Path, default=None)
    parser.add_argument("--visible-only", action="store_true")
    parser.add_argument("--invisible-only", action="store_true")
    parser.add_argument(
        "--geometry-scale",
        type=float,
        default=None,
        help="Override ILE-to-Materialist scene scale; normally inferred as 1.",
    )
    parser.add_argument(
        "--radiance-scale",
        type=float,
        default=1.0,
        help="Global calibration multiplier for ILE's relative RGB intensity.",
    )
    parser.add_argument(
        "--visible-offset",
        type=float,
        default=0.005,
        help="Move visible lamp meshes toward the camera to reduce coplanar overlap.",
    )
    parser.add_argument("--use-pred-normal", action="store_true")
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Keep the raw Monte Carlo image instead of applying OptiX denoising.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write projection diagnostics without loading CUDA/Mitsuba.",
    )
    args = parser.parse_args()
    if args.visible_only and args.invisible_only:
        parser.error("--visible-only and --invisible-only are mutually exclusive")
    if args.spp <= 0 or args.max_depth <= 0:
        parser.error("--spp and --max-depth must be positive")
    return args


def _resolve_mesh(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    preferred = root / "mesh_moge2.ply"
    if preferred.is_file():
        return preferred.resolve()
    candidates = sorted(root.glob("*.ply"))
    if len(candidates) == 1:
        return candidates[0].resolve()
    raise FileNotFoundError("Pass --mesh; no unique Materialist PLY was found")


def _find_material_maps(root: Path, explicit: Path | None) -> dict[str, Path]:
    directories = [explicit.expanduser().resolve()] if explicit is not None else []
    directories.extend([root / "best_results", root])
    names = {
        "albedo": ("albedo.exr", "albedoPred.exr"),
        "roughness": ("roughness.exr", "roughnessPred.exr"),
        "metallic": ("metallic.exr", "metallicPred.exr"),
        # "normal": ("normal.exr", "normalPred.exr"),
        "normal": ("normal.exr", "moge2_normal.exr"),
    }
    for directory in directories:
        result = {}
        for key, candidates in names.items():
            result[key] = next(
                (directory / name for name in candidates if (directory / name).is_file()),
                None,
            )
        if all(path is not None for path in result.values()):
            return {key: path.resolve() for key, path in result.items()}
    raise FileNotFoundError(
        "Could not find albedo/roughness/metallic/normal EXRs in --material-dir, "
        "best_results, or the Materialist run directory"
    )


def _load_camera(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        meta = json.load(stream)
    width, height = [int(value) for value in meta["film.size"]]
    K = np.asarray(meta["K"], dtype=np.float32)
    if K.shape != (3, 3) or min(width, height) <= 0:
        raise ValueError("Invalid Materialist camera metadata")
    return meta


def _bitmap_array(mi, path: Path) -> np.ndarray:
    return np.asarray(mi.Bitmap(str(path)), dtype=np.float32)


def _load_material_arrays(mi, paths: dict[str, Path], hw: tuple[int, int]):
    height, width = hw
    arrays = {key: _bitmap_array(mi, path) for key, path in paths.items()}
    for key, value in arrays.items():
        if value.shape[:2] != (height, width):
            raise ValueError(f"{key} map shape {value.shape[:2]} != film {(height, width)}")

    albedo = np.clip(arrays["albedo"][..., :3], 0.0, 1.0)
    roughness = np.clip(np.squeeze(arrays["roughness"]), 0.07, 1.0)[..., None]
    metallic = np.clip(np.squeeze(arrays["metallic"]), 0.0, 1.0)[..., None]
    normal = arrays["normal"][..., :3]
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
    return {
        "a": np.ascontiguousarray(albedo, dtype=np.float32),
        "r": np.ascontiguousarray(roughness, dtype=np.float32),
        "m": np.ascontiguousarray(metallic, dtype=np.float32),
        "n": np.ascontiguousarray(normal, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    materialist_dir = args.materialist_dir.expanduser().resolve()
    if not materialist_dir.is_dir():
        raise NotADirectoryError(materialist_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else materialist_dir / "hybrid_ile_render"
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

    include_visible = not args.invisible_only
    include_invisible = not args.visible_only
    light_set = load_ile_lights(
        args.lights_json,
        include_visible=include_visible,
        include_invisible=include_invisible,
        geometry_scale=args.geometry_scale,
    )
    mesh_path = _resolve_mesh(materialist_dir, args.mesh)
    material_paths = _find_material_maps(materialist_dir, args.material_dir)

    projection_report = []
    if light_set.image_path is not None and light_set.image_path.is_file():
        projection_report = render_projection_debug(
            light_set.image_path,
            light_set.lights,
            np.asarray(camera_meta["K"], dtype=np.float32),
            (width, height),
            output_dir / "lights_projected.png",
            visible_offset=args.visible_offset,
        )

    manifest = {
        "status": "validated" if args.dry_run else "rendering",
        "mode": args.mode,
        "materialist_dir": str(materialist_dir),
        "mesh": str(mesh_path),
        "camera_meta": str(camera_path),
        "materials": {key: str(path) for key, path in material_paths.items()},
        "lights_json": str(light_set.source_path),
        "geometry_scale": light_set.metadata["geometry_scale"],
        "radiance_scale": args.radiance_scale,
        "visible_offset": args.visible_offset,
        "windows_ignored": light_set.metadata["windows_ignored"],
        "lights": projection_report,
    }
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    print(f"Validated {len(light_set.lights)} ILE lamp(s)")
    print(f"Mesh: {mesh_path}")
    print(f"Projection debug: {output_dir / 'lights_projected.png'}")
    if args.dry_run:
        print(f"Dry run complete: {manifest_path}")
        return

    # Importing Materialist's BSDF selects cuda_ad_rgb, so keep it after the
    # dry-run path to allow CPU-only validation of camera/coordinates/files.
    import mitsuba as mi
    from myutils.mi_plugin import MatDiffBSDF

    mi.register_bsdf("MatDiffBSDF", lambda props: MatDiffBSDF(props))
    scene_dict = build_hybrid_scene_dict(
        mi,
        mesh_path=mesh_path,
        camera_meta_path=camera_path,
        camera_meta=camera_meta,
        lights=light_set.lights,
        mode=args.mode,
        envmap_path=args.envmap,
        radiance_scale=args.radiance_scale,
        visible_offset=args.visible_offset,
        use_mesh_normal=not args.use_pred_normal,
        max_depth=args.max_depth,
    )
    scene = mi.load_dict(scene_dict)
    material_arrays = _load_material_arrays(mi, material_paths, (height, width))
    params = mi.traverse(scene)
    for name, array in material_arrays.items():
        key = f"materialist_mesh.bsdf.{name}"
        if key not in params:
            raise KeyError(f"Missing Mitsuba parameter: {key}")
        params[key] = mi.TensorXf(array)
    params.update()

    rendered_raw = mi.render(scene, spp=args.spp, seed=args.seed)
    stem = f"render_{args.mode}"
    raw_exr_path = output_dir / f"{stem}_raw.exr"
    exr_path = output_dir / f"{stem}.exr"
    png_path = output_dir / f"{stem}.png"
    mi.util.write_bitmap(str(raw_exr_path), rendered_raw)

    rendered = rendered_raw
    if not args.no_denoise:
        denoiser = mi.OptixDenoiser(
            input_size=(int(rendered_raw.shape[1]), int(rendered_raw.shape[0])),
            albedo=False,
            normals=False,
            temporal=False,
        )
        rendered = denoiser(rendered_raw)
    mi.util.write_bitmap(str(exr_path), rendered)
    mi.util.write_bitmap(str(png_path), rendered)

    manifest["status"] = "complete"
    manifest["spp"] = args.spp
    manifest["max_depth"] = args.max_depth
    manifest["denoised"] = not args.no_denoise
    manifest["output_raw_exr"] = str(raw_exr_path)
    manifest["output_exr"] = str(exr_path)
    manifest["output_png"] = str(png_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Rendered EXR: {exr_path}")
    print(f"Rendered PNG: {png_path}")


if __name__ == "__main__":
    main()
