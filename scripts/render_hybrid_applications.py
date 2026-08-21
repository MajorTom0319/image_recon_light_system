#!/usr/bin/env python3
"""Render relighting, explicit-light edits, and object insertion.

The optimization MLP is not needed here.  This entry reconstructs the PBR
scene from a completed ``optimize_ile_farfield_materials.py`` run, loads its
selected ARM/normal maps and far-field EXR, applies relative scales to the ILE
lamps/windows, optionally inserts pre-positioned PLY/OBJ meshes, and performs a
plain forward path-tracing render.

Examples::

    # Re-render the optimized scene.
    python scripts/render_hybrid_applications.py --run-dir RUN

    # Turn off the visible window and dim all lamps to 40 percent.
    python scripts/render_hybrid_applications.py --run-dir RUN \
        --light-scale windows=0 --light-scale lamps=0.4

    # Replace the HDRI and insert a mesh already expressed in scene coordinates.
    python scripts/render_hybrid_applications.py --run-dir RUN \
        --envmap envmaps/night.exr --insert-object assets/oi.ply

Light-scale targets are ``all``, ``lamps``, ``windows``, or an exact light
name such as ``visible_lamp_0`` or ``invisible_window_0``.  Rules are applied
from left to right and the last matching rule wins.  A scale of zero disables
that emitter while retaining its geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hybrid_light.io import load_ile_lights
from hybrid_light.light_types import MeshAreaLight
from hybrid_light.mitsuba_builder import build_hybrid_scene_dict
from scripts.render_ile_lights import (
    _find_material_maps,
    _load_camera,
    _load_material_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Forward-render applications from an optimized hybrid-light run."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Completed far-field/material optimization output directory.",
    )
    parser.add_argument(
        "--envmap",
        type=Path,
        default=None,
        help="Replacement linear HDR/EXR; defaults to the optimized far field.",
    )
    parser.add_argument(
        "--lights-json",
        type=Path,
        default=None,
        help="Override the explicit-light JSON recorded by the optimization.",
    )
    parser.add_argument(
        "--light-scale",
        action="append",
        default=[],
        metavar="TARGET=SCALE",
        help=(
            "Relative light brightness. TARGET is all, lamps, windows, or an "
            "exact light name; repeat for multiple edits and use 0 to turn off."
        ),
    )
    parser.add_argument(
        "--insert-object",
        type=Path,
        action="append",
        default=[],
        help=(
            "Pre-positioned PLY/OBJ mesh to insert with a neutral rough-plastic "
            "BSDF; may be repeated."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--spp", type=int, default=512)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write the application manifest without loading Mitsuba.",
    )
    args = parser.parse_args()
    if args.spp <= 0:
        parser.error("--spp must be positive")
    return args


def parse_light_scale_rules(values: Sequence[str]) -> list[tuple[str, float]]:
    """Parse ordered ``TARGET=SCALE`` rules without binding them to a scene."""
    rules: list[tuple[str, float]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --light-scale {value!r}; expected TARGET=SCALE"
            )
        target, raw_scale = (part.strip() for part in value.split("=", 1))
        if not target:
            raise ValueError("Light-scale target cannot be empty")
        try:
            scale = float(raw_scale)
        except ValueError as error:
            raise ValueError(f"Invalid light scale in {value!r}") from error
        if not math.isfinite(scale) or scale < 0:
            raise ValueError(f"Light scale must be finite and non-negative: {value!r}")
        rules.append((target, scale))
    return rules


def apply_light_scale_rules(
    lights: Sequence[MeshAreaLight],
    rules: Sequence[tuple[str, float]],
) -> dict[str, float]:
    """Apply relative scales in place and return each light's final factor."""
    factors = {light.name: 1.0 for light in lights}
    available = sorted(factors)
    for target, scale in rules:
        if target == "all":
            matches = list(lights)
        elif target == "lamps":
            matches = [light for light in lights if not light.is_window]
        elif target == "windows":
            matches = [light for light in lights if light.is_window]
        else:
            matches = [light for light in lights if light.name == target]
        if not matches:
            raise ValueError(
                f"Light target {target!r} matched nothing; available: {available}"
            )
        for light in matches:
            factors[light.name] = scale

    for light in lights:
        factor = np.float32(factors[light.name])
        light.rgb = np.asarray(light.rgb, dtype=np.float32) * factor
        if light.is_window:
            if light.window_lobes is None:
                raise ValueError(f"Window {light.name} has no SG lobes")
            light.window_lobes = np.asarray(
                light.window_lobes, dtype=np.float32
            ).copy()
            light.window_lobes[:, :3] *= factor
    return factors


def _required_path(value: str | Path | None, label: str) -> Path:
    if value is None:
        raise ValueError(f"Optimization manifest does not record {label}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def _load_run(run_dir: Path) -> tuple[dict[str, Any], Path]:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    manifest_path = run_dir / "optimization_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("status") != "complete":
        raise ValueError(f"Optimization run is not complete: {manifest_path}")
    return manifest, run_dir


def _validate_objects(paths: Sequence[Path]) -> list[Path]:
    objects: list[Path] = []
    for value in paths:
        path = value.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in {".ply", ".obj"}:
            raise ValueError(f"Inserted object must be PLY or OBJ: {path}")
        objects.append(path)
    return objects


def _add_inserted_objects(
    scene_dict: dict[str, Any], objects: Sequence[Path]
) -> None:
    # Match Materialist's original IO example, but accept any number of meshes.
    for index, path in enumerate(objects):
        scene_dict[f"insert_object_{index}"] = {
            "type": path.suffix.lower()[1:],
            "filename": str(path),
            "bsdf": {
                "type": "roughplastic",
                "alpha": 0.3,
                "int_ior": 1.49,
                "diffuse_reflectance": {
                    "type": "rgb",
                    "value": [0.8, 0.8, 0.8],
                },
            },
        }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_manifest, run_dir = _load_run(args.run_dir)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "applications" / "relight"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dir = run_dir / "best_results"
    if not best_dir.is_dir():
        raise FileNotFoundError(best_dir)
    materialist_dir = Path(source_manifest["materialist_dir"]).expanduser().resolve()
    mesh_path = _required_path(source_manifest.get("mesh"), "mesh")
    camera_path = _required_path(source_manifest.get("camera_meta"), "camera metadata")
    camera_meta = _load_camera(camera_path)
    material_paths = _find_material_maps(materialist_dir, best_dir)

    # default_envmap = run_dir / "farfield_initial_32x16.exr"
    default_envmap = run_dir / "farfield_optimized_32x16.exr"
    
    envmap_path = _required_path(
        args.envmap if args.envmap is not None else default_envmap,
        "environment map",
    )
    lights_path = _required_path(
        args.lights_json
        if args.lights_json is not None
        else source_manifest.get("lights_json"),
        "explicit-light JSON",
    )
    objects = _validate_objects(args.insert_object)
    light_set = load_ile_lights(
        lights_path,
        include_windows=True,
        geometry_scale=source_manifest.get("geometry_scale"),
    )
    rules = parse_light_scale_rules(args.light_scale)
    light_factors = apply_light_scale_rules(light_set.lights, rules)

    base_radiance_scale = float(
        source_manifest.get("fixed_local_radiance_scale", 1.0)
    )
    if not math.isfinite(base_radiance_scale) or base_radiance_scale <= 0:
        raise ValueError("Recorded fixed_local_radiance_scale must be positive")
    max_depth = int(
        (source_manifest.get("optimization") or {}).get("max_depth", 8)
    )
    seed = int(
        (source_manifest.get("materials") or {}).get("final_seed", 500000)
    )
    use_mesh_normal = bool(source_manifest.get("use_mesh_normal", False))

    application_manifest: dict[str, Any] = {
        "status": "validated" if args.dry_run else "rendering",
        "source_run": str(run_dir),
        "source_manifest": str(run_dir / "optimization_manifest.json"),
        "mesh": str(mesh_path),
        "camera_meta": str(camera_path),
        "materials": {key: str(path) for key, path in material_paths.items()},
        "envmap": str(envmap_path),
        "lights_json": str(lights_path),
        "geometry_scale": light_set.metadata["geometry_scale"],
        "base_radiance_scale": base_radiance_scale,
        "relative_light_scales": light_factors,
        "effective_light_scales": {
            name: base_radiance_scale * factor
            for name, factor in light_factors.items()
        },
        "inserted_objects": [str(path) for path in objects],
        "inserted_object_bsdf": "roughplastic(alpha=0.3, int_ior=1.49)",
        "inserted_object_coordinates": "pre-positioned scene coordinates",
        "use_mesh_normal": use_mesh_normal,
        "integrator": "path",
        "max_depth": max_depth,
        "spp": args.spp,
        "seed": seed,
    }
    manifest_path = output_dir / "application_manifest.json"
    _write_json(manifest_path, application_manifest)

    print("Explicit lights:")
    for light in light_set.lights:
        print(f"  {light.name}: scale={light_factors[light.name]:g}")
    if args.dry_run:
        print(f"Dry run complete: {manifest_path}")
        return

    import mitsuba as mi
    from myutils.mi_plugin import MatDiffBSDF

    mi.register_bsdf("MatDiffBSDF", lambda props: MatDiffBSDF(props))
    scene_dict = build_hybrid_scene_dict(
        mi,
        mesh_path=mesh_path,
        camera_meta_path=camera_path,
        camera_meta=camera_meta,
        lights=light_set.lights,
        mode="combined",
        envmap_path=envmap_path,
        radiance_scale=base_radiance_scale,
        use_mesh_normal=use_mesh_normal,
        max_depth=max_depth,
    )
    _add_inserted_objects(scene_dict, objects)
    scene = mi.load_dict(scene_dict)

    width, height = [int(value) for value in camera_meta["film.size"]]
    material_arrays = _load_material_arrays(
        mi, material_paths, (height, width)
    )
    params = mi.traverse(scene)
    for name, array in material_arrays.items():
        key = f"materialist_mesh.bsdf.{name}"
        if key not in params:
            raise KeyError(f"Missing Mitsuba parameter: {key}")
        params[key] = mi.TensorXf(array)
    params.update()

    rendered_raw = mi.render(scene, spp=args.spp, seed=seed)
    raw_path = output_dir / "render_raw.exr"
    mi.util.write_bitmap(str(raw_path), rendered_raw)

    denoiser = mi.OptixDenoiser(
        input_size=(int(rendered_raw.shape[1]), int(rendered_raw.shape[0])),
        albedo=False,
        normals=False,
        temporal=False,
    )
    rendered_denoised = denoiser(rendered_raw)
    denoised_exr_path = output_dir / "render_denoised.exr"
    denoised_png_path = output_dir / "render_denoised.png"
    mi.util.write_bitmap(str(denoised_exr_path), rendered_denoised)
    mi.util.write_bitmap(str(denoised_png_path), rendered_denoised)

    application_manifest["status"] = "complete"
    application_manifest["outputs"] = {
        "raw_exr": str(raw_path),
        "denoised_exr": str(denoised_exr_path),
        "denoised_png": str(denoised_png_path),
    }
    _write_json(manifest_path, application_manifest)
    print(f"Application render complete: {output_dir}")


if __name__ == "__main__":
    main()
