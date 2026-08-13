#!/usr/bin/env python3
"""
how to use:
python flux2_opt.py --offline


Turn a PBR render into a photograph with FLUX.2 Klein Base 4B.

Two spatially aligned references are used by default:

1. a linear-HDR PBR color render, converted to display sRGB;
2. its signed ``[-1, 1]`` surface-normal EXR, converted to a normal-map preview.

The color render supplies appearance, lighting, and composition.  The normal
map is an auxiliary geometry constraint.  The prompt explicitly tells FLUX not
to copy its false colors into the photographic output.

The model is downloaded once to ``DEFAULT_MODEL_DIR``.  Diffusers then runs in
offline/local-only mode, so later invocations do not silently use another Hub
revision or put weights in Hugging Face's default cache directory.

Recommended environment (the repository's ``materialist5090`` conda env):

    /home/majortom/miniconda3/envs/materialist5090/bin/pip install \
        "diffusers>=0.39.0,<0.40" "transformers>=4.51,<6" \
        "accelerate>=0.31.0" "safetensors>=0.8.0"

Run with all defaults:

    /home/majortom/miniconda3/envs/materialist5090/bin/python flux2_opt.py

Download only:

    /home/majortom/miniconda3/envs/materialist5090/bin/python \
        flux2_opt.py --download-only

FLUX.2 Klein does reference-image editing, not classic SDEdit img2img.  There is
therefore no ``strength`` argument in this pipeline.  Reference fidelity is
controlled mainly by the instruction-style prompt and, secondarily, guidance.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

# OpenCV checks this variable when the module is imported.  It must be set
# before the lazy ``import cv2`` in load_render_rgb().
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
DEFAULT_MODEL_DIR = Path(
    "/home/majortom/project/datasets/ckpt/FLUX.2-klein-base-4B"
)
DEFAULT_INPUT_PATH = Path(
    "/home/majortom/project/Materialist/output_imgs/indoorlight_example1_ile2/best_results/rendered_img_hq.exr"
)
DEFAULT_NORMAL_PATH = Path(
    "/home/majortom/project/Materialist/output_imgs/indoorlight_example1_ile2/best_results/normal.exr"
)
DEFAULT_OUTPUT_PATH = Path(
    "/home/majortom/project/Materialist/output_imgs/indoorlight_example1_ile2/best_results/flux2_photoreal_with_normal.png"
)

# Image 1 defines the complete target scene; image 2 only helps retain geometry.
# Keep this instruction deliberately compact so FLUX focuses on subtle detail
# restoration instead of redesigning an already accurate PBR reconstruction.
DEFAULT_PROMPT = """Convert reference image 1, the PBR-rendered classroom, into
a natural realistic photograph of the same scene. Keep its composition, camera,
geometry, objects, lighting, shadows, brightness, color palette, and overall
color feeling unchanged. Only improve the missing photographic details: realistic
material texture, subtle surface imperfections, natural roughness, fine edges,
contact details, and mild camera realism. Reference image 2 is the aligned normal
map and should only help preserve shapes and boundaries; do not copy its colors.
Do not add, remove, move, or redesign anything. The result must remain very close
to reference image 1, but look like a clean real photo instead of a CGI render.""".replace(
    "\n", " "
)

# Download only the Diffusers-format components.  The model repository also
# carries a redundant single-file checkpoint and demonstration JPEGs; neither
# is used by Flux2KleinPipeline.
DIFFUSERS_MODEL_PATTERNS = (
    "model_index.json",
    "scheduler/**",
    "text_encoder/**",
    "tokenizer/**",
    "transformer/**",
    "vae/**",
    "README.md",
    "LICENSE.md",
)


@dataclass(frozen=True)
class ExposureInfo:
    """Information needed to reproduce the EXR-to-sRGB conversion."""

    source_is_hdr: bool
    exposure_ev: float
    auto_exposure: bool
    auto_exposure_percentile: float
    target_luminance: float
    measured_luminance: float | None
    tone_mapper: str
    linear_min: float
    linear_max: float


@dataclass(frozen=True)
class NormalInfo:
    """Information needed to reproduce the normal-map conversion."""

    requested_encoding: str
    detected_encoding: str
    source_min: float
    source_max: float
    invalid_value_count: int
    zero_vector_count: int
    length_before_p01: float
    length_before_median: float
    length_before_p99: float


def _dependency_error(package: str, detail: Exception | None = None) -> RuntimeError:
    suffix = f"\nOriginal import error: {detail}" if detail is not None else ""
    return RuntimeError(
        f"Required package '{package}' is not available. Install the FLUX "
        "dependencies with:\n"
        "/home/majortom/miniconda3/envs/materialist5090/bin/pip install "
        "'diffusers>=0.39.0,<0.40' 'transformers>=4.51,<6' "
        "'accelerate>=0.31.0' 'safetensors>=0.8.0'"
        f"{suffix}"
    )


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _dependency_error("numpy", exc) from exc
    return np


def _import_pil_image() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _dependency_error("Pillow", exc) from exc
    return Image


def _has_complete_safetensors(directory: Path) -> bool:
    """Check finalized weight files, including every shard named by an index."""

    index_files = tuple(directory.glob("*.safetensors.index.json"))
    if index_files:
        for index_path in index_files:
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                shard_names = set(index.get("weight_map", {}).values())
            except (OSError, json.JSONDecodeError):
                return False
            if not shard_names or not all(
                (directory / shard_name).is_file() for shard_name in shard_names
            ):
                return False
        return True
    return any(directory.glob("*.safetensors"))


def download_model(
    model_id: str,
    model_dir: Path,
    *,
    offline: bool = False,
) -> Path:
    """Ensure that a complete Diffusers snapshot exists in ``model_dir``.

    ``snapshot_download`` resumes interrupted downloads and uses Hugging Face's
    integrity/ETag checks.  ``local_dir`` makes every model file live under the
    requested checkpoint directory rather than the default user cache.
    """

    model_dir = model_dir.expanduser().resolve()
    required_files = (
        model_dir / "model_index.json",
        model_dir / "scheduler" / "scheduler_config.json",
        model_dir / "text_encoder" / "config.json",
        model_dir / "tokenizer" / "tokenizer_config.json",
        model_dir / "transformer" / "config.json",
        model_dir / "vae" / "config.json",
    )
    weight_dirs = (
        model_dir / "text_encoder",
        model_dir / "transformer",
        model_dir / "vae",
    )
    # A Hub download creates configs and directories before large shards finish.
    # Require at least one finalized safetensors file in every weighted module
    # so an interrupted download is resumed instead of mistaken for complete.
    looks_complete = all(path.is_file() for path in required_files) and all(
        directory.is_dir() and _has_complete_safetensors(directory)
        for directory in weight_dirs
    )
    if looks_complete:
        print(f"[model] Using existing local model: {model_dir}")
        return model_dir

    if offline:
        raise FileNotFoundError(
            f"The local model is incomplete at {model_dir}, but --offline was set."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _dependency_error("huggingface_hub", exc) from exc

    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"[model] Downloading {model_id}")
    print(f"[model] Destination: {model_dir}")
    snapshot_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        local_dir=str(model_dir),
        allow_patterns=list(DIFFUSERS_MODEL_PATTERNS),
    )
    resolved = Path(snapshot_path).resolve()
    if not all(path.is_file() for path in required_files) or not all(
        directory.is_dir() and _has_complete_safetensors(directory)
        for directory in weight_dirs
    ):
        raise RuntimeError(f"Downloaded Diffusers snapshot is incomplete: {resolved}")
    return resolved


def _read_hdr_rgb(path: Path) -> Any:
    """Read an EXR/HDR/PFM file as float32 RGB without changing its values."""

    np = _import_numpy()
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _dependency_error("opencv-python", exc) from exc

    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise RuntimeError(
            f"OpenCV could not decode HDR input {path}. For EXR, ensure "
            "OPENCV_IO_ENABLE_OPENEXR=1 and OpenCV was built with OpenEXR."
        )
    if bgr.ndim == 2:
        rgb = np.repeat(bgr[..., None], 3, axis=2)
    elif bgr.shape[2] == 4:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)
    elif bgr.shape[2] == 3:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported HDR channel count: shape={bgr.shape}")
    return np.asarray(rgb, dtype=np.float32)


def load_render_rgb(path: Path) -> tuple[Any, bool]:
    """Load an image as float32 RGB and report whether it is scene-linear HDR.

    EXR/HDR/PFM inputs are treated as linear radiance.  Common display image
    formats are decoded with Pillow and normalized to [0, 1] sRGB.
    """

    np = _import_numpy()
    Image = _import_pil_image()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {path}")

    is_hdr = path.suffix.lower() in {".exr", ".hdr", ".pfm"}
    if is_hdr:
        rgb = _read_hdr_rgb(path)
    else:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    non_finite = int(rgb.size - np.isfinite(rgb).sum())
    if non_finite:
        print(f"[input] Replacing {non_finite} NaN/Inf values with zero.")
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    if is_hdr:
        negative = int((rgb < 0.0).sum())
        if negative:
            print(f"[input] Clamping {negative} negative radiance values to zero.")
        rgb = np.maximum(rgb, 0.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)
    return rgb, is_hdr


def load_normal_rgb8(path: Path, encoding: str = "auto") -> tuple[Any, NormalInfo]:
    """Decode a data normal map and return its conventional RGB visualization.

    Signed normals are expected in ``[-1, 1]``.  Unsigned normals are expected
    in ``[0, 1]`` and are decoded with ``n = 2 * value - 1``.  Each vector is
    normalized after decoding; zero or invalid vectors become the neutral
    front-facing normal ``(0, 0, 1)``.

    No tone mapping or sRGB transfer function is applied: a normal map is data,
    not scene-linear color.  Directly storing ``(n + 1) / 2`` as RGB bytes gives
    FLUX the familiar false-color normal-map representation.
    """

    np = _import_numpy()
    Image = _import_pil_image()
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Normal image does not exist: {path}")
    if encoding not in {"auto", "signed", "unsigned"}:
        raise ValueError(f"Unsupported normal encoding: {encoding}")

    is_hdr = path.suffix.lower() in {".exr", ".hdr", ".pfm"}
    if is_hdr:
        source = _read_hdr_rgb(path)
    else:
        with Image.open(path) as image:
            source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

    invalid_count = int(source.size - np.isfinite(source).sum())
    source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
    source_min = float(source.min())
    source_max = float(source.max())
    if encoding == "auto":
        # Tolerate small interpolation/quantization overshoots around [0, 1].
        detected_encoding = (
            "signed" if source_min < -0.01 or source_max > 1.01 else "unsigned"
        )
    else:
        detected_encoding = encoding

    if detected_encoding == "signed":
        normals = np.clip(source, -1.0, 1.0)
    else:
        normals = np.clip(source, 0.0, 1.0) * 2.0 - 1.0

    lengths = np.linalg.norm(normals, axis=2)
    valid = lengths > 1e-6
    zero_count = int(valid.size - valid.sum())
    valid_lengths = lengths[valid]
    if not valid_lengths.size:
        raise ValueError(f"Normal image contains no non-zero vectors: {path}")
    length_percentiles = np.percentile(valid_lengths, (1.0, 50.0, 99.0))

    normalized = np.zeros_like(normals, dtype=np.float32)
    normalized[..., 2] = 1.0
    normalized[valid] = normals[valid] / lengths[valid, None]
    encoded = np.clip(normalized * 0.5 + 0.5, 0.0, 1.0)
    rgb8 = np.rint(encoded * 255.0).astype(np.uint8)
    info = NormalInfo(
        requested_encoding=encoding,
        detected_encoding=detected_encoding,
        source_min=source_min,
        source_max=source_max,
        invalid_value_count=invalid_count,
        zero_vector_count=zero_count,
        length_before_p01=float(length_percentiles[0]),
        length_before_median=float(length_percentiles[1]),
        length_before_p99=float(length_percentiles[2]),
    )
    return rgb8, info


def _linear_to_srgb(linear: Any) -> Any:
    np = _import_numpy()
    linear = np.maximum(linear, 0.0)
    return np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def _aces_fitted(linear: Any) -> Any:
    """Narkowicz ACES approximation, applied independently per RGB channel."""

    np = _import_numpy()
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((linear * (a * linear + b)) / (linear * (c * linear + d) + e), 0, 1)


def display_map(
    rgb: Any,
    *,
    source_is_hdr: bool,
    exposure_ev: float | None,
    auto_exposure: bool,
    auto_exposure_percentile: float,
    target_luminance: float,
    tone_mapper: str,
) -> tuple[Any, ExposureInfo]:
    """Convert float RGB to a display-referred uint8 sRGB reference image."""

    np = _import_numpy()
    rgb = np.asarray(rgb, dtype=np.float32)
    measured_luminance: float | None = None
    effective_ev = 0.0 if exposure_ev is None else float(exposure_ev)
    used_auto = bool(source_is_hdr and auto_exposure and exposure_ev is None)

    if source_is_hdr:
        luminance = (
            0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        )
        positive = luminance[np.isfinite(luminance) & (luminance > 1e-6)]
        if positive.size:
            measured_luminance = float(np.percentile(positive, auto_exposure_percentile))
        if used_auto and measured_luminance is not None:
            effective_ev = math.log2(target_luminance / max(measured_luminance, 1e-6))

        exposed = rgb * (2.0**effective_ev)
        if tone_mapper == "aces":
            mapped_linear = _aces_fitted(exposed)
        elif tone_mapper == "reinhard":
            mapped_linear = exposed / (1.0 + exposed)
        elif tone_mapper == "clip":
            mapped_linear = np.clip(exposed, 0.0, 1.0)
        else:  # protected by argparse choices
            raise ValueError(f"Unknown tone mapper: {tone_mapper}")
        display_rgb = _linear_to_srgb(mapped_linear)
    else:
        # LDR images read through Pillow are already display-referred sRGB.
        display_rgb = rgb

    display_rgb = np.clip(display_rgb, 0.0, 1.0)
    rgb8 = np.rint(display_rgb * 255.0).astype(np.uint8)
    info = ExposureInfo(
        source_is_hdr=source_is_hdr,
        exposure_ev=float(effective_ev),
        auto_exposure=used_auto,
        auto_exposure_percentile=float(auto_exposure_percentile),
        target_luminance=float(target_luminance),
        measured_luminance=measured_luminance,
        tone_mapper=tone_mapper if source_is_hdr else "none (input was already sRGB)",
        linear_min=float(rgb.min()),
        linear_max=float(rgb.max()),
    )
    return rgb8, info


def resize_reference(rgb8: Any, long_edge: int, multiple: int = 16) -> Any:
    """Resize without cropping, preserve aspect ratio, and align VAE dimensions."""

    Image = _import_pil_image()
    if long_edge < multiple:
        raise ValueError(f"--long-edge must be at least {multiple}, got {long_edge}")
    image = Image.fromarray(rgb8, mode="RGB")
    width, height = image.size
    scale = long_edge / max(width, height)
    new_width = max(multiple, int(round(width * scale / multiple)) * multiple)
    new_height = max(multiple, int(round(height * scale / multiple)) * multiple)
    # Klein automatically limits reference images to one megapixel.  Keeping the
    # requested output at or below that area avoids an unexpected internal crop.
    if new_width * new_height > 1024 * 1024:
        area_scale = math.sqrt((1024 * 1024) / (new_width * new_height))
        new_width = max(multiple, int(math.floor(new_width * area_scale / multiple)) * multiple)
        new_height = max(multiple, int(math.floor(new_height * area_scale / multiple)) * multiple)
    if (new_width, new_height) != image.size:
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


def _save_pil(image: Any, path: Path) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(path, quality=95, subsampling=0)
    elif suffix == ".png":
        image.save(path, compress_level=4)
    else:
        raise ValueError("Output must use .png, .jpg, or .jpeg (FLUX output is LDR).")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for package in ("torch", "diffusers", "transformers", "accelerate", "huggingface_hub"):
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[package] = "not installed"
    return versions


def load_pipeline(
    model_dir: Path,
    memory_mode: str,
    gpu_id: int,
    reference_count: int = 1,
) -> tuple[Any, Any]:
    """Load Flux2KleinPipeline in BF16 and place/offload it as requested."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _dependency_error("torch", exc) from exc
    try:
        from diffusers import Flux2KleinPipeline
    except (ImportError, RuntimeError) as exc:  # pragma: no cover
        raise _dependency_error("diffusers>=0.39.0", exc) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this BF16 FLUX.2 inference script.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected CUDA GPU does not support bfloat16.")
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        raise ValueError(f"Invalid --gpu-id {gpu_id}; found {torch.cuda.device_count()} GPU(s).")

    device = torch.device(f"cuda:{gpu_id}")
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_gib = free_bytes / 1024**3
    chosen_mode = memory_mode
    if memory_mode == "auto":
        # The official card reports about 13 GiB for this checkpoint.  Leave a
        # few GiB for reference tokens, VAE decode, and allocator fragmentation.
        # Each additional reference increases the attention sequence length.
        required_free_gib = 17.0 + max(reference_count - 1, 0) * 1.5
        chosen_mode = "cuda" if free_gib >= required_free_gib else "offload"

    print(f"[gpu] {torch.cuda.get_device_name(device)}; free={free_gib:.2f} GiB")
    print(f"[gpu] memory mode: {chosen_mode}; dtype: bfloat16")
    pipe = Flux2KleinPipeline.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.set_progress_bar_config(desc="FLUX.2 Klein")

    if chosen_mode == "offload":
        try:
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        except TypeError:
            # Compatibility with Accelerate/Diffusers versions whose method
            # does not expose gpu_id.
            if gpu_id != 0:
                raise RuntimeError(
                    "This Diffusers version supports offload only on GPU 0."
                )
            pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return pipe, device


def run_inference(args: argparse.Namespace) -> list[Path]:
    import torch

    input_path = Path(args.input).expanduser().resolve()
    normal_path = Path(args.normal).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    model_dir = download_model(args.model_id, Path(args.model_dir), offline=args.offline)

    rgb, source_is_hdr = load_render_rgb(input_path)
    rgb8, exposure = display_map(
        rgb,
        source_is_hdr=source_is_hdr,
        exposure_ev=args.exposure_ev,
        auto_exposure=args.auto_exposure,
        auto_exposure_percentile=args.auto_exposure_percentile,
        target_luminance=args.target_luminance,
        tone_mapper=args.tone_mapper,
    )
    reference = resize_reference(rgb8, args.long_edge)

    normal_rgb8, normal_info = load_normal_rgb8(
        normal_path, encoding=args.normal_encoding
    )
    if normal_rgb8.shape[:2] != rgb8.shape[:2]:
        raise ValueError(
            "The render and normal map must be pixel-aligned and have identical "
            f"dimensions; got render={rgb8.shape[1]}x{rgb8.shape[0]} and "
            f"normal={normal_rgb8.shape[1]}x{normal_rgb8.shape[0]}."
        )
    # Keep the auxiliary normal reference lower-resolution than the primary
    # color reference. Klein has no per-reference weight parameter, so using
    # fewer normal tokens is a practical way to make geometry guidance useful
    # without letting its false colors dominate scene semantics or appearance.
    normal_reference = resize_reference(normal_rgb8, args.normal_long_edge)
    print(
        f"[input] {input_path}\n"
        f"[input] source={rgb.shape[1]}x{rgb.shape[0]}, "
        f"FLUX reference/output={reference.width}x{reference.height}, "
        f"exposure={exposure.exposure_ev:+.3f} EV, tone-map={exposure.tone_mapper}"
    )
    print(
        f"[normal] {normal_path}\n"
        f"[normal] encoding={normal_info.detected_encoding}, "
        f"range=[{normal_info.source_min:.4f}, {normal_info.source_max:.4f}], "
        f"median-length={normal_info.length_before_median:.6f}, "
        f"FLUX auxiliary size={normal_reference.width}x{normal_reference.height}"
    )

    render_preview_path = output_path.with_name(
        f"{output_path.stem}_render_reference.png"
    )
    normal_preview_path = output_path.with_name(
        f"{output_path.stem}_normal_reference.png"
    )
    _save_pil(reference, render_preview_path)
    _save_pil(normal_reference, normal_preview_path)
    print(f"[output] FLUX reference 1 (PBR color): {render_preview_path}")
    print(f"[output] FLUX reference 2 (normal guide): {normal_preview_path}")

    pipe, device = load_pipeline(
        model_dir, args.memory_mode, args.gpu_id, reference_count=2
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    start = time.perf_counter()
    with torch.inference_mode():
        result = pipe(
            # Ordered multi-reference conditioning: the prompt defines the
            # first image as appearance and the second as geometry-only data.
            image=[reference, normal_reference],
            prompt=args.prompt,
            height=reference.height,
            width=reference.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=args.num_images,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        )
    elapsed = time.perf_counter() - start

    output_paths: list[Path] = []
    for index, image in enumerate(result.images):
        if len(result.images) == 1:
            current_path = output_path
        else:
            current_path = output_path.with_name(
                f"{output_path.stem}_{index:02d}{output_path.suffix}"
            )
        _save_pil(image, current_path)
        output_paths.append(current_path)
        print(f"[output] Generated image: {current_path}")

    metadata = {
        "model_id": args.model_id,
        "model_dir": str(model_dir),
        "input": str(input_path),
        "normal_input": str(normal_path),
        "reference_images": [
            {
                "index": 1,
                "role": "primary PBR color/lighting/composition reference",
                "path": str(render_preview_path),
                "width": reference.width,
                "height": reference.height,
            },
            {
                "index": 2,
                "role": "auxiliary surface-normal geometry reference",
                "path": str(normal_preview_path),
                "width": normal_reference.width,
                "height": normal_reference.height,
            },
        ],
        "outputs": [str(path) for path in output_paths],
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "num_images": args.num_images,
        "width": reference.width,
        "height": reference.height,
        "elapsed_seconds": elapsed,
        "exr_to_srgb": asdict(exposure),
        "normal_decoding": asdict(normal_info),
        "versions": _package_versions(),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[output] Metadata: {metadata_path}")
    print(f"[done] {len(output_paths)} image(s) generated in {elapsed:.2f} seconds")
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an aligned PBR color EXR + surface-normal EXR to a "
            "photorealistic image with FLUX.2 Klein Base 4B multi-reference editing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument(
        "--normal",
        type=Path,
        default=DEFAULT_NORMAL_PATH,
        help="Pixel-aligned surface-normal map used as the second FLUX reference.",
    )
    parser.add_argument(
        "--normal-encoding",
        choices=("auto", "signed", "unsigned"),
        default="auto",
        help="Normal storage: signed [-1,1], unsigned [0,1], or infer from range.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument(
        "--long-edge",
        type=int,
        default=1024,
        help="Long edge for the primary color reference and output; area is capped at 1 MP.",
    )
    parser.add_argument(
        "--normal-long-edge",
        type=int,
        default=512,
        help=(
            "Long edge of the auxiliary normal reference. Keeping this below "
            "--long-edge prevents the normal map from dominating appearance."
        ),
    )
    parser.add_argument(
        "--exposure-ev",
        type=float,
        default=None,
        help="Manual EXR exposure. Supplying it disables automatic exposure.",
    )
    parser.add_argument(
        "--auto-exposure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically map the selected luminance percentile to target luminance.",
    )
    parser.add_argument("--auto-exposure-percentile", type=float, default=50.0)
    parser.add_argument("--target-luminance", type=float, default=0.18)
    parser.add_argument("--tone-mapper", choices=("aces", "reinhard", "clip"), default="aces")
    parser.add_argument(
        "--memory-mode",
        choices=("auto", "cuda", "offload"),
        default="auto",
        help="Keep the full model on GPU or enable Accelerate CPU offload.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/check the model and exit without loading the render or GPU pipeline.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Require existing local weights and forbid a Hub download.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.guidance_scale <= 0:
        raise ValueError("--guidance-scale must be positive for the undistilled Base model")
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    if args.normal_long_edge < 16:
        raise ValueError("--normal-long-edge must be at least 16")
    if not 0.0 <= args.auto_exposure_percentile <= 100.0:
        raise ValueError("--auto-exposure-percentile must be in [0, 100]")
    if args.target_luminance <= 0.0:
        raise ValueError("--target-luminance must be positive")
    if not 1 <= args.max_sequence_length <= 512:
        raise ValueError("--max-sequence-length must be in [1, 512]")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
        if args.download_only:
            model_dir = download_model(
                args.model_id, Path(args.model_dir), offline=args.offline
            )
            print(f"[done] Model is ready at {model_dir}")
            return 0
        run_inference(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if os.environ.get("FLUX2_DEBUG") == "1":
            raise
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
